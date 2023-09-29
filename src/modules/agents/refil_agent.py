import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from src.utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class REFILAgent(nn.Module):
    def __init__(self, input_shape, args):
        # input_shape=local_state_size+n_actions
        super(REFILAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.hidden_dim = args.rnn_hidden_dim  # Whether this dim is approximate or not?
        self.n_heads = args.n_heads
        self.atten_dim = args.atten_dim
        assert self.atten_dim % self.n_heads == 0, "Attention dim must be divided by n_heads"
        self.head_dim = self.atten_dim // self.n_heads

        self.fc1 = nn.Sequential(nn.Linear(input_shape, self.hidden_dim), nn.ReLU())
        # q/k/v, hidden_dim --> atten_dim (n_heads * head_dim)
        self.query = nn.Linear(self.hidden_dim, self.atten_dim)
        self.key = nn.Linear(self.hidden_dim, self.atten_dim)
        self.value = nn.Sequential(nn.Linear(self.hidden_dim, self.atten_dim), nn.ReLU())
        self.scale_factor = th.scalar_tensor(self.head_dim).sqrt()
        # rnn
        self.rnn = nn.GRUCell(self.atten_dim, args.rnn_hidden_dim)
        # output_layer
        self.out = nn.Linear(args.rnn_hidden_dim, self.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.out.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def logical_not(self, inp):
        return 1 - inp

    def logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1        # both True return True, True or False return True
        return out

    # Calculate the mask that masks the un-existing elements in the score of the attention module
    def entitymask2attnmask(self, entity_mask):
        # entity_mask.shape=(bs, 1, n_entities), un-existing is 1, existing is 0
        bs, seq_t, ne = entity_mask.size()
        in1 = (th.tensor(1, dtype=th.float32, device=self.args.device) - entity_mask.to(th.float)).reshape(bs * seq_t, ne, 1)
        in2 = (th.tensor(1, dtype=th.float32, device=self.args.device) - entity_mask.to(th.float)).reshape(bs * seq_t, 1, ne)
        attn_mask = th.tensor(1, dtype=th.float32, device=self.args.device) - th.bmm(in1, in2)  # shape=(bs*seq_t, n_entities, n_entities)
        return attn_mask.reshape(bs, seq_t, ne, ne).to(th.uint8)

    def forward(self, inputs, obs_mask, scenario_mask, hidden_state, imagine=False):
        # inputs/transformed state.shape = (bs, seq_t, n_entities, local_state_size + n_actions),
        # which includes (local_state + last_actions), padded zero vector for non-agents' last actions
        # obs_mask.shape=(bs, seq_t, n_entities, n_entities), scenario_mask.shape=(bs, seq_t, n_entities)
        # h_interactive.shape=(bs, n_agents, rnn_hidden_dim)
        bs, seq_t, n_entities, edim = inputs.size()
        num_h = self.n_heads
        e = self.head_dim

        if not imagine:
            inputs = inputs.reshape(bs * seq_t, n_entities, -1)
            obs_mask = obs_mask.reshape(bs * seq_t, n_entities, n_entities)
            scenario_mask = scenario_mask.reshape(bs * seq_t, n_entities)

            embed_inputs = self.fc1(inputs)  # (bs*seq_t, n_entities, hidden_dim)
            embed_inputs_t = embed_inputs.transpose(0, 1)  # (n_entities, bs*seq_t, hidden_dim)

            queries = self.query(embed_inputs_t).reshape(n_entities, bs * seq_t * num_h, e).transpose(0, 1)  # (bs*seq_t*num_h, n_entities, e)
            keys = self.key(embed_inputs_t).reshape(n_entities, bs * seq_t * num_h, e).permute(1, 2, 0)  # (bs*seq_t*num_h, e, n_entities)
            values = self.value(embed_inputs_t).reshape(n_entities, bs * seq_t * num_h, e).transpose(0, 1)  # (bs*seq_t*num_h, n_entities, e)

            queries = queries[:, :self.n_agents, :]  # (bs*seq_t*num_h, n_agents, e)
            score = th.bmm(queries, keys) / self.scale_factor  # (bs*seq_t*num_h, n_agents, n_entities)
            assert score.size() == (bs * seq_t * num_h, self.n_agents, n_entities)

            # obs_mask.shape=(bs*seq_t, n_entities, n_entities), unobservable=1, observable=0
            obs_mask = obs_mask[:, :self.n_agents, :].to(th.bool)  # (bs*seq_t, n_agents, n_entities)
            obs_mask_rep = obs_mask.repeat_interleave(num_h, dim=0)     # (bs*seq_t*num_h, n_agents, n_entities)
            score = score.masked_fill(obs_mask_rep, -float('Inf'))  # mask the weights of un-observable entities's info
            weights = F.softmax(score, dim=-1)  # (bs*seq_t*num_h, n_agents, n_entities)
            # 对于当前环境中不存在的agent，obs_mask matrix中其相应行元素全部都为1，因为full_obs_mask=torch.ones(...)
            # Some weights might be NaN (if agent is inactive and all entities were masked)
            weights = weights.masked_fill(weights != weights, 0)

            agent_mask = scenario_mask[:, :self.n_agents]  # (bs*seq_t, n_agents)

            atten_out = th.bmm(weights, values)  # (bs*seq_t*num_h, n_agents, e)
            atten_out = atten_out.transpose(0, 1).reshape(self.n_agents, bs * seq_t, num_h * e)     # (n_agents, bs*seq_t, num_h*e)
            atten_out = atten_out.transpose(0, 1)       # (bs*seq_t, n_agents, atten_dim)

            atten_out = atten_out.masked_fill(agent_mask.unsqueeze(dim=2).to(th.bool), 0)  # Mask the un-existing agents

            atten_out = atten_out.reshape(bs, seq_t, self.n_agents, -1)  # (bs, seq_t, n_agents, atten_dim)
            hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)  # (bs * n_agents, hidden_dim)

            hs = []
            for t in range(seq_t):
                curr_inp = atten_out[:, t, :, :].reshape(-1, self.atten_dim)  # (bs*n_agents, atten_dim)
                hidden_state = self.rnn(curr_inp, hidden_state)
                hs.append(hidden_state.reshape(bs, self.n_agents, -1))
            hs = th.stack(hs, dim=1)  # (bs, seq_t, n_agents, hidden_dim)
            q = self.out(hs)  # (bs, seq_t, n_agents, n_actions)

            # Mask the un-existing agents' information
            agent_mask = agent_mask.reshape(bs, seq_t, self.n_agents).unsqueeze(dim=3).to(th.bool)
            hs = hs.masked_fill(agent_mask, 0)  # shape=(bs, seq_t, n_agents, hidden_dim)
            q = q.masked_fill(agent_mask, 0)  # shape=(bs, seq_t, n_agents, n_actions)

            return q, hs

        # Create random split of entities (once per episode), so should process the whole episode at a time.
        groupA_probs = th.rand(bs, 1, 1, device=self.args.device).repeat(1, 1, n_entities)      # (bs, 1, n_entities)

        groupA = th.bernoulli(groupA_probs).to(th.uint8)
        groupB = self.logical_not(groupA)
        # Mask out entities not present in env. Here we are confused, why don't use logical_and ?
        groupA = self.logical_or(groupA, scenario_mask[:, [0]])     # scenario_mask[:, [0]].shape=(bs, 1, n_entities)
        groupB = self.logical_or(groupB, scenario_mask[:, [0]])

        # Covert group entity mask to attention mask, shape=(bs, 1, n_entities, n_entities)
        groupA2attnmask = self.entitymask2attnmask(groupA)      # This means that only agents in the same group can observe each other.
        groupB2attnmask = self.entitymask2attnmask(groupB)

        # Create attention_mask for interactions between groups
        interact_attnmask = self.logical_or(self.logical_not(groupA2attnmask), self.logical_not(groupB2attnmask))       # All agents can observe each other.

        # Get within group attention mask
        within_attnmask = self.logical_not(interact_attnmask)

        active_attnmask = self.entitymask2attnmask(scenario_mask[:, [0]])       # (bs, 1, n_entities, n_entities)
        # Get masks to use for mixer (no obs_mask but mask out un-used entities)
        Wattnmask_noobs = self.logical_or(within_attnmask, active_attnmask)
        Iattnmask_noobs = self.logical_or(interact_attnmask, active_attnmask)
        # Get masks to use for mac. Mask out agents that are not observable (also expands time dim due to the shape of obs_mask)
        within_attnmask = self.logical_or(within_attnmask, obs_mask)
        interact_attnmask = self.logical_or(interact_attnmask, obs_mask)

        # Here inputs expand the first dim (including vanilla, within_group, interact_group)
        inputs = inputs.repeat(3, 1, 1, 1)      # (bs*3, seq_t, n_entities, local_state_size+n_actions)
        obs_mask = th.cat([obs_mask, within_attnmask, interact_attnmask], dim=0)    # (bs*3, seq_t, n_entities, n_entities)
        scenario_mask = scenario_mask.repeat(3, 1, 1)       # (bs*3, seq_t, n_entities)

        hidden_state = hidden_state.repeat(3, 1, 1)     # (bs*3, n_agents, hidden_dim)

        new_bs, seq_t, n_entities, edim = inputs.size()     # new_bs = bs * 3
        inputs = inputs.reshape(new_bs * seq_t, n_entities, -1)
        obs_mask = obs_mask.reshape(new_bs * seq_t, n_entities, n_entities)
        scenario_mask = scenario_mask.reshape(new_bs * seq_t, n_entities)
        agent_mask = scenario_mask[:, :self.n_agents]   # (new_bs*seq_t, n_agents)

        embed_inputs = self.fc1(inputs)     # (new_bs*seq_t, n_entities, hidden_dim)
        embed_inputs_t = embed_inputs.transpose(0, 1)  # (n_entities, new_bs*seq_t, hidden_dim)

        queries = self.query(embed_inputs_t).reshape(n_entities, new_bs * seq_t * num_h, e).transpose(0, 1)  # (new_bs*seq_t*num_h, n_entities, e)
        keys = self.key(embed_inputs_t).reshape(n_entities, new_bs * seq_t * num_h, e).permute(1, 2, 0)  # (new_bs*seq_t*num_h, e, n_entities)
        values = self.value(embed_inputs_t).reshape(n_entities, new_bs * seq_t * num_h, e).transpose(0, 1)  # (new_bs*seq_t*num_h, n_entities, e)

        queries = queries[:, :self.n_agents, :]  # (new_bs*seq_t*num_h, n_agents, e)
        score = th.bmm(queries, keys) / self.scale_factor  # (new_bs*seq_t*num_h, n_agents, n_entities)
        assert score.size() == (new_bs * seq_t * num_h, self.n_agents, n_entities)

        # obs_mask.shape=(new_bs*seq_t, n_entities, n_entities), unobservable=1, observable=0
        obs_mask = obs_mask[:, :self.n_agents, :].to(th.bool)  # (new_bs*seq_t, n_agents, n_entities)
        obs_mask_rep = obs_mask.repeat_interleave(num_h, dim=0)  # (new_bs*seq_t*num_h, n_agents, n_entities)
        score = score.masked_fill(obs_mask_rep, -float('Inf'))  # mask the weights of un-observable entities' info
        weights = F.softmax(score, dim=-1)  # (new_bs*seq_t*num_h, n_agents, n_entities)
        # 对于当前环境中不存在的agent，obs_mask matrix中其相应行元素全部都为1，因为full_obs_mask=torch.ones(...)
        # Some weights might be NaN (if agent is inactive and all entities were masked)
        weights = weights.masked_fill(weights != weights, 0)

        atten_out = th.bmm(weights, values)  # (new_bs*seq_t*num_h, n_agents, e)
        atten_out = atten_out.transpose(0, 1).reshape(self.n_agents, new_bs * seq_t, num_h * e)  # (n_agents, new_bs*seq_t, num_h*e)
        atten_out = atten_out.transpose(0, 1)  # (new_bs*seq_t, n_agents, atten_dim)

        atten_out = atten_out.masked_fill(agent_mask.unsqueeze(dim=2).to(th.bool), 0)  # Mask the un-existing agents

        atten_out = atten_out.reshape(new_bs, seq_t, self.n_agents, -1)     # (new_bs, seq_t, n_agents, atten_dim)
        hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)   # (new_bs * n_agents, hidden_dim)

        hs = []
        for t in range(seq_t):
            curr_inp = atten_out[:, t, :, :].reshape(-1, self.atten_dim)       # (new_bs*n_agents, atten_dim)
            hidden_state = self.rnn(curr_inp, hidden_state)
            hs.append(hidden_state.reshape(new_bs, self.n_agents, -1))
        hs = th.stack(hs, dim=1)        # (new_bs, seq_t, n_agents, hidden_dim)
        q = self.out(hs)        # (new_bs, seq_t, n_agents, n_actions)

        # Mask the un-existing agents' information
        agent_mask = agent_mask.reshape(new_bs, seq_t, self.n_agents).unsqueeze(dim=3).to(th.bool)
        hs = hs.masked_fill(agent_mask, 0)      # shape=(new_bs, seq_t, n_agents, hidden_dim)
        q = q.masked_fill(agent_mask, 0)        # shape=(new_bs, seq_t, n_agents, n_actions)

        return q, hs, (Wattnmask_noobs.repeat(1, seq_t, 1, 1), Iattnmask_noobs.repeat(1, seq_t, 1, 1))