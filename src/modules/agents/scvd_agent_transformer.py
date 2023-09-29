import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from src.utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


class ObsEncoder(nn.Module):
    def __init__(self, input_dim, args):
        super(ObsEncoder, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.hidden_dim = args.rnn_hidden_dim
        self.n_heads = args.n_heads
        self.atten_dim = args.atten_dim
        assert self.atten_dim % self.n_heads == 0, "Attention dim must be divided by n_heads"
        self.head_dim = self.atten_dim // self.n_heads

        self.fc1_state = nn.Sequential(nn.Linear(input_dim, self.hidden_dim), nn.ReLU())
        # q/k/v, hidden_dim --> atten_dim (n_heads * head_dim)
        self.query = nn.Linear(self.hidden_dim, self.atten_dim)
        self.key = nn.Linear(self.hidden_dim, self.atten_dim)
        self.value = nn.Sequential(nn.Linear(self.hidden_dim, self.atten_dim), nn.ReLU())
        self.scale_factor = th.scalar_tensor(self.head_dim).sqrt()
        # rnn
        self.rnn = nn.GRUCell((self.hidden_dim + self.atten_dim), args.rnn_hidden_dim)
        # q_alone
        self.q_alone = nn.Linear((args.rnn_hidden_dim + args.task_rep_dim), self.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.query.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs_interact, obs_mask, scenario_mask, h_interactive, task_z):
        # inputs_interact/transformed state.shape = (bs, n_entities, local_state_size + n_actions), including (local_state + last_actions), padded zero vector for non-agents' last actions
        # obs_mask.shape=(bs, n_entities, n_entities), scenario_mask.shape=(bs, n_entities), h_interactive.shape=(bs, n_agents, rnn_hidden_dim)
        bs, n_entities, edim = inputs_interact.size()
        num_h = self.n_heads
        e = self.head_dim

        embed_inputs = self.fc1_state(inputs_interact)              # (bs, n_entities, hidden_dim)
        embed_inputs_t = embed_inputs.transpose(0, 1)       # (n_entities, bs, hidden_dim)

        queries = self.query(embed_inputs_t).reshape(n_entities, bs * num_h, e).transpose(0, 1)  # (bs*num_h, n_entities, e)
        keys = self.key(embed_inputs_t).reshape(n_entities, bs * num_h, e).permute(1, 2, 0)  # (bs*num_h, e, n_entities)
        values = self.value(embed_inputs_t).reshape(n_entities, bs * num_h, e).transpose(0, 1)  # (bs*num_h, n_entities, e)
        queries = queries[:, :self.n_agents, :]  # (bs*num_h, n_agents, e)
        score = th.bmm(queries, keys) / self.scale_factor  # (bs*num_h, n_agents, n_entities)
        assert score.size() == (bs * num_h, self.n_agents, n_entities)

        # Mask agents themselves，then concatenate the embed_inputs and atten_out as the final inputs of RNN module.
        agent_index = th.eye(self.n_agents, device=self.args.device)
        padded_agent_index = th.zeros((self.n_agents, n_entities), device=self.args.device)
        padded_agent_index[:, :self.n_agents] = agent_index
        score = score.masked_fill(padded_agent_index.unsqueeze(dim=0).expand((bs * num_h), -1, -1).to(th.bool), -float('Inf'))

        # obs_mask.shape=(bs, n_entities, n_entities), unobservable=1, observable=0
        obs_mask = obs_mask[:, :self.n_agents, :].to(th.bool)                      # (bs, n_agents, n_entities)
        obs_mask_rep = obs_mask.repeat_interleave(num_h, dim=0)     # (bs*num_h, n_agents, n_entities)
        score = score.masked_fill(obs_mask_rep, -float('Inf'))                          # mask the weights of un-observable entities's info
        weights = F.softmax(score, dim=-1)  # (bs*num_h, n_agents, n_entities)
        # 对于当前环境中不存在的agent，obs_mask matrix中其相应行元素全部都为1，因为full_obs_mask=torch.ones(...)
        # Some weights might be NaN (if agent is inactive and all entities were masked)
        weights = weights.masked_fill(weights != weights, 0)

        atten_out = th.bmm(weights, values)     # (bs*num_h, n_agents, e)
        atten_out = atten_out.transpose(0, 1).reshape(self.n_agents, bs, num_h * e)
        atten_out = atten_out.transpose(0, 1).reshape(bs * self.n_agents, self.atten_dim)   # (bs*n_agents, atten_dim)
        h_interactive = h_interactive.reshape(-1, self.args.rnn_hidden_dim)     # (bs*n_agents, rnn_hidden_dim)

        agent_embed_inputs = embed_inputs[:, :self.n_agents, :].reshape(-1, self.hidden_dim)  # (bs*n_agents, hidden_dim)
        concat_inps = th.cat([agent_embed_inputs, atten_out], dim=-1)  # (bs*n_agents, (hidden_dim+atten_dim))

        h = self.rnn(concat_inps, h_interactive).reshape(bs, self.n_agents, self.hidden_dim)  # \tau, the output of [self-attention+GRUCell] module

        # scenario_mask.shape=(bs, n_entities), un-existing=1, existing=0
        agent_mask = scenario_mask[:, :self.n_agents].unsqueeze(dim=2).to(th.bool)      # (bs, n_agents, 1)
        h = h.masked_fill(agent_mask, 0)           # 将当前环境中不存在的agent相应的输出全部变为0

        # Calculate the corresponding q-value function
        q_alone_inputs = th.cat([h, task_z], dim=-1)     # (bs, n_agents, hidden_dim+task_rep_dim)

        all_q_alone = self.q_alone(q_alone_inputs)      # (bs, n_agents, n_actions)
        all_q_alone = all_q_alone.masked_fill(agent_mask, 0)

        return all_q_alone, h


class QValueDecoder(nn.Module):
    def __init__(self, args):
        super(QValueDecoder, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.hidden_dim = args.rnn_hidden_dim
        self.n_heads = args.n_heads
        self.atten_dim = args.atten_dim
        assert self.atten_dim % self.n_heads == 0, "Attention dim must be divided by n_heads"
        self.head_dim = self.atten_dim // self.n_heads

        # act_encoder
        self.act_encoder = nn.Sequential(nn.Linear((args.n_actions + 1), self.hidden_dim, bias=False), nn.ReLU())
        # q/k/v, hidden_dim --> atten_dim (n_heads * head_dim)
        self.query = nn.Linear(self.hidden_dim, self.atten_dim)
        self.key = nn.Linear(self.hidden_dim, self.atten_dim)
        self.value = nn.Sequential(nn.Linear(self.hidden_dim, self.atten_dim), nn.ReLU())
        self.scale_factor = th.scalar_tensor(self.head_dim).sqrt()

        # The output layer that calculates the corresponding q value function conditioned on the local history and the preceding agents' actions
        self.q_out = nn.Linear((self.hidden_dim + self.atten_dim + args.task_rep_dim), self.n_actions)   # \tau^{i}+encoder(a^{-i})+task_rep --> q^{i}

        self.mask = th.tril(th.ones(self.n_agents, self.n_agents)).view(1, self.n_agents, self.n_agents).to(args.device)

    def forward(self, h_interactive, onehot_actions, task_z, scenario_mask):
        # h_interactive.shape = (bs, n_agents, hidden_dim), onehot_actions.shape=(bs, n_agents, n_actions),
        # task_z.shape=(bs, n_agents, task_rep_dim), scenario_mask.shape=(bs, n_entities)
        bs, n_agents, n_actions = onehot_actions.size()
        num_h = self.n_heads
        e = self.head_dim

        shifted_action = th.zeros((bs, n_agents, n_actions + 1)).to(self.args.device)
        shifted_action[:, 0, 0] = 1         # 为第一个agent伪造一个action
        shifted_action[:, 1:, 1:] = onehot_actions[:, :-1, :]  # 这里少了最后一个agent的action，意思是最后一个agent是基于前(n-1)个agent的joint_action来进行决策
        act_embeddings = self.act_encoder(shifted_action)  # (bs, n_agents, hidden_dim)
        act_embeddings_t = act_embeddings.transpose(0, 1)   # (n_agents, bs, hidden_dim)

        h_interactive_t = h_interactive.transpose(0, 1)     # (n_agents, bs, hidden_dim)
        queries = self.query(h_interactive_t).reshape(n_agents, bs * num_h, e).transpose(0, 1)        # (bs*num_h, n_agents, e)

        keys = self.key(act_embeddings_t).reshape(n_agents, bs * num_h, e).permute(1, 2, 0)      # (bs*num_h, e, n_agents)
        values = self.value(act_embeddings_t).reshape(n_agents, bs * num_h, e).transpose(0, 1)       # (bs*num_h, n_agents, e)
        score = th.bmm(queries, keys) / self.scale_factor       # (bs*num_h, n_agents, n_agents)
        score = score.masked_fill(self.mask[:, :, :] == 0, -float('Inf'))

        # scenario_mask.shape=(bs, n_entities), un-exist is 1, exist=0
        agent_mask = scenario_mask[:, :self.n_agents]           # (bs, n_agents)
        agent_pair_mask = th.tensor(1, dtype=th.float32, device=self.args.device) - th.bmm(
            (th.tensor(1, dtype=th.float32, device=self.args.device) - agent_mask.to(th.float32)).unsqueeze(dim=2),
            (th.tensor(1, dtype=th.float32, device=self.args.device) - agent_mask.to(th.float32)).unsqueeze(dim=1))  # (bs, n_agents, n_agents)
        agent_pair_mask_rep = agent_pair_mask.repeat_interleave(num_h, dim=0)       # (bs*num_h, n_agents, n_agents)

        score = score.masked_fill(agent_pair_mask_rep.to(th.bool), -float('Inf'))  # mask the contribution of un-existing agents' actions info
        weights = F.softmax(score, dim=-1)  # (bs*num_h, n_agents, n_agents)
        # Some weights might be NaN (if agent is inactive and all entities were masked)
        weights = weights.masked_fill(weights != weights, 0)

        extra_act_embeddings = th.bmm(weights, values)                 # extra_action_embedding, (bs*num_h, n_agents, e)
        extra_act_embeddings = extra_act_embeddings.transpose(0, 1).reshape(n_agents, bs, num_h * e)
        extra_act_embeddings = extra_act_embeddings.transpose(0, 1)     # (bs, n_agents, atten_dim)

        concat_inps = th.cat([h_interactive, extra_act_embeddings, task_z], dim=-1)      # (bs, n_agents, hidden_dim+atten_dim+task_rep_dim)

        all_q = self.q_out(concat_inps)             # (bs, n_agents, n_actions)

        agent_mask = agent_mask.to(th.bool)  # (bs, n_agents)
        all_q[:, 0, :] = th.zeros((bs, n_actions), device=self.args.device)     # The first agent's q_interact has been calculated in q_alone, so mask it
        all_q = all_q.masked_fill(agent_mask.unsqueeze(dim=2), 0)  # 将当前环境中不存在的agent相应的q值输出全部变为0

        # TODO: q^i(\tau^i, i, a^{-i}), this term should equal to zero.
        # First way
        auxiliary_act_embeddings = extra_act_embeddings[:, 0, :].unsqueeze(dim=1).expand(-1, n_agents, -1)      # (bs, n_agents, atten_dim)
        # Second way
        # auxiliary_act_embeddings = th.zeros_like(extra_act_embeddings, device=self.args.device)

        # TODO: new auxiliary_q_inps
        auxiliary_q_inps = th.cat([h_interactive, auxiliary_act_embeddings, task_z], dim=-1)     # (bs, n_agents, hidden_dim+atten_dim+task_rep_dim)
        auxiliary_q = self.q_out(auxiliary_q_inps)      # (bs, n_agents, n_actions), these auxiliary_q should equal to zero.

        return all_q, auxiliary_q

    def act_forward(self, curr_inps, curr_query, shifted_action, agent_mask, task_z, curr_index):
        # curr_query.shape=(bs*num_h, 1, e), agent_mask.shape=(bs, n_agents)
        # curr_inps.shape=(bs, hidden_dim), shifted_action.shape=(bs, n_agents, n_actions+1), task_z.shape=(bs, task_rep_dim)
        bs = shifted_action.size()[0]

        shifted_action = shifted_action.masked_fill(agent_mask.unsqueeze(dim=2), 0)     # 将当前环境中不存在的agent相应的action置0
        act_embeddings = self.act_encoder(shifted_action)           # (bs, n_agents, hidden_dim)
        act_embeddings_t = act_embeddings.transpose(0, 1)           # (n_agents, bs, hidden_dim)

        keys = self.key(act_embeddings_t).reshape(self.n_agents, bs * self.n_heads, self.head_dim).permute(1, 2, 0)     # (bs*num_h, e, n_agents)
        values = self.value(act_embeddings_t).reshape(self.n_agents, bs * self.n_heads, self.head_dim).transpose(0, 1)          # (bs*num_h, n_agents, e)
        score = th.bmm(curr_query, keys) / self.scale_factor        # (bs*num_h, 1, n_agents)
        score = score.masked_fill(self.mask[:, curr_index, :].unsqueeze(dim=1) == 0, -float('Inf'))
        score = score.masked_fill(agent_mask.unsqueeze(dim=1).repeat_interleave(self.n_heads, dim=0), -float('Inf'))

        weights = F.softmax(score, dim=-1)  # (bs*num_h, 1, n_agents)
        weights = weights.masked_fill(weights != weights, 0)
        extra_act_embeddings = th.bmm(weights, values)      # (bs*num_h, 1, e)
        extra_act_embeddings = extra_act_embeddings.transpose(0, 1).reshape(1, bs, self.atten_dim)
        extra_act_embeddings = extra_act_embeddings.transpose(0, 1).squeeze(dim=1)      # (bs, atten_dim)

        concat_inps = th.cat([curr_inps, extra_act_embeddings, task_z], dim=-1)
        curr_q = self.q_out(concat_inps)

        if curr_index == 0:
            curr_q = th.zeros_like(curr_q, device=self.args.device)          # The first agent's q_interact equals to zero.

        return curr_q


class SCVDRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        # input_shape.dim=local_state_size+last actions (padding zero vector for non-agents)
        super(SCVDRNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_heads = args.n_heads
        self.atten_dim = args.atten_dim
        assert self.atten_dim % self.n_heads == 0, "Attention dim must be divided by n_heads"
        self.head_dim = self.atten_dim // self.n_heads

        self.obs_encoder = ObsEncoder(input_shape, args)
        self.q_decoder = QValueDecoder(args)

    def init_hidden(self):
        # make hidden states on same device as model, including the hidden states for both q1 and q2
        hidden_states = self.obs_encoder.init_hidden()
        return hidden_states

    def forward(self, inputs, onehot_actions, task_z, obs_mask, scenario_mask, hidden_states):
        # inputs.shape=(bs, n_entities, local_state_size+n_actions)，包括每个entity的local info以及last action, padded zero last_action for non-agents
        # task_z.shape=(bs, n_agents, task_rep_dim)
        all_q_alone, h = self.obs_encoder(inputs, obs_mask, scenario_mask, hidden_states, task_z)      # \tau, shape=(bs, n_agents, hidden_dim)
        all_q_interact, auxiliary_q = self.q_decoder(h, onehot_actions, task_z, scenario_mask)
        all_q = all_q_alone + all_q_interact
        return all_q, auxiliary_q, all_q_alone, all_q_interact, h

    def autoregressive_act(self, inputs, task_z, obs_mask, scenario_mask, hidden_states, avail_actions, bs_index, t_env, action_selector, test_mode):
        # inputs.shape=(bs, n_entities, local_state_size+n_actions)，包括每个entity的local info以及last action, padded zero last_action for non-agents
        # task_z.shape=(bs, n_agents, task_rep_dim)
        all_q_alone, h = self.obs_encoder(inputs, obs_mask, scenario_mask, hidden_states, task_z)

        bs, n_agents, n_actions = avail_actions.size()
        shifted_action = th.zeros((bs, n_agents, self.args.n_actions + 1), device=self.args.device)
        shifted_action[:, 0, 0] = 1
        output_action = th.zeros((bs, n_agents, 1), dtype=th.long, device=self.args.device)
        # scenario_mask.shape=(bs, n_entities), un-exist is 1, exist=0
        agent_mask = scenario_mask[:, :n_agents].to(th.bool)       # (bs, n_agents)
        h_t = h.transpose(0, 1)     # (n_agents, bs, hidden_dim)
        queries = self.q_decoder.query(h_t).reshape(n_agents, bs * self.n_heads, self.head_dim).transpose(0, 1)       # (bs*num_h, n_agents, e)

        for i in range(n_agents):
            curr_hidden = h[:, i, :]        # (bs, hidden_dim)
            curr_query = queries[:, i, :].unsqueeze(dim=1)      # (bs*num_h, 1, e)
            curr_shifted_action = shifted_action.clone()
            curr_q = self.q_decoder.act_forward(curr_hidden, curr_query, curr_shifted_action, agent_mask, task_z[:, i, :], curr_index=i)
            curr_q += all_q_alone[:, i, :]      # (bs, n_actions)
            chosen_action = action_selector.select_action(curr_q[bs_index], avail_actions[bs_index, i, :], t_env, test_mode=test_mode)
            output_action[bs_index, i, :] = chosen_action.unsqueeze(-1)     # Here only the unterminated envs have available actions.
            if i + 1 < n_agents:
                shifted_action[bs_index, i + 1, 1:] = F.one_hot(chosen_action, num_classes=n_actions).to(th.float32)
        output_action = output_action.masked_fill(agent_mask.unsqueeze(dim=2).to(th.bool), 0)                # 将当前环境中不存在的agent action置0
        output_action = output_action[bs_index, :, :]       # Only return actions for un-terminated env.

        return output_action, h