import torch as th
import torch.nn as nn
import torch.nn.functional as F


# class StateInpEncoder(nn.Module):
#     def __init__(self, args):
#         super(StateInpEncoder, self).__init__()
#         self.args = args
#         self.n_agents = args.n_agents
#         self.n_entities = args.n_entities
#         self.local_state_size = args.local_state_size
#         input_dim = args.local_state_size
#         if args.obs_last_action:
#             input_dim += args.n_actions
#         # fc_states is used to encoder each entity's info(local_state+last_action)
#         self.fc_states = nn.Sequential(nn.Linear(input_dim, args.rnn_hidden_dim), nn.ReLU())
#         # q/k/v
#         self.query = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.key = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.value = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim), nn.ReLU())
#         self.scale_factor = th.scalar_tensor(args.rnn_hidden_dim // 1).sqrt()
#         self.out = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
#
#     def forward(self, inputs, scenario_masks):
#         # inputs / transformed states.shape=(bs, t, n_entities, (local_state_size+last_action))
#         # scenario_masks.shape=(bs, t, n_entities)
#         bs, t, n_entities, e_dim = inputs.size()
#         inputs = inputs.reshape(bs * t, n_entities, -1)
#
#         embed_inputs = self.fc_states(inputs)       # (bs_t, n_entities, rnn_hidden_dim)
#
#         scenario_masks = scenario_masks.reshape(bs * t, self.n_entities)      # (bs_t, n_entities), existing is 1, otherwise is 0.
#         agent_masks = scenario_masks[:, :self.n_agents]                              # (bs_t, n_agents)
#         # create attn_mask from scenario_mask, th.bmm((bs_t, n_agents, 1), (bs_t, 1, n_entities)) = (bs_t, n_agents, n_entities)
#         atten_mask = th.tensor(1, dtype=th.float32, device=self.args.device) - th.bmm((th.tensor(1, dtype=th.float32, device=self.args.device) - agent_masks.to(th.float32)).unsqueeze(dim=2),
#                                                                                       (th.tensor(1, dtype=th.float32, device=self.args.device) - scenario_masks.to(th.float32)).unsqueeze(dim=1))
#         queries = self.query(embed_inputs).reshape(bs * t, self.n_entities, -1)  # (bs_t, n_entities, hidden_dim)
#         keys = self.key(embed_inputs).reshape(bs * t, self.n_entities, -1).permute(0, 2, 1)  # (bs_t, hidden_dim, n_entities)
#         values = self.value(embed_inputs).reshape(bs * t, self.n_entities, -1)  # (bs_t, n_entities, hidden_dim)
#         queries = queries[:, :self.n_agents, :]  # (bs_t, n_agents, hidden_dim)
#         score = th.bmm(queries, keys) / self.scale_factor  # (bs, n_agents, n_entities)
#
#         # attn_mask.shape=(bs_t, n_agents, n_entities), un-existing=1, existing=0
#         score = score.masked_fill(atten_mask.to(th.bool), -float('Inf'))  # mask the weights of un-existing entities's info
#         weights = F.softmax(score, dim=-1)  # (bs, n_agents, n_entities)
#         # Some weights might be NaN (if agent is inactive and all entities were masked)
#         weights = weights.masked_fill(weights != weights, 0)
#
#         atten_out = th.bmm(weights, values)  # (bs_t, n_agents, hidden_dim)
#         outputs = self.out(atten_out)       # (bs_t, n_agents, hidden_dim)
#         outputs = outputs.masked_fill(agent_masks.unsqueeze(dim=2).to(th.bool), 0).reshape(bs, t, self.n_agents, -1)    # (bs, t, n_agents, hidden_dim)
#         # outputs = outputs.mean(dim=2)       # (bs, t, rnn_hidden_dim)
#
#         return outputs


# StateInpEncoder with multi-head attention
class MultiHeadStateInpEncoder(nn.Module):
    def __init__(self, args):
        super(MultiHeadStateInpEncoder, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_entities = args.n_entities
        self.local_state_size = args.local_state_size
        input_dim = args.local_state_size
        if args.obs_last_action:
            input_dim += args.n_actions

        self.n_heads = args.n_heads
        self.atten_dim = args.atten_dim
        assert self.atten_dim % self.n_heads == 0, "Attention dim must be divided by n_heads"
        self.head_dim = self.atten_dim // self.n_heads

        # fc_states is used to encoder each entity's info(local_state+last_action)
        self.fc_states = nn.Sequential(nn.Linear(input_dim, args.rnn_hidden_dim), nn.ReLU())
        # q/k/v
        self.query = nn.Linear(args.rnn_hidden_dim, self.atten_dim)
        self.key = nn.Linear(args.rnn_hidden_dim, self.atten_dim)
        self.value = nn.Sequential(nn.Linear(args.rnn_hidden_dim, self.atten_dim), nn.ReLU())
        self.scale_factor = th.scalar_tensor(self.head_dim).sqrt()
        self.out = nn.Linear(self.atten_dim, args.rnn_hidden_dim)

    def forward(self, inputs, scenario_masks):
        # inputs / transformed states.shape=(bs, t, n_entities, (local_state_size+last_action))
        # scenario_masks.shape=(bs, t, n_entities)
        bs, t, n_entities, e_dim = inputs.size()
        inputs = inputs.reshape(bs * t, n_entities, -1)     # (bs_t, n_entities, edim)

        num_h = self.n_heads
        e = self.head_dim
        embed_inputs = self.fc_states(inputs)       # (bs_t, n_entities, rnn_hidden_dim)
        embed_inputs_t = embed_inputs.transpose(0, 1)   # (n_entities, bs_t, rnn_hidden_dim)

        scenario_masks = scenario_masks.reshape(bs * t, self.n_entities)      # (bs_t, n_entities), existing is 1, otherwise is 0.
        agent_masks = scenario_masks[:, :self.n_agents]                              # (bs_t, n_agents)
        # create attn_mask from scenario_mask, th.bmm((bs_t, n_agents, 1), (bs_t, 1, n_entities)) = (bs_t, n_agents, n_entities)
        atten_mask = th.tensor(1, dtype=th.float32, device=self.args.device) - th.bmm((th.tensor(1, dtype=th.float32, device=self.args.device) - agent_masks.to(th.float32)).unsqueeze(dim=2),
                                                                                      (th.tensor(1, dtype=th.float32, device=self.args.device) - scenario_masks.to(th.float32)).unsqueeze(dim=1))

        queries = self.query(embed_inputs_t).reshape(self.n_entities, bs * t * num_h,  e).transpose(0, 1)   # (bs_t*num_h, n_entities, e)
        keys = self.key(embed_inputs_t).reshape(self.n_entities, bs * t * num_h,  e).permute(1, 2, 0)  # (bs_t*num_h, e, n_entities)
        values = self.value(embed_inputs_t).reshape(self.n_entities, bs * t * num_h,  e).transpose(0, 1)  # (bs_t*num_h, n_entities, e)

        queries = queries[:, :self.n_agents, :]  # (bs_t*num_h, n_agents, e)
        score = th.bmm(queries, keys) / self.scale_factor  # (bs_t*num_h, n_agents, n_entities)
        assert score.size() == (bs * t * num_h, self.n_agents, self.n_entities)

        # attn_mask.shape=(bs_t, n_agents, n_entities), un-existing=1, existing=0
        atten_mask_rep = atten_mask.repeat_interleave(num_h, dim=0)     # shape=(bs_t*num_h, n_agents, n_entities)
        score = score.masked_fill(atten_mask_rep.to(th.bool), -float('Inf'))  # mask the weights of un-existing entities's info
        weights = F.softmax(score, dim=-1)  # (bs_t*num_h, n_agents, n_entities)
        # Some weights might be NaN (if agent is inactive and all entities were masked)
        weights = weights.masked_fill(weights != weights, 0)    # (bs_t*num_h, n_agents, n_entities)

        atten_out = th.bmm(weights, values)  # (bs_t*num_h, n_agents, e)
        atten_out = atten_out.transpose(0, 1).reshape(self.n_agents, bs * t, num_h * e)     # (n_agents, bs_t, num_h*e)
        atten_out = atten_out.transpose(0, 1)       # (bs_t, n_agents, atten_dim)

        outputs = self.out(atten_out)       # (bs_t, n_agents, rnn_hidden_dim)
        outputs = outputs.masked_fill(agent_masks.unsqueeze(dim=2).to(th.bool), 0).reshape(bs, t, self.n_agents, -1)    # (bs, t, n_agents, rnn_hidden_dim)

        return outputs