import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init

# class SelfAttention(nn.Module):
#     def __init__(self, args, hidden_dim):
#         super(SelfAttention, self).__init__()
#         self.hidden_dim = hidden_dim
#         # q/k/v
#         self.query = nn.Linear(hidden_dim, hidden_dim)
#         self.key = nn.Linear(hidden_dim, hidden_dim)
#         self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
#         self.scale_factor = th.scalar_tensor(hidden_dim // 1).sqrt()
#         # project layer
#         self.proj = nn.Linear(hidden_dim, hidden_dim)
#
#     def forward(self, embed_inputs):
#         # embed_inputs.shape = (bs*n_agents, n_entities+1, hidden_dim),
#         bs_na, n_entities_plus_one, edim = embed_inputs.size()
#
#         queries = self.query(embed_inputs).reshape(bs_na, n_entities_plus_one, -1)
#         keys = self.key(embed_inputs).reshape(bs_na, n_entities_plus_one, -1).permute(0, 2, 1)
#         values = self.value(embed_inputs).reshape(bs_na, n_entities_plus_one, -1)
#         score = th.bmm(queries, keys) / self.scale_factor  # (bs_na, n_entities_plus_one, n_entities_plus_one)
#
#         weights = F.softmax(score, dim=-1)  # (bs_na, n_entities_plus_one, n_entities_plus_one)
#         atten_out = th.bmm(weights, values)     # (bs_na, n_entities_plus_one, hidden_dim)
#
#         y = self.proj(atten_out)        # (bs_na, n_entities_plus_one, hidden_dim)
#         return y


class SelfAttention(nn.Module):
    def __init__(self, args, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = args.n_heads
        self.atten_dim = args.atten_dim
        assert self.atten_dim % self.n_heads == 0, "Attention dim must be divided by n_heads"
        self.head_dim = self.atten_dim // self.n_heads

        # q/k/v, hidden_dim --> atten_dim (n_heads * head_dim)
        self.query = nn.Linear(hidden_dim, self.atten_dim)
        self.key = nn.Linear(hidden_dim, self.atten_dim)
        self.value = nn.Sequential(nn.Linear(hidden_dim, self.atten_dim), nn.ReLU())
        self.scale_factor = th.scalar_tensor(self.head_dim).sqrt()
        # project layer
        self.proj = nn.Linear(self.atten_dim, hidden_dim)

    def forward(self, embed_inputs):
        # embed_inputs.shape = (bs*n_agents, n_entities+1, hidden_dim),
        bs_na, n_entities_plus_one, edim = embed_inputs.size()

        num_h = self.n_heads
        e = self.head_dim
        embed_inputs_t = embed_inputs.transpose(0, 1)   # (n_entities_plus_one, bs_na, edim)

        queries = self.query(embed_inputs_t).reshape(n_entities_plus_one, bs_na * num_h, e).transpose(0, 1)     # (bs_na*num_h, n_entities_plus_one, e)
        keys = self.key(embed_inputs_t).reshape(n_entities_plus_one, bs_na * num_h, e).permute(1, 2, 0) # (bs_na*num_h, e, n_entities_plus_one)
        values = self.value(embed_inputs_t).reshape(n_entities_plus_one, bs_na * num_h, e).transpose(0, 1)  # (bs_na*num_h, n_entities_plus_one, e)

        score = th.bmm(queries, keys) / self.scale_factor  # (bs_na*num_h, n_entities_plus_one, n_entities_plus_one)
        assert score.size() == (bs_na * num_h, n_entities_plus_one, n_entities_plus_one)

        weights = F.softmax(score, dim=-1)  # (bs_na*num_h, n_entities_plus_one, n_entities_plus_one)
        atten_out = th.bmm(weights, values)  # (bs_na*num_h, n_entities_plus_one, e)
        atten_out = atten_out.transpose(0, 1).reshape(n_entities_plus_one, bs_na, num_h * e)
        atten_out = atten_out.transpose(0, 1)   # (bs_na, n_entities_plus_one, atten_dim)

        y = self.proj(atten_out)  # (bs_na, n_entities_plus_one, hidden_dim)
        return y


class EncoderBlock(nn.Module):
    def __init__(self, args, hidden_dim):
        super(EncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        # Self attention
        self.attention = SelfAttention(args, hidden_dim)
        # mlp
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, 1 * hidden_dim), nn.ReLU(),
                                 nn.Linear(1 * hidden_dim, hidden_dim))

    def forward(self, embed_inputs):
        # embed_inputs.shape=(bs_na, n_entities+1, hidden_dim)
        x1 = self.ln1(embed_inputs + self.attention(embed_inputs))
        x2 = self.ln2(x1 + self.mlp(x1))
        return x2


class Transformer(nn.Module):
    def __init__(self, args, input_size, hidden_dim, n_blocks, output_size):
        super(Transformer, self).__init__()
        self.n_agents = args.n_agents
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_dim), nn.ReLU())
        # self.encoders = nn.Sequential(*[EncoderBlock(hidden_dim) for _ in range(n_blocks)])
        self.encoders = nn.ModuleList()
        for i in range(n_blocks):
            self.encoders.append(EncoderBlock(args, hidden_dim))
        self.q_out = nn.Linear(hidden_dim, output_size)

    def forward(self, obs_inputs, hidden_states, obs_mask, scenario_mask):
        # obs_inputs.shape=(bs*n_agents, n_entities, local_state_shape+n_actions), hidden_states.shape=(bs*n_agents, 1, rnn_hidden_dim)
        embed_inputs = self.fc1(obs_inputs)       # (bs*n_agents, n_entities, local_state_size+last_actions) --> (bs*n_agents, n_entities, hidden_dim)
        bs_na, n_entities, edim = embed_inputs.size()
        bs = bs_na // self.n_agents
        embed_inputs = th.cat([embed_inputs, hidden_states], dim=1)  # (bs_na, n_entities+1, hidden_dim)

        # Here we don't use obs_mask and scenario_mask owing to the obs inputs.
        x = embed_inputs        # (bs_na, n_entities+1, hidden_dim)
        for encoder in self.encoders:
            x = encoder(x)
        # the outputs of EncoderBlocks, shape=(bs_na, n_entities+1, hidden_dim)
        h = x[:, -1, :].view(bs, self.n_agents, -1)     # the last one in dim 1 is the pseudo-hidden-states in UpDeT
        q_inputs = x[:, :-1, :].view(bs, self.n_agents, n_entities, -1)     # (bs, n_agents, n_entities, hidden_dim)
        all_q = []
        for i in range(self.n_agents):
            curr_inp = q_inputs[:, i, i, :]     # Each agent uses its own aggregated features to calculate their Q-values.
            q_i = self.q_out(curr_inp)
            all_q.append(q_i)
        all_q = th.stack(all_q, dim=1)        # (bs, n_agents, n_actions)

        # scenario_mask.shape=(bs, n_entities), un-existing=1, existing=0
        agent_mask = scenario_mask[:, :self.n_agents].unsqueeze(dim=2).to(th.bool)  # (bs, n_agents, 1)
        all_q = all_q.masked_fill(agent_mask, 0)

        h = h.masked_fill(agent_mask, 0)        # (bs, n_agents, hidden_dim)

        return all_q, h


class UPDETAgent(nn.Module):
    def __init__(self, input_shape, args):
        # input_shape=local_state_size+n_actions
        super(UPDETAgent, self).__init__()
        self.args = args
        self.device = args.device
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.n_blocks = args.n_blocks
        self.hidden_dim = args.rnn_hidden_dim           # Whether this dim is approximate or not?
        self.transformer = Transformer(args, input_shape, self.hidden_dim, self.n_blocks, self.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.transformer.q_out.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, obs_mask, scenario_mask, hidden_state):
        # inputs.shape = (bs*n_agents, n_entities, local_state_size + n_actions),
        # which includes (local_state + last_actions), padded zero vector for non-agents' last actions
        # obs_mask.shape=(bs, n_entities, n_entities), scenario_mask.shape=(bs, n_entities)
        # h_interactive.shape=(bs*n_agents, 1, rnn_hidden_dim)
        q, h = self.transformer(inputs, hidden_state, obs_mask, scenario_mask)

        return q, h