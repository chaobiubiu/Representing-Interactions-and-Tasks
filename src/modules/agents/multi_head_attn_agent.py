import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import torch.nn.init as init
from src.utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


# Multi-head attention, where the outputs of the attention module serve as the inputs of RNN directly.

class ATTNRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        # input_shape=local_state_size+n_actions
        super(ATTNRNNAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        self.hidden_dim = args.rnn_hidden_dim           # Whether this dim is approximate or not?
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

    def forward(self, inputs, obs_mask, scenario_mask, hidden_state):
        # inputs/transformed state.shape = (bs, n_entities, local_state_size + n_actions),
        # which includes (local_state + last_actions), padded zero vector for non-agents' last actions
        # obs_mask.shape=(bs, n_entities, n_entities), scenario_mask.shape=(bs, n_entities)
        # hidden_state.shape=(bs, n_agents, rnn_hidden_dim)
        bs, n_entities, edim = inputs.size()

        num_h = self.n_heads
        e = self.head_dim
        embed_inputs = self.fc1(inputs)  # (bs, n_entities, hidden_dim)
        embed_inputs_t = embed_inputs.transpose(0, 1)       # (n_entities, bs, hidden_dim)

        queries = self.query(embed_inputs_t).reshape(n_entities, bs * num_h, e).transpose(0, 1)     # (bs*num_h, n_entities, e)
        keys = self.key(embed_inputs_t).reshape(n_entities, bs * num_h, e).permute(1, 2, 0)     # (bs*num_h, e, n_entities)
        values = self.value(embed_inputs_t).reshape(n_entities, bs * num_h, e).transpose(0, 1)  # (bs*num_h, n_entities, e)

        queries = queries[:, :self.n_agents, :]     # (bs*num_h, n_agents, e)
        score = th.bmm(queries, keys) / self.scale_factor       # (bs*num_h, n_agents, n_entities)
        assert score.size() == (bs * num_h, self.n_agents, n_entities)

        # obs_mask.shape=(bs, n_entities, n_entities), unobservable=1, observable=0
        obs_mask = obs_mask[:, :self.n_agents, :].to(th.bool)  # (bs, n_agents, n_entities)
        obs_mask_rep = obs_mask.repeat_interleave(num_h, dim=0)     # (bs*num_h, n_agents, n_entities)
        score = score.masked_fill(obs_mask_rep, -float('Inf'))      # Mask the weights of un-observable entities's info
        weights = F.softmax(score, dim=-1)      # (bs*num_h, n_agents, n_entities)
        # 对于当前环境中不存在的agent，obs_mask matrix中其相应行元素全部都为1，因为full_obs_mask=torch.ones(...)
        # Some weights might be NaN (if agent is inactive and all entities were masked)
        weights = weights.masked_fill(weights != weights, 0)        # (bs*num_h, n_agents, n_entities)

        atten_out = th.bmm(weights, values)     # (bs*num_h, n_agents, e)
        atten_out = atten_out.transpose(0, 1).reshape(self.n_agents, bs, num_h * e)     # (n_agents, bs, atten_dim)
        atten_out = atten_out.transpose(0, 1).reshape(-1, self.atten_dim)       # (bs*n_agents, atten_dim)
        hidden_state = hidden_state.reshape(-1, self.args.rnn_hidden_dim)       # (bs*n_agents, rnn_hidden_dim)

        h = self.rnn(atten_out, hidden_state).reshape(bs, self.n_agents, self.args.rnn_hidden_dim)  # \tau, the output of [self-attention+GRUCell]

        # scenario_mask.shape=(bs, n_entities), un-existing=1, existing=0
        agent_mask = scenario_mask[:, :self.n_agents].unsqueeze(dim=2).to(th.bool)      # (bs, n_agents, 1)
        h = h.masked_fill(agent_mask, 0)

        # output the final q value
        q = self.out(h)       # (bs, n_agents, n_actions)

        q = q.masked_fill(agent_mask, 0)

        return q, h