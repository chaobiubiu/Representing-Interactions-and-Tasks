import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm


# 该mixing network输入为state_embedding，环境中定义的scenario_mask在这里也会产生作用
class AttnMixer(nn.Module):
    def __init__(self, args, abs=True):
        super(AttnMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_entities = args.n_entities
        self.embed_dim = args.mixing_embed_dim
        self.local_state_size = args.local_state_size
        self.state_dim = int(np.prod(args.state_shape))
        if "scvd" in args.name and "rep" not in args.name:
            self.input_dim = args.rnn_hidden_dim + args.task_rep_dim
        else:
            self.input_dim = args.rnn_hidden_dim

        self.abs = abs  # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.embed_dim, 1))

        # # hyper w1 b1
        # self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        # self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))
        #
        # # hyper w2 b2
        # self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(args.hypernet_embed, self.embed_dim))
        # self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
        #                               nn.ReLU(inplace=True),
        #                               nn.Linear(self.embed_dim, 1))

    def forward(self, qvals, embed_inps, scenario_mask):
        # qvals.shape=(bs,(max_seq_length-1),n_agents), embed_inps.shape=(bs, t-1, n_agents, rnn_hidden_dim),
        # scenario_mask.shape=(bs, (max_seq_length-1),n_entities)
        b, t, _ = qvals.size()

        qvals = qvals.reshape(b * t, 1, self.n_agents)
        embed_inps = embed_inps.reshape(b * t, -1, self.input_dim)     # (bs * t, n_agents, rnn_hidden+dim)
        agent_mask = scenario_mask.reshape(b * t, -1)[:, :self.n_agents].unsqueeze(dim=2).to(th.bool)       # (bs * t, n_agents, 1)

        # First layer, w1: (bs*t, n_agents, input_dim) --> (bs*t, n_agents, self.embed_dim)
        w1 = self.hyper_w1(embed_inps)                      # (b * t, n_agents, embed_dim)
        w1 = w1.masked_fill(agent_mask, 0)

        b1 = self.hyper_b1(embed_inps)          # (b * t, n_agents, embed_dim)
        b1 = b1.masked_fill(agent_mask, 0).mean(dim=1).view(-1, 1, self.embed_dim)  # (bs*t, 1, embed_dim)

        # Second layer, w2: (bs*t, n_agents, input_dim) --> (bs*t, n_agents, embed_dim)
        w2 = self.hyper_w2(embed_inps)
        w2 = w2.masked_fill(agent_mask, 0).mean(dim=1).view(-1, self.embed_dim, 1)  # (b * t, emb, 1)

        b2 = self.hyper_b2(embed_inps)            # (bs*t, n_agents, 1)
        b2 = b2.masked_fill(agent_mask, 0).mean(dim=1).view(-1, 1, 1)       # (bs*t, 1, 1)

        if self.abs:  # the monotonic constraint
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1)  # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2  # b * t, 1, 1

        return y.view(b, t, -1)  # shape=(bs, t, 1)

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)