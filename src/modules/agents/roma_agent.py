import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch as th
import numpy as np
import torch.nn.init as init
from src.utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm
from torch.distributions import kl_divergence


class ROMAAgent(nn.Module):
    def __init__(self, input_shape, args):
        # input_shape=local_state_size+n_actions
        super(ROMAAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.rnn_hidden_dim           # Whether this dim is approximate or not?
        self.n_heads = args.n_heads
        self.atten_dim = args.atten_dim
        assert self.atten_dim % self.n_heads == 0, "Attention dim must be divided by n_heads"
        self.head_dim = self.atten_dim // self.n_heads
        self.bs = 0

        NN_HIDDEN_SIZE = args.NN_HIDDEN_SIZE
        activation_func = nn.LeakyReLU()

        # Separate attention module for role embedding.
        self.role_fc1 = nn.Sequential(nn.Linear(input_shape, self.hidden_dim), nn.ReLU())
        # q/k/v, hidden_dim --> atten_dim (n_heads * head_dim)
        self.role_query = nn.Linear(self.hidden_dim, self.atten_dim)
        self.role_key = nn.Linear(self.hidden_dim, self.atten_dim)
        self.role_value = nn.Sequential(nn.Linear(self.hidden_dim, self.atten_dim), nn.ReLU())
        self.role_scale_factor = th.scalar_tensor(self.head_dim).sqrt()

        self.embed_net = nn.Sequential(nn.Linear(self.atten_dim, NN_HIDDEN_SIZE),
                                       nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                       activation_func,
                                       nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))

        self.inference_net = nn.Sequential(nn.Linear((args.rnn_hidden_dim + self.atten_dim), NN_HIDDEN_SIZE),    # \tau^i+role_attention_out-->role
                                           nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                           activation_func,
                                           nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2))

        self.latent = th.rand(self.n_agents, self.latent_dim * 2)       # (n, mu+var)
        self.latent_infer = th.rand(self.n_agents, self.latent_dim * 2)     # (n, mu+var)

        self.latent_net = nn.Sequential(nn.Linear(args.latent_dim, NN_HIDDEN_SIZE), nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                        activation_func)        # A simple transformation

        self.fc1 = nn.Sequential(nn.Linear(input_shape, self.hidden_dim), nn.ReLU())

        # q/k/v, hidden_dim --> atten_dim (n_heads * head_dim)
        self.query = nn.Linear(self.hidden_dim, self.atten_dim)
        self.key = nn.Linear(self.hidden_dim, self.atten_dim)
        self.value = nn.Sequential(nn.Linear(self.hidden_dim, self.atten_dim), nn.ReLU())
        self.scale_factor = th.scalar_tensor(self.head_dim).sqrt()
        # rnn
        self.rnn = nn.GRUCell(self.atten_dim, args.rnn_hidden_dim)

        self.fc2_w_nn = nn.Linear(NN_HIDDEN_SIZE, args.rnn_hidden_dim * self.n_actions)
        self.fc2_b_nn = nn.Linear(NN_HIDDEN_SIZE, self.n_actions)

        # Dis net
        self.dis_net = nn.Sequential(nn.Linear(args.latent_dim * 2, NN_HIDDEN_SIZE),
                                     nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                     activation_func,
                                     nn.Linear(NN_HIDDEN_SIZE, 1))

        self.mi = th.rand(self.n_agents * self.n_agents)
        self.dissimilarity = th.rand(self.n_agents * self.n_agents)

        if args.dis_sigmoid:
            print('>>> sigmoid')
            self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_sigmoid
        else:
            self.dis_loss_weight_schedule = self.dis_loss_weight_schedule_step

    def init_hidden(self):
        # make hidden states on same device as model
        return self.query.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_latent(self, bs):
        self.bs = bs
        loss = 0

        self.trajectory = []

        var_mean = self.latent[:self.n_agents, self.args.latent_dim:].detach().mean()

        # mask = 1 - th.eye(self.n_agents).byte()
        # mi=self.mi.view(self.n_agents,self.n_agents).masked_select(mask)
        # di=self.dissimilarity.view(self.n_agents,self.n_agents).masked_select(mask)
        mi = self.mi
        di = self.dissimilarity
        indicator = [var_mean, mi.max(), mi.min(), mi.mean(), mi.std(), di.max(), di.min(), di.mean(), di.std()]
        return indicator, self.latent[:self.n_agents, :].detach(), self.latent_infer[:self.n_agents, :].detach()

    def forward(self, inputs, obs_mask, scenario_mask, hidden_state, t=0, batch=None, test_mode=None, t_glob=0, train_mode=False):
        # inputs/transformed state.shape = (bs, n_entities, local_state_size + n_actions),
        # which includes (local_state + last_actions), padded zero vector for non-agents' last actions
        # obs_mask.shape=(bs, n_entities, n_entities), scenario_mask.shape=(bs, n_entities)
        # h_interactive.shape=(bs, n_agents, rnn_hidden_dim)
        bs, n_entities, edim = inputs.size()
        h_in = hidden_state.reshape(bs * self.n_agents, self.hidden_dim)

        num_h = self.n_heads
        e = self.head_dim

        # =========================== role encoding ============================
        embed_inps = self.role_fc1(inputs)  # (bs, n_entities, hidden_dim)
        embed_inps_t = embed_inps.transpose(0, 1)   # (n_entities, bs, hidden_dim)

        role_queries = self.role_query(embed_inps_t).reshape(n_entities, bs * num_h, e).transpose(0, 1)  # (bs*num_h, n_entities, e)
        role_keys = self.role_key(embed_inps_t).reshape(n_entities, bs * num_h, e).permute(1, 2, 0)  # (bs*num_h, e, n_entities)
        role_values = self.role_value(embed_inps_t).reshape(n_entities, bs * num_h, e).transpose(0, 1)  # (bs*num_h, n_entities, e)

        role_queries = role_queries[:, :self.n_agents, :]  # (bs*num_h, n_agents, e)
        role_score = th.bmm(role_queries, role_keys) / self.role_scale_factor  # (bs*num_h, n_agents, n_entities)

        # obs_mask.shape=(bs, n_entities, n_entities), unobservable=1, observable=0
        obs_mask = obs_mask[:, :self.n_agents, :].to(th.bool)  # (bs, n_agents, n_entities)
        obs_mask_rep = obs_mask.repeat_interleave(num_h, dim=0)     # (bs*num_h, n_agents, n_entities)
        role_score = role_score.masked_fill(obs_mask_rep, -float('Inf'))  # mask the weights of un-observable entities's info
        role_weights = F.softmax(role_score, dim=-1)  # (bs*num_h, n_agents, n_entities)
        # 对于当前环境中不存在的agent，obs_mask matrix中其相应行元素全部都为1，因为full_obs_mask=torch.ones(...)
        # Some weights might be NaN (if agent is inactive and all entities were masked)
        role_weights = role_weights.masked_fill(role_weights != role_weights, 0)
        role_atten_out = th.bmm(role_weights, role_values)  # (bs*num_h, n_agents, e)
        role_atten_out = role_atten_out.transpose(0, 1).reshape(self.n_agents, bs, num_h * e)
        role_atten_out = role_atten_out.transpose(0, 1).reshape(bs * self.n_agents, self.atten_dim)     # (bs*n_agents, atten_dim)

        self.latent = self.embed_net(role_atten_out)        # (bs*n_agents, latent_dim * 2)
        self.latent[:, -self.latent_dim:] = th.clamp(th.exp(self.latent[:, -self.latent_dim:]), min=self.args.var_floor)  # var
        latent_embed = self.latent      # (bs*n_agents, latent_dim * 2)

        gaussian_embed = D.Normal(latent_embed[:, :self.latent_dim], (latent_embed[:, self.latent_dim:]) ** (1 / 2))
        latent = gaussian_embed.rsample()       # (bs*n_agents, latent_dim)

        c_dis_loss = th.tensor(0.0).to(self.args.device)
        ce_loss = th.tensor(0.0).to(self.args.device)
        loss = th.tensor(0.0).to(self.args.device)

        if train_mode and (not self.args.roma_raw):
            # TODO: Here we detach both the h_in and role_atten_out, whether this operation is reasonable?
            self.latent_infer = self.inference_net(th.cat([h_in.detach(), role_atten_out.detach()], dim=1))      # (bs*n_agents, hidden_dim+hidden_dim)
            self.latent_infer[:, -self.latent_dim:] = th.clamp(th.exp(self.latent_infer[:, -self.latent_dim:]), min=self.args.var_floor)
            gaussian_infer = D.Normal(self.latent_infer[:, :self.latent_dim], (self.latent_infer[:, self.latent_dim:]) ** (1 / 2))
            latent_infer = gaussian_infer.rsample()

            loss = gaussian_embed.entropy().sum(dim=-1).mean() * self.args.h_loss_weight + \
                   kl_divergence(gaussian_embed, gaussian_infer).sum(dim=-1).mean() * self.args.kl_loss_weight  # CE = H + KL
            loss = th.clamp(loss, max=2e3)
            ce_loss = th.log(1 + th.exp(loss))

            # Dis Loss
            cur_dis_loss_weight = self.dis_loss_weight_schedule(t_glob)
            if cur_dis_loss_weight > 0:
                dis_loss = 0
                dissimilarity_cat = None
                mi_cat = None
                latent_dis = latent.clone().view(bs, self.n_agents, -1)
                latent_move = latent.clone().view(bs, self.n_agents, -1)
                for agent_i in range(self.n_agents):
                    latent_move = th.cat([latent_move[:, -1, :].unsqueeze(1), latent_move[:, :-1, :]], dim=1)
                    latent_dis_pair = th.cat([latent_dis[:, :, :self.latent_dim], latent_move[:, :, :self.latent_dim]], dim=2)
                    mi = th.clamp(gaussian_embed.log_prob(latent_move.view(bs * self.n_agents, -1)) + 13.9, min=-13.9).sum(dim=1, keepdim=True) / self.latent_dim

                    dissimilarity = th.abs(self.dis_net(latent_dis_pair.view(-1, 2 * self.latent_dim)))

                    if dissimilarity_cat is None:
                        dissimilarity_cat = dissimilarity.view(bs, -1).clone()
                    else:
                        dissimilarity_cat = th.cat([dissimilarity_cat, dissimilarity.view(bs, -1)], dim=1)
                    if mi_cat is None:
                        mi_cat = mi.view(bs, -1).clone()
                    else:
                        mi_cat = th.cat([mi_cat, mi.view(bs, -1)], dim=1)

                mi_min = mi_cat.min(dim=1, keepdim=True)[0]
                mi_max = mi_cat.max(dim=1, keepdim=True)[0]
                di_min = dissimilarity_cat.min(dim=1, keepdim=True)[0]
                di_max = dissimilarity_cat.max(dim=1, keepdim=True)[0]

                mi_cat = (mi_cat - mi_min) / (mi_max - mi_min + 1e-12)
                dissimilarity_cat = (dissimilarity_cat - di_min) / (di_max - di_min + 1e-12)

                dis_loss = - th.clamp(mi_cat + dissimilarity_cat, max=1.0).sum() / bs / self.n_agents
                # dis_loss = ((mi_cat + dissimilarity_cat - 1.0 )**2).sum() / bs / self.n_agents
                dis_norm = th.norm(dissimilarity_cat, p=1, dim=1).sum() / bs / self.n_agents

                # c_dis_loss = (dis_loss + dis_norm) / self.n_agents * cur_dis_loss_weight
                c_dis_loss = (dis_norm + self.args.soft_constraint_weight * dis_loss) / self.n_agents * cur_dis_loss_weight
                loss = ce_loss + c_dis_loss

                self.mi = mi_cat[0]
                self.dissimilarity = dissimilarity_cat[0]
            else:
                c_dis_loss = th.zeros_like(loss)
                loss = ce_loss

        # Role --> fc2 layer's weights and biases
        latent = self.latent_net(latent)        # (bs*n_agents, NN_HIDDEN_DIM)

        fc2_w = self.fc2_w_nn(latent)       # (bs*n_agents, rnn_hidden_dim * self.n_actions)
        fc2_b = self.fc2_b_nn(latent)       # (bs*n_agents, self.n_actions)
        fc2_w = fc2_w.reshape(-1, self.args.rnn_hidden_dim, self.n_actions)
        fc2_b = fc2_b.reshape(-1, 1, self.n_actions)

        embed_inputs = self.fc1(inputs)  # (bs, n_entities, hidden_dim)
        embed_inputs_t = embed_inputs.transpose(0, 1)       # (n_entities, bs, hidden_dim)

        queries = self.query(embed_inputs_t).reshape(n_entities, bs * num_h, e).transpose(0, 1)  # (bs*num_h, n_entities, e)
        keys = self.key(embed_inputs_t).reshape(n_entities, bs * num_h, e).permute(1, 2, 0)  # (bs*num_h, e, n_entities)
        values = self.value(embed_inputs_t).reshape(n_entities, bs * num_h, e).transpose(0, 1)  # (bs*num_h, n_entities, e)
        queries = queries[:, :self.n_agents, :]  # (bs*num_h, n_agents, e)
        score = th.bmm(queries, keys) / self.scale_factor       # (bs*num_h, n_agents, n_entities)

        # obs_mask_rep.shape=(bs*num_h, n_agents, n_entities), unobservable=1, observable=0
        score = score.masked_fill(obs_mask_rep, -float('Inf'))  # mask the weights of un-observable entities's info
        weights = F.softmax(score, dim=-1)  # (bs*num_h, n_agents, n_entities)
        # 对于当前环境中不存在的agent，obs_mask matrix中其相应行元素全部都为1，因为full_obs_mask=torch.ones(...)
        # Some weights might be NaN (if agent is inactive and all entities were masked)
        weights = weights.masked_fill(weights != weights, 0)
        atten_out = th.bmm(weights, values)     # (bs*num_h, n_agents, e)
        atten_out = atten_out.transpose(0, 1).reshape(self.n_agents, bs, num_h * e)
        atten_out = atten_out.transpose(0, 1).reshape(bs * self.n_agents, self.atten_dim)   # (bs*n_agents, num_h*e)

        h = self.rnn(atten_out, h_in).reshape(bs, self.n_agents, self.args.rnn_hidden_dim)  # \tau, the output of [self-attention+GRUCell]

        # scenario_mask.shape=(bs, n_entities), un-existing=1, existing=0
        agent_mask = scenario_mask[:, :self.n_agents].unsqueeze(dim=2).to(th.bool)      # (bs, n_agents, 1)
        h = h.masked_fill(agent_mask, 0)

        h_view = h.reshape(bs * self.n_agents, 1, self.args.rnn_hidden_dim)

        q = th.bmm(h_view, fc2_w) + fc2_b        # (bs*n_agents, 1, n_actions)
        q = q.reshape(bs, self.n_agents, self.n_actions)

        q = q.masked_fill(agent_mask, 0)

        return q, h, loss, c_dis_loss, ce_loss

    def dis_loss_weight_schedule_step(self, t_glob):
        if t_glob > self.args.dis_time:
            return self.args.dis_loss_weight
        else:
            return 0

    def dis_loss_weight_schedule_sigmoid(self, t_glob):
        return self.args.dis_loss_weight / (1 + math.exp((1e7 - t_glob) / 2e6))