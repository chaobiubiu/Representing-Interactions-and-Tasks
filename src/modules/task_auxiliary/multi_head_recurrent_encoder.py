import torch as th
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F


class TaskRecurrentEncoder(nn.Module):
    def __init__(self, input_dim, args):
        super(TaskRecurrentEncoder, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.hidden_dim = args.rnn_hidden_dim
        self.task_rep_dim = args.task_rep_dim
        self.rew_latent_dim = (args.rnn_hidden_dim // 8)        # 8
        self.use_ib = args.use_ib
        self.out_dim = (self.task_rep_dim * 2) if self.use_ib else self.task_rep_dim

        self.n_heads = args.n_heads
        self.atten_dim = args.atten_dim
        assert self.atten_dim % self.n_heads == 0, "Attention dim must be divided by n_heads"
        self.head_dim = self.atten_dim // self.n_heads

        self.state_encoder = nn.Sequential(nn.Linear(input_dim, self.hidden_dim), nn.ReLU())
        self.rew_encoder = nn.Sequential(nn.Linear(1, self.rew_latent_dim), nn.ReLU())
        # q/k/v, hidden_dim --> atten_dim (n_heads * head_dim)
        self.query = nn.Linear(self.hidden_dim, self.atten_dim)
        self.key = nn.Linear(self.hidden_dim, self.atten_dim)
        self.value = nn.Sequential(nn.Linear(self.hidden_dim, self.atten_dim), nn.ReLU())
        self.scale_factor = th.scalar_tensor(self.head_dim).sqrt()
        # rnn
        self.rnn = nn.GRUCell((self.hidden_dim + self.atten_dim + self.rew_latent_dim), args.rnn_hidden_dim)
        # task_rep_out
        self.out = nn.Linear(args.rnn_hidden_dim, self.out_dim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.query.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, state_inputs, pre_rew_inputs, obs_mask, scenario_mask, pre_hidden, task_index):
        # state_inputs.shape = (bs, n_entities, local_state_size + n_actions), including (local_state + last_actions), padded zero vector for non-agents' last actions
        # obs_mask.shape=(bs, n_entities, n_entities), scenario_mask.shape=(bs, n_entities), pre_hidden.shape=(bs, n_agents, rnn_hidden_dim)
        bs, n_entities, edim = state_inputs.size()
        num_h = self.n_heads
        e = self.head_dim

        embed_inputs = self.state_encoder(state_inputs)              # (bs, n_entities, hidden_dim)
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
        weights = F.softmax(score, dim=-1)  # (bs, n_agents, n_entities)
        # 对于当前环境中不存在的agent，obs_mask matrix中其相应行元素全部都为1，因为full_obs_mask=torch.ones(...)
        # Some weights might be NaN (if agent is inactive and all entities were masked)
        weights = weights.masked_fill(weights != weights, 0)

        atten_out = th.bmm(weights, values)     # (bs*num_h, n_agents, e)
        atten_out = atten_out.transpose(0, 1).reshape(self.n_agents, bs, num_h * e)
        atten_out = atten_out.transpose(0, 1).reshape(-1, self.atten_dim)       # (bs, n_agents, atten_dim) --> (bs*n_agents, atten_dim)
        pre_hidden = pre_hidden.reshape(-1, self.args.rnn_hidden_dim)

        agent_embed_inputs = embed_inputs[:, :self.n_agents, :].reshape(-1, self.hidden_dim)  # (bs*n_agents, hidden_dim)
        pre_rew_inputs = pre_rew_inputs.unsqueeze(dim=1).expand(-1, self.n_agents, -1).reshape(-1, 1)        # (bs*n_agents, 1)
        embed_pre_rew_inps = self.rew_encoder(pre_rew_inputs)       # (bs*n_agents, rew_latent_dim)
        concat_inps = th.cat([agent_embed_inputs, atten_out, embed_pre_rew_inps], dim=-1)  # (bs*n_agents, hidden_dim+atten_dim+rew_latent_dim)

        h = self.rnn(concat_inps, pre_hidden).reshape(bs, self.n_agents, self.hidden_dim)  # \tau, the output of [self-attention+GRUCell] module

        # scenario_mask.shape=(bs, n_entities), un-existing=1, existing=0
        agent_mask = scenario_mask[:, :self.n_agents].unsqueeze(dim=2).to(th.bool)      # (bs, n_agents, 1)
        h = h.masked_fill(agent_mask, 0)           # 将当前环境中不存在的agent相应的输出全部变为0

        params = self.out(h)        # (bs, n_agents, task_rep_dim*2)
        params = params.masked_fill(agent_mask, 0)

        return params, h

    def infer_posterior(self, state_inputs, pre_rew_inputs, obs_mask, scenario_mask, pre_hidden, task_index, return_kl=False):
        params, hidden_states = self.forward(state_inputs, pre_rew_inputs, obs_mask, scenario_mask, pre_hidden, task_index)     # (bs, task_rep_dim*2)
        if self.use_ib:  # information bottleneck
            mu = params[..., :self.task_rep_dim]
            sigma = th.clamp(th.exp(params[..., -self.task_rep_dim:]), min=0.002)
            gaussian_posteriors = D.Normal(mu, sigma ** (1 / 2))
            task_z = gaussian_posteriors.rsample()      # (bs, n_agents, task_rep_dim)
            if return_kl:
                kl_div = self.compute_kl_div(gaussian_posteriors, mu)
            else:
                kl_div = None
        else:
            task_z = params
            kl_div = None
        return task_z, hidden_states, kl_div

    def compute_kl_div(self, posterior, mu):
        ''' compute KL( q(z|c) || p(z) ) '''
        prior = D.Normal(th.zeros_like(mu, device=self.args.device), th.ones_like(mu, device=self.args.device))
        kl_divs = D.kl.kl_divergence(posterior, prior)
        kl_div_sum = th.sum(kl_divs, dim=-1).mean()
        return kl_div_sum


class BinaryClassifier(nn.Module):
    def __init__(self, args):
        super(BinaryClassifier, self).__init__()
        self.input_dim = args.task_rep_dim * 2
        self.embed_dim = args.rnn_hidden_dim
        self.out_dim = 2

        self.fc1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim), nn.ReLU())
        self.fc2 = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, task_rep_pair):
        x1 = self.fc1(task_rep_pair)
        pred = self.fc2(x1)
        return pred