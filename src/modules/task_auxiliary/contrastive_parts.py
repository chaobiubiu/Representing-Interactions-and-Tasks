import torch as th
import torch.nn as nn
import torch.nn.functional as F


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = th.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / th.sum(th.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * th.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


class ContextEncoder(nn.Module):
    def __init__(self, args):
        super(ContextEncoder, self).__init__()
        self.args = args
        self.use_ib = args.use_ib
        self.n_agents = args.n_agents
        self.n_entities = args.n_entities
        self.local_state_size = args.local_state_size
        state_input_dim = args.local_state_size
        if args.obs_last_action:
            state_input_dim += args.n_actions
        self.state_latent_dim = args.rnn_hidden_dim
        self.act_latent_dim = (args.rnn_hidden_dim // 2)
        self.rew_latent_dim = (args.rnn_hidden_dim // 8)
        self.task_rep_dim = args.task_rep_dim
        self.out_dim = self.task_rep_dim if self.use_ib else (self.task_rep_dim * 2)
        # state, next_state encoder
        self.fc_states = nn.Sequential(nn.Linear(state_input_dim, self.state_latent_dim), nn.ReLU())
        # action encoder
        self.act_encoder = nn.Sequential(nn.Linear(args.n_actions, self.act_latent_dim), nn.ReLU())
        # reward encoder
        self.rew_encoder = nn.Sequential(nn.Linear(1, self.rew_latent_dim), nn.ReLU())
        # (state_embedding, act_embedding, next_state_embedding) are passed to the attention mechanism to get a fixed-length vector
        # q / k / v
        self.query = nn.Linear((self.state_latent_dim * 2 + self.act_latent_dim), args.rnn_hidden_dim)
        self.key = nn.Linear((self.state_latent_dim * 2 + self.act_latent_dim), args.rnn_hidden_dim)
        self.value = nn.Sequential(nn.Linear((self.state_latent_dim * 2 + self.act_latent_dim), args.rnn_hidden_dim), nn.ReLU())
        self.scale_factor = th.scalar_tensor(args.rnn_hidden_dim // 1).sqrt()
        # the output layer that outputs the task_rep distribution, mu and sigma
        self.out = nn.Linear((args.rnn_hidden_dim + self.rew_latent_dim), self.out_dim)

    def forward(self, states, onehot_actions, rewards, next_states, scenario_masks):
        # states / next_states.shape=(bs, t, n_entities, (local_state_size+last_action)), onehot_actions.shape=(bs, t, n_agents, n_actions),
        # rewards.shape=(bs, t, 1), scenario_masks.shape=(bs, t, n_entities)
        bs, t, n_agents, n_actions = onehot_actions.size()
        states = states.reshape(bs * t, self.n_entities, -1)
        onehot_actions = onehot_actions.reshape(bs * t, self.n_agents, -1)
        rewards = rewards.reshape(bs * t, -1)
        next_states = next_states.reshape(bs * t, self.n_entities, -1)

        embed_states = self.fc_states(states)   # (bs_t, n_entities, rnn_hidden_dim)
        embed_next_states = self.fc_states(next_states)     # (bs_t, n_entities, rnn_hidden_dim)
        act_embeddings = self.act_encoder(onehot_actions)  # (bs_t, n_agents, act_latent_dim)
        padded_act_embeddings = th.zeros((bs * t, self.n_entities, self.act_latent_dim), device=self.args.device)
        padded_act_embeddings[:, :n_agents, :] = act_embeddings  # (bs_t, n_entities, rnn_hidden_dim)
        embed_reward = self.rew_encoder(rewards)        # (bs_t, rew_hidden_dim)
        concat_sas_ = th.cat([embed_states, padded_act_embeddings, embed_next_states], dim=-1)  # (bs_t, n_entities, rnn_hidden_dim*2 + self.act_latent_dim)

        scenario_masks = scenario_masks.reshape(bs * t, self.n_entities)  # (bs_t, n_entities), existing is 1, otherwise is 0.
        # create attn_mask from scenario_mask, th.bmm((bs_t, n_entities, 1), (bs_t, 1, n_entities)) = (bs_t, n_entities, n_entities)
        atten_mask = th.tensor(1, dtype=th.float32, device=self.args.device) - th.bmm(
            (th.tensor(1, dtype=th.float32, device=self.args.device) - scenario_masks.to(th.float32)).unsqueeze(dim=2),
            (th.tensor(1, dtype=th.float32, device=self.args.device) - scenario_masks.to(th.float32)).unsqueeze(dim=1))

        queries = self.query(concat_sas_).reshape(bs * t, self.n_entities, -1)  # (bs_t, n_entities, hidden_dim)
        keys = self.key(concat_sas_).reshape(bs * t, self.n_entities, -1).permute(0, 2, 1)  # (bs_t, hidden_dim, n_entities)
        values = self.value(concat_sas_).reshape(bs * t, self.n_entities, -1)  # (bs_t, n_entities, hidden_dim)
        score = th.bmm(queries, keys) / self.scale_factor  # (bs, n_entities, n_entities)

        # attn_mask.shape=(bs_t, n_agents, n_entities), un-existing=1, existing=0
        score = score.masked_fill(atten_mask.to(th.bool), -float('Inf'))  # mask the weights of un-existing entities's info
        weights = F.softmax(score, dim=-1)  # (bs, n_entities, n_entities)
        # Some weights might be NaN (if agent is inactive and all entities were masked)
        weights = weights.masked_fill(weights != weights, 0)

        atten_out = th.bmm(weights, values)  # (bs_t, n_entities, rnn_hidden_dim)
        atten_out = atten_out.masked_fill(scenario_masks.unsqueeze(dim=2).to(th.bool), 0)
        pooling_atten_out = atten_out.mean(dim=1)  # (bs_t, rnn_hidden_dim)
        final_inps = th.cat([pooling_atten_out, embed_reward], dim=-1)      # (bs_t, rnn_hidden_dim+rew_latent_dim)
        params = self.out(final_inps).reshape(bs, t, -1)        # (bs, t, task_rep_dim*2)

        return params

    def sample_z(self):
        if self.use_ib:
            posteriors = [th.distributions.Normal(m, th.sqrt(s)) for m, s in zip(th.unbind(self.z_means), th.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = th.stack(z)
        else:
            self.z = self.z_means

    def infer_posterior(self, states, onehot_actions, rewards, next_states, scenario_masks):
        rep_dis_params = self.forward(states, onehot_actions, rewards, next_states, scenario_masks)     # (bs, t, task_rep_dim*2)
        if self.use_ib:  # information bottleneck
            mu = rep_dis_params[..., :self.task_rep_dim]
            sigma_squared = F.softplus(rep_dis_params[..., self.task_rep_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(th.unbind(mu), th.unbind(sigma_squared))]
            z_means = th.stack([p[0] for p in z_params])
            z_vars = th.stack([p[1] for p in z_params])
            posteriors = [th.distributions.Normal(m, th.sqrt(s)) for m, s in zip(th.unbind(z_means), th.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            task_rep = th.stack(z)
        else:
            z_means = th.mean(rep_dis_params, dim=1)
            task_rep = z_means
        return task_rep     # (bs, task_rep_dim)


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


class StateInpEncoder(nn.Module):
    def __init__(self, args):
        super(StateInpEncoder, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_entities = args.n_entities
        self.local_state_size = args.local_state_size
        input_dim = args.local_state_size
        if args.obs_last_action:
            input_dim += args.n_actions
        # fc_states is used to encoder each entity's info(local_state+last_action)
        self.fc_states = nn.Sequential(nn.Linear(input_dim, args.rnn_hidden_dim), nn.ReLU())
        # q/k/v
        self.query = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.key = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.value = nn.Sequential(nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim), nn.ReLU())
        self.scale_factor = th.scalar_tensor(args.rnn_hidden_dim // 1).sqrt()
        self.out = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def forward(self, inputs, scenario_masks):
        # inputs / transformed states.shape=(bs, t, n_entities, (local_state_size+last_action))
        # scenario_masks.shape=(bs, t, n_entities)
        bs, t, n_entities, e_dim = inputs.size()
        inputs = inputs.reshape(bs * t, n_entities, -1)

        embed_inputs = self.fc_states(inputs)       # (bs_t, n_entities, rnn_hidden_dim)

        scenario_masks = scenario_masks.reshape(bs * t, self.n_entities)      # (bs_t, n_entities), existing is 1, otherwise is 0.
        agent_masks = scenario_masks[:, :self.n_agents]                              # (bs_t, n_agents)
        # create attn_mask from scenario_mask, th.bmm((bs_t, n_agents, 1), (bs_t, 1, n_entities)) = (bs_t, n_agents, n_entities)
        atten_mask = th.tensor(1, dtype=th.float32, device=self.args.device) - th.bmm((th.tensor(1, dtype=th.float32, device=self.args.device) - agent_masks.to(th.float32)).unsqueeze(dim=2),
                                                                                      (th.tensor(1, dtype=th.float32, device=self.args.device) - scenario_masks.to(th.float32)).unsqueeze(dim=1))
        queries = self.query(embed_inputs).reshape(bs * t, self.n_entities, -1)  # (bs_t, n_entities, hidden_dim)
        keys = self.key(embed_inputs).reshape(bs * t, self.n_entities, -1).permute(0, 2, 1)  # (bs_t, hidden_dim, n_entities)
        values = self.value(embed_inputs).reshape(bs * t, self.n_entities, -1)  # (bs_t, n_entities, hidden_dim)
        queries = queries[:, :self.n_agents, :]  # (bs_t, n_agents, hidden_dim)
        score = th.bmm(queries, keys) / self.scale_factor  # (bs, n_agents, n_entities)

        # attn_mask.shape=(bs_t, n_agents, n_entities), un-existing=1, existing=0
        score = score.masked_fill(atten_mask.to(th.bool), -float('Inf'))  # mask the weights of un-existing entities's info
        weights = F.softmax(score, dim=-1)  # (bs, n_agents, n_entities)
        # Some weights might be NaN (if agent is inactive and all entities were masked)
        weights = weights.masked_fill(weights != weights, 0)

        atten_out = th.bmm(weights, values)  # (bs_t, n_agents, hidden_dim)
        outputs = self.out(atten_out)       # (bs_t, n_agents, hidden_dim)
        outputs = outputs.masked_fill(agent_masks.unsqueeze(dim=2).to(th.bool), 0).reshape(bs, t, self.n_agents, -1)    # (bs, t, n_agents, hidden_dim)
        # outputs = outputs.mean(dim=2)       # (bs, t, rnn_hidden_dim)

        return outputs