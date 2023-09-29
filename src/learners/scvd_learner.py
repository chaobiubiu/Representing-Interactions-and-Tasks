import copy
from src.components.episode_buffer import EpisodeBatch
from src.modules.mixers.nmix import Mixer
from src.modules.mixers.vdn import VDNMixer
from src.modules.mixers.scvd_mix import SCVDMixer
from src.modules.mixers.attn_mix import AttnMixer
from src.modules.task_auxiliary.rep_dynamics import MultiHeadStateInpEncoder
from src.modules.task_auxiliary.recurrent_encoder import BinaryClassifier
from src.utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from src.utils.th_utils import get_parameters_num


CrossEntropyLoss = th.nn.CrossEntropyLoss()


class SCVDLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_entities = args.n_entities

        self.last_target_update_episode = 0
        # self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.device = args.device
        self.params = list(mac.parameters())

        if args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        elif args.mixer == "svn_mix":
            self.mixer = SCVDMixer(args)
        elif args.mixer == "attn_mix":
            self.mixer = AttnMixer(args)
        else:
            raise Exception("mixer error")
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        self.state_encoder = MultiHeadStateInpEncoder(args)
        # params includes: mac (agent) + mixer + mixer's input_encoder
        self.params += list(self.state_encoder.parameters())

        self.classifier = BinaryClassifier(args)      # [task_rep_dim^i, task_rep_dim^j] --> [0, 1]
        self.params += list(self.classifier.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

    def _build_inputs_state(self, batch_states, batch_actions):
        # batch_states.shape=(bs, max_seq_length, state_shape), batch_actions.shape=(bs, max_seq_length, n_agents, n_actions)
        bs, seq_t, n_agents, n_actions = batch_actions.size()
        batch_states = batch_states.reshape(bs, seq_t, self.n_entities, -1)     # (bs, t, n_entities, local_state_size)
        inputs = []
        inputs.append(batch_states)
        if self.args.obs_last_action:
            last_actions = th.zeros((bs, seq_t, self.n_entities, n_actions), device=self.device, dtype=batch_actions.dtype)
            last_actions[:, 1:, :n_agents, :] = batch_actions[:, :-1, :, :]  # 注意t=0时, last_action=zero vector.
            inputs.append(last_actions)
        inputs = th.cat(inputs, dim=-1)          # (bs, t, n_entities, local_state_size+n_actions)
        return inputs

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):    # t_env, episode_num are used to log.
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]       # shape=(bs, max_seq_length, 1)
        actions = batch["actions"][:, :-1]      # shape=(bs, max_seq_length, n_agents, n_actions)
        terminated = batch["terminated"][:, :-1].float()        # shape=(bs, max_seq_length, 1)
        mask = batch["filled"][:, :-1].float()                  # shape=(bs, max_seq_length, 1), normal is 1, padded is 0.
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]                      # shape=(bs, max_seq_length, n_agents, n_actions)
        actions_onehot = batch["actions_onehot"]        # shape=(bs, max_seq_length, n_agents, n_actions)

        state_inps = self._build_inputs_state(batch["state"], actions_onehot)           # shape=(bs, seq_t, n_entities, local_state_size+n_actions)

        mixer_inps = self.state_encoder(state_inps, batch["scenario_mask"])  # shape=(bs, seq_t, n_agents, rnn_hidden_dim)

        # Calculate estimated Q-Values
        self.mac.agent.train()
        self.mac.task_rep.train()

        mac_out = []
        auxiliary_mac_out = []
        all_q_alone, all_q_interact = [], []
        all_task_z = []
        kl_div_sum = []
        self.mac.init_hidden(batch.batch_size)
        self.mac.init_encoder_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, auxiliary_agent_outs, q_alone, q_interact, task_z, kl_div = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            auxiliary_mac_out.append(auxiliary_agent_outs)
            all_q_alone.append(q_alone)
            all_q_interact.append(q_interact)
            all_task_z.append(task_z)
            if kl_div is not None:
                kl_div_sum.append(kl_div)
            else:
                kl_div_sum.append(th.tensor(0.0, device=self.device))
        mac_out = th.stack(mac_out, dim=1)  # Concat over time, shape=(bs, max_seq_length, n_agents, n_actions)
        auxiliary_mac_out = th.stack(auxiliary_mac_out, dim=1)     # (bs, max_seq_length, n_agents, n_actions)
        # Only used to record these two q values.
        all_q_alone = th.stack(all_q_alone, dim=1)
        all_q_interact = th.stack(all_q_interact, dim=1)
        all_task_z = th.stack(all_task_z, dim=1)        # (bs, max_seq_length, n_agents, task_rep_dim)
        kl_div_sum = th.stack(kl_div_sum, dim=0)        # (max_seq_length, 1)

        concat_mixer_inps = th.cat([mixer_inps, all_task_z.detach()], dim=-1)       # (bs, max_seq_length, n_agents, (rnn_hidden_dim + task_rep_dim))

        # Pick the Q-Values for the actions taken by each agent, shape=(bs, (max_seq_length - 1), n_agents)
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        all_target_task_z = []
        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            self.target_mac.task_rep.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            self.target_mac.init_encoder_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs, _, _, _, target_task_z, _ = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
                all_target_task_z.append(target_task_z)

            all_target_task_z = th.stack(all_target_task_z, dim=1)      # (bs, max_seq_length, n_agents, task_rep_dim)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time, (bs, max_seq_length, n_agents, n_actions)

            # Mask out unavailable actions
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)         # shape=(bs, max_seq_length, n_agents)

            # Calculate n-step Q-Learning targets
            if self.args.mixer == "attn_mix":
                target_max_qvals = self.target_mixer(target_max_qvals, concat_mixer_inps[:, 1:], batch["scenario_mask"][:, 1:, :])
            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

            if getattr(self.args, 'td_lambda', False):
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma, self.args.td_lambda)
            else:
                # one-step td target
                targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Mixer
        if self.args.mixer == "attn_mix":
            chosen_action_qvals = self.mixer(chosen_action_qvals, concat_mixer_inps[:, :-1], batch["scenario_mask"][:, :-1, :])
        else:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()

        auxiliary_target = th.zeros_like(auxiliary_mac_out, device=self.args.device)
        extra_loss = 0.5 * (auxiliary_target - auxiliary_mac_out).pow(2).mean()

        all_task_z_a = all_task_z.view(batch.batch_size * batch.max_seq_length * self.n_agents, -1)       # (bs*seq_t*n_agents, task_rep_dim)
        rand = th.randperm(batch.batch_size * batch.max_seq_length * self.n_agents).numpy()
        all_task_z_b = all_task_z_a[rand]
        concat_pairs = th.cat([all_task_z_a, all_task_z_b], dim=-1)     # (bs*seq_t*n_agents, task_rep_dim * 2)
        pred = self.classifier(concat_pairs)      # (bs*seq_t*n_agents, 2)

        target_label_a = batch["task_index"].unsqueeze(dim=2).expand(-1, -1, self.n_agents, -1).reshape(-1, self.args.n_tasks)     # (bs*seq_t*n_agents, n_tasks)
        target_label_a = th.topk(target_label_a, 1)[1]
        target_label_b = target_label_a[rand]
        target_label = (target_label_a == target_label_b).long().squeeze(dim=-1)
        contrastive_loss = CrossEntropyLoss(pred, target_label)

        kl_loss = kl_div_sum.mean()
        final_loss = loss + 1 * extra_loss + 1 * contrastive_loss + 1 * kl_loss

        # Optimise
        self.optimiser.zero_grad()
        final_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("extra_loss", extra_loss.item(), t_env)
            self.logger.log_stat("contrastive_loss", contrastive_loss.item(), t_env)
            self.logger.log_stat("kl_loss", kl_loss.item(), t_env)
            self.logger.log_stat("q_alone_mean", all_q_alone.mean(), t_env)
            self.logger.log_stat("q_interact_mean", all_q_interact.mean(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

        # return info
        info = {}
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.state_encoder.to(self.device)
        self.classifier.to(self.device)
        if self.mixer is not None:
            self.mixer.to(self.device)
            self.target_mixer.to(self.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.state_encoder.state_dict(), "{}/state_encoder.th".format(path))
        th.save(self.classifier.state_dict(), "{}/classifier.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))