import copy
from src.components.episode_buffer import EpisodeBatch
from src.modules.mixers.vdn import VDNMixer
from src.modules.mixers.qmix import QMixer
from src.modules.mixers.attn_mix import AttnMixer
from src.modules.mixers.flex_qmix import FlexQMixer
from src.utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np


class REFILLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_entities = args.n_entities

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0
        self.device = args.device

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "flex_qmix":
                self.mixer = FlexQMixer(args)
            else:
                raise Exception("The REFIILLeaner must use the flex_qmix mixer.")
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.train_t = 0

    def _build_inputs_mixer(self, batch_states, batch_actions):
        # batch_states.shape=(bs, max_seq_length, state_shape), batch_actions.shape=(bs, max_seq_length, n_agents, n_actions)
        bs, seq_t, n_agents, n_actions = batch_actions.size()
        batch_states = batch_states.reshape(bs, seq_t, self.n_entities, -1)     # (bs, t, n_entities, local_state_size)
        inputs = []
        inputs.append(batch_states)
        if self.args.obs_last_action:
            last_actions = th.zeros((bs, seq_t, self.n_entities, n_actions), dtype=batch_actions.dtype, device=self.device)
            last_actions[:, 1:, :n_agents, :] = batch_actions[:, :-1, :, :]  # 注意t=0时, last_action=zero vector.
            inputs.append(last_actions)
        inputs = th.cat(inputs, dim=-1)          # (bs, t, n_entities, local_state_size+n_actions)
        return inputs

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        self.mac.init_hidden(batch.batch_size)

        # all_mac_out.shape=(bs*3, seq_t, n_agents, n_actions), groups include Wgroup_noobs and Igroup_noobs, each shape=(bs, seq_t, n_entities, n_entities)
        all_mac_out, groups = self.mac.forward(batch, t=None, imagine=True)
        rep_actions = actions.repeat(3, 1, 1, 1)        # (bs * 3, seq_t, n_agents, n_actions)
        # all_chosen_action_qvals.shape=(bs*3, seq_t, n_agents)
        all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)   # Remove the last dim

        mac_out, moW, moI = all_mac_out.chunk(3, dim=0)     # The first dim is bs*3, including vanilla, within_group and interact_group
        chosen_action_qvals, caqW, caqI = all_chosen_action_qvals.chunk(3, dim=0)   # each of them, shape (bs, seq_t, n_agents)
        caq_imagine = th.cat([caqW, caqI], dim=2)       # shape=(bs, seq_t, n_agents * 2) ?

        # Calculate the Q-Values necessary for the target
        self.target_mac.init_hidden(batch.batch_size)
        target_mac_out = self.target_mac.forward(batch, t=None)
        avail_actions_targ = avail_actions
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions_targ[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            state_inps = self._build_inputs_mixer(batch["state"], batch["actions_onehot"])      # shape=(bs, seq_t, n_entities, local_state_size+n_actions)
            curr_mixer_inps = (state_inps[:, :-1], batch["scenario_mask"][:, :-1, :])
            target_mixer_inps = (state_inps[:, 1:], batch["scenario_mask"][:, 1:, :])
            chosen_action_qvals = self.mixer(chosen_action_qvals, curr_mixer_inps)

            groups = [gr[:, :-1] for gr in groups]  # two elements, each one.shape=(bs, seq_t-1, n_entities, n_entities)
            caq_imagine = self.mixer(caq_imagine, curr_mixer_inps, imagine_groups=groups)

            target_max_qvals = self.target_mixer(target_max_qvals, target_mixer_inps)

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()

        # Auxiliary loss brough by imagine.
        im_prop = self.args.lmbda
        im_td_error = (caq_imagine - targets.detach())
        im_masked_td_error = im_td_error * mask
        im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
        loss = (1 - im_prop) * loss + im_prop * im_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.to(device=self.device)
            self.target_mixer.to(device=self.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))