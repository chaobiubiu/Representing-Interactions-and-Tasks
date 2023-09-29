from src.envs import REGISTRY as env_REGISTRY
from functools import partial
from src.components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th


# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])

        # if 'scvd' in self.args.name:
        #     assert self.args.env == 'randomsc2custom', "SCVD must run in randomsc2custom."
        # else:
        #     assert self.args.env != 'randomsc2custom', 'Other algs except for SCVD must run in sc2custom.'

        env_fn = env_REGISTRY[self.args.env]
        self.ps = []

        base_seed = self.args.env_args.pop('seed')  # 注意这里随机化env的seed时，seed + rank
        for i, worker_conn in enumerate(self.worker_conns):
            if args.env == 'mpe':
                ps = Process(target=env_worker_mpe,
                             args=(worker_conn, CloudpickleWrapper(partial(env_fn, env_args=args.env_args, seed=(args.env_args["seed"] + i * 100)))))
            elif args.env == 'sc2custom':
                # By default entity_scheme is True
                ps = Process(target=env_worker_sc2custom, args=(worker_conn, True,
                                                      CloudpickleWrapper(partial(env_fn, seed=(base_seed + i), **self.args.env_args))))
            elif args.env == 'randomsc2custom':
                # If alg is scvd, use random sc2custom
                ps = Process(target=env_worker_random_sc2custom, args=(worker_conn, True,
                                                                CloudpickleWrapper(partial(env_fn, seed=(base_seed + i), **self.args.env_args))))
            else:
                raise Exception("Un-supported env.")
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]
        # Use this to label the selected task, which is updated per call reset() method.
        self.task_ids = [None for _ in range(self.batch_size)]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            # TODO: 如果测试算法到未知任务的泛化性，则需要将evaluate设置为True
            parent_conn.send(("reset", {"evaluate": False}))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            # "obs": [],
            "obs_mask": [],
            "scenario_mask": [],
            "task_index": [],
        }
        # Get the obs, state and avail_actions back
        for i, parent_conn in enumerate(self.parent_conns):
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])                      # shape=(state_shape)
            pre_transition_data["avail_actions"].append(data["avail_actions"])      # shape=(n_agents, n_actions)
            # pre_transition_data["obs"].append(data["obs"])                          # shape=(n_agents, state_shape)
            obs_mask, scenario_mask = data["masks"]
            pre_transition_data["obs_mask"].append(np.squeeze(obs_mask, axis=-1))   # shape=(n_entities, n_entities)
            pre_transition_data["scenario_mask"].append(scenario_mask)              # shape=(n_entities)
            pre_transition_data["task_index"].append(data["task_index"])            # shape=(n_tasks)
            self.task_ids[i] = data["task_index"]           # different threads may render different envs

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        if "scvd" in self.args.name and "rep" not in self.args.name:        # For task_rep ablation studies, args.name must contain 'rep' string.
            self.mac.init_encoder_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        
        save_probs = getattr(self.args, "save_probs", False)
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)

            if len(actions.size()) == 3:            # To make the shape of actions equal to (bs, n_agents)
                actions = actions.squeeze(dim=-1)
            # actions.shape=(bs, n_agents)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                # "actions": actions.unsqueeze(1).to("cpu"),
                "actions": actions.unsqueeze(2).to("cpu"),      # TODO: 修改unsqueeze(dim=2)，暂时不确定这样做是否正确, shape=(bs, n_agents, 1)
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env, 因为有bs=envs_not_terminated这个标识符，所以需要跨过未终止的env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                # "obs": [],
                "obs_mask": [],
                "scenario_mask": [],
                "task_index": [],
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))        # shape=(1)

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])                        # data["info"]={}
                    if self.args.env == "mpe":
                        if data["terminated"]:
                            env_terminated = True
                    else:
                        if data["terminated"] and not data["info"].get("episode_limit", False):
                            env_terminated = True
                    terminated[idx] = data["terminated"]                            # shape=(1)
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])              # shape=(state_shape)
                    pre_transition_data["avail_actions"].append(data["avail_actions"])  # shape=(n_agents, n_actions)
                    # pre_transition_data["obs"].append(data["obs"])                      # shape=(n_agents, state_shape)
                    obs_mask, scenario_mask = data["masks"]
                    pre_transition_data["obs_mask"].append(np.squeeze(obs_mask, axis=-1))   # shape=(n_entities, n_entities)
                    pre_transition_data["scenario_mask"].append(scenario_mask)              # shape=(n_entites)
                    pre_transition_data["task_index"].append(self.task_ids[idx])        # The task index of each thread keeps fixed until next calling reset() method.

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)      # 这里其实是对batch_size_run个并行环境相应的参数做了平均和标准差计算
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()                                                                 # Note that the returns and stats are cleaned after log.

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


def env_worker_sc2custom(remote, entity_scheme, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            if entity_scheme:
                actions = data
                # Take a step in the environment
                reward, terminated, env_info = env.step(actions)
                next_global_states = env.get_entities()
                full_obs_mask, scenario_mask = env.get_masks()
                avail_actions = env.get_avail_actions()     # (max_n_agents, n_actions)

                next_obs_n = []
                full_obs_mask_ = np.expand_dims(full_obs_mask, axis=-1)
                for i in range(env.max_n_agents):
                    next_obs_i = next_global_states * (1 - full_obs_mask_[i])
                    next_obs_i = np.concatenate(next_obs_i, axis=0)
                    next_obs_n.append(next_obs_i)
                # next_obs_n.shape=(max_n_agents, (max_n_agents+max_n_enemies)xnf_entities)

                next_global_states = np.concatenate(next_global_states, axis=0)
                send_dict = {"state": next_global_states,
                             "avail_actions": avail_actions,
                             "obs": next_obs_n,
                             "masks": (full_obs_mask_, scenario_mask),
                             "reward": reward,
                             "terminated": terminated,
                             "info": env_info}
                remote.send(send_dict)
            else:
                raise Exception("Un-supported env setting except for the entity_scheme.")
        elif cmd == "reset":
            if entity_scheme:
                global_states, masks, task_index = env.reset(**data)
                full_obs_mask, scenario_mask = masks
                # masks=(obs_mask, scenario_mask), obs_mask.shape=(max_num_allies+max_num_enemies, max_num_allies+max_num_enemies), scenario_mask.shape=(max_num_allies+max_num_enemies)
                # TODO:如果要根据entities(states)得到obs，那么需要将obs_mask.unsqueeze(dim=-1)，后续存储时需要将该维度squeeze(dim=-1)
                avail_actions = env.get_avail_actions()     # (max_n_agents, n_actions)

                obs_n = []
                full_obs_mask_ = np.expand_dims(full_obs_mask, axis=-1)
                for i in range(env.max_n_agents):
                    obs_i = global_states * (1 - full_obs_mask_[i])
                    obs_i = np.concatenate(obs_i, axis=0)
                    obs_n.append(obs_i)
                # obs_n.shape=(max_n_agents, (max_n_agents+max_n_enemies)xnf_entities)

                global_states = np.concatenate(global_states, axis=0)
                send_dict = {
                    "state": global_states,
                    "avail_actions": avail_actions,
                    "obs": obs_n,
                    "masks": (full_obs_mask_, scenario_mask),
                    "task_index": task_index,
                }
                remote.send(send_dict)
            else:
                raise Exception('Unknown env setting except for the entity_scheme.')
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info(data))
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        # TODO: unused now?
        # elif cmd == "agg_stats":
        #     agg_stats = env.get_agg_stats(data)
        #     remote.send(agg_stats)
        else:
            raise NotImplementedError


def env_worker_random_sc2custom(remote, entity_scheme, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            if entity_scheme:
                actions = data
                # Take a step in the environment
                next_global_states, masks, avail_actions, reward, terminated, env_info = env.step(actions)
                full_obs_mask, scenario_mask = masks

                next_obs_n = []
                full_obs_mask_ = np.expand_dims(full_obs_mask, axis=-1)
                for i in range(env.max_n_agents):
                    next_obs_i = next_global_states * (1 - full_obs_mask_[i])
                    next_obs_i = np.concatenate(next_obs_i, axis=0)
                    next_obs_n.append(next_obs_i)
                # next_obs_n.shape=(max_n_agents, (max_n_agents+max_n_enemies)xnf_entities)

                next_global_states = np.concatenate(next_global_states, axis=0)
                send_dict = {"state": next_global_states,
                             "avail_actions": avail_actions,
                             "obs": next_obs_n,
                             "masks": (full_obs_mask_, scenario_mask),
                             "reward": reward,
                             "terminated": terminated,
                             "info": env_info}
                remote.send(send_dict)
            else:
                raise Exception("Un-supported env setting except for the entity_scheme.")
        elif cmd == "reset":
            if entity_scheme:
                global_states, masks, avail_actions, task_index = env.reset(**data)
                full_obs_mask, scenario_mask = masks
                # masks=(obs_mask, scenario_mask), obs_mask.shape=(max_num_allies+max_num_enemies, max_num_allies+max_num_enemies), scenario_mask.shape=(max_num_allies+max_num_enemies)
                obs_n = []
                full_obs_mask_ = np.expand_dims(full_obs_mask, axis=-1)
                for i in range(env.max_n_agents):
                    obs_i = global_states * (1 - full_obs_mask_[i])
                    obs_i = np.concatenate(obs_i, axis=0)
                    obs_n.append(obs_i)
                # obs_n.shape=(max_n_agents, (max_n_agents+max_n_enemies)xnf_entities)

                global_states = np.concatenate(global_states, axis=0)
                send_dict = {
                    "state": global_states,
                    "avail_actions": avail_actions,
                    "obs": obs_n,
                    "masks": (full_obs_mask_, scenario_mask),
                    "task_index": task_index,
                }
                remote.send(send_dict)
            else:
                raise Exception('Unknown env setting except for the entity_scheme.')
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info(data))
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        # TODO: unused now?
        # elif cmd == "agg_stats":
        #     agg_stats = env.get_agg_stats(data)
        #     remote.send(agg_stats)
        else:
            raise NotImplementedError


def env_worker_mpe(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            state, obs, masks, reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            # state = env.get_state()
            avail_actions = env.get_avail_actions()
            # obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "masks": masks,         # masks=(full_obs_mask, scenario_mask)
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            state, obs, masks, task_index = env.reset()     # (full_obs_mask, scenario_mask)
            avail_actions = env.get_avail_actions()
            # remote.send({
            #     "state": env.get_state(),
            #     "avail_actions": env.get_avail_actions(),
            #     "obs": env.get_obs()
            # })
            remote.send({
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "masks": masks,
                "task_index": task_index,
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

