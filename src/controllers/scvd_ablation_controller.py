import torch as th
from src.modules.agents import REGISTRY as agent_REGISTRY
from src.components.action_selectors import REGISTRY as action_REGISTRY


# This multi-agent controller shares parameters between agents
class SCVDMAC_Ablation:
    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents           # max_num_agents
        self.n_entities = args.n_entities       # max_num_agents + max_num_landmarks
        self.n_actions = args.n_actions
        self.local_state_size = args.local_state_size       # local_state_size
        self.state_shape = args.state_shape                 # the shape of global state and local obs
        input_shape = self._get_input_shape(scheme)   # dim=local_state_size + last actions of all agents (padding zero vector for non-agents)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None

    # 这里需要auto-regressive action selection
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        obs_mask = ep_batch["obs_mask"][:, t_ep, :, :]  # (bs, n_entities, n_entities)
        scenario_mask = ep_batch["scenario_mask"][:, t_ep, :]  # (bs, n_entities)

        if test_mode:
            self.agent.eval()

        inputs = self._build_inputs_interact(ep_batch, t_ep)     # (bs, n_entities, local_state_size+n_actions)
        output_actions, self.hidden_states = self.agent.autoregressive_act(inputs, obs_mask, scenario_mask, self.hidden_states, avail_actions, bs, t_env, self.action_selector, test_mode)
        return output_actions

    def forward(self, ep_batch, t, test_mode=False):
        inputs_interact = self._build_inputs_interact(ep_batch, t)      # (bs, n_entities, local_state_size+n_actions)
        onehot_actions = ep_batch["actions_onehot"][:, t, :, :]         # (bs, n_agents, n_actions)
        obs_mask = ep_batch["obs_mask"][:, t, :, :]               # (bs, n_entities, n_entities)
        scenario_mask = ep_batch["scenario_mask"][:, t, :]       # (bs, n_entities)

        if test_mode:
            self.agent.eval()

        agent_outs, auxiliary_agent_outs, all_q_alone, all_q_interact, self.hidden_states = self.agent(inputs_interact, onehot_actions, obs_mask, scenario_mask, self.hidden_states)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1), auxiliary_agent_outs.view(ep_batch.batch_size, self.n_agents, -1),\
               all_q_alone.view(ep_batch.batch_size, self.n_agents, -1), all_q_interact.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.to(self.args.device)

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs_interact(self, batch, t):
        bs = batch.batch_size
        inputs = []
        states = batch["state"][:, t]       # (bs, state_shape)
        local_states = states.reshape(bs, self.n_entities, -1)      # (bs, n_entities, local_state_size)
        inputs.append(local_states)
        if self.args.obs_last_action:
            entities_last_acts = th.zeros((bs, self.n_entities, self.n_actions), dtype=batch["actions_onehot"].dtype, device=batch.device)
            # if t == 0:
            #     entities_last_acts[:, :self.n_agents, :] = th.zeros_like(batch["actions_onehot"][:, t])      # can be omited, batch["actions_onehot"].shape=(bs, n_agents, n_actions)
            if t > 0:
                last_actions = batch["actions_onehot"][:, t - 1]  # shape=(bs, n_agents, n_actions)
                entities_last_acts[:, :self.n_agents, :] = last_actions
            inputs.append(entities_last_acts)
        inputs_interact = th.cat(inputs, dim=-1)     # (bs, n_entities, local_state_size+n_actions)
        return inputs_interact

    def _get_input_shape(self, scheme):
        input_shape = self.local_state_size
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        return input_shape