from src.modules.agents import REGISTRY as agent_REGISTRY
from src.components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# This multi-agent controller shares parameters between agents
class REFILMAC:
    def __init__(self, scheme, groups, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_entities = args.n_entities
        self.n_actions = args.n_actions
        self.local_state_size = args.local_state_size
        self.state_shape = args.state_shape
        input_shape = self._get_input_shape(scheme)     # (local_state_size+n_actions)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False, imagine=False):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True
        else:
            raise Exception("Unknown setting in the refil controller.")

        agent_inputs = self._build_inputs(ep_batch, t)  # (bs, seq_t, n_entities, local_state_size+n_actions)
        avail_actions = ep_batch["avail_actions"][:, t]
        obs_mask = ep_batch["obs_mask"][:, t, :, :]  # (bs, seq_t, n_entities, n_entities)
        scenario_mask = ep_batch["scenario_mask"][:, t, :]  # (bs, seq_t, n_entities)
        if test_mode:
            self.agent.eval()
        # Set imagine to False when training.
        if imagine:
            # groups include (withingroup_noobs, interactgroup_noobs), which are masks for the mixer
            agent_outs, self.hidden_states, groups = self.agent(agent_inputs, obs_mask, scenario_mask, self.hidden_states, imagine=True)
        else:
            agent_outs, self.hidden_states = self.agent(agent_inputs, obs_mask, scenario_mask, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                agent_outs = agent_outs.reshape(ep_batch.batch_size * self.n_agents, -1)
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        if int_t:
            return agent_outs.squeeze(dim=1)        # shape=(bs, n_agents, n_actions)
        if imagine:
            return agent_outs, groups       # Here agent_outs.shape=(bs*3, seq_t, n_agents, n_actions)
        return agent_outs

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.to(device=self.args.device)

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        states = batch["state"][:, t]           # (bs, seq_t, state_shape)
        seq_t = states.shape[1]
        local_states = states.reshape(bs, seq_t, self.n_entities, self.local_state_size)   # (bs, seq_t, n_entities, local_state_size)
        inputs.append(local_states)
        if self.args.obs_last_action:
            entities_last_acts = th.zeros((bs, (t.stop - t.start), self.n_entities, self.n_actions), dtype=batch["actions_onehot"].dtype, device=self.args.device)
            if t.start == 0:
                entities_last_acts[:, 1:, :self.n_agents, :] = (batch["actions_onehot"][:, slice(0, t.stop - 1)])
            else:
                entities_last_acts[:, :, :self.n_agents, :] = (batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)])
            inputs.append(entities_last_acts)

        inputs = th.cat(inputs, dim=-1)         # (bs, seq_t, n_entities, local_state_size+n_actions)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = self.local_state_size
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]

        return input_shape