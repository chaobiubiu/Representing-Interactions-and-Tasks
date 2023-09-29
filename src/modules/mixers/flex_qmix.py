import torch as th
import torch.nn as nn
import torch.nn.functional as F


class EntityAttentionLayer(nn.Module):
    def __init__(self, in_dim, embed_dim, out_dim, args):
        super(EntityAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_heads = args.n_heads
        self.n_agents = args.n_agents
        self.args = args

        assert self.embed_dim % self.n_heads == 0, "Embed dim must be divisible by n_heads"
        self.head_dim = self.embed_dim // self.n_heads
        self.register_buffer('scale_factor', th.scalar_tensor(self.head_dim).sqrt())

        self.in_trans = nn.Linear(self.in_dim, self.embed_dim * 3, bias=False)
        self.out_trans = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, entities, pre_mask=None, post_mask=None):
        """
        entities: Entity representations
            shape: batch size, # of entities, embedding dimension
        pre_mask: Which agent-entity pairs are not available (observability and/or padding).
                  Mask out before attention.
            shape: batch_size, # of agents, # of entities
        post_mask: Which agents/entities are not available. Zero out their outputs to
                   prevent gradients from flowing back. Shape of 2nd dim determines
                   whether to compute queries for all entities or just agents.
            shape: batch size, # of agents (or entities)
        ret_attn_logits: whether to return attention logits
            None: do not return
            "max": take max over heads
            "mean": take mean over heads

        ====== For agent's initialization:
        entities.shape=(bs*t, n_entities, embedding_dim), pre_mask=obs_mask.shape=(bs*t, n_entities, n_entities),
        post_mask=agent_mask.shape=(bs*t, n_agents)

        ====== For initialization in mixer:
        ...entities.shape=(bs*t, n_entities, hypernet_embed), pre_mask=attn_mask.shape=(bs*t, n_agents, n_entities),
        post_mask=agent_mask.shape=(bs*t, n_agents)

        Return shape: batch size, # of agents, embedding dimension
        """
        entities_t = entities.transpose(0, 1)       # (n_entities, bs*t, embedding_dim)
        n_queries = post_mask.shape[1]              # n_agents
        pre_mask = pre_mask[:, :n_queries]          # (bs*t, n_agents, n_entities)
        ne, bs, ed = entities_t.shape
        query, key, value = self.in_trans(entities_t).chunk(3, dim=2)

        query = query[:n_queries]               # (n_agents, bs*t, embedding_dim)

        query_spl = query.reshape(n_queries, bs * self.n_heads, self.head_dim).transpose(0, 1)      # bst*n_heads, n_agents, head_dim    q
        key_spl = key.reshape(ne, bs * self.n_heads, self.head_dim).permute(1, 2, 0)                # bst*n_heads, head_dim, n_entities  k
        value_spl = value.reshape(ne, bs * self.n_heads, self.head_dim).transpose(0, 1)             # bst*n_heads, n_entities, head_dim  v

        attn_logits = th.bmm(query_spl, key_spl) / self.scale_factor            # (bst*n_heads, n_agents, n_entities)
        if pre_mask is not None:    # (bs*t, n_agents, n_entities), the partial observable mask of each agent, unobservable=1, observable=0
            pre_mask_rep = pre_mask.repeat_interleave(self.n_heads, dim=0)      # (bs*t*n_heads, n_agents, n_entities)
            attn_logits = attn_logits.masked_fill(pre_mask_rep[:, :, :ne].to(th.bool), -float('Inf'))            # 将unobservble entity相应的attention weight变为0
        attn_weights = F.softmax(attn_logits, dim=2)
        # 注意：当前环境中未包括的agent以及landmark，obs_mask中相应行所有元素均为1
        # some weights might be NaN (if agent is inactive and all entities were masked)
        attn_weights = attn_weights.masked_fill(attn_weights != attn_weights, 0)
        attn_outs = th.bmm(attn_weights, value_spl)     # (bs*t*n_heads, n_agents, head_dim)
        attn_outs = attn_outs.transpose(0, 1).reshape(n_queries, bs, self.embed_dim)    # (n_agents, bs*t, n_heads*head_dim)
        attn_outs = attn_outs.transpose(0, 1)               # (bs*t, n_agents, n_heads*head_dim)
        attn_outs = self.out_trans(attn_outs)               # (bs*t, n_agents, out_dim)
        if post_mask is not None:           # (bs*t, n_agents), in-existing is 1, existing is 0
            attn_outs = attn_outs.masked_fill(post_mask.unsqueeze(2).to(th.bool), 0)        # 将不存在的agents相应的聚合info置0

        return attn_outs


class AttentionHyperNet(nn.Module):
    """
    mode='matrix' gets you a <n_agents x mixing_embed_dim> sized matrix
    mode='vector' gets you a <mixing_embed_dim> sized vector by averaging over agents
    mode='alt_vector' gets you a <n_agents> sized vector by averaging over embedding dim
    mode='scalar' gets you a scalar by averaging over agents and embed dim
    ...per set of entities
    """
    def __init__(self, args, extra_dims=0, mode='matrix'):
        super(AttentionHyperNet, self).__init__()
        self.args = args
        self.mode = mode
        self.extra_dims = extra_dims
        self.entity_dim = args.local_state_size
        if self.args.obs_last_action:
            self.entity_dim += args.n_actions
        if extra_dims > 0:
            self.entity_dim += extra_dims

        hypernet_embed = args.hypernet_embed
        atten_dim = args.atten_dim
        self.fc1 = nn.Linear(self.entity_dim, hypernet_embed)
        self.attn = EntityAttentionLayer(hypernet_embed,
                                         atten_dim,
                                         hypernet_embed,
                                         args)
        self.fc2 = nn.Linear(hypernet_embed, args.mixing_embed_dim)

    def forward(self, entities, entity_mask, attn_mask=None):
        # entities.shape=(bs*t, n_entities, local_state_size+n_actions), entity_mask.shape=(bs*t, n_entities)
        x1 = F.relu(self.fc1(entities))     # (bs*t, n_entities, hypernet_embed)
        agent_mask = entity_mask[:, :self.args.n_agents]    # (bs*t, n_agents)
        if attn_mask is None:
            # create attn_mask from entity mask,  shape=(bs*t, n_agents, n_entities), non-existing is 1, otherwise is 0;
            attn_mask = th.tensor(1, dtype=th.float32, device=self.args.device) - \
                        th.bmm((th.tensor(1, dtype=th.float32, device=self.args.device) - agent_mask.to(th.float)).unsqueeze(2),
                               (th.tensor(1, dtype=th.float32, device=self.args.device) - entity_mask.to(th.float)).unsqueeze(1))
        x2 = self.attn(x1, pre_mask=attn_mask, post_mask=agent_mask)        # (bs*t, n_agents, hypernet_embed)
        x3 = self.fc2(x2)                           # (bs*t, n_agents, mixing_embed_dim)
        x3 = x3.masked_fill(agent_mask.unsqueeze(2).to(th.bool), 0)     # Mask the un-existing agents
        if self.mode == 'vector':
            return x3.mean(dim=1)
        elif self.mode == 'alt_vector':
            return x3.mean(dim=2)
        elif self.mode == 'scalar':
            return x3.mean(dim=(1, 2))
        return x3       # (bs*t, n_agents, mixing_embed_dim)


class FlexQMixer(nn.Module):
    def __init__(self, args):
        super(FlexQMixer, self).__init__()
        self.args = args

        self.n_agents = args.n_agents

        self.embed_dim = args.mixing_embed_dim

        self.hyper_w_1 = AttentionHyperNet(args, mode='matrix')
        self.hyper_w_final = AttentionHyperNet(args, mode='vector')
        self.hyper_b_1 = AttentionHyperNet(args, mode='vector')
        # V(s) instead of a bias for the last layers
        self.V = AttentionHyperNet(args, mode='scalar')

        self.non_lin = F.elu
        # if getattr(self.args, "mixer_non_lin", "elu") == "tanh":
        #     self.non_lin = F.tanh

    def forward(self, agent_qs, inputs, imagine_groups=None):
        # inputs include state_inps and scenario_mask
        # state_inps.shape=(bs, seq_t-1, n_entities, local_state_size+n_actions), scenario_mask.shape=(bs, seq_t-1, n_entities)
        # If imagine_groups is None, agent_qs.shape=(bs, seq_t-1, n_agents), otherwise agent_qs.shape=(bs, seq_t-1, n_agents*2)
        entities, entity_mask = inputs      # entities.shape=(bs, seq_t-1, n_entities, local_state_size+n_actions)
        bs, max_t, ne, ed = entities.shape

        entities = entities.reshape(bs * max_t, ne, ed)
        entity_mask = entity_mask.reshape(bs * max_t, ne)
        if imagine_groups is not None:
            agent_qs = agent_qs.view(-1, 1, self.n_agents * 2)      # shape=(bs*t, 1, n_agents*2)
            Wmask, Imask = imagine_groups       # each of them, shape=(bs, t, n_entities, n_entities)
            w1_W = self.hyper_w_1(entities, entity_mask, attn_mask=Wmask.reshape(bs * max_t, ne, ne))
            w1_I = self.hyper_w_1(entities, entity_mask, attn_mask=Imask.reshape(bs * max_t, ne, ne))
            w1 = th.cat([w1_W, w1_I], dim=1)        # (bs*t, n_agents*2, mixing_embed_dim)
        else:
            agent_qs = agent_qs.view(-1, 1, self.n_agents)      # (bs*t, 1, n_agents)
            # First layer
            w1 = self.hyper_w_1(entities, entity_mask)       # (bs*t, n_agents, mixing_embed_dim)
        b1 = self.hyper_b_1(entities, entity_mask)          # (bs*t, mixing_embed_dim)
        w1 = w1.view(bs * max_t, -1, self.embed_dim)        # (bs*t, n_agents, mixing_embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)                 # (bs*t, 1, mixing_embed_dim)

        w1 = th.abs(w1)
        hidden = self.non_lin(th.bmm(agent_qs, w1) + b1)        # (bs*t, 1, mixing_embed_dim) + (bs*t, 1, mixing_embed_dim)

        # Second layer
        w_final = th.abs(self.hyper_w_final(entities, entity_mask))     # (bs*t, mixing_embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)       # (bs*t, mixing_embed_dim, 1)

        # State-dependent bias
        v = self.V(entities, entity_mask).view(-1, 1, 1)        # (bs*t, 1, 1)

        # Compute final output
        y = th.bmm(hidden, w_final) + v     # (bs*t, 1, 1) + (bs*t, 1, 1)
        # Reshape and return
        q_tot = y.view(bs, -1, 1)       # (bs, seq_t, 1)

        return q_tot