import torch as th
import torch.nn as nn
import torch.nn.functional as F

FLOAT_MIN, FLOAT_MAX = -3.4e38, 3.4e38


class EntityAttentionLayer(nn.Module):
    def __init__(self, args, in_dim, embed_dim, out_dim):
        super(EntityAttentionLayer, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_agents = args.n_agents

        # q/k/v soft attention / or multi-head attention?
        self.query = nn.Linear(self.in_dim, self.embed_dim)
        self.key = nn.Linear(self.in_dim, self.embed_dim)
        self.value = nn.Sequential(nn.Linear(self.in_dim, self.embed_dim), nn.ReLU())
        self.scale_factor = th.scalar_tensor(self.embed_dim).sqrt()

        self.out = nn.Linear(self.embed_dim * 2, self.out_dim)

    def forward(self, inputs, pre_mask=None, post_mask=None):
        """
        inputs: the embeded state representations, shape: (bs_t*n_entities, hypernet_embed)
        pre_mask: Which agent-entity pairs are not available. shape: (bs_t, n_agents, n_entities), existing agent is 0, non-exist agent is 1
        post_mask: Which agents/entities are not available. shape=(bs_t, n_agents), existing agent is 0, non-exist agent is 1
        Return shape: (bs_t, n_agents, out_dim), bs_t is writed as bs
        """
        bs, n_agents, n_entities = pre_mask.size()
        queries = self.query(inputs).reshape(bs, n_entities, -1)[:, :n_agents, :]  # (bs, n_agents, embed_dim)
        keys = self.key(inputs).reshape(bs, n_entities, -1).permute(0, 2, 1)  # (bs, embed_dim, n_entities)
        values = self.value(inputs).reshape(bs, n_entities, -1)  # (bs, n_entities, embed_dim)
        score = th.bmm(queries, keys) / self.scale_factor       # shape=(bs, n_agents, n_entities)
        # TODO: 注意所有entities相应的local state即便mask后值为0，但是有bias的存在仍然可能会导致其weight以及value不为0，所以需要特意mask掉，之前svn_agent文件中的attention部分也要注意这个问题
        if pre_mask is not None:
            score = score.masked_fill(pre_mask[:, :, :], -float('Inf'))            # 将mask中entity_info为0的部分(即不存在的entity)，其相应attention weight变为0
        # 去除每个agent各自的contribution
        full_mask = th.zeros((bs, n_agents, n_entities), device=inputs.device)
        mask = th.eye(n_agents, device=inputs.device)[None, :, :]  # shape=(1, n_agents, n_agents)
        mask_inf = th.clamp(th.log(th.tensor(1) - mask), FLOAT_MIN, FLOAT_MAX)  # 对角为-inf,其余位置为0
        full_mask[:, :n_agents, :n_agents] = mask_inf
        masked_score = score + full_mask        # 到这一步已经去除agent自身的contribution以及当前环境中不包括的entity
        weights = nn.functional.softmax(masked_score, dim=2)  # shape=(bs, n_agents, n_entities)
        attended_values = th.bmm(weights, values).reshape(bs, n_agents, -1)  # (bs, n_agents, embed_dim)
        inputs = inputs.reshape(bs, n_entities, -1)[:, :n_agents, :]
        concat_outputs = th.cat([inputs, attended_values], dim=-1).reshape(bs * n_agents, -1)  # shape=(bs*n_agents, embed_dim*2)
        outputs = self.out(concat_outputs).reshape(bs, n_agents, self.out_dim)
        # TODO: 所有attention相关的部分都需要仔细检查，需要根据scenario_mask来过滤掉needless information, remaining to be done tomorrow.
        if post_mask is not None:
            outputs = outputs.masked_fill(post_mask.unsqueeze(2), 0)        # non-existing entity is 0, otherwise 1.
        return outputs

# a.masked_fill(mask_index, value)方法会将mask_index中元素1所在的index相应的a中元素，替换为value值

class EntityPoolingLayer(nn.Module):
    def __init__(self, args, in_dim, embed_dim, out_dim, pooling_type):
        super(EntityPoolingLayer, self).__init__()
        self.args = args
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.pooling_type = pooling_type
        self.n_agents = args.n_agents

        self.in_trans = nn.Linear(self.in_dim, self.embed_dim)
        self.out_trans = nn.Linear(self.embed_dim, self.out_dim)

    def forward(self, inputs, pre_mask=None, post_mask=None):
        """
        inputs: the embeded state representations, shape: (bs*n_entities, hypernet_embed)
        pre_mask: Which agent-entity pairs are not available, mask the padded info. shape: (bs, n_agents, n_entities)
        post_mask: Which agents/entities are not available, mask the extra info. shape: (bs, n_agents)
        Return shape: (bs, n_agents, out_dim)
        """
        bs, n_agents, n_entities = pre_mask.size()
        inputs_trans = self.in_trans(inputs)        # shape=(bs*n_entities, embed_dim)
        inputs_trans_rep = inputs_trans.reshape(bs, n_entities, -1).unsqueeze(dim=1).expand(-1, n_agents, -1, -1)
        if pre_mask is not None:
            inputs_trans_rep = inputs_trans_rep.masked_fill(pre_mask.unsqueeze(dim=3), 0)       # 对于padded的entity info，处理掉
        if self.pooling_type == 'max':
            pool_outs = inputs_trans_rep.max(dim=2)[0]      # 取所有entities local state的最大值
        elif self.pooling_type == 'mean':
            pool_outs = inputs_trans_rep.mean(dim=2)        # 取所有entities local state的平均值
        else:
            raise Exception("Unkonwn pooling operation.")
        # pool_outs.shape=(bs, n_agents, embed_dim)，由于max/mean操作，所以原本dim=2的n_entities消失
        pool_outs = self.out_trans(pool_outs)

        if post_mask is not None:
            pool_outs = pool_outs.masked_fill(post_mask.unsqueeze(2), 0)

        return pool_outs