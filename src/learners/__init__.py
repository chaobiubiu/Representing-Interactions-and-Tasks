from .q_learner import QLearner
from .nq_learner import NQLearner
from .scvd_learner import SCVDLearner
from .attn_learner import AttnQLearner
from .refil_learner import REFILLearner
from .roma_learner import ROMALearner
from .scvd_ablation_learner import Ablation_SCVDLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["nq_learner"] = NQLearner
REGISTRY["scvd_learner"] = SCVDLearner
REGISTRY["attn_learner"] = AttnQLearner
REGISTRY["refil_learner"] = REFILLearner
REGISTRY['roma_learner'] = ROMALearner
REGISTRY['scvd_ablation_learner'] = Ablation_SCVDLearner