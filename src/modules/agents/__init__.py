REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .mlp_agent import MLPAgent
from .scvd_agent_transformer import SCVDRNNAgent
from .attn_agent import ATTNRNNAgent
from .updet_agent import UPDETAgent
from .refil_agent import REFILAgent
from .roma_agent import ROMAAgent
from .scvd_agent_ablation import SCVDRNNAgent_Ablation

REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["scvd_rnn"] = SCVDRNNAgent
REGISTRY["attn_rnn"] = ATTNRNNAgent
REGISTRY['updet'] = UPDETAgent
REGISTRY['refil'] = REFILAgent
REGISTRY['roma'] = ROMAAgent
REGISTRY['scvd_ablation_rnn'] = SCVDRNNAgent_Ablation