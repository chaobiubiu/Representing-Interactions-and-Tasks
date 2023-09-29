REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .scvd_controller import SCVDMAC
from .attn_controller import AttnMAC
from .updet_controller import UpDeTMAC
from .refil_controller import REFILMAC
from .roma_controller import ROMAMAC
from .scvd_ablation_controller import SCVDMAC_Ablation

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["scvd_mac"] = SCVDMAC
REGISTRY["attn_mac"] = AttnMAC
REGISTRY['updet_mac'] = UpDeTMAC
REGISTRY['refil_mac'] = REFILMAC
REGISTRY['roma_mac'] = ROMAMAC
REGISTRY['scvd_ablation_mac'] = SCVDMAC_Ablation