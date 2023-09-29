from .run import run as default_run
from .on_off_run import run as on_off_run
from .per_run import run as per_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["on_off"] = on_off_run
REGISTRY["per_run"] = per_run