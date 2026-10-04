from dataclasses import dataclass
from typing import Callable

from .data.sampling import BatchSpec
from .models.mgcn import MGCN, MGCNConfig
from .models.lirdrec import LIRDRec, LIRDRecConfig


@dataclass(frozen=True)
class ModelEntry:
    factory: Callable
    modalities: tuple[str, ...]
    batch_spec: BatchSpec
    parse_config: Callable


MODELS = {
    "mgcn": ModelEntry(MGCN, ("image", "text"), BatchSpec("pairwise", 1), MGCNConfig.parse),
    "lirdrec": ModelEntry(LIRDRec, ("image", "text"), BatchSpec("pairwise", 1), LIRDRecConfig.parse),
}


def get_model(name):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(MODELS)}")
    return MODELS[name]
