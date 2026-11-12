from dataclasses import dataclass
from typing import Callable

from .data.sampling import BatchSpec
from .models.mgcn import MGCN, MGCNConfig
from .models.lirdrec import LIRDRec, LIRDRecConfig
from .models.pgl import PGL, PGLConfig
from .models.lgmrec import LGMRec, LGMRecConfig
from .models.bm3 import BM3, BM3Config
from .models.smore import SMORE, SMOREConfig
from .models.freedom import FREEDOM, FREEDOMConfig


@dataclass(frozen=True)
class ModelEntry:
    factory: Callable
    modalities: tuple[str, ...]
    batch_spec: BatchSpec
    parse_config: Callable


MODELS = {
    "bm3": ModelEntry(BM3, ("image", "text"), BatchSpec("positive"), BM3Config.parse),
    "smore": ModelEntry(SMORE, ("image", "text"), BatchSpec("pairwise", 1), SMOREConfig.parse),
    "freedom": ModelEntry(FREEDOM, ("image", "text"), BatchSpec("pairwise", 1), FREEDOMConfig.parse),
    "pgl": ModelEntry(PGL, ("image", "text"), BatchSpec("pairwise", 1), PGLConfig.parse),
    "lgmrec": ModelEntry(LGMRec, ("image", "text"), BatchSpec("pairwise", 1), LGMRecConfig.parse),
    "mgcn": ModelEntry(MGCN, ("image", "text"), BatchSpec("pairwise", 1), MGCNConfig.parse),
    "lirdrec": ModelEntry(LIRDRec, ("image", "text"), BatchSpec("pairwise", 1), LIRDRecConfig.parse),
}


def get_model(name):
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(MODELS)}")
    return MODELS[name]
