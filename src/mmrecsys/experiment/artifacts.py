import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch


def array_digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(f"{array.shape}:{array.dtype}".encode())
    view = memoryview(array).cast("B")
    for start in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def content_key(metadata: dict) -> str:
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class GraphCache:
    """Content-addressed CPU COO storage; never pickle model objects."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def get_or_build(self, metadata: dict, build) -> torch.Tensor:
        metadata = {"algorithm_version": 1, **metadata}
        path = self.root / (content_key(metadata) + ".pt")
        if path.exists():
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["metadata"] != metadata:
                raise ValueError(f"Graph cache metadata mismatch: {path}")
            return torch.sparse_coo_tensor(record["indices"], record["values"],
                                           record["shape"], check_invariants=True).coalesce()
        graph = build().detach().cpu().coalesce()
        atomic_torch_save({"metadata": metadata, "indices": graph.indices(),
                           "values": graph.values(), "shape": list(graph.shape)}, path)
        return graph
