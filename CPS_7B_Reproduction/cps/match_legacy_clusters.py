#!/usr/bin/env python3
"""Match new CPS cluster IDs to historical 7B IDs and freeze localBest.

KMeans labels are permutation-invariant, so copying ``-0.6*C2`` by numeric ID
would be invalid. Both prototype sets live in the same model/layer space. We
subtract each run's six-prototype centroid, cosine-match the residual shapes
with a Hungarian assignment, then transfer the prespecified localBest direction
to the corresponding new prototype.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as functional
from scipy.optimize import linear_sum_assignment


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_write(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def cosine_matrix(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return functional.normalize(left.float(), dim=1) @ functional.normalize(right.float(), dim=1).T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-vectors", type=Path, required=True)
    parser.add_argument("--new-vectors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-localbest-cluster", type=int, default=2)
    parser.add_argument("--legacy-localbest-coef", type=float, default=-0.6)
    args = parser.parse_args()

    legacy = torch.load(args.legacy_vectors, map_location="cpu", weights_only=True).float()
    new = torch.load(args.new_vectors, map_location="cpu", weights_only=True).float()
    if legacy.shape != new.shape or legacy.ndim != 2:
        raise ValueError(f"prototype shape mismatch: legacy={tuple(legacy.shape)}, new={tuple(new.shape)}")

    raw_cosine = cosine_matrix(legacy, new)
    legacy_centered = legacy - legacy.mean(dim=0, keepdim=True)
    new_centered = new - new.mean(dim=0, keepdim=True)
    centered_cosine = cosine_matrix(legacy_centered, new_centered)
    legacy_indices, new_indices = linear_sum_assignment(-centered_cosine.numpy())
    mapping = {int(old): int(new_id) for old, new_id in zip(legacy_indices, new_indices)}

    old_id = args.legacy_localbest_cluster
    if old_id not in mapping:
        raise ValueError(f"legacy cluster {old_id} was not matched")
    new_id = mapping[old_id]
    localbest = args.legacy_localbest_coef * new[new_id]
    vector_path = args.output_dir / "localbest_transferred.pt"
    atomic_torch_save(localbest, vector_path)

    assignment = []
    for old in sorted(mapping):
        matched = mapping[old]
        assignment.append(
            {
                "legacy_cluster": old,
                "new_cluster": matched,
                "centered_cosine": float(centered_cosine[old, matched]),
                "raw_cosine": float(raw_cosine[old, matched]),
                "centered_row_margin": float(
                    centered_cosine[old, matched]
                    - torch.topk(centered_cosine[old], k=2).values[1]
                ),
            }
        )

    manifest = {
        "matching_rule": "Hungarian assignment maximizing cosine after centering the six prototypes within each run",
        "legacy_vectors": str(args.legacy_vectors),
        "new_vectors": str(args.new_vectors),
        "legacy_to_new_cluster": {str(key): value for key, value in sorted(mapping.items())},
        "assignments": assignment,
        "legacy_localbest": {
            "cluster_id": old_id,
            "coefficient": args.legacy_localbest_coef,
            "historical_formula": f"{args.legacy_localbest_coef:g}*C{old_id}",
            "historical_math500_accuracy": 0.884,
        },
        "transferred_localbest": {
            "new_cluster_id": new_id,
            "coefficient": args.legacy_localbest_coef,
            "formula": f"{args.legacy_localbest_coef:g}*C{new_id}",
            "path": str(vector_path),
            "norm": float(localbest.norm()),
        },
        "raw_cosine_matrix": raw_cosine.tolist(),
        "centered_cosine_matrix": centered_cosine.tolist(),
    }
    atomic_json_write(manifest, args.output_dir / "match_manifest.json")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
