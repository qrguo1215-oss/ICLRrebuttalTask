import argparse
import json
import os
from collections import Counter, defaultdict

import torch


def load_assignments(assignments_path):
    rows = []
    with open(assignments_path) as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No assignments found in {assignments_path}")
    return rows


def cluster_ids_from_rows(rows):
    return sorted({int(row["cluster_id"]) for row in rows})


def summarize_rows(rows):
    cluster_sizes = Counter(int(row["cluster_id"]) for row in rows)
    example_counts = defaultdict(set)
    source_counts = defaultdict(Counter)
    for row in rows:
        cluster_id = int(row["cluster_id"])
        example_counts[cluster_id].add((row["source_name"], int(row["example_id"])))
        source_counts[cluster_id][row["source_name"]] += 1
    return {
        "num_rows": len(rows),
        "cluster_sizes": {int(k): int(v) for k, v in sorted(cluster_sizes.items())},
        "num_examples_per_cluster": {int(k): int(len(v)) for k, v in sorted(example_counts.items())},
        "source_breakdown_per_cluster": {
            int(k): {source: int(count) for source, count in sorted(counter.items())}
            for k, counter in sorted(source_counts.items())
        },
    }


def compute_cluster_means(hidden_paths, source_names, layer, rows):
    if len(hidden_paths) != len(source_names):
        raise ValueError("hidden_paths and source_names must have the same length")

    hidden_by_source = {}
    for hidden_path, source_name in zip(hidden_paths, source_names):
        hidden = torch.load(hidden_path, map_location="cpu")
        if layer >= len(hidden):
            raise ValueError(f"Layer {layer} not available in {hidden_path}")
        hidden_by_source[source_name] = hidden[layer]

    sums = {}
    counts = Counter()
    dim = None

    for row in rows:
        source_name = row["source_name"]
        example_id = int(row["example_id"])
        step_index = int(row["step_index"])
        cluster_id = int(row["cluster_id"])

        step_tensor = hidden_by_source[source_name][example_id]["step"][step_index].float()
        if dim is None:
            dim = int(step_tensor.shape[0])
        if cluster_id not in sums:
            sums[cluster_id] = torch.zeros_like(step_tensor)
        sums[cluster_id] += step_tensor
        counts[cluster_id] += 1

    means = {}
    for cluster_id in sorted(sums):
        means[cluster_id] = sums[cluster_id] / counts[cluster_id]
    return means, counts, dim


def save_outputs(output_dir, means, counts, metadata):
    os.makedirs(output_dir, exist_ok=True)

    ordered_cluster_ids = sorted(means)
    stacked = torch.stack([means[cid] for cid in ordered_cluster_ids], dim=0)
    torch.save(stacked, os.path.join(output_dir, "cluster_mean_vectors.pt"))

    index_to_cluster = {}
    for idx, cluster_id in enumerate(ordered_cluster_ids):
        vector = means[cluster_id]
        torch.save(vector, os.path.join(output_dir, f"cluster_{cluster_id}_mean.pt"))
        index_to_cluster[int(idx)] = int(cluster_id)

    metadata = dict(metadata)
    metadata["vector_dim"] = int(stacked.shape[1])
    metadata["num_clusters"] = int(stacked.shape[0])
    metadata["cluster_order"] = ordered_cluster_ids
    metadata["stack_index_to_cluster_id"] = index_to_cluster
    metadata["mean_vector_norms"] = {
        int(cluster_id): float(means[cluster_id].norm().item()) for cluster_id in ordered_cluster_ids
    }
    metadata["cluster_sizes"] = {
        int(cluster_id): int(counts[cluster_id]) for cluster_id in ordered_cluster_ids
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as fout:
        json.dump(metadata, fout, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-cluster mean vectors from multiple hidden sources."
    )
    parser.add_argument("--hidden_paths", type=str, nargs="+", required=True)
    parser.add_argument("--source_names", type=str, nargs="+", required=True)
    parser.add_argument("--assignments_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--source_cluster_dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_assignments(args.assignments_path)
    means, counts, dim = compute_cluster_means(args.hidden_paths, args.source_names, args.layer, rows)
    summary = summarize_rows(rows)
    metadata = {
        "hidden_paths": args.hidden_paths,
        "source_names": args.source_names,
        "assignments_path": args.assignments_path,
        "source_cluster_dir": args.source_cluster_dir,
        "layer": args.layer,
        "vector_type": "raw_cluster_mean",
        "summary": summary,
        "cluster_ids": cluster_ids_from_rows(rows),
        "vector_dim": dim,
    }
    save_outputs(args.output_dir, means, counts, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
