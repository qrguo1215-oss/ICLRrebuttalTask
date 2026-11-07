import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from cluster_multi_hidden_sources import (
    attach_cluster_ids,
    build_records_from_sources,
    cluster_composition,
    fit_preprocessor,
    summarize_step_counts,
    transform_vectors,
    write_json,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fixed-k KMeans for multiple PCA dims and seeds, saving every seed's assignments."
    )
    parser.add_argument("--hidden_paths", type=str, nargs="+", required=True)
    parser.add_argument("--source_names", type=str, nargs="+", required=True)
    parser.add_argument("--prompts_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--pca_dims", type=int, nargs="+", default=[0, 20, 50, 100, 200, 400])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--max_steps_per_example", type=int, default=30)
    parser.add_argument("--n_init", type=int, default=50)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--think_only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def fit_labels(vectors, k, seed, n_init, max_iter):
    model = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
    labels = model.fit_predict(vectors)
    return model, labels


def balance_from_labels(labels):
    counts = Counter(int(label) for label in labels.tolist())
    if not counts:
        return 0.0
    values = list(counts.values())
    return min(values) / max(values)


def write_run_dir(
    run_dir,
    args,
    pca_dim,
    seed,
    effective_pca_dim,
    explained_variance_ratio_sum,
    source_summary,
    step_count_stats,
    sampled_rows,
    full_rows,
    sampled_labels,
    full_labels,
    silhouette,
    inertia,
    stability_row,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "hidden_paths": args.hidden_paths,
        "source_names": args.source_names,
        "prompts_paths": args.prompts_paths,
        "layer": args.layer,
        "num_sources": len(args.hidden_paths),
        "source_summary": source_summary,
        "num_sampled_steps": len(sampled_rows),
        "num_full_steps": len(full_rows),
        "max_steps_per_example_for_k_search": args.max_steps_per_example,
        "requested_pca_dim": pca_dim,
        "effective_pca_dim": effective_pca_dim,
        "explained_variance_ratio_sum": explained_variance_ratio_sum,
        "seed": seed,
        "stability_seeds": args.seeds,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "best_k_by_silhouette": args.k,
        "step_count_summary": summarize_step_counts(step_count_stats),
    }
    write_json(run_dir / "config_summary.json", config)
    write_json(
        run_dir / "kmeans_metrics.json",
        [
            {
                "k": args.k,
                "silhouette": float(silhouette),
                "inertia": float(inertia),
                "cluster_sizes": {
                    int(key): int(value)
                    for key, value in sorted(Counter(sampled_labels.tolist()).items())
                },
            }
        ],
    )
    write_json(run_dir / "kmeans_seed_stability.json", [stability_row])
    write_json(
        run_dir / "best_k.json",
        {
            "best_k_by_silhouette": args.k,
            "silhouette": float(silhouette),
            "inertia": float(inertia),
            "stability_ari_mean": stability_row["ari_mean"],
            "stability_ari_min": stability_row["ari_min"],
        },
    )
    write_json(
        run_dir / "cluster_sizes_sampled.json",
        {int(key): int(value) for key, value in sorted(Counter(sampled_labels.tolist()).items())},
    )
    write_json(
        run_dir / "cluster_sizes_full.json",
        {int(key): int(value) for key, value in sorted(Counter(full_labels.tolist()).items())},
    )
    write_json(run_dir / "cluster_heuristic_composition_sampled.json", cluster_composition(sampled_rows))
    write_json(run_dir / "cluster_heuristic_composition_full.json", cluster_composition(full_rows))
    write_jsonl(run_dir / "assignments_sampled.jsonl", sampled_rows)
    write_jsonl(run_dir / "assignments_full.jsonl", full_rows)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    sampled_vectors_t, sampled_records, step_count_stats, source_summary = build_records_from_sources(
        hidden_paths=args.hidden_paths,
        source_names=args.source_names,
        prompts_paths=args.prompts_paths,
        layer=args.layer,
        max_steps_per_example=args.max_steps_per_example,
        seed=0,
        think_only=args.think_only,
    )
    full_vectors_t, full_records, _, _ = build_records_from_sources(
        hidden_paths=args.hidden_paths,
        source_names=args.source_names,
        prompts_paths=args.prompts_paths,
        layer=args.layer,
        max_steps_per_example=None,
        seed=0,
        think_only=args.think_only,
    )

    sampled_vectors = sampled_vectors_t.numpy()
    full_vectors = full_vectors_t.numpy()
    all_summary_rows = []

    for pca_dim in args.pca_dims:
        mean, pca, sampled_transformed, effective_pca_dim, explained_variance_ratio_sum = fit_preprocessor(
            sampled_vectors,
            pca_dim,
        )
        full_transformed = transform_vectors(full_vectors, mean, pca)

        sampled_seed_payloads = []
        for seed in args.seeds:
            sampled_model, sampled_labels = fit_labels(
                sampled_transformed,
                args.k,
                seed,
                args.n_init,
                args.max_iter,
            )
            silhouette = float(silhouette_score(sampled_transformed, sampled_labels))
            sampled_seed_payloads.append(
                {
                    "seed": seed,
                    "sampled_model": sampled_model,
                    "sampled_labels": sampled_labels,
                    "silhouette": silhouette,
                    "inertia": float(sampled_model.inertia_),
                    "balance": balance_from_labels(sampled_labels),
                }
            )

        pairwise_ari = []
        for item_a, item_b in combinations(sampled_seed_payloads, 2):
            pairwise_ari.append(
                float(adjusted_rand_score(item_a["sampled_labels"], item_b["sampled_labels"]))
            )
        stability_row = {
            "k": args.k,
            "seeds": args.seeds,
            "num_runs": len(args.seeds),
            "silhouette_mean": float(
                sum(item["silhouette"] for item in sampled_seed_payloads) / len(sampled_seed_payloads)
            ),
            "silhouette_std": float(
                torch.tensor([item["silhouette"] for item in sampled_seed_payloads]).std(unbiased=False).item()
            ),
            "inertia_mean": float(
                sum(item["inertia"] for item in sampled_seed_payloads) / len(sampled_seed_payloads)
            ),
            "inertia_std": float(
                torch.tensor([item["inertia"] for item in sampled_seed_payloads]).std(unbiased=False).item()
            ),
            "ari_mean": float(sum(pairwise_ari) / len(pairwise_ari)) if pairwise_ari else 1.0,
            "ari_std": float(torch.tensor(pairwise_ari).std(unbiased=False).item()) if pairwise_ari else 0.0,
            "ari_min": float(min(pairwise_ari)) if pairwise_ari else 1.0,
            "ari_max": float(max(pairwise_ari)) if pairwise_ari else 1.0,
        }

        for item in sampled_seed_payloads:
            seed = item["seed"]
            run_dir = args.output_root / f"pca{pca_dim}" / f"seed{seed}"
            if (run_dir / "assignments_full.jsonl").exists() and (run_dir / "best_k.json").exists():
                continue
            full_model, full_labels = fit_labels(
                full_transformed,
                args.k,
                seed,
                args.n_init,
                args.max_iter,
            )
            sampled_rows = attach_cluster_ids(sampled_records, item["sampled_labels"])
            full_rows = attach_cluster_ids(full_records, full_labels)
            write_run_dir(
                run_dir=run_dir,
                args=args,
                pca_dim=pca_dim,
                seed=seed,
                effective_pca_dim=effective_pca_dim,
                explained_variance_ratio_sum=explained_variance_ratio_sum,
                source_summary=source_summary,
                step_count_stats=step_count_stats,
                sampled_rows=sampled_rows,
                full_rows=full_rows,
                sampled_labels=item["sampled_labels"],
                full_labels=full_labels,
                silhouette=item["silhouette"],
                inertia=item["inertia"],
                stability_row=stability_row,
            )
            all_summary_rows.append(
                {
                    "run_dir": str(run_dir),
                    "pca_dim": pca_dim,
                    "seed": seed,
                    "k": args.k,
                    "silhouette": item["silhouette"],
                    "inertia": item["inertia"],
                    "balance": item["balance"],
                    "stability_ari_mean": stability_row["ari_mean"],
                    "stability_ari_min": stability_row["ari_min"],
                }
            )

    write_json(args.output_root / "fixed_k_seed_runs_summary.json", all_summary_rows)
    print(json.dumps({"output_root": str(args.output_root), "runs_written": len(all_summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
