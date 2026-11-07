import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    with open(path) as fin:
        return json.load(fin)


def cluster_balance(cluster_sizes):
    sizes = [int(value) for value in cluster_sizes.values()]
    if not sizes:
        return 0.0
    return min(sizes) / max(sizes)


def main():
    parser = argparse.ArgumentParser(
        description="Select a fixed-k clustering run by intrinsic KMeans quality metrics."
    )
    parser.add_argument("--run_dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--min_balance", type=float, default=0.02)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_dir in args.run_dirs:
        config_path = run_dir / "config_summary.json"
        best_path = run_dir / "best_k.json"
        sizes_path = run_dir / "cluster_sizes_sampled.json"
        if not (config_path.exists() and best_path.exists() and sizes_path.exists()):
            continue

        config = load_json(config_path)
        best = load_json(best_path)
        sizes = load_json(sizes_path)
        stability = load_json(run_dir / "kmeans_seed_stability.json")
        stability_row = next((row for row in stability if int(row["k"]) == args.k), {})
        metrics = load_json(run_dir / "kmeans_metrics.json")
        metric_row = next((row for row in metrics if int(row["k"]) == args.k), {})

        balance = cluster_balance(sizes)
        rows.append(
            {
                "run_dir": str(run_dir),
                "requested_pca_dim": int(config["requested_pca_dim"]),
                "effective_pca_dim": int(config["effective_pca_dim"]),
                "seed": int(config["seed"]),
                "k": int(best["best_k_by_silhouette"]),
                "silhouette": float(best["silhouette"]),
                "inertia": float(best["inertia"]),
                "stability_ari_mean": None
                if "ari_mean" not in stability_row
                else float(stability_row["ari_mean"]),
                "stability_ari_min": None
                if "ari_min" not in stability_row
                else float(stability_row["ari_min"]),
                "silhouette_mean_across_stability_seeds": None
                if "silhouette_mean" not in stability_row
                else float(stability_row["silhouette_mean"]),
                "inertia_mean_across_stability_seeds": None
                if "inertia_mean" not in stability_row
                else float(stability_row["inertia_mean"]),
                "cluster_balance_min_over_max": float(balance),
                "cluster_sizes": sizes,
                "metric_cluster_sizes": metric_row.get("cluster_sizes"),
            }
        )

    if not rows:
        raise ValueError("No completed clustering run dirs were found.")

    valid_rows = [row for row in rows if row["k"] == args.k and row["cluster_balance_min_over_max"] >= args.min_balance]
    ranked_rows = valid_rows or [row for row in rows if row["k"] == args.k] or rows
    ranked_rows.sort(
        key=lambda row: (
            row["silhouette"],
            row["stability_ari_mean"] if row["stability_ari_mean"] is not None else -1.0,
            row["stability_ari_min"] if row["stability_ari_min"] is not None else -1.0,
            row["cluster_balance_min_over_max"],
            -row["inertia"],
        ),
        reverse=True,
    )
    best = ranked_rows[0]

    with open(args.output_dir / "comparison.json", "w") as fout:
        json.dump(rows, fout, ensure_ascii=False, indent=2)

    fieldnames = [
        "run_dir",
        "requested_pca_dim",
        "effective_pca_dim",
        "seed",
        "k",
        "silhouette",
        "inertia",
        "stability_ari_mean",
        "stability_ari_min",
        "silhouette_mean_across_stability_seeds",
        "inertia_mean_across_stability_seeds",
        "cluster_balance_min_over_max",
    ]
    with open(args.output_dir / "comparison.csv", "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["silhouette"], reverse=True):
            writer.writerow({key: row.get(key) for key in fieldnames})

    payload = {
        "selection_rule": (
            "rank by silhouette, then stability ARI mean/min, then cluster balance, "
            "with inertia as a final tie-breaker"
        ),
        "k": args.k,
        "min_balance": args.min_balance,
        "best_run_dir": best["run_dir"],
        "requested_pca_dim": best["requested_pca_dim"],
        "effective_pca_dim": best["effective_pca_dim"],
        "seed": best["seed"],
        "silhouette": best["silhouette"],
        "inertia": best["inertia"],
        "stability_ari_mean": best["stability_ari_mean"],
        "stability_ari_min": best["stability_ari_min"],
        "cluster_balance_min_over_max": best["cluster_balance_min_over_max"],
        "cluster_sizes": best["cluster_sizes"],
        "num_candidate_runs": len(rows),
        "num_valid_runs": len(valid_rows),
    }
    with open(args.output_dir / "best_run.json", "w") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)

    lines = [
        "# Fixed K=6 Clustering Selection",
        "",
        f"- Best run: `{payload['best_run_dir']}`",
        f"- PCA dim: `{payload['requested_pca_dim']}`",
        f"- Seed: `{payload['seed']}`",
        f"- Silhouette: `{payload['silhouette']:.6f}`",
        f"- Inertia: `{payload['inertia']:.6f}`",
        f"- Stability ARI mean: `{payload['stability_ari_mean']}`",
        f"- Stability ARI min: `{payload['stability_ari_min']}`",
        f"- Cluster balance min/max: `{payload['cluster_balance_min_over_max']:.6f}`",
        "",
        "Selection uses intrinsic clustering quality only: separation/cohesion via silhouette and inertia, plus seed stability.",
    ]
    (args.output_dir / "best_run.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
