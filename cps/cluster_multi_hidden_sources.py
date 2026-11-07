import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score


HEURISTIC_LABELS = {
    "reflection": "reflection",
    "transition": "transition",
    "execution": "execution",
}


def load_prompts(prompts_path):
    if not prompts_path:
        return None
    with open(prompts_path) as fin:
        return json.load(fin)


def analysis_span(text, think_only):
    if not think_only:
        return 0, len(text)

    think_begin = text.find("<think>")
    if think_begin < 0:
        return None

    start = think_begin + len("<think>")
    think_end = text.find("</think>", start)
    end = len(text) if think_end < 0 else think_end
    return start, end


def step_matches(text, start_char, end_char):
    matches = []
    segment = text[start_char:end_char]
    for match in re.finditer(r"\n\s*\n+", segment):
        boundary_start = start_char + match.start()
        boundary_end = start_char + match.end()
        step_start = boundary_end
        while step_start < end_char and text[step_start] in " \t\r\n":
            step_start += 1
        if step_start < end_char:
            matches.append((boundary_start, boundary_end, step_start))
    return matches


def extract_step_texts(text, think_only=True):
    span = analysis_span(text, think_only)
    if span is None:
        return []

    start_char, end_char = span
    matches = step_matches(text, start_char, end_char)
    if not matches:
        return []

    steps = []
    for step_id, (_, _, step_start) in enumerate(matches):
        next_boundary_start = matches[step_id + 1][0] if step_id + 1 < len(matches) else end_char
        steps.append(text[step_start:next_boundary_start].strip())
    return steps


def heuristic_labels_for_entry(entry):
    step_count = int(entry["step"].shape[0])
    labels = [HEURISTIC_LABELS["execution"]] * step_count
    for idx in entry["check_index"].tolist():
        if 0 <= idx < step_count:
            labels[idx] = HEURISTIC_LABELS["reflection"]
    for idx in entry["switch_index"].tolist():
        if 0 <= idx < step_count:
            labels[idx] = HEURISTIC_LABELS["transition"]
    return labels


def sample_local_indices(step_count, max_steps_per_example, seed, source_index, example_id):
    if max_steps_per_example is None or max_steps_per_example <= 0 or step_count <= max_steps_per_example:
        return list(range(step_count))
    generator = torch.Generator().manual_seed(seed + source_index * 1000003 + example_id)
    picked = torch.randperm(step_count, generator=generator)[:max_steps_per_example].tolist()
    picked.sort()
    return picked


def build_records_from_sources(
    hidden_paths,
    source_names,
    prompts_paths,
    layer,
    max_steps_per_example,
    seed,
    think_only,
):
    vectors = []
    records = []
    step_count_stats = []
    source_example_counts = {}
    source_step_counts = {}

    for source_index, (hidden_path, source_name, prompts_path) in enumerate(
        zip(hidden_paths, source_names, prompts_paths)
    ):
        hidden = torch.load(hidden_path, map_location="cpu")
        if layer >= len(hidden):
            raise ValueError(f"Layer {layer} not available in {hidden_path}")
        layer_data = hidden[layer]
        prompts = load_prompts(prompts_path)

        source_example_counts[source_name] = 0
        source_step_counts[source_name] = 0

        for example_id in sorted(layer_data):
            entry = layer_data[example_id]
            step_hidden = entry["step"].float()
            step_count = int(step_hidden.shape[0])
            if step_count == 0:
                continue

            source_example_counts[source_name] += 1
            source_step_counts[source_name] += step_count
            step_count_stats.append(step_count)

            labels = heuristic_labels_for_entry(entry)
            step_texts = None
            if prompts is not None and example_id < len(prompts):
                step_texts = extract_step_texts(prompts[example_id], think_only=think_only)

            selected_indices = sample_local_indices(
                step_count=step_count,
                max_steps_per_example=max_steps_per_example,
                seed=seed,
                source_index=source_index,
                example_id=example_id,
            )
            for local_idx in selected_indices:
                record = {
                    "source_name": source_name,
                    "source_index": source_index,
                    "hidden_path": hidden_path,
                    "example_id": int(example_id),
                    "step_index": int(local_idx),
                    "heuristic_label": labels[local_idx],
                    "step_text": step_texts[local_idx] if step_texts is not None and local_idx < len(step_texts) else "",
                }
                records.append(record)
                vectors.append(step_hidden[local_idx])

        del hidden

    if not vectors:
        raise ValueError("No step embeddings were collected from the provided hidden sources.")

    source_summary = {
        source_name: {
            "num_examples_with_steps": int(source_example_counts[source_name]),
            "num_steps_total": int(source_step_counts[source_name]),
        }
        for source_name in source_names
    }
    return torch.stack(vectors, dim=0), records, step_count_stats, source_summary


def fit_preprocessor(train_vectors, pca_dim):
    mean = train_vectors.mean(axis=0, keepdims=True)
    centered = train_vectors - mean
    if pca_dim and pca_dim > 0 and pca_dim < centered.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=0)
        reduced = pca.fit_transform(centered)
        explained_variance_ratio_sum = float(pca.explained_variance_ratio_.sum())
        effective_pca_dim = int(pca.n_components_)
    else:
        pca = None
        reduced = centered
        explained_variance_ratio_sum = None
        effective_pca_dim = int(centered.shape[1])
    return mean, pca, reduced, effective_pca_dim, explained_variance_ratio_sum


def transform_vectors(vectors, mean, pca):
    centered = vectors - mean
    if pca is None:
        return centered
    return pca.transform(centered)


def evaluate_kmeans(vectors, k_values, seed, n_init, max_iter):
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
        labels = model.fit_predict(vectors)
        silhouette = silhouette_score(vectors, labels)
        counts = Counter(labels.tolist())
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette),
                "inertia": float(model.inertia_),
                "cluster_sizes": {int(key): int(val) for key, val in sorted(counts.items())},
            }
        )
    return rows


def evaluate_seed_stability(vectors, k_values, seeds, n_init, max_iter):
    rows = []
    for k in k_values:
        assignments = []
        silhouettes = []
        inertias = []

        for seed in seeds:
            model = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
            labels = model.fit_predict(vectors)
            assignments.append(labels)
            silhouettes.append(float(silhouette_score(vectors, labels)))
            inertias.append(float(model.inertia_))

        pairwise_ari = []
        for label_a, label_b in combinations(assignments, 2):
            pairwise_ari.append(float(adjusted_rand_score(label_a, label_b)))

        rows.append(
            {
                "k": k,
                "seeds": list(seeds),
                "num_runs": len(seeds),
                "silhouette_mean": float(sum(silhouettes) / len(silhouettes)),
                "silhouette_std": float(torch.tensor(silhouettes).std(unbiased=False).item()) if len(silhouettes) > 1 else 0.0,
                "inertia_mean": float(sum(inertias) / len(inertias)),
                "inertia_std": float(torch.tensor(inertias).std(unbiased=False).item()) if len(inertias) > 1 else 0.0,
                "ari_mean": float(sum(pairwise_ari) / len(pairwise_ari)) if pairwise_ari else 1.0,
                "ari_std": float(torch.tensor(pairwise_ari).std(unbiased=False).item()) if len(pairwise_ari) > 1 else 0.0,
                "ari_min": float(min(pairwise_ari)) if pairwise_ari else 1.0,
                "ari_max": float(max(pairwise_ari)) if pairwise_ari else 1.0,
            }
        )
    return rows


def choose_best_k(metric_rows):
    return max(metric_rows, key=lambda row: row["silhouette"])["k"]


def fit_final_model(vectors, k, seed, n_init, max_iter):
    model = KMeans(n_clusters=k, random_state=seed, n_init=n_init, max_iter=max_iter)
    labels = model.fit_predict(vectors)
    return model, labels


def attach_cluster_ids(records, labels):
    enriched = []
    for record, cluster_id in zip(records, labels.tolist()):
        row = dict(record)
        row["cluster_id"] = int(cluster_id)
        enriched.append(row)
    return enriched


def cluster_composition(records):
    by_cluster = defaultdict(Counter)
    for row in records:
        by_cluster[row["cluster_id"]][row["heuristic_label"]] += 1
    return {
        int(cluster_id): {
            label: int(count)
            for label, count in sorted(counter.items(), key=lambda item: item[0])
        }
        for cluster_id, counter in sorted(by_cluster.items())
    }


def representative_steps(records, transformed_vectors, centers, top_n):
    cluster_to_items = defaultdict(list)
    for idx, row in enumerate(records):
        center = centers[row["cluster_id"]]
        distance = float(((transformed_vectors[idx] - center) ** 2).sum())
        cluster_to_items[row["cluster_id"]].append((distance, row))

    output = {}
    for cluster_id, items in sorted(cluster_to_items.items()):
        items.sort(key=lambda item: item[0])
        output[int(cluster_id)] = []
        for distance, row in items[:top_n]:
            output[int(cluster_id)].append(
                {
                    "distance": distance,
                    "source_name": row["source_name"],
                    "example_id": int(row["example_id"]),
                    "step_index": int(row["step_index"]),
                    "heuristic_label": row["heuristic_label"],
                    "step_text": row["step_text"],
                }
            )
    return output


def write_json(path, payload):
    with open(path, "w") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    with open(path, "w") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_metrics_csv(path, rows):
    with open(path, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=["k", "silhouette", "inertia"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"k": row["k"], "silhouette": row["silhouette"], "inertia": row["inertia"]})


def plot_metric_curves(metric_rows, output_path):
    import matplotlib.pyplot as plt

    ks = [row["k"] for row in metric_rows]
    silhouettes = [row["silhouette"] for row in metric_rows]
    inertias = [row["inertia"] for row in metric_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(ks, silhouettes, marker="o", linewidth=2)
    axes[0].set_title("Silhouette Score")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.25)

    axes[1].plot(ks, inertias, marker="o", linewidth=2)
    axes[1].set_title("SSE / Inertia")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Inertia")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_stability_curves(stability_rows, output_path):
    import matplotlib.pyplot as plt

    ks = [row["k"] for row in stability_rows]
    ari_means = [row["ari_mean"] for row in stability_rows]
    ari_stds = [row["ari_std"] for row in stability_rows]
    silhouette_means = [row["silhouette_mean"] for row in stability_rows]
    silhouette_stds = [row["silhouette_std"] for row in stability_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].errorbar(ks, ari_means, yerr=ari_stds, marker="o", linewidth=2, capsize=3)
    axes[0].set_title("Seed Stability (ARI)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Pairwise ARI")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(ks, silhouette_means, yerr=silhouette_stds, marker="o", linewidth=2, capsize=3)
    axes[1].set_title("Silhouette Across Seeds")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pca_scatter(vectors, labels, output_path, title):
    import matplotlib.pyplot as plt

    plot_vectors = vectors
    if vectors.shape[1] > 2:
        plot_vectors = PCA(n_components=2, random_state=0).fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(6, 5))
    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10", len(unique_labels))
    for idx, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        ax.scatter(
            plot_vectors[mask, 0],
            plot_vectors[mask, 1],
            s=10,
            alpha=0.65,
            color=cmap(idx),
            label=f"Cluster {cluster_id}",
            edgecolors="none",
        )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize_step_counts(step_count_stats):
    sorted_counts = sorted(step_count_stats)
    n = len(sorted_counts)
    return {
        "num_examples_with_steps": n,
        "min": int(sorted_counts[0]),
        "median": int(sorted_counts[n // 2]),
        "max": int(sorted_counts[-1]),
        "mean": float(sum(sorted_counts) / n),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster thought-step embeddings from multiple hidden.pt sources."
    )
    parser.add_argument("--hidden_paths", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prompts_paths", type=str, nargs="*", default=None)
    parser.add_argument("--source_names", type=str, nargs="*", default=None)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--think_only", action="store_true", default=True)
    parser.add_argument("--max_steps_per_example", type=int, default=30)
    parser.add_argument("--k_min", type=int, default=2)
    parser.add_argument("--k_max", type=int, default=12)
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--n_init", type=int, default=50)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--top_n_representatives", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.source_names and len(args.source_names) != len(args.hidden_paths):
        raise ValueError("--source_names must match the number of --hidden_paths")
    if args.prompts_paths and len(args.prompts_paths) != len(args.hidden_paths):
        raise ValueError("--prompts_paths must match the number of --hidden_paths")

    source_names = args.source_names or [
        os.path.basename(os.path.dirname(hidden_path.rstrip("/")))
        for hidden_path in args.hidden_paths
    ]
    prompts_paths = args.prompts_paths or [None] * len(args.hidden_paths)

    sampled_vectors_t, sampled_records, step_count_stats, source_summary = build_records_from_sources(
        hidden_paths=args.hidden_paths,
        source_names=source_names,
        prompts_paths=prompts_paths,
        layer=args.layer,
        max_steps_per_example=args.max_steps_per_example,
        seed=args.seed,
        think_only=args.think_only,
    )
    full_vectors_t, full_records, _, _ = build_records_from_sources(
        hidden_paths=args.hidden_paths,
        source_names=source_names,
        prompts_paths=prompts_paths,
        layer=args.layer,
        max_steps_per_example=None,
        seed=args.seed,
        think_only=args.think_only,
    )

    sampled_vectors = sampled_vectors_t.numpy()
    full_vectors = full_vectors_t.numpy()

    mean, pca, sampled_transformed, effective_pca_dim, explained_variance_ratio_sum = fit_preprocessor(
        sampled_vectors, args.pca_dim
    )
    full_transformed = transform_vectors(full_vectors, mean, pca)

    k_values = list(range(args.k_min, args.k_max + 1))
    metric_rows = evaluate_kmeans(
        vectors=sampled_transformed,
        k_values=k_values,
        seed=args.seed,
        n_init=args.n_init,
        max_iter=args.max_iter,
    )
    stability_rows = evaluate_seed_stability(
        vectors=sampled_transformed,
        k_values=k_values,
        seeds=args.stability_seeds,
        n_init=args.n_init,
        max_iter=args.max_iter,
    )
    best_k = choose_best_k(metric_rows)

    sampled_best_model, sampled_best_labels = fit_final_model(
        sampled_transformed, best_k, args.seed, args.n_init, args.max_iter
    )
    full_best_model, full_best_labels = fit_final_model(
        full_transformed, best_k, args.seed, args.n_init, args.max_iter
    )

    sampled_rows = attach_cluster_ids(sampled_records, sampled_best_labels)
    full_rows = attach_cluster_ids(full_records, full_best_labels)

    summary = {
        "hidden_paths": args.hidden_paths,
        "source_names": source_names,
        "prompts_paths": prompts_paths,
        "layer": args.layer,
        "num_sources": len(args.hidden_paths),
        "source_summary": source_summary,
        "num_sampled_steps": len(sampled_rows),
        "num_full_steps": len(full_rows),
        "max_steps_per_example_for_k_search": args.max_steps_per_example,
        "requested_pca_dim": args.pca_dim,
        "effective_pca_dim": effective_pca_dim,
        "explained_variance_ratio_sum": explained_variance_ratio_sum,
        "seed": args.seed,
        "stability_seeds": args.stability_seeds,
        "n_init": args.n_init,
        "max_iter": args.max_iter,
        "best_k_by_silhouette": best_k,
        "step_count_summary": summarize_step_counts(step_count_stats),
    }

    write_json(os.path.join(args.output_dir, "config_summary.json"), summary)
    write_json(os.path.join(args.output_dir, "kmeans_metrics.json"), metric_rows)
    write_metrics_csv(os.path.join(args.output_dir, "kmeans_metrics.csv"), metric_rows)
    plot_metric_curves(metric_rows, os.path.join(args.output_dir, "kmeans_metrics.png"))
    write_json(os.path.join(args.output_dir, "kmeans_seed_stability.json"), stability_rows)
    plot_stability_curves(stability_rows, os.path.join(args.output_dir, "kmeans_seed_stability.png"))

    best_row = next(row for row in metric_rows if row["k"] == best_k)
    best_stability_row = next((row for row in stability_rows if row["k"] == best_k), None)
    write_json(
        os.path.join(args.output_dir, "best_k.json"),
        {
            "best_k_by_silhouette": best_k,
            "silhouette": best_row["silhouette"],
            "inertia": best_row["inertia"],
            "stability_ari_mean": best_stability_row["ari_mean"] if best_stability_row else None,
        },
    )

    write_json(
        os.path.join(args.output_dir, "cluster_sizes_sampled.json"),
        {int(k): int(v) for k, v in sorted(Counter(sampled_best_labels.tolist()).items())},
    )
    write_json(
        os.path.join(args.output_dir, "cluster_sizes_full.json"),
        {int(k): int(v) for k, v in sorted(Counter(full_best_labels.tolist()).items())},
    )
    write_json(
        os.path.join(args.output_dir, "cluster_heuristic_composition_sampled.json"),
        cluster_composition(sampled_rows),
    )
    write_json(
        os.path.join(args.output_dir, "cluster_heuristic_composition_full.json"),
        cluster_composition(full_rows),
    )
    write_json(
        os.path.join(args.output_dir, "representative_steps_sampled.json"),
        representative_steps(
            sampled_rows,
            sampled_transformed,
            sampled_best_model.cluster_centers_,
            args.top_n_representatives,
        ),
    )
    write_json(
        os.path.join(args.output_dir, "representative_steps_full.json"),
        representative_steps(
            full_rows,
            full_transformed,
            full_best_model.cluster_centers_,
            args.top_n_representatives,
        ),
    )
    write_jsonl(os.path.join(args.output_dir, "assignments_sampled.jsonl"), sampled_rows)
    write_jsonl(os.path.join(args.output_dir, "assignments_full.jsonl"), full_rows)

    plot_pca_scatter(
        sampled_transformed,
        sampled_best_labels,
        os.path.join(args.output_dir, f"sampled_scatter_k{best_k}.png"),
        f"Sampled Steps, K={best_k}",
    )
    plot_pca_scatter(
        full_transformed,
        full_best_labels,
        os.path.join(args.output_dir, f"full_scatter_k{best_k}.png"),
        f"Full Steps, K={best_k}",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
