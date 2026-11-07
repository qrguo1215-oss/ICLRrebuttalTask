import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Create +/- signed copies of cluster mean vectors.")
    parser.add_argument("--vector_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--num_clusters", type=int, default=6)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cluster_id in range(args.num_clusters):
        src = args.vector_dir / f"cluster_{cluster_id}_mean.pt"
        if not src.exists():
            raise FileNotFoundError(src)
        vector = torch.load(src, map_location="cpu", weights_only=True).float()
        for sign, signed in [("pos", vector), ("neg", -vector)]:
            out_path = args.output_dir / f"cluster_{cluster_id}_{sign}.pt"
            torch.save(signed, out_path)
            manifest.append(
                {
                    "cluster_id": cluster_id,
                    "sign": sign,
                    "source_vector": str(src),
                    "path": str(out_path),
                    "formula": f"{'+' if sign == 'pos' else '-'}C{cluster_id}_raw_mean",
                    "norm": float(signed.norm().item()),
                }
            )

    with open(args.output_dir / "manifest.json", "w") as fout:
        json.dump(manifest, fout, ensure_ascii=False, indent=2)

    lines = ["# Signed Cluster Vectors", ""]
    for row in manifest:
        lines.append(
            f"- `{row['formula']}`: `{row['path']}`; norm={row['norm']:.6f}"
        )
    (args.output_dir / "manifest.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
