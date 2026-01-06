from __future__ import annotations

import json
from pathlib import Path


def plot_results(input_path: str, output_path: str) -> None:
    import matplotlib.pyplot as plt

    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    conditions = data.get("conditions", [])
    names = [item["condition"] for item in conditions]
    scores = [item["pass_at_1"] for item in conditions]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, scores, color="#4C72B0")
    ax.set_ylim(0, 1)
    ax.set_ylabel("pass@1")
    ax.set_title("Evaluation Results")
    for idx, score in enumerate(scores):
        ax.text(idx, score + 0.02, f"{score:.2f}", ha="center")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
