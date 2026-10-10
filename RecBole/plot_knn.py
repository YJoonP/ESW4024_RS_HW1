import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ks = [1, 5, 10, 100, 500]
metrics = {
    "recall@10": "Recall@10",
    "mrr@10": "MRR@10",
    "ndcg@10": "NDCG@10",
    "hit@10": "Hit@10",
    "precision@10": "Precision@10",
}
results = {}

paths = sorted(
    Path("log/ItemKNN").glob("*.log"),
    key=lambda p: p.stat().st_mtime,
)

for path in paths:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    k_match = re.search(r"^\s*k\s*=\s*(\d+)\s*$", text, re.M)
    method_match = re.search(
        r"^\s*knn_method\s*=\s*['\"]?(item|user)['\"]?\s*$",
        text, re.M,
    )
    test_lines = re.findall(r"test result\s*:\s*([^\r\n]+)", text)

    if not (k_match and method_match and test_lines):
        continue

    k = int(k_match.group(1))
    if k not in ks:
        continue

    raw = test_lines[-1].strip()
    if raw.startswith("OrderedDict(") and raw.endswith(")"):
        values = dict(ast.literal_eval(raw[len("OrderedDict("):-1]))
    else:
        values = dict(ast.literal_eval(raw))

    results[(method_match.group(1), k)] = {
        str(key).lower(): float(value)
        for key, value in values.items()
    }

missing = [
    (method, k)
    for method in ["item", "user"]
    for k in ks
    if (method, k) not in results
]
if missing:
    raise SystemExit(f"Missing test results: {missing}")

for key, values in results.items():
    missing_metrics = [m for m in metrics if m not in values]
    if missing_metrics:
        raise SystemExit(f"{key}: missing metrics {missing_metrics}")

x = list(range(len(ks)))
styles = ["o", "s", "^", "D", "v"]

for method, title in [("item", "ItemKNN"), ("user", "UserKNN")]:
    fig, ax = plt.subplots(figsize=(8, 5))

    for (metric, label), marker in zip(metrics.items(), styles):
        y = [results[(method, k)][metric] for k in ks]
        ax.plot(x, y, marker=marker, linewidth=2, label=label)

    ax.set_xticks(x, [str(k) for k in ks])
    ax.set_xlabel("Number of neighbors (k)")
    ax.set_ylabel("Test metric score")
    ax.set_ylim(0, 1)
    ax.set_title(f"{title} on ML-100K")
    ax.legend(title="Evaluation metrics", loc="upper center", ncol=5, fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    filename = f"{method}_knn_metrics.png"
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures", filename), dpi=300)
    plt.close(fig)
    print(f"Saved: figures/{filename}")