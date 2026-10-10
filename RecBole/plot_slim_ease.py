import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

metrics = {
    "recall@10": "Recall@10",
    "mrr@10": "MRR@10",
    "ndcg@10": "NDCG@10",
    "hit@10": "Hit@10",
    "precision@10": "Precision@10",
}
# (log dir, label, required "key = value" lines in the log)
models = [
    ("ItemKNN", "ItemKNN (k=100)", {"k": "100", "knn_method": "item"}),
    ("SLIMElastic", "SLIM", {}),
    ("EASE", "EASE", {}),
]


def latest_test_result(log_dir, required):
    paths = sorted(
        Path("log", log_dir).glob("*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    result = None
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)

        if not all(
            re.search(rf"^\s*{key}\s*=\s*['\"]?{value}['\"]?\s*$", text, re.M)
            for key, value in required.items()
        ):
            continue
        test_lines = re.findall(r"test result\s*:\s*([^\r\n]+)", text)
        if not test_lines:
            continue

        raw = test_lines[-1].strip()
        if raw.startswith("OrderedDict(") and raw.endswith(")"):
            values = dict(ast.literal_eval(raw[len("OrderedDict("):-1]))
        else:
            values = dict(ast.literal_eval(raw))
        result = {str(k).lower(): float(v) for k, v in values.items()}
    return result


results = {}
for log_dir, label, required in models:
    result = latest_test_result(log_dir, required)
    if result is None:
        raise SystemExit(f"Missing test result: {label}")
    results[label] = result

for label, values in results.items():
    print(label, {m: values[m] for m in metrics})

# All three models in one plot, in the style of the example plots.
colors = ["#FFB000", "#F26B1D", "#E8364F"]
x = list(range(len(metrics)))

fig, ax = plt.subplots(figsize=(7, 4.5))
for (_, label, _), color in zip(models, colors):
    y = [results[label][m] for m in metrics]
    ax.plot(x, y, marker="o", markersize=5, linewidth=1.5, color=color,
            label=label)

ax.set_xticks(x, list(metrics.values()))
ax.set_xlabel("Evaluation Metric", fontsize=12)
ax.set_ylabel("Test Score", fontsize=12)
ax.set_ylim(0, 1)
ax.set_title("ItemKNN vs SLIM vs EASE on ML-100K", fontsize=16,
             color="#333333")
ax.tick_params(labelsize=11)
ax.grid(linestyle="--", alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper left")

fig.tight_layout()
filename = "item_to_item_metrics.png"
Path("figures").mkdir(exist_ok=True)
fig.savefig(Path("figures", filename), dpi=300)
plt.close(fig)
print(f"Saved: figures/{filename}")

# SLIM vs EASE side by side. Bar heights are zoomed per metric (cut at 97%
# of the SLIM score) to emphasize the gap. Real scores are written on the bars.
pair = [("SLIM", "#F26B1D"), ("EASE", "#E8364F")]
fig, ax = plt.subplots(figsize=(8, 4.5))
width = 0.36

for i, (label, color) in enumerate(pair):
    offset = (i - 0.5) * width
    heights, values = [], []
    for m in metrics:
        base = results["SLIM"][m] * 0.97
        value = results[label][m]
        heights.append((value - base) / (results["SLIM"][m] - base))
        values.append(value)
    bars = ax.bar([p + offset for p in x], heights, width, color=color,
                  label=label, edgecolor="white", linewidth=1)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.4f}", (bar.get_x() + bar.get_width() / 2,
                                     bar.get_height()),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=9)

ax.set_xticks(x, list(metrics.values()))
ax.set_xlabel("Evaluation Metric", fontsize=12)
ax.set_ylabel("Test Score (zoomed)", fontsize=12)
ax.set_yticks([])
ax.set_ylim(0, 2.2)
ax.set_title("SLIM vs EASE on ML-100K", fontsize=16, color="#333333")
ax.tick_params(labelsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", ncol=2)

fig.tight_layout()
filename = "item_to_item_slim_vs_ease.png"
fig.savefig(Path("figures", filename), dpi=300)
plt.close(fig)
print(f"Saved: figures/{filename}")
