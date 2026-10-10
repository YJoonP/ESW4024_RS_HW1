import ast
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

metric = "ndcg@10"
sweeps = [
    # (swept param, values, fixed param, fixed value, x label, filename)
    ("embedding_size", [32, 64, 128, 256], "reg_lambda", 0.1,
     "Embedding Size", "bpr_ndcg_embedding_size.png"),
    ("reg_lambda", [0.01, 0.1, 0.5], "embedding_size", 64,
     "Regularization Lambda", "bpr_ndcg_reg_lambda.png"),
]
results = {}

paths = sorted(
    Path("log/BPR").glob("*.log"),
    key=lambda p: p.stat().st_mtime,
)

for path in paths:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    emb_match = re.search(r"^\s*embedding_size\s*=\s*(\d+)\s*$", text, re.M)
    reg_match = re.search(r"^\s*reg_lambda\s*=\s*([0-9.eE+-]+)\s*$", text, re.M)
    test_lines = re.findall(r"test result\s*:\s*([^\r\n]+)", text)

    if not (emb_match and reg_match and test_lines):
        continue

    raw = test_lines[-1].strip()
    if raw.startswith("OrderedDict(") and raw.endswith(")"):
        values = dict(ast.literal_eval(raw[len("OrderedDict("):-1]))
    else:
        values = dict(ast.literal_eval(raw))

    # Later logs overwrite earlier ones for the same setting.
    key = (int(emb_match.group(1)), float(reg_match.group(1)))
    results[key] = {str(k).lower(): float(v) for k, v in values.items()}

for param, xs, fixed, fixed_value, xlabel, filename in sweeps:
    def key_of(x):
        if param == "embedding_size":
            return (x, fixed_value)
        return (fixed_value, x)

    missing = [x for x in xs if key_of(x) not in results]
    if missing:
        print(f"Skip {filename}: missing {param}={missing}")
        continue

    y = [results[key_of(x)][metric] for x in xs]

    # Match the style of the example plots in the assignment.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(xs, y, marker="o", markersize=5, linewidth=1.5,
            color="#FFB000", label="BPR")

    ax.set_xticks(xs)
    if param == "reg_lambda":
        ax.set_xticklabels([f"{x:.2f}" for x in xs])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("NDCG@10", fontsize=12)
    ax.set_title(f"NDCG vs {xlabel}", fontsize=16, color="#333333")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.tick_params(labelsize=12)
    ax.grid(linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best")

    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(Path("figures", filename), dpi=300)
    plt.close(fig)
    print(f"Saved: figures/{filename}")
