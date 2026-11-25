"""README charts, built from the aggregate result files in the outputs folder."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import data_source

SURFACE, INK, INK_2, INK_3 = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, BLUE = "#e1e0d9", "#c3c2b7", "#2a78d6"
LABELS = {
    "event_count": "Open house / event attendance", "campus_visits": "Campus visits",
    "decision_plan": "Decision plan (ED / EA / RD)", "merit_amount_k": "Merit award amount",
    "act": "ACT score", "sport": "Sport recruited for", "gpa": "High school GPA",
    "legacy": "Legacy status", "days_inquiry_to_apply": "Days from inquiry to application",
    "merit_tier": "Merit award type", "class_top_percent": "Class rank percentile",
    "class_rank_missing": "No class rank reported", "religion": "Religion",
    "intends_financial_aid": "Plans to apply for aid", "academic_index": "Academic index",
}
plt.rcParams["font.family"] = "DejaVu Sans"


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=INK_3, labelsize=9)
    ax.set_title(title, fontsize=13, color=INK, fontweight="bold", loc="left", pad=14)


def deciles(out, base_rate):
    d = pd.read_csv(out / "deciles.csv")
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    bars = ax.bar(d["decile"], d["actual_rate"] * 100, color=BLUE, width=0.62, zorder=3, label="Actual enrollment rate")
    ax.scatter(d["decile"], d["predicted_rate"] * 100, color=INK, s=28, zorder=4, label="Model's predicted rate")
    for b, v, p in zip(bars, d["actual_rate"], d["predicted_rate"]):
        ax.text(b.get_x() + b.get_width() / 2, max(v, p) * 100 + 2, f"{v:.0%}", ha="center", fontsize=8.5, color=INK_2)
    ax.axhline(base_rate * 100, color=INK_3, linestyle="--", linewidth=1, zorder=2)
    ax.text(10.45, base_rate * 100 + 1.2, f"Overall: {base_rate:.0%}", ha="right", fontsize=8.5, color=INK_2)
    ax.set_xticks(d["decile"])
    ax.set_xticklabels([f"{i}" for i in d["decile"]])
    ax.set_xlabel("Admitted students grouped into tenths, most likely to enroll (1) to least likely (10)", fontsize=9.5, color=INK_2)
    ax.set_ylabel("Enrolled (%)", fontsize=10, color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    style(ax, "Test set: enrollment rate by predicted likelihood")
    plt.tight_layout()
    plt.savefig(out / "figures" / "enrollment_by_decile.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def importance(out):
    f = pd.read_csv(out / "feature_importance.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.barh([LABELS.get(x, x) for x in f["feature"]], f["auc_drop"], color=BLUE, height=0.62, zorder=3)
    for y, v in enumerate(f["auc_drop"]):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8.5, color=INK_2)
    ax.set_xlabel("Drop in test AUC when the feature is shuffled (higher = more important)", fontsize=9.5, color=INK_2)
    style(ax, "What predicts enrollment: top 10 features")
    ax.tick_params(axis="y", colors=INK, labelsize=9.5, length=0)
    plt.tight_layout()
    plt.savefig(out / "figures" / "feature_importance.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    _, out, _ = data_source()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "summary.json").read_text())
    deciles(out, summary["test_enroll_rate"])
    importance(out)
    print(f"Saved charts to {out / 'figures'}")
