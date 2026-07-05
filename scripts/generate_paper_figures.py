"""
generate_paper_figures.py
=========================
Generates all publication-quality figures for:
"Beyond Dice: Structural Evaluation and Failure Analysis of Retinal Vessel Segmentation"

Figures produced:
  fig1_motivation.png       — Same Dice, different topology (synthetic illustration)
  fig2_metric_overview.png  — 10-metric suite visual taxonomy
  fig3_dice_bpr_combined.png— Dice vs BPR scatter (all models, DRIVE+STARE)
  fig4_correlation_heatmap.png — Pearson-r heatmap across models/metrics
  fig5_failure_taxonomy.png — Grouped bar chart of failure mode frequencies
  fig6_ablation.png         — Ablation study bar chart
  fig7_drive_vs_stare.png   — DRIVE vs STARE cross-dataset comparison
  fig8_radar.png            — Radar/spider chart comparing 3 models

Run from project root:
    python scripts/generate_paper_figures.py
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy import stats
from scipy.stats import pearsonr

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BM_DIR   = os.path.join(ROOT, "results", "paper1_benchmark")
FIG_DIR  = os.path.join(ROOT, "files")
os.makedirs(FIG_DIR, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────────
C = {
    "unet":   "#2979FF",   # vivid blue
    "unetpp": "#FF6D00",   # vivid orange
    "retina": "#00C853",   # vivid green
    "drive":  "#7C4DFF",   # purple
    "stare":  "#F06292",   # pink
    "bg":     "#F8F9FA",
    "grid":   "#E0E0E0",
}
MODEL_LABELS = {"UNet": "U-Net", "UNetPP": "U-Net++", "RetinaUNet": "Retina-UNet"}
MODEL_COLORS = {"UNet": C["unet"], "UNetPP": C["unetpp"], "RetinaUNet": C["retina"]}
MODEL_MARKERS= {"UNet": "o",       "UNetPP": "s",        "RetinaUNet": "^"}

def set_style(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_facecolor(C["bg"])
    if title:  ax.set_title(title,  fontsize=12, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    if grid:   ax.grid(True, color=C["grid"], linewidth=0.8, zorder=0)
    ax.spines[["top","right"]].set_visible(False)

def load_per_image(model, dataset="DRIVE"):
    path = os.path.join(BM_DIR, f"per_image_metrics_{model}_{dataset}.json")
    if not os.path.exists(path): return None
    with open(path) as f: return json.load(f)

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  {name}")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Motivation: Same Dice, Different Topology
# ══════════════════════════════════════════════════════════════════════════════
def fig1_motivation():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), facecolor="white")
    fig.suptitle("Same Dice Score — Vastly Different Topology",
                 fontsize=14, fontweight="bold", y=1.02)

    def vessel_image(ax, connected=True, title="", dice="", cca="", bpr=""):
        img = np.zeros((100, 100))
        # main horizontal vessel
        img[45:55, 10:90] = 1
        # vertical branch
        img[20:55, 38:44] = 1
        if connected:
            # two intact branches from main trunk
            img[55:80, 55:61] = 1
            img[55:75, 68:74] = 1
        else:
            # fragmented — gaps introduced
            img[45:55, 35:45] = 0   # gap in main
            img[55:80, 55:61] = 1
            img[70:80, 55:61] = 0   # break branch 1
            img[55:75, 68:74] = 1
            img[60:65, 68:74] = 0   # break branch 2
        ax.imshow(img, cmap="Reds", vmin=0, vmax=1.5, interpolation="nearest")
        ax.set_title(title, fontsize=11, fontweight="bold")
        info = f"Dice = {dice}\nCCA  = {cca}\nBPR  = {bpr}"
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
                va="bottom", ha="left", family="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85))
        ax.axis("off")

    vessel_image(axes[0], connected=True,
                 title="Ground Truth",    dice="—",     cca="1.00", bpr="1.00")
    vessel_image(axes[1], connected=True,
                 title="Model A  (high Dice)", dice="0.814", cca="0.83", bpr="0.88")
    vessel_image(axes[2], connected=False,
                 title="Model B  (same Dice)", dice="0.812", cca="0.20", bpr="0.46")

    # Arrow annotations
    for ax, lbl in zip(axes[1:], ["Topologically\ncorrect ✓", "Topologically\nbroken ✗"]):
        col = "#00C853" if "✓" in lbl else "#D50000"
        ax.text(0.98, 0.98, lbl, transform=ax.transAxes, fontsize=9,
                va="top", ha="right", color=col, fontweight="bold")

    plt.tight_layout()
    save(fig, "fig1_motivation.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Metric Suite Overview (visual taxonomy)
# ══════════════════════════════════════════════════════════════════════════════
def fig2_metric_overview():
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.set_title("Structural Metric Suite — 8 Topology-Aware Metrics",
                 fontsize=14, fontweight="bold")

    cats = [
        ("Connectivity", "#3F51B5", 0.5,
         ["CCA\n(Connected Component Acc.)", "Betti-0\nError"]),
        ("Skeleton /\nCentreline", "#009688", 3.6,
         ["SkelDice", "SkelHD\n(Hausdorff)"]),
        ("Branch /\nJunction", "#E91E63", 6.5,
         ["BPR\n(Branch Pres. Rate)", "JPR\n(Junction Pres. Rate)", "BFR\n(Branch Focal Ratio)"]),
        ("Graph\nDistance", "#FF5722", 9.1,
         ["GED\n(Graph Edit Dist.)"]),
    ]

    for label, col, x, metrics in cats:
        box = mpatches.FancyBboxPatch((x-0.45, 1.5), 2.6, 2.8,
            boxstyle="round,pad=0.15", facecolor=col+"22", edgecolor=col, lw=2)
        ax.add_patch(box)
        ax.text(x + 0.85, 4.1, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=col)
        for i, m in enumerate(metrics):
            my = 3.4 - i * 0.85
            mp = mpatches.FancyBboxPatch((x-0.3, my-0.32), 2.4, 0.62,
                boxstyle="round,pad=0.08", facecolor="white", edgecolor=col+"88", lw=1.2)
            ax.add_patch(mp)
            ax.text(x + 0.9, my, m, ha="center", va="center",
                    fontsize=8.2, color="#212121")

    ax.text(5, 0.4, "↑ higher = better   (except SkelHD, GED, BFR → lower/closer-to-0 = better)",
            ha="center", va="center", fontsize=9, color="#555", style="italic")
    plt.tight_layout()
    save(fig, "fig2_metric_overview.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Dice vs BPR Scatter (DRIVE + STARE combined)
# ══════════════════════════════════════════════════════════════════════════════
def fig3_dice_bpr():
    models = ["UNet", "UNetPP", "RetinaUNet"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), facecolor="white", sharey=True)
    fig.suptitle("Dice vs. Branch Preservation Rate (BPR) — Dice ↑ ≠ Topology ↑",
                 fontsize=13, fontweight="bold")

    for ax, model in zip(axes, models):
        dice_all, bpr_all, dset_colors = [], [], []
        for dataset, col in [("DRIVE", C["drive"]), ("STARE", C["stare"])]:
            data = load_per_image(model, dataset)
            if data is None: continue
            d = [x["Dice"] for x in data]
            b = [x["BPR"]  for x in data]
            ax.scatter(d, b, c=col, s=55, alpha=0.8, edgecolors="white",
                       linewidths=0.6, label=dataset, zorder=3)
            dice_all.extend(d); bpr_all.extend(b)

        if len(dice_all) >= 4:
            slope, intercept, r, p, se = stats.linregress(dice_all, bpr_all)
            xl = np.linspace(min(dice_all), max(dice_all), 100)
            yl = slope * xl + intercept
            ax.plot(xl, yl, "r--", lw=2, zorder=2)
            ax.text(0.04, 0.95, f"r = {r:.3f}\np = {p:.3f}\nn = {len(dice_all)}",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        set_style(ax, title=MODEL_LABELS[model],
                  xlabel="Dice Coefficient", ylabel="BPR" if ax == axes[0] else "")
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    save(fig, "fig3_dice_bpr_combined.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Correlation Heatmap (Pearson r, DRIVE)
# ══════════════════════════════════════════════════════════════════════════════
def fig4_correlation_heatmap():
    metrics   = ["CCA", "BPR", "JPR", "GED", "SkelDice", "SkelHD"]
    models    = ["UNet", "UNetPP", "RetinaUNet"]
    corr_keys = {"CCA":"CCA","BPR":"BPR","JPR":"JPR","GED":"GED_approx","SkelDice":"SkelDice","SkelHD":"SkelHD"}

    matrix = np.zeros((len(models), len(metrics)))
    pmat   = np.zeros_like(matrix)

    for i, model in enumerate(models):
        data = load_per_image(model, "DRIVE")
        if data is None: continue
        dice = np.array([x["Dice"] for x in data])
        for j, met in enumerate(metrics):
            key = corr_keys[met]
            vals = np.array([x[key] for x in data])
            r, p = pearsonr(dice, vals)
            matrix[i, j] = r
            pmat[i, j]   = p

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="white")
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks(range(len(models)));  ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=10)
    ax.set_title("Dice vs. Structural Metric Correlations (DRIVE, Pearson r)", fontsize=12, fontweight="bold")

    for i in range(len(models)):
        for j in range(len(metrics)):
            sig = "**" if pmat[i,j]<0.01 else ("*" if pmat[i,j]<0.05 else "")
            ax.text(j, i, f"{matrix[i,j]:.2f}{sig}", ha="center", va="center",
                    fontsize=9, color="black" if abs(matrix[i,j])<0.6 else "white")

    ax.text(1.01, -0.06, "* p<0.05  ** p<0.01", transform=ax.transAxes,
            fontsize=8, color="#555", style="italic")
    plt.tight_layout()
    save(fig, "fig4_correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Failure Taxonomy Grouped Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
def fig5_failure_taxonomy():
    failure_labels = {
        "F1":"F1: Capillary\nDropout",
        "F2":"F2: Branch\nMerge",
        "F3":"F3: False\nBridge",
        "F4":"F4: Peripheral\nLoss",
        "F5":"F5: Junction\nError",
        "F6":"F6: Crossing\nError"
    }
    models = ["UNet", "UNetPP", "RetinaUNet"]
    rates  = {}
    for m in models:
        path = os.path.join(BM_DIR, f"failure_freq_{m}_DRIVE.json")
        with open(path) as f: rates[m] = json.load(f)["failure_rates"]

    fkeys = list(failure_labels.keys())
    x     = np.arange(len(fkeys))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    for i, model in enumerate(models):
        vals = [rates[model].get(k, 0) for k in fkeys]
        bars = ax.bar(x + i*width - width, vals, width, label=MODEL_LABELS[model],
                      color=MODEL_COLORS[model], alpha=0.88, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    set_style(ax, title="Failure Mode Frequencies (DRIVE, n=20 per model)",
              xlabel="", ylabel="Frequency (%)")
    ax.set_xticks(x); ax.set_xticklabels([failure_labels[k] for k in fkeys], fontsize=9)
    ax.set_ylim(0, 115); ax.legend(fontsize=9)

    # Annotation
    ax.axhspan(90, 115, alpha=0.06, color="red")
    ax.text(0.5, 109, "Systemic Failures (>90%)", ha="center", fontsize=9,
            color="red", style="italic")
    plt.tight_layout()
    save(fig, "fig5_failure_taxonomy.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Ablation Study
# ══════════════════════════════════════════════════════════════════════════════
def fig6_ablation():
    configs = ["A: Baseline", "B: +Curriculum", "C: +Topo Only", "D: Full (Ours)"]
    data = {
        "Dice":    [79.94, 80.04, 79.35, 80.48],
        "CCA":     [0.205, 0.224, 0.205, 0.228],
        "BPR":     [0.811, 0.814, 0.779, 0.828],
        "JPR":     [0.645, 0.633, 0.587, 0.682],
        "GED":     [1031.8, 1019.4, 1132.5, 956.1],
    }
    palette = ["#90A4AE", "#42A5F5", "#AB47BC", "#EF5350"]  # grey, blue, purple, red
    x = np.arange(len(configs))

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.5), facecolor="white")
    fig.suptitle("Ablation Study — Curriculum Learning × Topological Loss (DRIVE)",
                 fontsize=13, fontweight="bold")

    arrows = {"CCA": True, "BPR": True, "JPR": True, "GED": False, "Dice": True}
    for ax, (metric, vals) in zip(axes, data.items()):
        bars = ax.bar(x, vals, color=palette, alpha=0.9, zorder=3, width=0.65)
        best_idx = np.argmax(vals) if arrows[metric] else np.argmin(vals)
        bars[best_idx].set_edgecolor("#212121"); bars[best_idx].set_linewidth(2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002*max(vals),
                    f"{v:.3f}" if v<1 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        better = "↑ better" if arrows[metric] else "↓ better"
        set_style(ax, title=f"{metric}  ({better})", ylabel=metric)
        ax.set_xticks(x)
        ax.set_xticklabels(["A","B","C","D"], fontsize=9)
        ymin, ymax = min(vals)*0.97, max(vals)*1.04
        ax.set_ylim(ymin, ymax)

    handles = [mpatches.Patch(color=c, label=l) for c,l in zip(palette, configs)]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0,0.06,1,1])
    save(fig, "fig6_ablation.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — DRIVE vs STARE Cross-Dataset Comparison
# ══════════════════════════════════════════════════════════════════════════════
def fig7_drive_vs_stare():
    metrics = ["Dice", "BPR", "JPR", "SkelDice", "CCA"]
    models  = ["UNet", "UNetPP", "RetinaUNet"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4.5), facecolor="white")
    fig.suptitle("DRIVE vs. STARE — Cross-Dataset Generalisation",
                 fontsize=13, fontweight="bold")

    drive_data = {m: load_per_image(m, "DRIVE") for m in models}
    stare_data = {m: load_per_image(m, "STARE") for m in models}

    def mean_metric(data, key):
        if data is None: return 0
        return np.mean([x[key] for x in data])

    key_map = {"Dice":"Dice","BPR":"BPR","JPR":"JPR","SkelDice":"SkelDice","CCA":"CCA"}
    x = np.arange(len(models)); width = 0.38

    for ax, met in zip(axes, metrics):
        key = key_map[met]
        drive_vals = [mean_metric(drive_data[m], key) for m in models]
        stare_vals = [mean_metric(stare_data[m], key) for m in models]
        b1 = ax.bar(x - width/2, drive_vals, width, label="DRIVE", color=C["drive"], alpha=0.85, zorder=3)
        b2 = ax.bar(x + width/2, stare_vals, width, label="STARE", color=C["stare"], alpha=0.85, zorder=3)
        for bar, v in zip(list(b1)+list(b2), drive_vals+stare_vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
        set_style(ax, title=met, ylabel=met if ax==axes[0] else "")
        ax.set_xticks(x); ax.set_xticklabels(["U-Net","U-Net++","Retina\nUNet"], fontsize=8)
        ax.legend(fontsize=7)

    plt.tight_layout()
    save(fig, "fig7_drive_vs_stare.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Radar Chart (3 models, DRIVE)
# ══════════════════════════════════════════════════════════════════════════════
def fig8_radar():
    labels  = ["Dice", "CCA", "BPR", "JPR", "SkelDice", "AUC"]
    models  = ["UNet", "UNetPP", "RetinaUNet"]
    # normalised 0-1 values (higher = better for all, GED excluded)
    raw = {
        "UNet":       [79.89, 0.188, 0.816, 0.647, 0.481, 95.68],
        "UNetPP":     [79.99, 0.198, 0.818, 0.634, 0.480, 95.44],
        "RetinaUNet": [80.42, 0.204, 0.831, 0.686, 0.487, 95.85],
    }
    # normalise per column
    arr = np.array([raw[m] for m in models], dtype=float)
    mins, maxs = arr.min(0), arr.max(0)
    denom = maxs - mins
    denom[denom == 0] = 1
    norm = (arr - mins) / denom

    N = len(labels)
    angles = [n/N * 2*np.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="white",
                           subplot_kw=dict(polar=True))
    ax.set_facecolor(C["bg"])

    for i, model in enumerate(models):
        vals = list(norm[i]) + [norm[i][0]]
        col  = list(MODEL_COLORS.values())[i]
        ax.plot(angles, vals, color=col, linewidth=2, linestyle="solid")
        ax.fill(angles, vals, color=col, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=7, color="#777")
    ax.set_title("Model Comparison — DRIVE (normalised scores)",
                 fontsize=12, fontweight="bold", pad=18)

    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m]) for m in models]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.15),
              ncol=3, fontsize=9, framealpha=0.9)
    plt.tight_layout()
    save(fig, "fig8_radar.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Per-Image Metric Distribution (box plots, DRIVE)
# ══════════════════════════════════════════════════════════════════════════════
def fig9_boxplots():
    models  = ["UNet", "UNetPP", "RetinaUNet"]
    metrics = [("Dice","Dice Coefficient"), ("BPR","BPR"), ("JPR","JPR"),
               ("SkelDice","SkelDice"), ("CCA","CCA")]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), facecolor="white")
    fig.suptitle("Per-Image Metric Distributions — DRIVE Test Set (n=20)",
                 fontsize=13, fontweight="bold")

    for ax, (key, label) in zip(axes, metrics):
        all_vals = []
        for model in models:
            data = load_per_image(model, "DRIVE")
            vals = [x[key] for x in data] if data else []
            all_vals.append(vals)

        bp = ax.boxplot(all_vals, patch_artist=True, notch=False,
                        medianprops=dict(color="#212121", lw=2),
                        whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2))
        for patch, model in zip(bp["boxes"], models):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.75)

        set_style(ax, title=label, ylabel=label if ax==axes[0] else "")
        ax.set_xticks([1,2,3])
        ax.set_xticklabels(["U-Net","U-Net++","Retina\nUNet"], fontsize=9)

    plt.tight_layout()
    save(fig, "fig9_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nGenerating figures → {FIG_DIR}\n")
    fig1_motivation()
    fig2_metric_overview()
    fig3_dice_bpr()
    fig4_correlation_heatmap()
    fig5_failure_taxonomy()
    fig6_ablation()
    fig7_drive_vs_stare()
    fig8_radar()
    fig9_boxplots()
    print("\nAll figures generated successfully!\n")
