#!/bin/bash
# =============================================================
# run_multiseed.sh  —  Multi-Seed Experiment Runner
# Runs Exp A (baseline) and Exp D (full method) across 3 seeds
# and reports mean ± std for all structural metrics.
#
# Required for statistical credibility.
# Without multiple seeds, a reviewer can dismiss every result
# as single-seed noise.
#
# Usage:
#   bash scripts/run_multiseed.sh
#
# Output:
#   results/multiseed/seed_{42,123,456}/
#   results/multiseed/summary_mean_std.json
# =============================================================

set -e
SEEDS=(42 123 456)
EXPS=("A" "D")
OUTDIR="results/multiseed"
mkdir -p $OUTDIR

echo ""
echo "============================================================"
echo "  Multi-Seed Evaluation Runner"
echo "  Experiments: A (baseline BCE+Dice) and D (full method)"
echo "  Seeds: ${SEEDS[*]}"
echo "============================================================"

# ----------------------------------------------------------
# TRAIN each combination (skip if checkpoint already exists)
# ----------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    for EXP in "${EXPS[@]}"; do
        CKPT="results/multiseed/seed_${SEED}/exp${EXP}/best_model.pth"
        if [ -f "$CKPT" ]; then
            echo "[SKIP] Seed=${SEED} Exp=${EXP}: checkpoint exists"
            continue
        fi
        echo ""
        echo "[TRAIN] Seed=${SEED}  Exp=${EXP}"
        mkdir -p "results/multiseed/seed_${SEED}/exp${EXP}"

        # Base training call - curriculum flag and topo flag depend on Exp
        USE_CURRICULUM=false
        USE_TOPO=false
        USE_SKEL=false
        if [ "$EXP" == "D" ]; then
            USE_CURRICULUM=true
            USE_SKEL=true
        fi

        python train_unetpp.py \
            --seed        $SEED \
            --save_dir    "results/multiseed/seed_${SEED}/exp${EXP}" \
            --curriculum  $USE_CURRICULUM \
            --use_skel    $USE_SKEL \
            2>&1 | tee "${OUTDIR}/train_seed${SEED}_exp${EXP}.log"

        echo "[TRAIN] Done: Seed=${SEED} Exp=${EXP}"
    done
done

# ----------------------------------------------------------
# EVALUATE each combination
# ----------------------------------------------------------
echo ""
echo "[EVAL] Running structural evaluation on all checkpoints ..."

for SEED in "${SEEDS[@]}"; do
    for EXP in "${EXPS[@]}"; do
        CKPT="results/multiseed/seed_${SEED}/exp${EXP}/best_model.pth"
        if [ ! -f "$CKPT" ]; then
            echo "[SKIP] $CKPT not found"
            continue
        fi
        echo "  Evaluating Seed=${SEED} Exp=${EXP} ..."
        python scripts/evaluate_unetpp.py \
            --checkpoint  "$CKPT" \
            --output_json "${OUTDIR}/seed_${SEED}/exp${EXP}/metrics.json" \
            2>&1 | tee "${OUTDIR}/eval_seed${SEED}_exp${EXP}.log"
    done
done

# ----------------------------------------------------------
# AGGREGATE: compute mean ± std across seeds
# ----------------------------------------------------------
echo ""
echo "[AGGREGATE] Computing mean ± std ..."
python - <<'PYEOF'
import json, os, glob, numpy as np

outdir = "results/multiseed"
seeds  = [42, 123, 456]
exps   = ["A", "D"]

summary = {}
for exp in exps:
    files = [
        f"{outdir}/seed_{s}/exp{exp}/metrics.json"
        for s in seeds
    ]
    available = [f for f in files if os.path.exists(f)]
    if not available:
        print(f"  [SKIP] No results found for Exp {exp}")
        continue

    all_metrics = [json.load(open(f)) for f in available]
    keys = [k for k in all_metrics[0].keys() if isinstance(all_metrics[0][k], (int, float))]

    stats = {}
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        stats[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                    "n": len(vals)}
    summary[f"Exp{exp}"] = stats

# Print paper table
print("\n" + "="*75)
print("  MULTI-SEED RESULTS TABLE  (mean ± std across 3 seeds)")
print("="*75)

metric_display = [
    ("Dice",       "Dice        ", "{:.2f}%", 100),
    ("CCA",        "CCA         ", "{:.3f}",    1),
    ("BPR",        "BPR         ", "{:.3f}",    1),
    ("JPR",        "JPR         ", "{:.3f}",    1),
    ("GED_approx", "GED         ", "{:.1f}",    1),
    ("SkelDice",   "SkelDice    ", "{:.3f}",    1),
    ("SkelHD",     "SkelHD      ", "{:.2f}px",  1),
]

print(f"  {'Metric':<14}  {'Exp A (Baseline)':>22}  {'Exp D (Ours)':>22}  {'Delta':>10}")
print("  " + "-"*68)
for key, label, fmt, scale in metric_display:
    row_a = summary.get("ExpA", {}).get(key)
    row_d = summary.get("ExpD", {}).get(key)
    if row_a is None or row_d is None:
        print(f"  {label} {'[no data]':>22}  {'[no data]':>22}")
        continue
    a_m, a_s = row_a["mean"] * scale, row_a["std"] * scale
    d_m, d_s = row_d["mean"] * scale, row_d["std"] * scale
    delta = d_m - a_m
    fmt_a = fmt.format(a_m) + f" \u00b1 {a_s:.2f}"
    fmt_d = fmt.format(d_m) + f" \u00b1 {d_s:.2f}"
    arrow = "↑" if delta > 0 else "↓"
    if key == "GED_approx" or key == "SkelHD":
        arrow = "↓" if delta < 0 else "↑"  # lower is better
    print(f"  {label:<14} {fmt_a:>22}  {fmt_d:>22}  {arrow}{abs(delta):.3f}")
print("="*75)

out_path = f"{outdir}/summary_mean_std.json"
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Summary saved to: {out_path}")
PYEOF

echo ""
echo "============================================================"
echo "  Multi-seed run complete."
echo "  Fill these numbers into Table 1 and Table 2 of the paper."
echo "============================================================"
