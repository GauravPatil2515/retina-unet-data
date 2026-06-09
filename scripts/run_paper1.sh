#!/bin/bash
# =============================================================
# run_paper1.sh  —  Complete Paper 1 Execution Script
# Paper: "Beyond Dice: Structural Evaluation and Failure
#         Analysis of Retinal Vessel Segmentation"
# =============================================================
# Usage:
#   bash scripts/run_paper1.sh
#
# Prerequisites:
#   source .venv/bin/activate    (or conda activate your-env)
#   DRIVE data at data/DRIVE/
#   STARE data at data/STARE/    (optional for cross-dataset)
#   Model checkpoints at paths defined in benchmark_models.py
# =============================================================

set -e   # stop on first error
LOG_DIR="results/paper1_benchmark/logs"
mkdir -p $LOG_DIR

echo ""
echo "============================================================"
echo "  Paper 1 Execution Pipeline"
echo "  Beyond Dice: Structural Evaluation of Retinal Vessels"
echo "============================================================"
date

# ----------------------------------------------------------
# STEP 1: Metric Sanity Tests (synthetic shapes)
# Must all pass before running on real data.
# ----------------------------------------------------------
echo ""
echo "[STEP 1] Running synthetic metric sanity tests ..."
python tests/test_metrics_sanity.py 2>&1 | tee $LOG_DIR/step1_sanity.log
echo "[STEP 1] DONE"

# ----------------------------------------------------------
# STEP 2: Benchmark — DRIVE
# ----------------------------------------------------------
echo ""
echo "[STEP 2] Running benchmark on DRIVE ..."
python scripts/benchmark_models.py --dataset DRIVE \
    2>&1 | tee $LOG_DIR/step2_drive_benchmark.log
echo "[STEP 2] DONE"

# ----------------------------------------------------------
# STEP 3: Benchmark — STARE (skip if data not available)
# ----------------------------------------------------------
if [ -d "data/STARE" ]; then
    echo ""
    echo "[STEP 3] Running benchmark on STARE ..."
    python scripts/benchmark_models.py --dataset STARE \
        2>&1 | tee $LOG_DIR/step3_stare_benchmark.log
    echo "[STEP 3] DONE"
else
    echo "[STEP 3] SKIPPED: data/STARE not found"
fi

# ----------------------------------------------------------
# STEP 4: Failure Taxonomy — UNet on DRIVE
# ----------------------------------------------------------
echo ""
echo "[STEP 4] Running failure taxonomy for UNet on DRIVE ..."
# Adjust --pred_dir to wherever your model saves predictions
python scripts/failure_taxonomy.py \
    --pred_dir results/predictions/UNet/DRIVE \
    --gt_dir   data/DRIVE/test/mask \
    --model    UNet \
    --dataset  DRIVE \
    2>&1 | tee $LOG_DIR/step4_taxonomy_unet.log
echo "[STEP 4] DONE"

# ----------------------------------------------------------
# STEP 5: Failure Taxonomy — UNetPP on DRIVE
# ----------------------------------------------------------
echo ""
echo "[STEP 5] Running failure taxonomy for UNetPP on DRIVE ..."
python scripts/failure_taxonomy.py \
    --pred_dir results/predictions/UNetPP/DRIVE \
    --gt_dir   data/DRIVE/test/mask \
    --model    UNetPP \
    --dataset  DRIVE \
    2>&1 | tee $LOG_DIR/step5_taxonomy_unetpp.log
echo "[STEP 5] DONE"

# ----------------------------------------------------------
# STEP 6: Aggregate and print final tables
# ----------------------------------------------------------
echo ""
echo "[STEP 6] Aggregating results ..."
python - <<'PYEOF'
import json, os, glob

result_dir = "results/paper1_benchmark"
files = glob.glob(f"{result_dir}/failure_freq_*.json")
print("\n" + "="*70)
print("  FAILURE TAXONOMY TABLE (Paper Section 4.3)")
print("="*70)
print(f"  {'Model':<14} {'Dataset':<8} F1   F2   F3   F4   F5   F6")
print("  " + "-"*54)
for fp in sorted(files):
    with open(fp) as f:
        d = json.load(f)
    rates = d["failure_rates"]
    print(f"  {d['model']:<14} {d['dataset']:<8} "
          f"{rates.get('F1',0):4.0f}% "
          f"{rates.get('F2',0):4.0f}% "
          f"{rates.get('F3',0):4.0f}% "
          f"{rates.get('F4',0):4.0f}% "
          f"{rates.get('F5',0):4.0f}% "
          f"{rates.get('F6',0):4.0f}%")
print("="*70)
print("  F1=CapDrop F2=BranchBreak F3=FalseBridge F4=PeriphLoss F5=JunctionErr F6=CrossErr")
PYEOF
echo "[STEP 6] DONE"

echo ""
echo "============================================================"
echo "  All Paper 1 experiments complete."
echo "  Results saved to: results/paper1_benchmark/"
echo "  Next step: fill in docs/PAPER1_OUTLINE.md with numbers"
echo "============================================================"
date
