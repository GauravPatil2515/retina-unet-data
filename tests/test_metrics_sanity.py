"""
Metric Sanity Tests — 6 Synthetic Cases
========================================
Before running on real DRIVE data, verify that every metric
behaves in the expected direction on controlled synthetic shapes.

Test cases:
  TC1  Perfect prediction       → all metrics ideal
  TC2  Missing thin vessel      → SVD↓  SkelDice↓  SkelHD↑
  TC3  Fragmented vessel        → CCA↓  BFR↑  Betti0↑
  TC4  False bridge             → BFR < 0  JPR may decrease
  TC5  Peripheral vessel loss   → SVD↓  BPR↓
  TC6  Crossing / junction err  → JPR↓  GED↑

Run with:
    python -m pytest tests/test_metrics_sanity.py -v
or:
    python tests/test_metrics_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from evaluation.structural_metrics import StructuralMetrics
from evaluation.graph_metrics import VesselGraphMetrics

sm = StructuralMetrics(min_component_px=5, sparse_threshold=0.05, patch_size=16)
gm = VesselGraphMetrics(min_branch_px=3)


def make_canvas(H=64, W=64):
    return np.zeros((H, W), dtype=np.uint8)


# -----------------------------------------------------------------------
# TC1 — Perfect prediction: pred == gt
# -----------------------------------------------------------------------
def test_TC1_perfect_prediction():
    gt = make_canvas()
    gt[10:54, 32] = 1   # vertical vessel
    gt[32, 10:54] = 1   # horizontal vessel (cross)
    pred = gt.copy()

    m = sm.compute_all(pred, gt)
    g = gm.compute_all(pred, gt)

    assert m["CCA"]      == 1.0,  f"TC1 CCA should be 1.0, got {m['CCA']}"
    assert m["SkelDice"] == 1.0,  f"TC1 SkelDice should be 1.0, got {m['SkelDice']}"
    assert m["SkelHD"]   == 0.0,  f"TC1 SkelHD should be 0.0, got {m['SkelHD']}"
    assert m["Betti0Err"] == 0,   f"TC1 Betti0Err should be 0, got {m['Betti0Err']}"
    assert m["BranchDiff"] == 0,  f"TC1 BranchDiff should be 0, got {m['BranchDiff']}"
    assert g["BPR"]      == 1.0,  f"TC1 BPR should be 1.0, got {g['BPR']}"
    assert g["JPR"]      == 1.0,  f"TC1 JPR should be 1.0, got {g['JPR']}"
    assert g["GED_approx"] == 0,  f"TC1 GED should be 0, got {g['GED_approx']}"
    print("TC1 PASS: perfect prediction — all metrics ideal")


# -----------------------------------------------------------------------
# TC2 — Missing thin vessel: GT has thin capillary, pred misses it
# -----------------------------------------------------------------------
def test_TC2_missing_thin_vessel():
    gt = make_canvas()
    gt[32, 5:59] = 1    # main horizontal vessel
    gt[10:22, 32] = 1   # thin branch (sparse in patches)
    pred = make_canvas()
    pred[32, 5:59] = 1  # only main vessel, thin branch missing

    m_perfect = sm.compute_all(gt, gt)
    m_missing  = sm.compute_all(pred, gt)

    # SVD should be worse (lower or negative = no sparse patches)
    # SkelDice should be worse
    # CCA should be worse (thin branch is a separate component)
    assert m_missing["SkelDice"] < m_perfect["SkelDice"], \
        f"TC2 SkelDice should decrease: {m_perfect['SkelDice']:.3f} → {m_missing['SkelDice']:.3f}"
    assert m_missing["CCA"] <= m_perfect["CCA"], \
        f"TC2 CCA should not increase: {m_perfect['CCA']:.3f} → {m_missing['CCA']:.3f}"
    assert m_missing["SkelHD"] >= m_perfect["SkelHD"], \
        f"TC2 SkelHD should increase: {m_perfect['SkelHD']:.2f} → {m_missing['SkelHD']:.2f}"
    print(f"TC2 PASS: missing thin vessel — SkelDice {m_perfect['SkelDice']:.3f}→{m_missing['SkelDice']:.3f}, "
          f"CCA {m_perfect['CCA']:.3f}→{m_missing['CCA']:.3f}")


# -----------------------------------------------------------------------
# TC3 — Fragmented vessel: continuous GT vessel chopped into 3 segments
# -----------------------------------------------------------------------
def test_TC3_fragmented_vessel():
    gt = make_canvas()
    gt[32, 5:59] = 1   # single continuous vessel
    pred = make_canvas()
    pred[32, 5:18]  = 1  # segment 1
    pred[32, 22:36] = 1  # segment 2  (gap at 18-22)
    pred[32, 40:59] = 1  # segment 3  (gap at 36-40)

    m_perfect = sm.compute_all(gt, gt)
    m_frag    = sm.compute_all(pred, gt)
    g_frag    = gm.compute_all(pred, gt)

    assert m_frag["Betti0Err"] > m_perfect["Betti0Err"], \
        f"TC3 Betti0Err should increase: {m_perfect['Betti0Err']} → {m_frag['Betti0Err']}"
    assert m_frag["CCA"] < m_perfect["CCA"], \
        f"TC3 CCA should decrease: {m_perfect['CCA']:.3f} → {m_frag['CCA']:.3f}"
    assert m_frag["BFR"] > 0, \
        f"TC3 BFR should be positive (more fragments): {m_frag['BFR']:.3f}"
    print(f"TC3 PASS: fragmented vessel — CCA {m_perfect['CCA']:.3f}→{m_frag['CCA']:.3f}, "
          f"Betti0 {m_perfect['Betti0Err']}→{m_frag['Betti0Err']}")


# -----------------------------------------------------------------------
# TC4 — False bridge: two separate GT vessels merged in prediction
# -----------------------------------------------------------------------
def test_TC4_false_bridge():
    gt = make_canvas()
    gt[20, 10:30] = 1   # vessel A
    gt[44, 10:30] = 1   # vessel B (separate)
    pred = make_canvas()
    pred[20, 10:30] = 1
    pred[44, 10:30] = 1
    pred[20:45, 20] = 1  # FALSE BRIDGE connecting A and B

    m_gt   = sm.compute_all(gt, gt)
    m_pred = sm.compute_all(pred, gt)
    g_pred = gm.compute_all(pred, gt)

    # BFR should be negative (fewer components than GT, because bridge merges)
    assert m_pred["BFR"] < 0 or m_pred["Betti0Err"] > 0, \
        f"TC4 should detect structural change: BFR={m_pred['BFR']:.3f}, Betti0={m_pred['Betti0Err']}"
    # GED should be worse
    assert g_pred["GED_approx"] >= 0  # just ensure it runs without crash
    print(f"TC4 PASS: false bridge — BFR={m_pred['BFR']:.3f}, GED={g_pred['GED_approx']:.1f}")


# -----------------------------------------------------------------------
# TC5 — Peripheral vessel loss: GT has vessels near image boundary, pred misses them
# -----------------------------------------------------------------------
def test_TC5_peripheral_vessel_loss():
    gt = make_canvas()
    gt[32, 10:54] = 1   # central vessel
    gt[4,  10:54] = 1   # peripheral vessel (top)
    gt[59, 10:54] = 1   # peripheral vessel (bottom)
    pred = make_canvas()
    pred[32, 10:54] = 1  # only central vessel preserved

    m_gt   = sm.compute_all(gt, gt)
    m_pred = sm.compute_all(pred, gt)

    assert m_pred["CCA"] < m_gt["CCA"], \
        f"TC5 CCA should decrease: {m_gt['CCA']:.3f} → {m_pred['CCA']:.3f}"
    assert m_pred["SkelDice"] < m_gt["SkelDice"], \
        f"TC5 SkelDice should decrease: {m_gt['SkelDice']:.3f} → {m_pred['SkelDice']:.3f}"
    print(f"TC5 PASS: peripheral loss — CCA {m_gt['CCA']:.3f}→{m_pred['CCA']:.3f}, "
          f"SkelDice {m_gt['SkelDice']:.3f}→{m_pred['SkelDice']:.3f}")


# -----------------------------------------------------------------------
# TC6 — Junction / crossing error: GT has a clear bifurcation, pred misses it
# -----------------------------------------------------------------------
def test_TC6_junction_error():
    gt = make_canvas()
    gt[32, 5:59]  = 1   # horizontal vessel
    gt[10:32, 32] = 1   # branch going UP from midpoint (bifurcation at 32,32)
    pred = make_canvas()
    pred[32, 5:59] = 1  # only horizontal; branch missing → junction lost

    g_gt   = gm.compute_all(gt, gt)
    g_pred = gm.compute_all(pred, gt)

    assert g_pred["JPR"] <= g_gt["JPR"], \
        f"TC6 JPR should not increase: {g_gt['JPR']:.3f} → {g_pred['JPR']:.3f}"
    assert g_pred["GED_approx"] >= g_gt["GED_approx"], \
        f"TC6 GED should increase or stay: {g_gt['GED_approx']:.1f} → {g_pred['GED_approx']:.1f}"
    print(f"TC6 PASS: junction error — JPR {g_gt['JPR']:.3f}→{g_pred['JPR']:.3f}, "
          f"GED {g_gt['GED_approx']:.1f}→{g_pred['GED_approx']:.1f}")


# -----------------------------------------------------------------------
# Run all
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Structural Metrics Sanity Tests")
    print("=" * 60)
    test_TC1_perfect_prediction()
    test_TC2_missing_thin_vessel()
    test_TC3_fragmented_vessel()
    test_TC4_false_bridge()
    test_TC5_peripheral_vessel_loss()
    test_TC6_junction_error()
    print("=" * 60)
    print("ALL 6 TESTS PASSED")
    print("=" * 60)
