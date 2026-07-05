# Audit Report: "Beyond Dice" (Retina-Unet) — Paper + Supporting Docs

**Scope reviewed:** the IEEE-style paper PDF, `PROJECT_OVERVIEW.md`, `PROJECT_SUMMARY.md`, `ABLATION_STUDY.md`, `ARCHITECTURE_AND_MODEL.md`, `LOSS_FUNCTIONS.md`, `METRICS_AND_EVALUATION.md`, `CODEBASE_STRUCTURE.md`.

**Bottom line:** The *idea* is genuinely good and publishable — "Dice can't see topology, here's a metric suite + fix" is a real, well-motivated gap, especially for a target like *Computers in Biology and Medicine*. But right now the paper is **not internally consistent**, and a reviewer who cross-checks two tables (which reviewers do) will find contradictions that undermine trust in every other number in the manuscript. Fix data integrity before touching prose polish. Below is everything, ranked by severity.

---

## P0 — Data-integrity issues (must fix before anyone else reads this)

### 1. The paper's central claim contradicts its own table (critical)
This is the most serious problem in the whole project.

- **Table III / Fig. 4** (Dice vs. structural metric correlation): Dice–BPR row is reported as **positive** for all three models: U-Net r=+0.426, U-Net++ r=+0.435, Retina-UNet r=+0.418.
- **Body text, same section**: *"we observe a strong negative correlation between Dice and Branch Preservation Rate (BPR) across all models (pooled r = −0.708, p < 0.001), revealing a structural anti-incentive."*
- **Fig. 5 caption** (same relationship, DRIVE+STARE combined): *"A strong positive correlation (r ≈ 0.91–0.93, p < 0.001)."*
- **Conclusion / Abstract** repeat the −0.708 figure as the paper's flagship finding.

Three different signs/magnitudes for the same statistical relationship, in the same document. This isn't a rounding issue — it's a sign flip on your core contribution. Possibilities:
- The −0.708 was computed on a *different pair* (e.g., mixed up with a residual/error term) and pasted into the BPR narrative by mistake.
- "Pooled" here means something non-obvious (e.g., pooling across a hidden low-Dice STARE subset causes Simpson's-paradox reversal) — if so, this needs its own explicit explanation and a supporting table, not a throwaway sentence.
- The synthetic "anti-incentive mechanism" (Section V-B) was written *before* the correlation was actually computed, and the number was never reconciled against real output.

**Action:** Re-run the correlation analysis script (`correlation_analysis.py`) from scratch, log the exact inputs/outputs, and make Table III, Fig. 4, Fig. 5, and every text reference use the *same* number. If the true relationship is positive, the "anti-incentive" framing needs to be rebuilt around a different piece of evidence (e.g., BFR going negative, or a specific failure-mode rate) rather than a fabricated-looking correlation.

### 2. GED (and other metrics) reported on two incompatible numeric scales
- **Paper Table II / V, PROJECT_OVERVIEW.md, CODEBASE_STRUCTURE.md**: GED values in the **900–1130** range (e.g., baseline 1031.8 → full 956.1).
- **ABLATION_STUDY.md, LOSS_FUNCTIONS.md**: GED values in the **12–15** range for the *same* experiments (baseline 15.2 → full 12.3).

That's a ~68x scale mismatch for the same metric on the same experiment. Same problem, smaller magnitude, for BPR (paper: 0.811–0.828 vs. ABLATION_STUDY.md: 0.58–0.72) and JPR/SkelDice.

This strongly suggests ABLATION_STUDY.md was generated independently (possibly per-image mean vs. summed/whole-graph GED, or a different metric normalization) and never reconciled with the paper. Two live, contradictory ablation tables exist in your repo right now.

**Action:** Pick one canonical GED definition (per-image average is more standard for reporting a mean±std), document the normalization explicitly, regenerate every downstream table from the *same* JSON output (`per_image_metrics_*.json`), and delete/replace the stale table.

### 3. STARE results appear in the paper without any STARE methodology
Fig. 5 and Fig. 8 both present quantitative DRIVE-vs-STARE comparisons ("Dice drops by ~44 percentage points on STARE," r≈0.91–0.93 on n=24 combined) — but:
- Section IV-A ("Dataset and Setup") only describes DRIVE. STARE is never introduced, its size/split/preprocessing is never stated.
- `CODEBASE_STRUCTURE.md` explicitly lists `scripts/dataloader_stare.py — STARE dataset loader **(planned)**`, i.e., not yet implemented.

If the STARE loader doesn't exist yet, these figures can't reflect real evaluation output. This is the kind of thing that turns a "sloppy writeup" review into a "results may not be reproducible" review — the more damaging category.

**Action:** Either (a) implement the STARE loader, actually run zero-shot cross-dataset eval, and report it as a proper subsection with methodology, or (b) remove Figs. 5 (STARE portion) and 8 entirely from this submission and flag cross-dataset validation as future work (it already is listed there — don't claim results you don't have).

### 4. RETFound backbone is incompatible with the stated 9M parameter count
`ARCHITECTURE_AND_MODEL.md` states the default encoder is **RETFound**, "pretrained on 1.6M retinal images," inside a model with **~9M total parameters** (8.5M trainable). RETFound (Zhou et al., *Nature* 2023) is a ViT-Large MAE encoder with **~300M+ parameters**. There's no way a RETFound-backed U-Net++ totals 9M params — either RETFound isn't actually wired in (and the doc is aspirational/wrong), or the model variant actually used for the reported numbers is a lightweight from-scratch encoder and "RETFound" is a mislabel.

This matters a lot for credibility because RETFound is a specific, checkable, named public model — a reviewer familiar with it will catch this in one glance at your parameter table.

**Action:** State plainly which encoder produced the reported numbers. If it's a small from-scratch CNN encoder (which the 9M param count and 28-minute RTX 3050 training time both suggest), say so, and move RETFound/Swin/ViT to "supported alternative backbones, not used for reported results" or "future work."

### 5. Training-time numbers don't add up against the stated protocol
- Section IV-C says training uses **60 epochs** (3 curriculum stages × 20 epochs) with early stopping.
- `ARCHITECTURE_AND_MODEL.md` says **"Training Time: ~28 min (20 epochs, RTX 3050)."**
- `LOSS_FUNCTIONS.md`'s loss-ablation table shows the *full* loss (BCE+Tversky+Topo+SkelDice) taking only 28 min total — barely more than BCE-only's 22 min — despite (a) 3x the epoch count implied by curriculum, and (b) a topological loss that does GPU→CPU→numpy persistent-homology computation per image, which is normally the single most expensive component in the pipeline by a wide margin.

**Action:** Re-benchmark and report actual wall-clock time for the true 60-epoch curriculum runs, broken out by stage, and don't undersell/oversell the topo-loss overhead — a large but honest overhead is a legitimate and expected result, not a weakness.

### 6. Copy-paste error in PROJECT_SUMMARY.md
> "CCA: 0.204 (U-Net++) → 0.204 (Ours) [+11.2% vs baseline]"

Same number on both sides of an arrow, labeled as an 11.2% increase. This conflates the *model-comparison* Table II number (U-Net++ 0.198 → Retina-UNet 0.204, actually +3%) with the *ablation* Table V number (baseline 0.205 → full 0.228, +11.2%). These are two different comparisons (different models vs. different training configs of the same model) and shouldn't be merged into one sentence.

**Action:** Keep model-comparison deltas and ablation deltas in clearly separate tables/sentences everywhere — this same conflation appears in softer forms elsewhere (double-check PROJECT_OVERVIEW.md's equivalent paragraph).

---

## P1 — Methodological rigor gaps (a 2026 reviewer will flag these even if P0 is fixed)

### 7. n=20, single seed, and yet you're reporting Pearson r to 3 decimal places with significance stars
With n=20 test images, your own p-values (0.05–0.22 range for several metrics) already honestly show you're underpowered — good, that's honest. But:
- **Single random seed** for every model/config (already listed in your own Limitations — good that you know this) means every point estimate (Dice 79.94% vs 80.04% vs 80.48%, a 0.1–0.5% spread) is indistinguishable from seed noise. You cannot currently claim "Full method achieves best Dice" with a straight face without at least 3–5 seeds and a paired test (e.g., Wilcoxon signed-rank per-image, which n=20 paired samples actually supports well).
- **Action, concretely:** Re-run each of the 4 ablation configs × 3–5 seeds. Report mean ± std. For the headline Dice/structural deltas, use a paired non-parametric test (Wilcoxon) on the 20 per-image pairs rather than (or in addition to) comparing single-run means. This is the single highest-leverage thing you can do to make the paper "stand out" as rigorous rather than "here's a table."

### 8. The "10-metric suite" double-counts
BPR and JPR are defined once under "Topology-Aware Metrics" and then re-listed verbatim under "Graph-Based Metrics" ("same as topology-aware... included for completeness"). Counting duplicates to reach "10" reads as padding to a reviewer. You actually have **8 distinct structural metrics** (CCA, BFR, SVD, SkelDice, SkelHD, BPR, JPR, GED) + **4 clinical/descriptive** ones (Length, Width, Tortuosity, Bifurcation Count) = 12 total, no duplicates.

**Action:** Drop the "10-metric" framing; say "8-metric structural/graph suite plus 4 clinical descriptors," and remove the duplicate BPR/JPR listing. It's a small thing but "why did you count the same metric twice to inflate the number" is a cheap, easy criticism to avoid.

### 9. TC3's "Betti-0 error" isn't in your metric list
Synthetic validation (Section III-C) says TC3 causes "Betti-0 error increases" — but Betti-0 error was never defined as one of your metrics. Either define it (it's a natural, legitimate addition — Betti-0 = number of connected components, closely related to CCA) or replace the sentence with a metric you actually report (CCA drop, presumably).

### 10. Related work section is thin for a 2026 submission
You cite clDice (2021) and the persistent-homology loss (2020) but the topology-preserving-segmentation space has moved fast since — e.g., work on centerline Dice variants, Betti-matching losses (you already cite Stucki et al. 2023 in the future-work section — that citation should probably be pulled *into* related work, since it's directly relevant prior art for "differentiable topological loss," not just a future direction), skeleton-recall losses, and graph-based vessel segmentation metrics published 2023–2025. A CBM/ISBI reviewer in 2026 will expect at least 3–4 more recent (2023–2025) topology-aware segmentation citations, and ideally a discussion of why your metric suite differs from/complements clDice's loss-side approach rather than just citing it.

**Action:** Add 3–5 more recent citations (2023–2025) on topology-aware medical segmentation losses/metrics, and explicitly differentiate your contribution (evaluation-focused metric suite + failure taxonomy) from loss-focused prior work (clDice, Clough et al., Stucki et al.).

### 11. The "gradient-breaking" topological loss is a real limitation, presented too gently
You correctly disclose (Limitations) that the topo loss does tensor→numpy conversion and breaks strict gradient flow, calling it "an approximate, epoch-gated regularizer." Good — that's honest. But this deserves more prominence: it means Exp. C and D's "topological loss" isn't actually being backpropagated through in the normal sense; it's closer to a scheduled hard-example/consistency signal. Reviewers reading the loss formula (Section III) with no caveat until deep in Discussion may feel misled. Move a one-line caveat next to the loss formula itself (Section III / LOSS_FUNCTIONS.md), not just buried in Limitations.

### 12. FOV masking caveat is a good catch — but undersold
The fact that Dice collapses to ~43% without FOV masking is a legitimately interesting, honest engineering finding (shows you understand a common pitfall in DRIVE evaluation). Currently it's mentioned only in the internal docs (CODEBASE_STRUCTURE.md, METRICS_AND_EVALUATION.md), not in the paper itself. This is free, easy credibility — consider a one-sentence footnote or appendix note in the paper: "we note FOV masking is essential; without it, measured Dice deflates to ~43% due to background noise outside the retinal field, a common source of inconsistent reporting across prior work." This actively signals rigor to a reviewer.

### 13. Failure taxonomy thresholds look ad hoc / unvalidated
F1–F6 are defined by fixed thresholds (e.g., "vessel width < 3px," "BFR < −0.05," "JPR < 0.5 per image") with no stated justification for why those specific cutoffs were chosen, and no sensitivity analysis. Given F1/F2 hit 95–100% of images under your thresholds, a reviewer will reasonably ask "would F1 hit 40% of images under a slightly different threshold?" A brief threshold-sensitivity plot (F-mode rate vs. threshold) would pre-empt this and is cheap to produce since you already have the per-image data.

---

## P2 — Presentation / polish (worth doing, lower risk if skipped)

- **Contradiction risk from having 7 near-duplicate summary docs.** PROJECT_OVERVIEW.md and PROJECT_SUMMARY.md are ~90% identical and already diverged in places (e.g., the CCA copy-paste error appears in one, not verified as fixed in the other). Maintaining two overlapping "source of truth" docs is exactly how P0 issues #1–#2 happened. **Recommendation: merge into one canonical `PROJECT_OVERVIEW.md`, delete the duplicate, and treat the paper PDF (or a `RESULTS.md` generated directly from JSON) as the single source of truth for every number.**
- **BFR vs BPR naming collision.** "Branch Focal Ratio" (BFR) and "Branch Preservation Rate" / "Branch Point Ratio" (both abbreviated BPR) are easy to visually confuse in tables, especially since BPR is defined twice under two category headings. Consider renaming Branch Point Ratio → "Branch Count Ratio (BCR)" or similar to remove the ambiguity, or clearly footnote every table.
- **Fig. 9 radar chart** plots "AUC" as an axis alongside Dice/CCA/BPR/JPR/SkelDice, but AUC was never discussed as a structural metric anywhere else — it's a standard pixel metric, mixing categories on a "structural comparison" radar chart muddies the message. Consider swapping in GED (normalized) instead.
- **Author/affiliation is single-author, single-institution, no external validation.** Not fixable in docs, but worth knowing: reviewers at CBM tend to weight external clinical input for a paper making clinical-implication claims (Section V-A). Even an informal note that vessel-topology relevance was sanity-checked against existing clinical literature (which you already partially do) or, ideally, a short qualitative review by an ophthalmologist/retina specialist collaborator, would meaningfully strengthen the clinical framing versus resting purely on citations.
- **Target venue calibration:** the current combination (n=20, single dataset, single seed, one architecture family) is a reasonable fit for an **ISBI/MICCAI workshop paper** (shorter, methods/insight-focused, lower bar on statistical power) but is under-powered for a full **Computers in Biology and Medicine** journal submission, which will expect multi-seed, ideally multi-dataset, more thorough statistics. Given your own venue list already includes ISBI 2027 workshop as a secondary target — consider **submitting the workshop version first** once P0 fixes are done, and using reviewer feedback + the additional runs (P1 items) to build the stronger journal version. This is a lower-risk path to a real accepted publication than going straight for the journal with current data.

---

## Suggested Priority Order

1. **Re-run correlation analysis** and reconcile the Dice–BPR sign contradiction (#1) — this alone determines whether contribution #4 survives.
2. **Regenerate all tables from one canonical JSON output**, delete the stale ABLATION_STUDY.md numbers (#2), delete/reconcile PROJECT_SUMMARY.md vs PROJECT_OVERVIEW.md (P2).
3. **Resolve the RETFound/param-count and STARE-loader claims** (#3, #4) — say only what the code you actually ran supports.
4. **Add multi-seed runs** (3–5 seeds × 4 ablation configs) and switch headline comparisons to paired Wilcoxon tests on the n=20 per-image data (#7). This is the highest-value single addition for making the paper feel like a "proper" 2026 paper rather than a single-run demo.
5. Tighten related work (#10), fix the metric-counting/naming issues (#8, #9, P2 BFR/BPR), and surface the FOV-masking and gradient-approximation caveats more prominently in the paper body (#11, #12).
6. Consider the ISBI workshop route first, journal submission after strengthening with #4 and #7.

None of this requires new ideas — the contribution is sound. It requires making the numbers in every document agree with each other and with what the code actually produces, and adding the statistical power (seeds, paired tests) that turns "promising result" into "defensible claim."
