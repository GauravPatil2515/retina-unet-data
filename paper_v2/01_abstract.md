# Abstract

We introduce a topology-aware structural evaluation protocol for retinal vessel segmentation comprising 8 complementary metrics (CCA, BPR, JPR, GED, SkelDice, SkelHD, SVD, and morphological failure modes). We demonstrate that models achieving near-identical Dice scores (79.89%–80.42%) differ substantially on junction preservation (up to 5.2%) and graph edit distance (up to 77.6 points). Statistical analysis confirms Dice is significantly independent of structural metrics (p > 0.05). We release an open-source topology-aware evaluation toolkit for the community.

**Contributions:**
1. We define and validate an 8-metric structural evaluation protocol for retinal vessel segmentation.
2. We prove that Dice-based and structure-based rankings diverge across all evaluated architectures.
3. We release the first open-source topology-aware retinal vessel evaluation toolkit for community use.