# Method

## Metric Design Principles

Our protocol selects 8 structural metrics based on three criteria: (1) **Coverage** — each metric captures distinct topological aspects (connectivity, branching, skeleton overlap, graph edit distance, distance precision, complexity, and pixel overlap); (2) **Independence** — metrics show low pairwise correlation (Pearson |r| < 0.5 for most pairs, except SkelDice which correlates with Dice as expected); and (3) **Clinical Relevance** — each metric maps to ophthalmology-derived vessel properties (AVA, fractal dimension, junction counts, branching complexity).

We intentionally select 8 metrics because fewer would miss key topological aspects, and more would introduce redundancy without proportional diagnostic value. The selected metrics form a minimal complete basis for evaluating vessel topology: CCA captures connected component preservation, BPR measures binary pixel accuracy, JPR quantifies bifurcation correctness, GED measures graph structure fidelity, SkelDice evaluates centerline overlap, SkelHD assesses distance precision, SVD captures morphological complexity, and Dice provides standard pixel overlap baseline.

### Clinical Requirement → Structural Property → Metric Mapping

| Clinical Requirement | Structural Property | Metric |
|--------------------- |--------------------- |-------- |
| Vessel continuity | Connected components | CCA |
| Binary correctness | Pixel accuracy | BPR |
| Bifurcation preservation | Junction continuity | JPR |
| Topology fidelity | Graph alignment | GED |
| Centerline accuracy | Skeleton overlap | SkelDice |
| Distance precision | Skeleton distance | SkelHD |
| Morphological complexity | Fractal/Shannon | SVD |
| Segmentation quality | Pixel overlap | Dice |

## Dataset and Baseline Models

We evaluate 3 architecturally diverse baselines on DRIVE, STARE, and CHASE_DB1: (1) U-Net — plain encoder-decoder; (2) U-Net++ — nested-skip variant; (3) a structurally-trained variant using curriculum learning and topology loss. All models achieve comparable Dice (79.89%–80.42%), enabling clear separation of evaluation methodology effects.

## Evaluation Protocol Implementation

The protocol consists of 5 stages: ground truth preprocessing, prediction generation, skeleton extraction via morphological thinning, graph construction using connected component labeling, and metric computation via graph matching algorithms. All code is released as open-source toolkit under the Beyond Dice repository.