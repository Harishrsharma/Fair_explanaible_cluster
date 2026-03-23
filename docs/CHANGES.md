# Changes, Design Decisions & Reference Paper Mapping

**Thesis:** Fair Explainable Clustering: from Fairness to Explainability
**Author:** Harish Sharma
**Supervisors:** Prof. Dr. Frank Hopfgartner & Dr. Tai Le Quy
**Date of last update:** March 2026

---

## 1. Overview of the Pipeline

The pipeline compares **5 methods** across **4 datasets**, evaluating each on three axes:
clustering quality, fairness, and explainability.

| # | Method | Type | Reference |
|---|--------|------|-----------|
| 1 | K-Means | Baseline | Lloyd (1982) |
| 2 | K-Medoids (PAM) | Baseline | Kaufman & Rousseeuw (1987) |
| 3 | K-Medians | Baseline (prof. requested) | Kaufman & Rousseeuw (1987) |
| 4 | Twagner Fairlets + K-Medoids | Fair | Chierichetti et al. (2017) |
| 5 | Twagner Fairlets + K-Medians | Fair | Chierichetti et al. (2017) |

---

## 2. Changes Made and Why

### 2.1 `src/config.py` – Centralised Configuration

**Change:** Added full dataset registry with sensitive attribute definitions for all 4 datasets.
**Why:** Previously only the bank dataset was configured. Centralising config avoids
hard-coding in data loaders and makes it easy to swap sensitive attributes for ablation studies.

**Sensitive attribute choices** (aligned with thesis proposal, Nov 2025):

| Dataset | Attribute | Value | Reason |
|---------|-----------|-------|--------|
| Bank Marketing | age_group | age > 35 = 1 | Proposal lists "age group"; bank has no gender column |
| Adult | gender | Male = 1 | Proposal lists "gender, race"; gender is the primary axis |
| COMPAS | race | African-American = 1 | Standard fairness benchmark attribute in recidivism literature |
| German Credit | sex | male = 1 | Proposal lists "age, gender"; sex column is directly available |

**Note on Bank dataset:** Earlier experiments used `marital = married` (balance ~0.45)
and `housing = yes` (better balance with p=1, q=2). The proposal specifies "age group"
as the protected attribute. Age group (>35) was chosen to align with the proposal while
maintaining interpretable group semantics. If balance is still low, switch to `housing`
in `SENSITIVE_CONFIGS["bank"]`.

**P/Q ratio:** Set to `p=1, q=2` across all datasets.
This means at least 1 minority member for every 2 majority members per cluster —
a stricter constraint than `p=q=1` (equal representation) but more realistic for
imbalanced real-world datasets.

---

### 2.2 `src/data_processing.py` – Unified Data Loaders

**Change:** Replaced single `load_bank_dataset()` with separate loaders for all 4
datasets, plus a unified `load_dataset(name, path)` entry point.

**Why:**
- Each dataset has a different file format (bank uses `;` separator), different column
  names, and different preprocessing needs (COMPAS needs column selection to remove
  identifiers).
- Unified entry point used by the pipeline for clean iteration over datasets.

**COMPAS column selection:** Only predictive features kept (`age`, `priors_count`, etc.).
Identifiers (`id`, `name`), dates, and the raw COMPAS score columns are dropped to
prevent data leakage.

---

### 2.3 `src/clustering.py` – Added K-Medians

**Change:** Added `k_medians()` function and `cluster_fairlet_centers()` helper.

**Why:**
- Professor explicitly requested K-Medians as an additional baseline.
- K-Medians uses component-wise **medians** as cluster centres and **L1 (Manhattan)**
  distance for assignment. This is consistent: the L1 minimiser is the median.
- K-Means uses mean centres + L2; K-Medoids uses actual data points + L2;
  K-Medians uses median centres + L1. Together they cover the space of common
  partitional clustering variants.
- `cluster_fairlet_centers()` is a unified dispatcher so the pipeline can call
  any algorithm on the reduced fairlet centre set without duplication.

**Reference:** The L1 consistency of the median is described in
Weiszfeld (1937) and revisited in Vardi & Zhang (2000).

---

### 2.4 `src/fairness.py` – Twagner Quadtree Fairlet Decomposition

**Change:** Replaced simple greedy `fairlet_decomposition()` with the full Twagner
quadtree method. Added `min_balance` to fairness metrics.

**Why (algorithmic):**
The simple greedy approach matched red/blue pairs globally without considering
spatial locality. This increases intra-fairlet cost because matched points may be
far apart in feature space.

The **Twagner quadtree method** (Chierichetti et al. NIPS 2017, Algorithm 1) partitions
the feature space into a hierarchy of hypercubes. Points in the same leaf node are
spatially close. Fairlets are formed within nodes first, so matched points are nearby,
minimising the intra-fairlet cost that contributes to the overall clustering cost.

**Three-phase leftover bubbling (Lemma 3, Chierichetti et al.):**
After greedy fairlet formation at a node, leftover points (incomplete groups) bubble
UP to the parent node. This ensures:
1. No points are ever dropped.
2. The spatial locality guarantee is maintained as tightly as possible.
3. The final result provably satisfies the (p,q)-balance constraint.

**Why `min_balance` was added:**
`avg_balance` can hide a single severely imbalanced cluster. `min_balance` is the
worst-case cluster balance — the key metric from Chierichetti et al. Definition 2.1.
A fair clustering must guarantee `min_balance >= p/q` for ALL clusters.

**Random shift:** Enabled by default (`QUADTREE_RANDOM_SHIFT=True`).
Theorem 3.2 of Chierichetti et al. shows that a random translation of the bounding
box reduces the expected intra-fairlet cost by a logarithmic factor.

---

### 2.5 `src/metrics.py` – Added Inertia

**Change:** Added `inertia` (sum of squared distances to centroids) alongside
silhouette and Davies-Bouldin.

**Why:** Inertia is a natural companion metric for K-Means and is requested in the
thesis proposal evaluation framework. It tracks how tightly points cluster around
their centres, independent of between-cluster separation.

**Silhouette subsampling:** For datasets with >10,000 points, silhouette is computed
on a random 10,000-point subsample for speed. The metric is statistically consistent
at this scale.

---

### 2.6 `src/explainability.py` – Rule Extraction & Exemplar Coverage

**Change:** Added `export_text()` rule extraction and `exemplar_metrics()` function.
Sensitive attribute columns are now **excluded** from the decision tree features.

**Why (rule extraction):** The thesis proposal requires human-readable "if-then" rules
as part of the explainability evaluation. `sklearn.tree.export_text()` produces exactly
this output, aligned with Moshkovitz et al. (2020).

**Why (sensitive exclusion):** A fair explanation should not reference the protected
attribute in its rules. Removing sensitive columns ensures the surrogate tree explains
clusters in terms of non-sensitive features only — consistent with the fairness-aware
explainability goal of the thesis.

**Why (exemplar coverage):** Exemplar-based explanation is the second method in the
proposal. The medoid is the most representative point in each cluster. "Coverage" measures
what fraction of cluster members are within one standard deviation of the medoid's
distance distribution — a proxy for how well the single exemplar represents its cluster.

**Reference:**
- Moshkovitz, M., Dasgupta, S., Rashtchian, C., & Frost, N. (2020).
  "Explainable k-Means and k-Medians Clustering." ICML 2020.
  https://arxiv.org/abs/2002.12538

---

### 2.7 `src/pipeline.py` – New Orchestrator Module

**Change:** Created new `src/pipeline.py` replacing the ad-hoc logic in `main.py`.

**Why:** The old `main.py` had a partially implemented `run_experiment()` that called
`fairlet_decomposition()` but then ran K-Means again instead of using the fairlet
centres — a bug. The new pipeline correctly:
1. Decomposes data into fairlets via quadtree.
2. Clusters only the fairlet centres (reduced set).
3. Propagates labels back to original points.
4. Evaluates all three metric categories.
5. Saves per-dataset and combined CSVs plus rules text files.

---

### 2.8 `main.py` – CLI Entry Point

**Change:** Replaced hardcoded `run_experiment("bank")` with an `argparse`-based CLI.

**Why:** Allows running one dataset at a time (`python main.py --dataset german`) for
faster iteration, or all four together for the full thesis comparison.

---

## 3. Reference Papers

| Paper | Used For | Link |
|-------|----------|------|
| Chierichetti, F., Kumar, R., Lattanzi, S., & Vassilvitskii, S. (2017). *Fair Clustering Through Fairlets.* NeurIPS. | Core fairlet algorithm, quadtree construction, balance definition | https://proceedings.neurips.cc/paper/2017/hash/978fce5bcc4501f762b2523a5f23b66c-Abstract.html |
| Moshkovitz, M., Dasgupta, S., Rashtchian, C., & Frost, N. (2020). *Explainable k-Means and k-Medians Clustering.* ICML. | Rule-based explainability, surrogate decision trees | https://arxiv.org/abs/2002.12538 |
| Kaufman, L., & Rousseeuw, P.J. (1987). *Clustering by Means of Medoids.* | K-Medoids (PAM), K-Medians, medoid concept | Book: Statistical Data Analysis Based on the L1-Norm |
| Rousseeuw, P.J. (1987). *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.* | Silhouette score | Journal of Computational and Applied Mathematics |
| Davies, D.L., & Bouldin, D.W. (1979). *A Cluster Separation Measure.* | Davies-Bouldin index | IEEE Transactions on Pattern Analysis and Machine Intelligence |
| Weiszfeld, E. (1937) / Vardi & Zhang (2000) | L1 minimiser = median (K-Medians justification) | — |

---

## 4. Dataset Sensitive Attribute Justification

### Bank Marketing
- No explicit gender column in `bank-full.csv`.
- Thesis proposal says "age group" → binary split at age 35 (median-ish for this dataset).
- Previous experiments: `marital=married` gave low balance; `housing=yes` gave better
  balance with p=1, q=2. Age group provides a clean, semantically meaningful split.

### Adult (Census Income)
- `gender` column present (Male/Female). Male=1, Female=0.
- Standard protected attribute in the fairness literature for this dataset
  (e.g., Hardt et al. 2016).

### COMPAS
- `race` column present. African-American = 1, all others = 0.
- This is the primary fairness concern for COMPAS highlighted in ProPublica's
  2016 analysis and subsequent fairness ML literature.

### German Credit
- `sex` column present (male/female). male=1, female=0.
- Standard sensitive attribute for this dataset in the fairness literature.

---

## 5. Metrics Summary Table

| Metric | Category | Better | Formula |
|--------|----------|--------|---------|
| Silhouette | Quality | Higher ↑ | (b−a)/max(a,b) per point |
| Davies-Bouldin | Quality | Lower ↓ | Average cluster similarity ratio |
| Inertia | Quality | Lower ↓ | Σ‖xᵢ − μₖ‖² |
| min_balance | Fairness | Higher ↑ | min over clusters of min(n₁,n₀)/max(n₁,n₀) |
| avg_balance | Fairness | Higher ↑ | Mean of per-cluster balance |
| violation_rate | Fairness | Lower ↓ | Fraction of clusters with balance < p/q |
| avg_dp_gap | Fairness | Lower ↓ | Mean |P(sensitive=1\|cluster=k) − P(sensitive=1)| |
| tree_fidelity | Explainability | Higher ↑ | Decision tree accuracy on cluster labels |
| tree_depth | Explainability | Lower ↓ | Depth of surrogate tree (simpler = better) |
| tree_leaves | Explainability | Lower ↓ | Leaf count (fewer = simpler rules) |
| avg_exemplar_coverage | Explainability | Higher ↑ | Fraction of points near their cluster medoid |
