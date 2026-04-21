# CSEN 140 PR1 — Text Classification (k-NN)

4-class news classification. Scored by macro F1 on leaderboard (5 submissions / 24h).
**Constraint: k-NN only** — no `sklearn.KNeighborsClassifier`, no non-k-NN classifiers.

## Current status
- Local val F1: **0.9295** (13-model, regularized 7-param optimizer, K_MAX=101)
- Leaderboard F1: **0.9608** (best — +0.0001 from regularized optimizer)
- LB is consistently ~+0.034 higher than local val — gap is stable.
- Target: 0.98 LB → need ~0.946 local val.
- **Active predictions**: 13-model ensemble (cells `cell-11` optimizer + `cell-12` prediction use `optimal_weights_13`)

## LB experiment history (do not repeat failures)
| Config | Local val | LB | Notes |
|---|---|---|---|
| 13-model baseline | 0.9292 | **0.9607** | Best so far — keep as baseline |
| +M15/M16/M17 (600d LSI) | 0.9290 | 0.9596 | Hurt — removed from ensemble |
| +3 centroid models (C-M1/8/12) | 0.9294 | 0.9599 | Marginally better local, hurt LB |
| K_MAX=101 + regularized optimizer (7-param) | 0.9295 | **0.9608** | Current best — marginal +0.0001 LB gain |
| BM25 expanded grid (k1/b search on 12k subset) | 0.9280 | -- | Hurt -- subset search too noisy; M12 collapsed to 0; reverted |
| 5-fold CV + SVD-500 M9/M10 + SVD-300 M13 | 0.9290 (mean CV) | 0.9601 | Hurt -- CV averaging too conservative; SVD-500 also hurt; reverted all |
| M18 asymmetric BM25 (raw-TF query vs BM25 doc) | 0.9208 standalone | -- | Not added — standalone too weak, not complementary to M1 |
| M19 3vs4 discriminative chi2 vocabulary | 0.3628 standalone | -- | Complete failure — see lesson below |

**Key lesson: adding more models consistently hurts LB even when local val improves.**
The Powell optimizer overfits when given more weight parameters. Do NOT add more models to the ensemble without a strategy to prevent optimizer overfitting.

**Key lesson: BM25 param tuning via 12k subset search is too noisy.**
The subset search found params that hurt full val F1 (0.9295->0.9280) and caused M12 to collapse to weight=0. Hardcode k1=1.2, b=0.5 -- do not run grid searches on subsets.

**Key lesson: 5-fold CV weights hurt LB (0.9608->0.9601).**
CV averaging makes weights more conservative and loses the sharp signal the single-split optimizer finds. Single-split 7-param Powell on 80/20 is the right approach.

**Key lesson: SVD-500 hurt on all LSI models.**
Both SVD-500 for M13 (crowded out M12) and SVD-500 for M9/M10 (combined with CV) hurt LB. Keep all models at SVD-300.

**Key lesson: binary chi2 features do NOT work for 4-class k-NN (M19 failure).**
Computing chi2 on a binary class-3-vs-4 subset then projecting all 4-class docs to those features causes class 1/2 docs to scatter randomly (they have incidental business/tech vocab that gets amplified by L2 normalization). Result: model predicts only class 3/4 for everything → macro F1=0.36. Do NOT try pairwise-focused feature selection for a 4-class k-NN model.

**Key lesson: asymmetric BM25 (M18) is not complementary enough.**
Raw-TF query vs BM25 doc shares the same chi2 feature space as M1. Standalone F1=0.9208 — lower than M1, insufficient new signal. Not worth the optimizer parameter cost.

## Project layout
- `p1.ipynb` — main notebook. Cells run top-to-bottom.
- `train.dat` / `test.dat` — input data.
- `format.dat` — submission format reference.
- `predictions.dat` — final test predictions (written by save cell).
- `generate_writeup.py` — builds `CSEN140_PR1_Writeup.docx`. **Edit this, not the DOCX directly.**
- `p1_backup.ipynb` — historical snapshot; don't edit.

## Architecture

### k-NN core
- Cosine similarity via dot product on **L2-normalized** vectors (not Euclidean).
- Sparse path: `scipy.sparse` CSR → `sparse_batch_to_gpu_dense` → `torch.sparse.mm` on GPU.
- Dense path (for LSI): `l2_normalize_rows` + dense matmul on GPU.
- **Cache-based validation** (added 2026-04-20): `knn_cache_topk` / `knn_cache_topk_dense` do ONE GPU pass and cache top-K_MAX neighbors. All k-sweeps and sim_power sweeps then run in pure numpy via `topk_to_scores` / `best_k_search_cached`. ~30x faster than the old per-k GPU pass approach.

### The models (cells 9a–9p)
- **M1** BM25 (k1=1.2, b=0.5) + trigrams — highest weight model
- **M2** sublinear TF-IDF + bigrams + chi²
- **M3** BM25 + unigrams
- **M4** BM25 + bigrams
- **M5** lemma + BM25 + bigrams
- **M6** raw (no stemming) + BM25 + bigrams — surprisingly useful (weight ~0.08)
- **M7** lemma + TF-IDF + bigrams
- **M8** TF-IDF + trigrams
- **M9** LSI (SVD-300) on TF-IDF+trigrams
- **M10** LSI (SVD-300) on BM25+trigrams
- **M11** PRF (Rocchio) on M1: `expanded = 0.9*query + 0.1*centroid(top-10)`
- **M12** char 3-5-gram TF-IDF + chi² — second highest weight model
- **M13** LSI (SVD-300) on char n-gram matrix
- **M15/M16/M17** — SVD-600 upgrades of M10/M9/M13 (cells exist but **NOT in ensemble** — hurt LB)
- **Centroid C-M1/C-M8/C-M12** — `build_centroid_matrix` exists in cell `3461e281`, but **NOT in active ensemble** — added 0.0002 local but hurt LB

### Optimizer-revealed weights (sim_power=2.0, T=1.0, 13-model)
```
# 5-fold CV averaged (SVD-500 M9/M10, SVD-300 M13) — CV mean 0.9290
M1=0.342, M12=0.179, M13=0.165, M11=0.092, M10=0.085, M3=0.082, M9=0.055
best_k: M1=11 M3=21 M9=7 M10=7 M11=11 M12=9 M13=9

# Single-split regularized run (K_MAX=101, 7 free params) — local val 0.9295
M1=0.3832, M3=0.1393, M10=0.1335, M12=0.1364, M13=0.1410, M11=0.0382, M9=0.0284
M2/M4/M5/M6/M7/M8 = 0 (fixed to zero)

# Prior 13-param run (K_MAX=41) — local val 0.9292
M1=0.3433, M9=0.1355, M12=0.1336, M3=0.1247, M13=0.1062, M11=0.0490, M10=0.0463
```
Note: M10 jumped significantly (0.0463→0.1335) and M9 dropped (0.1355→0.0284) with the new config.

### Ensemble (active 13-model config)
- **Score-level soft voting**. Per-model: sum cosine sims of top-k per class → L1-normalize.
- Weights: Powell-optimized (`optimal_weights_13`) from cell `cell-11`.
- sim_power=2.0, T=1.0 (best from sweep).
- `CLASS_LABELS = np.array([1, 2, 3, 4])` — canonical class order.

### Standalone F1 per model (single-split 80/20, sim_power=2.0, best k)
| Model | Best k | Standalone F1 | Active in ensemble |
|---|---|---|---|
| M1 BM25+trigrams | 11 | **0.9216** | yes (w=0.3832) |
| M2 TF-IDF+bigrams | 9 | 0.9174 | no (zeroed) |
| M3 BM25+unigrams | 11 | 0.9135 | yes (w=0.1393) |
| M4 BM25+bigrams | 9 | 0.9180 | no (zeroed) |
| M5 lemma+BM25+bigrams | 11 | 0.9178 | no (zeroed) |
| M6 raw+BM25+bigrams | 11 | 0.9181 | no (zeroed) |
| M7 lemma+TF-IDF+bigrams | 9 | 0.9172 | no (zeroed) |
| M8 TF-IDF+trigrams | 11 | 0.9209 | no (zeroed) |
| M9 LSI-300 TF-IDF | 11 | 0.8813 | yes (w=0.0284) |
| M10 LSI-300 BM25 | 7 | 0.8876 | yes (w=0.1335) |
| M11 PRF on M1 | 11 | 0.9202 | yes (w=0.0382) |
| M12 char 3-5-grams | 9 | 0.9111 | yes (w=0.1364) |
| M13 LSI-300 char n-grams | 7 | 0.8902 | yes (w=0.1410) |

### Confusion matrix (val set, optimized 13-model ensemble, 2026-04-20)
```
Val accuracy: 18978/20416 = 0.9296  Errors: 1438

       pred1  pred2  pred3  pred4
true1:  4694    119    208    127    (8.8% error)
true2:    31   5017      8      8    (0.9% error)  <- nearly perfect
true3:   136     41   4621    357   (10.4% error)  <- worst class
true4:   116     18    269   4646    (8.0% error)

Error breakdown by pair:
  Class 3 <-> 4: 626 errors (43.5% of all errors) -- dominant confusion
  Class 1 <-> 3: 344 errors (23.9%)
  Class 1 <-> 4: 243 errors (16.9%)
  Everything else: 225 errors (15.6%)
```
Class 2 (Sports) is easily separated. The entire hard problem is World/Business/Sci-Tech overlap.
The 3<->4 confusion is a fundamental semantic overlap issue -- Business and Sci/Tech share vocabulary about companies, products, and markets. Not easily fixable with bag-of-words features alone.

### Cell structure
- `3461e281` — score helpers: `knn_predict_scores`, `knn_predict_dense_scores`, `knn_cache_topk`, `knn_cache_topk_dense`, `topk_to_scores`, `best_k_search_cached`, `l1_normalize_rows`, `build_centroid_matrix`
- `71cdabb0` — validation: caches top-K_MAX per model, k-search in numpy, sim_power sweep in numpy, stores `score_list_v` (16 entries but only first 13 used by optimizer)
- `cell-11` — optimizer: uses `score_list_v[:13]`, outputs `optimal_weights_13`. Also prints confusion matrix + per-class error rates.
- `cell-viz` — 2x2 matplotlib figure: standalone F1 bar chart, ensemble weights bar chart, confusion matrix heatmap, per-class error breakdown. Uses `%matplotlib inline`.
- `cell-12` (prediction) — uses `optimal_weights_13`, 13 models

### PRF params (conservative — aggressive hurt)
```
PRF_N=10, PRF_ALPHA=0.9, PRF_BETA=0.1
```

## Suggested next steps (pick up here next session)

**Current assessment**: 0.9295 local / 0.9608 LB appears to be near the ceiling for bag-of-words k-NN on this dataset. The dominant remaining errors (43.5%) are class 3 vs 4 (Business vs Sci/Tech) — a fundamental semantic overlap problem.

### DONE (2026-04-20)
- K_MAX=101, K_LIST extended with [51,71,101]
- Regularized optimizer (7 free params: M1/M3/M9/M10/M11/M12/M13)
- Confusion matrix added to optimizer cell output
- Visualization cell (cell-viz) added after optimizer
- M18 asymmetric BM25 tried and discarded (standalone 0.9208, not complementary)
- M19 3vs4 discriminative chi2 tried and failed (standalone 0.3628, fundamental flaw)

### Unexplored options (if attempting to push further)
- **PRF on M12** (M20): Apply Rocchio pseudo-relevance feedback to char n-grams instead of M1. May capture character-level expansion useful for 3/4 boundary (corp/inc vs tech/net suffixes). Same risk as adding any model.
- **Post-processing correction**: After ensemble prediction, for samples where p(class3) and p(class4) are within threshold, apply a separate binary k-NN trained only on class 3/4 data to break the tie. Does not touch optimizer parameters. Complex to implement correctly.

## Commands

```bash
python generate_writeup.py   # regenerate DOCX after editing script
```

## Conventions / gotchas
- Don't edit `.docx` — edit `generate_writeup.py` and regenerate.
- Don't use `Edit` tool on `.ipynb` — use `NotebookEdit` (or Python JSON edit via Bash if NotebookEdit fails due to stale file).
- Writeup keeps full trial-and-error history including failures. Don't prune.
- New models must define train/test matrices at module scope (`train_mN`, `test_mN`).
- Notebook must run top-to-bottom from cell 1. Kernel restart → rerun from start.
- K_LIST currently `[1,3,5,7,9,11] + list(range(21,42)) + [51,71,101]` (K_MAX=101 sparse, K_MAX_DENSE=301).
- Val scores cached as `score_list_v` (16 entries: 13 k-NN + 3 centroid). Optimizer uses `[:13]`.
