# CSEN 140 PR1 — Text Classification (k-NN)

4-class news classification. Scored by macro F1 on leaderboard (5 submissions / 24h).
**Constraint: k-NN only** — no `sklearn.KNeighborsClassifier`, no non-k-NN classifiers.

## Current status
- Local val F1: **0.9294** (16-model run with centroids) / **0.9292** (13-model)
- Leaderboard F1: **0.9607** (best — 13-model baseline; reverted to this after centroid run dropped to 0.9599)
- LB is consistently ~+0.034 higher than local val — gap is stable.
- Target: 0.98 LB → need ~0.946 local val.
- **Active predictions**: 13-model ensemble (cells `cell-11` optimizer + `cell-12` prediction use `optimal_weights_13`)

## LB experiment history (do not repeat failures)
| Config | Local val | LB | Notes |
|---|---|---|---|
| 13-model baseline | 0.9292 | **0.9607** | Best so far — keep as baseline |
| +M15/M16/M17 (600d LSI) | 0.9290 | 0.9596 | Hurt — removed from ensemble |
| +3 centroid models (C-M1/8/12) | 0.9294 | 0.9599 | Marginally better local, hurt LB |

**Key lesson: adding more models consistently hurts LB even when local val improves.**
The Powell optimizer overfits when given more weight parameters. Do NOT add more models to the ensemble without a strategy to prevent optimizer overfitting.

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
M1=0.3433, M9=0.1355, M12=0.1336, M3=0.1247, M13=0.1062
M11=0.0490, M10=0.0463
M2/M4/M5/M6/M7/M8 ≈ 0 (correlated with M1/M3)
```
Note: in 16-model run (with centroids), M6 rose to 0.0772 and C-M1 got 0.2096 — but those weights overfit.

### Ensemble (active 13-model config)
- **Score-level soft voting**. Per-model: sum cosine sims of top-k per class → L1-normalize.
- Weights: Powell-optimized (`optimal_weights_13`) from cell `cell-11`.
- sim_power=2.0, T=1.0 (best from sweep).
- `CLASS_LABELS = np.array([1, 2, 3, 4])` — canonical class order.

### Cell structure
- `3461e281` — score helpers: `knn_predict_scores`, `knn_predict_dense_scores`, `knn_cache_topk`, `knn_cache_topk_dense`, `topk_to_scores`, `best_k_search_cached`, `l1_normalize_rows`, `build_centroid_matrix`
- `71cdabb0` — validation: caches top-K_MAX per model, k-search in numpy, sim_power sweep in numpy, stores `score_list_v` (16 entries but only first 13 used by optimizer)
- `cell-11` — optimizer: uses `score_list_v[:13]`, outputs `optimal_weights_13`
- `cell-12` (prediction) — uses `optimal_weights_13`, 13 models

### PRF params (conservative — aggressive hurt)
```
PRF_N=10, PRF_ALPHA=0.9, PRF_BETA=0.1
```

## Suggested next steps (pick up here next session)

**Problem**: Adding more models overfits the optimizer. Need to improve model quality, not quantity.

### Option A — Larger k for top models (try first, low risk)
Current K_MAX=41 for sparse models. M1 and M12 are the two highest-weight models.
Try K_MAX=101, K_LIST extended to include [51, 71, 101] for M1 and M12 only.
Larger k may improve individual model F1 and ensemble.
Implementation: modify `K_LIST` in cell `71cdabb0` to `[1,3,5,7,9,11] + list(range(21,42)) + [51,71,101]`
and set `K_MAX=101`.

### Option B — Regularized optimizer (reduce overfitting)
Instead of Powell with 13 free weights, constrain the optimizer:
- Fix weights for low-value models (M2/M4/M5/M6/M7/M8) to 0, only optimize the 7 high-weight models
- This reduces from 13 to 7 free parameters → less overfitting
- Implementation: modify `cell-11` to zero out low-weight models before optimization

### Option C — Cross-validated weights (most principled, most work)
Run 5-fold CV: fit/val split 5 times, average the optimal weights across folds.
This gives more reliable weights than single-split optimization.
Reduces optimizer overfitting at the cost of 5x more compute.

### Option D — Raw+trigrams model (M18)
M6 (raw, no stemming + BM25 + bigrams) has unexpectedly high weight ~0.08.
Try M18: raw + BM25 + trigrams (same as M6 but trigrams instead of bigrams).
Add ONLY this one model. May capture proper-noun/entity patterns that stemming loses.

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
- K_LIST currently `[1,3,5,7,9,11] + list(range(21,42))` (K_MAX=41 sparse, K_MAX_DENSE=301).
- Val scores cached as `score_list_v` (16 entries: 13 k-NN + 3 centroid). Optimizer uses `[:13]`.
