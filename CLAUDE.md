# CSEN 140 PR1 — Text Classification (k-NN)

4-class news classification. Scored by macro F1 on leaderboard (5 submissions / 24h).
**Constraint: k-NN only** — no `sklearn.KNeighborsClassifier`, no non-k-NN classifiers.

## Current status
- Local val F1: **0.9292** (13-model optimized ensemble, sim_power=2.0, T=1.0)
- Leaderboard F1: **0.9602** (last submitted; 0.9292 submission pending)
- LB is consistently ~+0.034 higher than local val — gap is stable, not overfit.

## Project layout
- `p1.ipynb` — main notebook. Cells run top-to-bottom; model cells are `9a`–`9l`, ensemble val is cell `10`, final test predictions is cell `11`.
- `train.dat` / `test.dat` — input data (label + text per line for train; text only for test).
- `format.dat` — submission format reference.
- `predictions.dat` — final test predictions (written by cell 11).
- `generate_writeup.py` — builds `CSEN140_PR1_Writeup.docx` from an in-script structured spec. **Edit this, not the DOCX directly.**
- `CSEN140_PR1_Writeup.docx` — generated; regenerate after editing the script.
- `p1_backup.ipynb` — historical snapshot; don't edit.

## Architecture

### k-NN core
- Cosine similarity via dot product on **L2-normalized** vectors (not Euclidean).
- Sparse path: `scipy.sparse` CSR, batched `batch @ train.T` on GPU via `sparse_batch_to_gpu_dense`.
- Dense path (for LSI): `l2_normalize_rows` + `knn_predict_dense` with pre-loaded train on GPU.
- Per-model k tuned on val split from `K_LIST` / `K_LIST_DENSE`.

### The 13 models
- **M1** BM25 (k1=1.2, b=0.5) + trigrams
- **M2** sublinear TF-IDF + trigrams + chi² feature selection
- **M3–M8** permutations of {BM25, TF-IDF} × {unigram, bigram, trigram} × {raw, Porter, WordNet}
- **M9** LSI (TruncatedSVD, 300 dims) on TF-IDF+trigrams
- **M10** LSI (TruncatedSVD, 300 dims) on BM25+trigrams
- **M11** PRF (Rocchio) on M1: `expanded = 0.9*query + 0.1*centroid(top-10)`
- **M12** char 3-5-gram TF-IDF + chi²
- **M13** LSI (TruncatedSVD, 300 dims) on M12's char n-gram matrix

### Optimizer-revealed weights (sim_power=2.0)
M1=0.3433, M3=0.1247, M9=0.1355, M12=0.1336, M13=0.1062, M11=0.0490, M10=0.0463
M2/M4/M5/M6/M7/M8 ≈ 0 (correlated with M1/M3 — near-zero weight)

### Ensemble
- **Score-level soft voting** (not hard vote). Per-model: sum cosine similarities of top-k per class → L1-normalize to per-class probabilities.
- Weighted by per-model validation F1.
- Temperature T=1.5 applied as `p ** T` then renormalize.
- `CLASS_LABELS = np.array([1, 2, 3, 4])` is the canonical class order for all score matrices.

### PRF params (conservative — aggressive hurt)
```
PRF_N=10, PRF_ALPHA=0.9, PRF_BETA=0.1
```
Aggressive (α=0.7, β=0.3, N=20) dropped ensemble to 0.9236. Keep conservative drift.

## Commands

```bash
# Regenerate writeup DOCX after editing generate_writeup.py
python generate_writeup.py
```

Notebook must be run top-to-bottom from cell 1 — later cells (9g+) depend on `train_m1_full`, `train_m8_full`, etc. built earlier. Kernel restart → rerun from start.

## Conventions / gotchas
- Don't edit `.docx` — edit `generate_writeup.py` and regenerate.
- Don't use `Edit` tool on `.ipynb` — use `NotebookEdit`.
- Writeup keeps **full trial-and-error history** including null results (temperature sweep) and failed experiments (aggressive PRF). Don't prune failures — they're pedagogical.
- New models must define train/test matrices at module scope (`train_mN`, `test_mN`) so cells 10/11 can reference them.
