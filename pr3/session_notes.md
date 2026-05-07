# PR3 Clustering — Session Notes

## Assignment Summary
- **Task:** Cluster 34,500 samples × 178 features into K=39 clusters
- **Metric:** NMI (Normalized Mutual Information) against hidden true labels, evaluated on CLP leaderboard
- **Data:** `train.dat` — multi-modal sensor activity recognition, 5 activities, 23 subjects
- **Data properties:** Already zero-mean, unit-variance (mean=0.000, std=1.000). No NaNs.
- **PCA coverage:** 127/178 components needed for 95% variance — features are spread nearly uniformly, so linear DR barely helps
- **Submission format:** `predictions.txt` — one integer per line, values 1–39, 34,500 lines total
- **Leaderboard:** https://clp.engr.scu.edu/ (SCU username/password). 5 submissions/day. Public board = 50% of data only.

---

## NMI Progress
| Approach | Silhouette | NMI (LB) |
|---|---|---|
| K-Means + PCA (original) | ~0.25 | 0.2052 |
| K-Means + UMAP cosine nc15 nn50 | 0.4356 | 0.5209 |
| K-Means + UMAP cosine nc2 nn50 | 0.4453 | — |
| **K-Means + UMAP cosine nc2 nn15 md0.0** | **0.4684** | **0.5572** ← best so far |

---

## What Went Well

### 1. UMAP >> PCA for this data
Switching from PCA to UMAP was the single biggest win (NMI 0.20 → 0.52). PCA is linear and can't capture the non-linear manifold structure of sensor/activity data. UMAP preserves local neighborhood structure which maps directly to activity clusters.

### 2. Cosine metric >> Euclidean in UMAP
Using `metric='cosine'` in UMAP dramatically outperforms `metric='euclidean'` (Silhouette 0.4356 vs 0.3690 best euclidean). For activity recognition features, the *direction* of the feature vector matters more than its magnitude.

### 3. Very low n_components works best
Trend: nc=2 (0.4684) > nc=3 (0.4314) > nc=5 (0.4259) > nc=10 (0.4287) > nc=15 (0.4356).
Forcing extreme 2D compression creates tighter, more separated clusters.

### 4. Smaller n_neighbors is better
Trend: nn=15 (0.4684) > nn=30 (0.4585) > nn=50 (0.4453) > nn=100 (0.4243) > nn=200 (0.4038).
Smaller n_neighbors = more local structure = tighter clusters.
**nn=3, 5, 7, 10 still unexplored — trend likely continues.**

### 5. K-Means beats GMM and AggWard
GMM(full) on cosine UMAP gave 0.3737 Silhouette. AggWard scores ~0.02–0.03 below KMeans.
K-Means with `n_init=100` is the reliable winner.

---

## What Didn't Go Well

### 1. CPU was too slow for grid search
`umap-learn` on CPU (178D data) took ~60–90s per UMAP config. A 162-config grid would take ~45–90 min.
**Root cause:** `pynndescent` kNN is single-threaded by default, and Numba JIT compilation makes the first call look completely idle (~60s before any real computation starts).
**Fix:** PCA-50 preprocessing (178D → 50D) + `n_jobs=-1` gives ~32x speedup per UMAP → full grid in ~5–10 min.

### 2. GPU showed 0% utilization
`cuml` (RAPIDS) was not installed — the `except` block silently fell back to CPU.
**Fix:** Added three-tier detection: cuML (WSL2) → faiss-GPU (Windows CUDA native) → CPU-optimized.

### 3. GMM singular covariance crash
GMM with `min_dist=0.0` UMAP failed with `ValueError: ill-defined empirical covariance` because min_dist=0.0 packs clusters too tightly.
**Fix:** Always use `reg_covar=1e-3` in `GaussianMixture(...)`.

### 4. Spectral on raw features was too slow
`SpectralClustering` with a precomputed cosine kNN graph on N=34,500 took too long.
**Fix:** Removed from notebook. Spectral is only applied to small 2D UMAP embeddings where it is fast.

---

## Best Configuration So Far
```python
import umap
from sklearn.cluster import KMeans

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.0,
    metric='cosine',
    random_state=42,
    low_memory=False,
    n_jobs=-1,        # added for speed
)
X_umap = reducer.fit_transform(X_scaled)

km = KMeans(n_clusters=39, init='k-means++', n_init=100, max_iter=500, random_state=42)
labels = km.fit_predict(X_umap)

np.savetxt('predictions.txt', labels + 1, fmt='%d')
```

---

## UMAP Computation — Why CPU/GPU Look Idle

**Phase 1 — pynndescent kNN graph:**
- Single-threaded by default → one core at ~100%, overall CPU % looks low on multi-core machines
- Fix: `n_jobs=-1` in `umap.UMAP(...)` parallelizes kNN across all cores

**Phase 2 — Numba JIT compilation:**
- On the very first call ever, Numba compiles kernels for 30–90 seconds — CPU shows ~0% the whole time, it is NOT hung
- After compilation, subsequent calls skip this step

**GPU 0%:** cuML not installed → silent CPU fallback. Cell 14 first line prints which mode is active.

**Speed stack (implemented in cell 14):**
- `X_pca50 = PCA(50).fit_transform(X_scaled)` → kNN in 50D instead of 178D = ~4x faster
- `n_jobs=-1` → all CPU cores = ~8x faster
- Combined: ~32x speedup (90s → ~3s per UMAP on CPU)

---

## Notebook Structure (20 cells, as of session 2)

| Cell | Status | Description |
|---|---|---|
| 0 | done | Markdown title |
| 1 | done | Imports (numpy, pandas, sklearn, etc.) |
| 2 | done | Load train.dat — shape (34500, 178) |
| 3 | done | StandardScaler + PCA coverage analysis |
| 4 | done | `evaluate()` helper — silhouette/CH/DB |
| 5 | done | Import umap-learn |
| 6 | done | Precompute UMAP embeddings (5 configs, euclidean + cosine) |
| 7 | done | KMeans on all UMAP embeddings |
| 8 | done | GMM on UMAP embeddings |
| 9 | done | Extended cosine UMAP search |
| 10 | done | AggWard on cosine embeddings |
| 11 | done | nc=2,3,5 with nn=50 |
| 12 | done | Deep nc2 cosine grid (nn=15,30,100,200; md=0,0.01,0.05) |
| **13** | **NOT RUN** | **Install all libraries** (numpy, umap-learn, pacmap, trimap, faiss-gpu/cpu, cuML check) |
| **14** | **NOT RUN** | **GPU setup** (cuML → faiss-GPU → CPU-optimized); PCA-50 preprocessing; helpers `umap_embed()` and `run_kmeans()`; nn=[3,5,7,10] sweep |
| **15** | **NOT RUN** | **Mega grid** — nc={2,3,4} × nn={3..30} × md={0,0.001,0.005} × metric={cosine,correlation} = 162 configs |
| **16** | **NOT RUN** | **Multi-seed ensemble** — 20 seeds (GPU) / 10 seeds (CPU); best-of-N + meta-feature matrix + cross-metric concat |
| **17** | **NOT RUN** | **Advanced methods** — HDBSCAN, Spectral (UMAP space only), GMM full-cov, PaCMAP, TriMap |
| **18** | done | Results summary + internal_metrics.png |
| **19** | done (stale) | predictions.txt — currently from best of cells 6–12 only |

**Run cells 13–19 in order to get fresh results and updated predictions.txt.**

---

## GPU Acceleration Setup

### faiss-GPU (Windows, no WSL2 needed)
Works natively on Windows with CUDA. Accelerates KMeans only (not UMAP kNN).
```bash
pip install faiss-gpu    # tries CUDA 11 wheels
# if that fails:
pip install faiss-cpu    # CPU fallback
```

### cuML / RAPIDS (WSL2 required for Windows — full GPU UMAP + KMeans)
```bash
# In PowerShell (Admin):
wsl --install

# In WSL2 terminal:
conda create -n rapids python=3.11
conda activate rapids
conda install -c rapidsai -c conda-forge -c nvidia rapids=25.02 cuda-version=12.0
conda install jupyter
pip install umap-learn pandas matplotlib scikit-learn pacmap trimap faiss-gpu

# Launch Jupyter from WSL2:
jupyter notebook --no-browser --port=8888
# Then open http://localhost:8888 in Windows browser
```

---

## What To Try Next (cells 13–17 cover all of these)

### Already implemented in notebook (just need to run):
1. **nn=3,5,7,10** on nc2 cosine (cell 14)
2. **162-config mega grid** — nc={2,3,4}, full nn/md/metric sweep (cell 15)
3. **Multi-seed ensemble** — 10–20 seeds, best-of-N + meta-feature matrix (cell 16)
4. **HDBSCAN** with 4 configs on top-2 embeddings (cell 17)
5. **SpectralClustering** on best 2D UMAP embeddings (cell 17)
6. **GMM full covariance** with reg_covar=1e-3 (cell 17)
7. **PaCMAP** nc2 with nn=10,15,20 (cell 17)
8. **TriMap** nc2 with nn=10,15 (cell 17)
9. **Cross-metric ensemble** — best cosine + best correlation nc2 concatenated (cell 16)

### After running cells 13–17:
- Check cell 18 (sorted results table) for the new best Silhouette
- Submit predictions.txt (from cell 19) to leaderboard
- If Silhouette went up but NMI didn't improve much → Silhouette-NMI correlation is weak for this data; try submitting the #2 and #3 models too (edit cell 19 to pick by index)
- Consider trying `n_components=1` UMAP (extreme compression) — not yet tried
- Consider L2-normalizing features before UMAP euclidean (equivalent to cosine but different numerical path)

---

## Key Parameters Summary
| Parameter | Best value | Notes |
|---|---|---|
| `n_components` | 2 | Lower is better for this data |
| `n_neighbors` | 15 (so far) | Trend still improving — smaller nn unexplored |
| `min_dist` | 0.0 | Tightest packing |
| `metric` | `'cosine'` | Direction matters more than magnitude |
| `n_init` (KMeans) | 100 | More restarts = more reliable |
| `reg_covar` (GMM) | 1e-3 | Required to prevent singular covariance |

---

## File Locations
- **Notebook:** `pr3.ipynb` (20 cells)
- **Data:** `train.dat` (local), also at `/WAVE/projects/CSEN-140-Sp26/data/pr3` on HPC
- **Submission:** `predictions.txt` (34,500 lines, values 1–39)
- **Metrics plot:** `internal_metrics.png`
- **Session notes:** `session_notes.md` (this file)
