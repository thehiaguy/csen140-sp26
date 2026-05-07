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
| Approach | Silhouette | NMI |
|---|---|---|
| K-Means + PCA (original) | ~0.25 | 0.2052 |
| K-Means + UMAP cosine nc15 nn50 | 0.4356 | 0.5209 |
| K-Means + UMAP cosine nc2 nn50 | 0.4453 | — |
| K-Means + UMAP cosine nc2 nn15 | 0.4684 | 0.5572 |

---

## What Went Well

### 1. UMAP >> PCA for this data
Switching from PCA to UMAP was the single biggest win (NMI 0.20 → 0.52). PCA is linear and can't capture the non-linear manifold structure of sensor/activity data. UMAP preserves local neighborhood structure which maps directly to activity clusters.

### 2. Cosine metric >> Euclidean in UMAP
Using `metric='cosine'` in UMAP dramatically outperforms `metric='euclidean'` (Silhouette 0.4356 vs 0.3690 best euclidean). For activity recognition features, the *direction* of the feature vector matters more than its magnitude, which is exactly what cosine distance captures.

### 3. Very low n_components works best
Trend: nc=2 (0.4684) > nc=3 (0.4314) > nc=5 (0.4259) > nc=10 (0.4287) > nc=15 (0.4356).
Forcing extreme 2D compression creates tighter, more separated clusters.

### 4. Smaller n_neighbors is better
Trend: nn=15 (0.4684) > nn=30 (0.4585) > nn=50 (0.4453) > nn=100 (0.4243) > nn=200 (0.4038).
Smaller n_neighbors = more local structure = tighter clusters.
**Still unexplored: nn=5, 7, 10 — likely to improve further.**

### 5. K-Means beats GMM and AggWard on UMAP embeddings
GMM(full) on cosine UMAP gave 0.3737 Silhouette (worse than KMeans 0.4356).
AggWard consistently scores ~0.02–0.03 below KMeans.
K-Means with `n_init=100` is the reliable winner.

---

## What Didn't Go Well

### 1. GMM singular covariance crash
GMM with `min_dist=0.0` UMAP failed with `ValueError: ill-defined empirical covariance` because clusters were packed too tightly. Fix: add `reg_covar=1e-3` to `GaussianMixture(...)`.

### 2. Diminishing returns from parameter search
Each round of tuning gives smaller Silhouette gains (+0.01 to +0.02). We may be near the K-Means ceiling on 2D cosine UMAP. HDBSCAN is the logical next step since it can find non-spherical/non-convex clusters.

### 3. NMI ceiling unknown
Internal Silhouette correlates with NMI but imperfectly. We don't know the true label structure (39 clusters from 5 activities × 23 subjects doesn't divide cleanly).

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
    low_memory=False
)
X_umap = reducer.fit_transform(X_scaled)

km = KMeans(n_clusters=39, init='k-means++', n_init=100, max_iter=500, random_state=42)
labels = km.fit_predict(X_umap)

# Output: 1-indexed
np.savetxt('predictions.txt', labels + 1, fmt='%d')
```

---

## What To Try Next (Priority Order)

### 1. Smaller n_neighbors on nc2 cosine (CURRENTLY IN NOTEBOOK)
Try nn=5, 7, 10 — trend hasn't bottomed out yet.
```python
for nn in [5, 7, 10]:
    reducer = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=0.0,
                        metric='cosine', random_state=42)
    ...
```

### 2. HDBSCAN on 2D cosine UMAP
K-Means assumes spherical clusters. HDBSCAN finds arbitrary shapes.
Complication: must force exactly 39 clusters. Post-process:
- Noise points (label=-1): assign to nearest cluster centroid via KNN
- If N > 39 clusters: merge smallest into nearest centroid
- If N < 39 clusters: fall back to K-Means on that embedding
```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=10,
                              cluster_selection_method='eom')
labels = clusterer.fit_predict(X_umap)
```

### 3. GPU-accelerated search with RAPIDS cuML (RTX 5080 machine)
Allows exploring 50+ UMAP configs in the time CPU takes for 5.
```python
from cuml.manifold import UMAP as cuUMAP
from cuml.cluster import KMeans as cuKMeans
```
Setup on Windows 11: WSL2 + Miniconda + `conda install -c rapidsai -c conda-forge -c nvidia rapids=25.02 cuda-version=12.0`

### 4. Wider nc2 cosine grid search (with GPU)
Now that nc2 cosine is confirmed best, try a fine grid:
- n_neighbors: 3, 5, 7, 10, 12, 15, 20, 25
- min_dist: 0.0, 0.001, 0.005, 0.01
- metric: cosine, correlation

### 5. GMM with full covariance on best 2D embedding
With `reg_covar=1e-3` to avoid singular matrices:
```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=39, covariance_type='full',
                      n_init=10, reg_covar=1e-3, random_state=42)
labels = gmm.fit_predict(X_umap)
```

---

## Environment Notes
- **Local Mac:** Python 3.13, no CUDA, `umap-learn` installed with `--break-system-packages`
- **Windows RTX 5080 machine:** Windows 11, needs WSL2 for RAPIDS
  - Install WSL2: `wsl --install` (PowerShell as Admin)
  - Install Miniconda in WSL2
  - `conda create -n rapids python=3.11`
  - `conda install -c rapidsai -c conda-forge -c nvidia rapids=25.02 cuda-version=12.0`
  - `conda install jupyter`
  - `pip install umap-learn pandas matplotlib scikit-learn`
  - Launch: `jupyter notebook --no-browser --port=8888`

## File Locations
- Notebook: `pr3.ipynb`
- Data: `train.dat` (local), also at `/WAVE/projects/CSEN-140-Sp26/data/pr3` on HPC
- Submission file: `predictions.txt` (34,500 lines, values 1–39)
- Internal metrics plot: `internal_metrics.png`
