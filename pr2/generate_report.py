from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

for section in doc.sections:
    section.top_margin    = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin   = Inches(1.1)
    section.right_margin  = Inches(1.1)

def set_font(run, size=11, bold=False, color=None):
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)

def heading(text, size=13, color=(31, 73, 125)):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after  = Pt(3)
    run = p.add_run(text)
    set_font(run, size=size, bold=True, color=color)
    return p

def subheading(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after  = Pt(2)
    run = p.add_run(text)
    set_font(run, size=11, bold=True, color=(31, 73, 125))
    return p

def body(text, space_after=4):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(space_after)
    for run in p.runs:
        set_font(run, size=10.5)
    return p

def bullet(text):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)
    for run in p.runs:
        set_font(run, size=10.5)
    return p

# ── Title ─────────────────────────────────────────────────────────────────────
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_after = Pt(2)
r = title.add_run("Airbnb Nightly Price Prediction")
set_font(r, size=18, bold=True, color=(31, 73, 125))

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub.paragraph_format.space_after = Pt(10)
r = sub.add_run("CSEN 140 — Programming Report 2  |  Spring 2026")
set_font(r, size=11, color=(89, 89, 89))

# ── My Strategy ───────────────────────────────────────────────────────────────
heading("My Strategy")
body(
    "To predict the nightly rental price of Airbnb listings and reach the top of the "
    "leaderboard, the strategy I approached was utilizing multiple diverse model "
    "architectures to break through the limitations of relying on just one. I utilized a "
    "3-model family ensemble blending GPU-accelerated LightGBM, XGBoost, and a PyTorch MLP. "
    "A critical part of my strategy was engineering an implied price feature derived from "
    "booking-history columns already present in the provided dataset "
    "(estimated_revenue_l365d and estimated_occupancy_l365d), and completely avoiding "
    "target variable capping, which allowed the algorithm to accurately predict the expensive "
    "outlier listings that drive the leaderboard gap."
)

# ── Preprocessing ─────────────────────────────────────────────────────────────
heading("Preprocessing")

subheading("Data Cleaning")
body(
    "I parsed dollar and percentage string columns into numeric values. Columns with more "
    "than 60% missing values or identifier-like free text were dropped. I also created a "
    "has_revenue_data flag before imputing NaNs to distinguish genuine booking histories "
    "from listings that would be filled with the training median — this was essential to "
    "prevent a fake constant signal in the derived implied price feature."
)

subheading("Feature Encoding")
body(
    "To manage categorical data, I applied Out-of-fold (OOF) Bayesian target encoding for "
    "high-cardinality categoricals to capture mean-price signals without causing target "
    "leakage. Low-cardinality categoricals were one-hot encoded, and remaining "
    "high-cardinality categoricals were frequency encoded."
)

subheading("Feature Engineering")
body(
    "I created interaction features like accommodates-per-bedroom and quality-volume "
    "(review score × log reviews). Most importantly, I derived an implied_price_per_night "
    "feature by dividing estimated_revenue_l365d by estimated_occupancy_l365d — both "
    "columns already provided in the dataset. This ratio directly recovers the actual "
    "nightly price for listings with real booking history. To avoid noise for listings "
    "without data, the has_revenue_data flag (set before NaN imputation) zeroed out the "
    "implied price for those rows rather than using a misleading median-filled value."
)

subheading("Dimensionality Reduction")
body(
    "After median imputation and standardization, features for the MLP were reduced via "
    "PCA, retaining 95% of the variance and compressing 86 columns down to roughly 70. "
    "The tree-based models operated on the raw scaled features."
)

# ── The Models ────────────────────────────────────────────────────────────────
heading("The Models")
body(
    "I created a combined ensemble architecture with the idea that the structural "
    "diversity of the neural network would complement the tree-based models."
)

subheading("LightGBM")
body(
    "A GPU-accelerated histogram-based gradient boosting model running up to 5000 "
    "estimators with early stopping (patience=50), learning_rate=0.02, num_leaves=255, "
    "subsample=0.8, and colsample_bytree=0.7."
)

subheading("XGBoost")
body(
    "CUDA-accelerated gradient boosted trees, also running up to 5000 estimators with "
    "early_stopping_rounds=50, learning_rate=0.02, max_depth=7, subsample=0.8, and "
    "colsample_bytree=0.7."
)

subheading("PyTorch MLP")
body(
    "A feedforward network (512→256→128→64→1) with BatchNorm1d, GELU activations, and "
    "Dropout, trained on an RTX 5080 GPU using AdamW (lr=3e-3), HuberLoss(δ=0.5), and "
    "CosineAnnealingLR. To reduce variance from random initialization — a single run could "
    "vary by ±50 RMSE points — I trained three independent models across seeds 42, 123, "
    "and 456 and averaged their predictions, reducing the ensemble val RMSE from ~355 "
    "(single run) to ~349."
)

subheading("Baselines")
body(
    "Random Forest and CatBoost were evaluated for ensemble inclusion but were ultimately "
    "dropped. Random Forest (val RMSE ~376) was too weak. CatBoost (val RMSE ~348) was "
    "individually competitive, but because it is also a gradient boosting tree method, "
    "its predictions were too correlated with LightGBM and XGBoost to add meaningful "
    "diversity — adding it raised the leaderboard RMSE from 1010.07 to 1013.55."
)

# ── Optimizing ────────────────────────────────────────────────────────────────
heading("Optimizing and Actually Doing the Regression")

subheading("Removing the Price Cap")
body(
    "When testing, I realized that capping prices at the 99th percentile artificially "
    "lowered my local RMSE while keeping the leaderboard RMSE high, because tree models "
    "cannot extrapolate above the capped training maximum. The local RMSE appeared as low "
    "as ~133, masking a true leaderboard RMSE above 1055. Removing this cap was the single "
    "largest score improvement, dropping the leaderboard from 1055 to 1014."
)

subheading("Breaking Estimator Ceilings")
body(
    "Early experiments showed that both LightGBM and XGBoost were reporting "
    "best_iteration ≈ 2990–2997 out of n_estimators=3000, meaning they never actually "
    "triggered early stopping — they simply ran out of trees. Raising the ceiling to 5000 "
    "allowed XGBoost to find its true optimum, improving its validation RMSE from 338.10 "
    "to 336.33. LightGBM showed less consistent improvement across runs, likely due to "
    "GPU non-determinism, but the principle held: always verify models are stopping early "
    "rather than hitting the estimator limit."
)

subheading("Score Level Blending")
body(
    "Instead of a simple average, the final predictions combined the three primary models "
    "using inverse-RMSE² weighted averaging in log space, then applying expm1 to recover "
    "dollar values. Final blend weights were XGBoost ~0.344, LightGBM ~0.336, and MLP "
    "ensemble ~0.320. The PyTorch MLP was essential despite its weaker individual score — "
    "its architectural difference from the boosting models added genuine diversity, and "
    "removing it from the blend worsened the leaderboard RMSE from 1010 to 1015."
)

# ── What I Learned ────────────────────────────────────────────────────────────
heading("What I Learned")
bullet(
    "Never cap the target variable: Capping prices hides errors on expensive listings, "
    "masking a 3× local-to-leaderboard gap. The test set consistently contains more "
    "unusual and expensive listings than the training split."
)
bullet(
    "Exploit booking-history columns already in the data: Deriving implied_price_per_night "
    "from the provided estimated_revenue_l365d and estimated_occupancy_l365d columns "
    "recovers the true nightly price and strongly targets the tail of the price "
    "distribution, which has an outsized impact on the leaderboard."
)
bullet(
    "Check n_estimators ceilings: Always verify whether models are actually triggering "
    "early stopping or just running out of estimators. A best_iteration near the "
    "n_estimators limit is a sign the model needs more room to train."
)
bullet(
    "MLP adds diversity but is variance-prone: A single MLP run can vary by ±50 RMSE "
    "points across random seeds due to random initialization. Using a multi-seed ensemble "
    "stabilizes predictions and is a reliable improvement."
)
bullet(
    "Do not change regularization and capacity simultaneously: Changing num_leaves and "
    "reg_lambda at the same time made it impossible to isolate what helped, and the "
    "combination hurt test-set generalization on out-of-distribution listings even though "
    "the local validation score improved."
)

doc.save("PR2_Report.docx")
print("Saved PR2_Report.docx")
