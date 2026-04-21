from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# ── Styles ────────────────────────────────────────────────────────────────────
def h1(text):
    p = doc.add_heading(text, level=1)
    p.runs[0].font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)

def h2(text):
    p = doc.add_heading(text, level=2)
    p.runs[0].font.color.rgb = RGBColor(0x2E, 0x74, 0xB5)

def h3(text):
    doc.add_heading(text, level=3)

def body(text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(6)

def code(text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x00, 0x00, 0x80)

def bullet(text, level=0):
    p = doc.add_paragraph(text, style='List Bullet')
    p.paragraph_format.left_indent = Inches(0.4 + level * 0.2)

def add_divider():
    doc.add_paragraph('─' * 80)

# ── Title ─────────────────────────────────────────────────────────────────────
title = doc.add_heading('CSEN-140 PR1: Text Classification — Full Technical Write-Up', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph(
    'How we built a 13-model k-NN ensemble for news classification — combining surface '
    'n-grams, character n-grams, latent semantic analysis (LSI), and pseudo-relevance '
    'feedback (PRF) — maximized GPU utilization on an RTX 5080, and iteratively refined '
    'score-level ensembling through Powell-optimized weight search and cache-based '
    'validation. Final leaderboard score: 0.9608 (13-model, regularized 7-param Powell '
    'optimizer). Local F1: 0.9295.',
    style='Intense Quote'
)

doc.add_page_break()

# ══════════════════════════════════════════════════════════════════════════════
h1('PART 1 — THE ASSIGNMENT AND THE GOAL')
# ══════════════════════════════════════════════════════════════════════════════

body(
    'The assignment (PR1) requires building a k-Nearest Neighbours (k-NN) text classifier '
    'from scratch — no sklearn KNeighborsClassifier allowed. The dataset is a collection of '
    'news abstracts belonging to 4 classes (e.g. World, Sports, Business, Science/Tech). '
    'The goal is to maximize macro F1-score on a held-out test set, measured on a class '
    'leaderboard with 5 submissions per 24 hours.'
)
body(
    'The core idea of k-NN is: given a new document, find the k most similar training '
    'documents, and let them vote on the label. The tricky parts are (1) how you represent '
    'documents as vectors, and (2) how you measure similarity efficiently when you have '
    '100,000+ training examples and a tight submission budget.'
)

# ══════════════════════════════════════════════════════════════════════════════
h1('PART 2 — HOW DOCUMENTS BECOME VECTORS')
# ══════════════════════════════════════════════════════════════════════════════

h2('2.1  Tokenization and Preprocessing')
body(
    'Raw text can\'t go directly into math. We first convert each abstract into a list of '
    'tokens. For most models we:'
)
bullet('Lowercase everything.')
bullet('Strip punctuation (using str.maketrans).')
bullet('Remove stopwords ("the", "is", "at", etc.) — words so common they carry no signal.')
bullet('Apply a stemmer or lemmatizer to collapse word variants.')

body('Three different normalization strategies were used across models:')
bullet('Porter Stemmer — heuristically chops suffixes: "running" → "run", "prices" → "price".')
bullet('WordNet Lemmatizer — looks up the dictionary base form: "better" → "good".')
bullet('Raw words — no normalization at all; keeps "running", "runner", "runs" separate.')

body(
    'We also generate n-grams — sequences of consecutive tokens that capture phrases. '
    '"stock market" as a bigram is more informative than "stock" and "market" separately.'
)
bullet('Unigrams: single words ("oil", "price").')
bullet('Bigrams: two-word phrases ("oil_price", "stock_market").')
bullet('Trigrams: three-word phrases ("stock_market_crash") — used in Models 1 and 8.')

h2('2.2  Bag of Words — Turning Tokens into Numbers')
body(
    'Once we have tokens, we build a vocabulary: a mapping from every unique token to an '
    'integer index. Then each document becomes a sparse vector: index i holds the weight '
    'of word i in that document. Most entries are zero (a document only contains a tiny '
    'fraction of all vocabulary words), so we use scipy sparse CSR matrices to store only '
    'the non-zero entries — this is critical for memory efficiency.'
)

h2('2.3  BM25 Weighting')
body(
    'Not all word occurrences are equally informative. BM25 (Best Match 25) is the gold '
    'standard weighting scheme from information retrieval. The formula for term t in '
    'document d is:'
)
code('score(t, d) = IDF(t) × tf × (k1 + 1) / (tf + k1 × (1 - b + b × |d| / avgdl))')
body('Where:')
bullet('IDF(t) = log((N - df + 0.5) / (df + 0.5) + 1)  — inverse document frequency. '
       'Rare words get high IDF; extremely common words get low IDF.')
bullet('tf = term frequency in document d.')
bullet('|d| = document length, avgdl = average document length across all docs.')
bullet('k1 controls term frequency saturation (we found k1=1.2 optimal).')
bullet('b controls length normalization (we found b=0.5 optimal).')
body(
    'After computing BM25 scores, every row (document) is L2-normalized '
    '(divided by its Euclidean length). This means cosine similarity between two '
    'document vectors is simply their dot product — which we can compute very fast.'
)

h2('2.4  TF-IDF Weighting (Alternative)')
body(
    'For some models we use sublinear TF-IDF instead of BM25:'
)
code('score(t, d) = (1 + log(tf)) / total_terms × IDF(t)')
body(
    'This compresses large term frequencies logarithmically. BM25 tends to outperform '
    'TF-IDF for k-NN because its length normalization is more principled, but TF-IDF '
    'adds useful diversity in an ensemble. The key difference: BM25\'s saturation term '
    'dampens high-frequency words more aggressively than sublinear TF-IDF, so the two '
    'scoring functions sometimes disagree on borderline documents — which is exactly '
    'what makes combining them valuable.'
)

h2('2.5  Chi-Squared Feature Selection')
body(
    'Even after capping vocabulary at 150,000–300,000 words, many features are noise. '
    'Chi-squared (χ²) feature selection measures the statistical dependence between each '
    'feature and the class labels. We keep only the top k features by χ² score — '
    '200,000 for Models 1 and 8, 120,000 for most others, 50,000 for Model 3. '
    'This removes noise, reduces memory, and often improves accuracy because k-NN is '
    'sensitive to irrelevant dimensions.'
)

h2('2.6  Character N-gram Features')
body(
    'Beyond word-level features, character n-grams capture sub-word morphology. '
    'By sliding a window of 3–5 characters over each word (with boundary markers), '
    'we generate features like "#sci", "scie", "cien", "ienc", "ence", "nce#" for '
    '"science". These features appear in "science", "scientific", and "scientist" — '
    'capturing morphological relationships that word-level models miss. Character n-grams '
    'are especially useful for distinguishing Business vs. Science/Tech vocabulary, where '
    'suffixes like "corp", "inc", "tech", "net" are class-discriminative at the character level.'
)

# ══════════════════════════════════════════════════════════════════════════════
h1('PART 3 — K-NN CLASSIFICATION')
# ══════════════════════════════════════════════════════════════════════════════

h2('3.1  The Algorithm')
body(
    'After building normalized sparse matrices for training and test, k-NN works as follows '
    'for each test document:'
)
bullet('Compute cosine similarity to every training document (dot product, since both are L2-normalized).')
bullet('Find the top-k most similar training documents.')
bullet('Let those k neighbors vote on the label, weighted by their similarity score.')
bullet('Predict the label with the highest total weight.')

body(
    'Distance-weighted voting means a neighbor with similarity 0.95 has much more '
    'influence than one with similarity 0.60 — this consistently outperforms simple '
    'majority voting.'
)

h2('3.2  Similarity Power (sim_power)')
body(
    'Beyond distance weighting, we raise each neighbor\'s cosine similarity to a power '
    'before accumulating votes: vote_weight = similarity ** sim_power. With sim_power=1.0 '
    'this is standard distance weighting. With sim_power=2.0 (our final setting), '
    'neighbors with higher similarity are amplified quadratically — a neighbor at 0.9 '
    'similarity gets 0.81 weight, while one at 0.5 gets only 0.25. This sharpens '
    'the voting so that very close neighbors dominate and distant neighbors contribute '
    'almost nothing. We swept sim_power ∈ {1.0, 1.5, 2.0, 3.0} and found 2.0 consistently '
    'best across models.'
)

h2('3.3  Choosing k — The Journey')
body(
    'k is a critical hyperparameter. Too small: predictions are noisy. Too large: '
    'you start including documents from the wrong class. We sweep over a list of k '
    'values and pick the one that maximizes validation F1. Our K_LIST evolved through '
    'several iterations:'
)
bullet('Initial: K_LIST = [3, 5, 7, 9, 11] — only small k values, first baseline.')
bullet('Expanded: K_LIST = list(range(21, 42)) — found larger k values were better, but accidentally dropped k=11.')
bullet('Bug fix: K_LIST = [11] + list(range(21, 42)) — re-added k=11 explicitly after score dropped.')
bullet('Attempted extension: K_LIST = [3, 5, 7, 9, 11] + list(range(21, 42)) — added small k values back to test all options.')
bullet('Reverted: K_LIST = [11] + list(range(21, 42)) — the extended sweep caused overfitting on the leaderboard.')
bullet('Final (current): K_LIST = [1,3,5,7,9,11] + list(range(21,42)) + [51,71,101] — extended to K_MAX=101. '
       'Sparse models use this list; dense LSI models use K_LIST_DENSE = [11,21,31,41,51,71,101,151,201,251,301].')
body(
    'The extension to K_MAX=101 was motivated by the char n-gram and LSI models, which '
    'benefit from larger neighborhoods because their similarity scores are smoother. '
    'With cache-based validation (see Part 5.9), sweeping a wider k range adds no GPU '
    'cost — all sweeps run on cached top-K_MAX results in numpy.'
)

# ══════════════════════════════════════════════════════════════════════════════
h1('PART 4 — THE ENSEMBLE: 13 MODELS')
# ══════════════════════════════════════════════════════════════════════════════

body(
    'A single k-NN model is limited by its feature representation. An ensemble combines '
    'multiple models, each with a different "view" of the data. When one model makes a '
    'mistake, the others may still get it right — and the weighted vote corrects the error.'
)
body(
    'Each model\'s weight in the ensemble is determined by a Powell optimizer that '
    'directly maximizes validation macro F1. Stronger and more complementary models '
    'receive higher weight; redundant models are driven to zero.'
)

h2('4.1  The 13 Models')

models = [
    ('M1', 'BM25 + unigrams + bigrams + trigrams + chi2(200k), vocab 300k',
     'Porter stemmer. The kitchen-sink model — largest feature set, includes trigrams. '
     'Started at vocab 200k / chi2(150k), expanded to 300k / 200k to capture more rare '
     'trigrams like "interest_rate_hike". Individual F1: ~0.9216. Highest weight in '
     'ensemble (w=0.3832).'),
    ('M2', 'TF-IDF + unigrams + bigrams + chi2(120k)',
     'Porter stemmer. Same preprocessing as M1 but TF-IDF scoring, no trigrams. '
     'Individual F1: ~0.9174. Zeroed by Powell optimizer (redundant with M1/M3).'),
    ('M3', 'BM25 + unigrams only + chi2(50k)',
     'Porter stemmer. Unigrams only — completely ignores phrase structure. '
     'Weakest individual model (~0.9135) but contributes in the ensemble (w=0.1393). '
     'We tried removing it — local improved slightly but leaderboard dropped. '
     'Captures something the other models miss.'),
    ('M4', 'BM25 + unigrams + bigrams + chi2(120k)',
     'Porter stemmer. Like M1 but no trigrams, smaller feature set. Individual F1: ~0.9180. '
     'Zeroed by Powell optimizer.'),
    ('M5', 'BM25 + unigrams + bigrams + chi2(120k)',
     'WordNet lemmatizer. Same scoring as M4 but dictionary-form normalization instead '
     'of heuristic stemming. Individual F1: ~0.9178. Zeroed by Powell optimizer.'),
    ('M6', 'BM25 + unigrams + bigrams + chi2(120k)',
     'Raw words — NO stemming or lemmatizing. Keeps "stocks", "markets", "running" as-is. '
     'Completely different vocabulary from M1–M5. Individual F1: ~0.9181. '
     'Zeroed by Powell optimizer despite good standalone performance.'),
    ('M7', 'TF-IDF + unigrams + bigrams + chi2(120k)',
     'WordNet lemmatizer + TF-IDF scoring. Combines M5\'s preprocessing with M2\'s '
     'scoring. Individual F1: ~0.9172. Zeroed by Powell optimizer.'),
    ('M8', 'TF-IDF + unigrams + bigrams + trigrams + chi2(200k), vocab 300k',
     'Porter stemmer + TF-IDF scoring. The trigram counterpart to M1 — same features, '
     'different scoring function. Individual F1: ~0.9209. Zeroed by Powell optimizer '
     '(its signal appears captured by M1 + M12 combination).'),
    ('M9', 'LSI (TruncatedSVD, 300 dims) on TF-IDF + trigrams, dense cosine k-NN',
     'First latent-semantic model. Takes the pre-chi2 TF-IDF trigram matrix and reduces '
     'it to 300 dense dimensions via randomized truncated SVD. Captures latent topic '
     'structure — synonyms and topically-related terms end up close in the reduced space '
     'even when they never co-occur in the same document. '
     'Individual F1: ~0.8813. Active in ensemble (w=0.0284).'),
    ('M10', 'LSI (TruncatedSVD, 300 dims) on BM25 + trigrams',
     'Second latent-semantic model. Same SVD-300 reduction but applied to BM25-scored '
     'features instead of TF-IDF. BM25\'s saturation function emphasizes different '
     'term-document structure, so the latent topics SVD extracts are slightly different. '
     'Individual F1: ~0.8876. Active in ensemble (w=0.1335) — the optimizer assigns it '
     'substantially more weight than M9 because BM25-based LSI topics are cleaner.'),
    ('M11', 'Pseudo-Relevance Feedback (PRF) on M1 — query expansion + 2nd-pass k-NN',
     'Rocchio-style query expansion. For each query, the first k-NN pass finds the top-N '
     'nearest neighbors (similarity-weighted), the query is replaced by '
     'alpha*query + beta*centroid(top-N neighbors), and a second k-NN pass retrieves '
     'against the expanded query. Conservative tuning (alpha=0.9, beta=0.1, N=10) was '
     'required — aggressive values (alpha=0.7, beta=0.3, N=20) caused query drift to '
     'wrong-class clusters. Individual F1: ~0.9202. Active in ensemble (w=0.0382).'),
    ('M12', 'TF-IDF + character 3–5-grams + chi2, vocab ~150k',
     'Character n-gram model. Slides a window of 3, 4, and 5 characters over each '
     'word with boundary markers (e.g. "#sci", "scie", "cien", "ence", "nce#") and '
     'builds a TF-IDF weighted bag of character n-grams. Captures morphological patterns '
     'across word forms — "scien" appears in "science", "scientific", "scientist". '
     'Completely different representation from all word-level models. '
     'Individual F1: ~0.9111 (below word models solo, but highly complementary). '
     'Second highest ensemble weight (w=0.1364).'),
    ('M13', 'LSI (TruncatedSVD, 300 dims) on char n-gram matrix',
     'Latent-semantic analysis applied to the character n-gram space. SVD-300 on the '
     'M12 feature matrix produces a dense 300-dim topic signature in character-space. '
     'Captures latent morphological structure — related character patterns compress '
     'into shared topic dimensions. Individual F1: ~0.8902. '
     'Highest ensemble weight after M1 (w=0.1410) — the optimizer discovered that '
     'char-level LSI is more complementary to the word-level models than word-level '
     'LSI (M9/M10), because it operates on a fundamentally different feature space.'),
]

for name, config, desc in models:
    p = doc.add_paragraph()
    run = p.add_run(f'{name}: {config}')
    run.bold = True
    run.font.size = Pt(11)
    body(f'    → {desc}')

body(
    'The key insight: every model uses a different combination of (preprocessing × scoring × '
    'n-gram order × feature space × reduction × query transformation). When they disagree, '
    'the ensemble votes. The diversity is what makes ensembles work — M12\'s char n-grams and '
    'M13\'s char-space LSI add signal that no word-level model can replicate.'
)

h2('4.2  Latent Semantic Indexing (LSI) — M9, M10, M13')
body(
    'Three models operate via LSI (Truncated SVD). M1–M8 and M12 operate in sparse n-gram '
    'or char-gram space. Two documents that discuss the same topic using different words '
    '(e.g. "stocks rise" vs "shares climb") may share very few n-grams and look dissimilar '
    'despite meaning the same thing. LSI addresses this: Truncated Singular Value '
    'Decomposition (SVD) factors the term-document matrix into a much smaller dense '
    'representation where each dimension captures a latent topic — a weighted combination '
    'of terms that tend to co-occur.'
)
body(
    'M9 takes the TF-IDF + trigram matrix and computes a rank-300 randomized SVD, '
    'producing a dense matrix where each row is a 300-dim topic signature. '
    'M10 applies the same SVD to the BM25 matrix — the latent topics differ slightly '
    'because BM25\'s saturation function changes which term-document entries dominate. '
    'M13 applies SVD-300 to the character n-gram matrix — a third distinct topic space '
    'based on morphological rather than lexical structure.'
)
body(
    'Why 300 dimensions? A sweet spot: small enough that topics are meaningful and '
    'retrieval is fast, large enough to preserve class structure. We tried SVD-500 for '
    'M9/M10 and SVD-600 for additional models — both hurt leaderboard performance. '
    'More components add noise along with signal, and the optimizer overfits to the '
    'additional parameters. All LSI models stay at SVD-300.'
)
body(
    'M9 alone scores 0.8813, clearly below the sparse models. M10 scores 0.8876 solo. '
    'M13 scores 0.8902. The value of these models is purely ensemble diversity — they make '
    'different mistakes from the sparse models because their similarity metric is '
    'fundamentally different. The optimizer confirms this: M13\'s ensemble weight (0.1410) '
    'exceeds M10\'s (0.1335) and far exceeds M9\'s (0.0284) — char-space LSI is the '
    'most complementary of the three.'
)

h2('4.3  Pseudo-Relevance Feedback (PRF) — M11')
body(
    'PRF is an information-retrieval technique that assumes the top-N retrieved '
    'documents for a query are "relevant" (even without ground truth labels), then '
    'uses them to refine the query. The classic Rocchio formula:'
)
code('expanded_query = alpha * original_query + beta * centroid(top_N_neighbors)')
body(
    'The refined query is then re-issued against the corpus, producing a second set '
    'of nearest neighbors that (in theory) are more topically coherent with the '
    'query\'s actual class. The 2-pass approach effectively "pulls" noisy queries '
    'toward their topical cluster before final ranking.'
)
body(
    'Parameter tuning was critical. Initial values (alpha=0.7, beta=0.3, N=20) caused '
    'drift into neighbor classes — M11 scored only 0.9182 solo, and the ensemble F1 '
    'dropped from 0.9240 to 0.9236. The expanded queries were pulled too strongly '
    'toward the centroid, which occasionally straddled class boundaries.'
)
body(
    'We tuned to conservative drift (alpha=0.9, beta=0.1, N=10): the original query '
    'dominates, and only the 10 closest neighbors contribute a small nudge. M11 '
    'rose to 0.9202 solo (still slightly below M1\'s 0.9216 — PRF does not improve '
    'M1 directly), but the ensemble climbed. The 2-pass k-NN produces predictions '
    'that are just different enough from M1 to contribute diversity.'
)

h2('4.4  Char N-gram Models (M12 and M13)')
body(
    'M12 and M13 operate below the word level, on character sequences. '
    'For each word, we generate all character n-grams of length 3, 4, and 5, '
    'with boundary markers (#) prepended and appended. For "science": '
    '"#sc", "sci", "cie", "ien", "enc", "nce", "ce#", "#sci", "scie", "cien", '
    '"ienc", "ence", "nce#", "#scie", "scien", "cienc", "ience", "ence#", etc. '
    'These character sequences appear across inflected forms of the same root word, '
    'capturing morphological family relationships that word n-grams miss.'
)
body(
    'Why char n-grams help on this specific dataset: the 4-class confusion is dominated '
    'by Business (class 3) vs Science/Tech (class 4). These classes share many words '
    'but differ in morphological patterns — "corporation", "incorporated", "quarterly" '
    'vs "technology", "processor", "bandwidth". Character n-grams pick up these suffixes '
    '("corp", "inc", "quar" vs "tech", "cess", "widt") without requiring exact word matches.'
)
body(
    'M12 (sparse TF-IDF char n-grams) has standalone F1=0.9111 — below all word models '
    'except M9/M10/M13. But its ensemble weight (0.1364) is the second highest after M1. '
    'M13 (LSI-300 on char n-grams) has F1=0.8902 standalone but the highest weight of '
    'all models except M1 (0.1410). The optimizer\'s preference for these models reveals '
    'that their errors are the most decorrelated from the word-level models — they are '
    'the most complementary models in the ensemble despite weaker standalone performance.'
)

h2('4.5  Models We Tried and Removed')

body(
    'Not every model made the final ensemble. Several were added and removed after testing:'
)
p = doc.add_paragraph()
run = p.add_run('Original M8 (char n-grams): BM25 + character 3–4-grams + chi2(80k)')
run.bold = True
run.font.size = Pt(11)
body(
    '    → Extracted character sequences from stemmed words using boundary markers. '
    'Local ensemble improved from 0.9201 → 0.9212 (best local score yet). '
    'But leaderboard dropped from 0.9570 → 0.9567. Classic overfitting to the validation '
    'split. Removed permanently. (Later replaced by M12 with better char n-gram design.)'
)

p = doc.add_paragraph()
run = p.add_run('M15/M16/M17: SVD-600 upgrades of M10/M9/M13')
run.bold = True
run.font.size = Pt(11)
body(
    '    → Attempted to improve the LSI models by doubling the SVD dimensions from 300 '
    'to 600. Local F1 was 0.9290 (marginal drop from 0.9292 baseline). '
    'Leaderboard dropped from 0.9607 to 0.9596 — a clear hurt. '
    'SVD-600 adds noise along with signal at this corpus size. Removed permanently. '
    'All LSI models stay at SVD-300.'
)

p = doc.add_paragraph()
run = p.add_run('Centroid Models C-M1/C-M8/C-M12: per-class centroid k-NN')
run.bold = True
run.font.size = Pt(11)
body(
    '    → Built per-class centroid vectors (average of all training documents in each '
    'class) and used them as additional "anchor" models. Local F1 improved marginally '
    '(0.9292 → 0.9294). Leaderboard dropped from 0.9607 to 0.9599. '
    'The centroid models are too correlated with the full k-NN models to add new signal. '
    'Not in active ensemble.'
)

p = doc.add_paragraph()
run = p.add_run('M18: Asymmetric BM25 (raw-TF query vs BM25 document)')
run.bold = True
run.font.size = Pt(11)
body(
    '    → Applied BM25 weighting to training documents but used raw term frequencies '
    'for test queries (asymmetric scoring). Standalone F1=0.9208 — below M1\'s 0.9216. '
    'Shares the same chi2 feature space as M1, so adds no new signal. '
    'Not worth the additional optimizer parameter cost. Not added.'
)

p = doc.add_paragraph()
run = p.add_run('M19: 3vs4 discriminative chi2 vocabulary')
run.bold = True
run.font.size = Pt(11)
body(
    '    → Attempted to build a model with features selected specifically to discriminate '
    'class 3 (Business) from class 4 (Sci/Tech) — the dominant source of confusion. '
    'Computed chi2 on a binary class-3-vs-4 subset, then projected all 4-class documents '
    'to those features. Complete failure: standalone F1=0.3628. '
    'Root cause: class 1/2 documents have incidental business/tech vocabulary that, when '
    'amplified by L2 normalization in a business/tech-focused feature space, scatters them '
    'randomly. The model predicted only class 3/4 for everything. '
    'Do NOT try pairwise-focused feature selection for a 4-class k-NN model.'
)

h2('4.6  Ensemble Weighting: From F1 Weights to Powell Optimizer')
body(
    'Early in the project, model weights in the ensemble were simply set proportional '
    'to each model\'s validation F1 score. This is intuitive — stronger models get '
    'more influence — but it ignores the complementarity between models. Two models '
    'with identical F1 but decorrelated errors are worth more together than two high-F1 '
    'models that always agree.'
)
body('Early weighting experiments before the optimizer:')
bullet(
    'Linear F1 weights: w = best_f1_mX. Simple and robust. '
    'Proportional influence based on validation performance.'
)
bullet(
    'Squared F1 weights (tried, reverted): w = best_f1_mX ** 2. '
    'Intended to amplify stronger models more aggressively. '
    'Made zero difference — the individual F1 scores are too tightly '
    'clustered (0.88–0.92) for squaring to flip any votes. Reverted.'
)
body(
    'The major upgrade was replacing F1-based weights with a Powell optimizer. '
    'scipy.optimize.minimize(method="Powell") directly maximizes validation macro F1 '
    'by searching the 13-dimensional weight space. The objective function is cheap '
    'because it runs on cached top-K_MAX scores (no GPU) — each evaluation is pure '
    'numpy matrix operations. With ~1000 optimizer evaluations feasible per minute, '
    'the optimizer thoroughly explores the weight space and finds complementarity '
    'patterns that F1 weights cannot capture.'
)
code(
    'def objective(weights):\n'
    '    weights = np.maximum(weights, 0)          # no negative weights\n'
    '    weights = weights / weights.sum()          # normalize to simplex\n'
    '    combined = sum(w * scores for w, scores in zip(weights, score_list_v[:13]))\n'
    '    preds = CLASS_LABELS[combined.argmax(axis=1)]\n'
    '    return -f1_score(val_labels, preds, average="macro")\n'
    '\n'
    'result = minimize(objective, x0=initial_weights, method="Powell")'
)
body(
    'The optimizer immediately zeroed out M2/M4/M5/M6/M7/M8 — finding that these '
    'models\' signal is fully subsumed by M1, M3, M12, and M13. The active weights '
    'in the final regularized 7-param run are: M1=0.3832, M3=0.1393, M10=0.1335, '
    'M12=0.1364, M13=0.1410, M11=0.0382, M9=0.0284.'
)
body(
    'Regularization: after discovering that the optimizer drives M2–M8 to zero, we '
    'fixed those weights at zero and ran the optimizer on only 7 free parameters '
    '(M1/M3/M9/M10/M11/M12/M13). Fewer parameters means less optimizer overfitting '
    'to the validation split. Local F1: 0.9295 (same as 13-param), LB: 0.9608 '
    '(+0.0001 over 13-param run). The regularization helped generalization.'
)

h2('4.7  Score-Level Ensembling (Soft Probabilities)')
body(
    'The ensemble combines models at the score level, not the label level. '
    'Each model returns per-class similarity sums (shape n_test × 4) '
    'before argmax. L1-normalize each row to turn the sums into a probability '
    'distribution, weight-sum across models by their optimizer weights, then '
    'argmax the combined probabilities. Confident models now contribute more to the '
    'decision than unsure ones, on a per-query basis.'
)
code(
    'probs_m = l1_normalize(per_class_similarity_sums_m)  # shape (n_test, 4) — one per model\n'
    'combined = sum(w_m * probs_m for m in models)        # weighted soft vote\n'
    'predictions = CLASS_LABELS[argmax(combined, axis=1)]'
)
body(
    'With sim_power=2.0, similarity scores are raised to the second power before '
    'accumulation — amplifying high-confidence neighbors and suppressing noise. '
    'This is applied per-model before L1 normalization, so each model\'s probability '
    'distribution already reflects the sharpened similarity landscape.'
)

h2('4.8  Temperature Sharpening Sweep')
body(
    'A natural extension of score-level ensembling: raise each model\'s probability '
    'distribution to a power T before combining. T=1 preserves the distribution; '
    'T > 1 sharpens it; T < 1 flattens it.'
)
body('We swept T ∈ {1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0} on the validation set:')
bullet('At 10 models (before M11): T=1.0 and T=2.0 both hit 0.9240 exactly. Higher T hurt.')
bullet('At 11 models (after M11): T=1.5 became marginally best at 0.9245.')
bullet('At 13 models (after M12/M13 + Powell weights): T=1.0 is best. Sharpening hurts.')
body(
    'Interpretation: with Powell-optimized weights (not F1 weights), the weight calibration '
    'is already sharp and accurate. The optimizer has found the right balance. '
    'Additional temperature sharpening just trades diversity for confidence without '
    'net benefit. Final config: T=1.0.'
)

# ══════════════════════════════════════════════════════════════════════════════
h1('PART 5 — GPU ACCELERATION: THE FULL STORY')
# ══════════════════════════════════════════════════════════════════════════════

h2('5.1  Why GPU?')
body(
    'The core operation in k-NN is: for each test document, compute its similarity to '
    'ALL 100,000+ training documents. With 25,520 test documents and 102,080 training '
    'documents, that is 2.6 billion similarity computations per model, per k value. '
    'We sweep many values of k per model, across 13 models — the total is enormous. '
    'On CPU this would take many hours. On GPU it takes minutes.'
)
body(
    'Because our vectors are L2-normalized, cosine similarity = dot product. '
    'The full similarity matrix for a batch is: train_matrix @ batch.T. '
    'This is a matrix multiplication — exactly what GPUs are designed to do at maximum speed.'
)

h2('5.2  How a GPU Works (Simplified)')
body(
    'A CPU has ~8–16 cores, each very fast and optimized for complex sequential logic. '
    'A GPU has thousands of smaller cores (the RTX 5080 Laptop has 10,752 CUDA cores) '
    'designed to do the same simple operation on thousands of numbers simultaneously — '
    'perfect for matrix math.'
)
body(
    'VRAM (Video RAM) is the GPU\'s own memory — separate from CPU RAM. Data must be '
    'explicitly transferred from CPU RAM to VRAM before the GPU can operate on it. '
    'This transfer happens over the PCIe bus (~16 GB/s bandwidth). '
    'If your program constantly transfers large amounts of data per batch, '
    'that transfer time dominates — the GPU sits idle waiting for data.'
)
body(
    'The RTX 5080 Laptop GPU has 16 GB of VRAM. Our training matrix (sparse) is only '
    'about 120 MB on the GPU — tiny. The bottleneck was never the computation itself.'
)

h2('5.3  The Original Code — Where the Bottleneck Was')
body('The original knn_predict loop looked like this:')
code(
    'for start in range(0, n_test, batch_size):        # loop over batches\n'
    '    chunk = test_mat[start:end].toarray()          # [CPU] sparse → dense: zero-fill 1.2 GB\n'
    '    batch = torch.tensor(chunk, device=DEVICE)     # [CPU→GPU] transfer 1.2 GB over PCIe\n'
    '    sims  = torch.sparse.mm(train_gpu, batch.T).T  # [GPU] matrix multiply — fast\n'
    '    ...                                            # [GPU→CPU] transfer tiny top-k results'
)
body('The timeline per batch looked like this:')
code(
    '|████████████████████████████████████░░░░░░░░░░|  CPU (toarray + PCIe transfer)\n'
    '|░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████░░░░░░|  GPU (matmul)\n'
    '\nGPU active = ~20% of total time. 80% is the CPU+PCIe bottleneck.'
)
body(
    '.toarray() on a 2000×150,000 sparse matrix must allocate and zero-fill a '
    '2000×150,000 float32 array = 1.2 GB, even though a news article only touches '
    '~200–500 of those 150,000 columns. Then it ships all 1.2 GB over PCIe. '
    'By the time the transfer finishes, the GPU has been idle for the entire duration.'
)

h2('5.4  How We Diagnosed It')
body(
    'We observed the GPU utilization percentage in Task Manager while the code '
    'was running. It showed ~20% utilization even with batch_size=2000. '
    'The pattern was clear: utilization spiked briefly then dropped to near-zero repeatedly, '
    'in sync with the batch loop. That "sawtooth" pattern is the signature of a '
    'CPU-bound transfer bottleneck — the GPU is starved for data.'
)
body(
    'We also noticed that when running a denser model (char n-grams, ~7x more non-zeros '
    'per document), the CPU utilization dropped and GPU utilization rose — confirming '
    'the bottleneck was the CPU zero-filling work, not the GPU matmul.'
)

h2('5.5  First Attempt: Pre-Loading Test Data as float16')
body(
    'Our first fix attempt: load the ENTIRE test matrix onto GPU before the loop, '
    'storing in float16 to halve VRAM usage.'
)
code(
    '# Attempt 1: pre-load entire test set as float16\n'
    'PREFETCH = 5000\n'
    'parts = []\n'
    'for i in range(0, n_test, PREFETCH):\n'
    '    chunk = test_mat[i:i+PREFETCH].toarray()                          # CPU sparse → dense\n'
    '    parts.append(torch.tensor(chunk, dtype=torch.float16, device=DEVICE))  # → GPU float16\n'
    'test_gpu = torch.cat(parts, dim=0)  # full test matrix lives on GPU\n'
    '\n'
    'for start in range(0, n_test, batch_size):\n'
    '    batch = test_gpu[start:end].to(torch.float32)   # GPU slice + upcast (nanoseconds)\n'
    '    sims  = torch.sparse.mm(train_gpu, batch.T).T   # GPU matmul'
)
body(
    'Theory: A 25,520×150,000 test matrix in float32 = 15.3 GB (exceeds 16 GB VRAM). '
    'In float16 it is 7.7 GB — fits. The float16→float32 upcast per batch is a pure '
    'GPU operation taking microseconds. One-time transfer instead of per-batch transfer.'
)
body(
    'Reality: On Windows, .toarray() on chunks of 5,000 rows required ~3 GB CPU RAM '
    'allocations per chunk. The chunked CPU allocation was actually slower than the '
    'original per-batch approach — we were waiting longer, not shorter. '
    'Also, the code was also accidentally re-uploading the train matrix inside every '
    'knn_predict call (26 times during a k sweep), compounding the slowdown. Reverted.'
)

h2('5.6  Root Cause #2: Train Matrix Re-Uploaded 26x Per Model')
body(
    'While investigating the slowdown, we found a second bottleneck: the training '
    'matrix was being converted and uploaded to the GPU inside every knn_predict call. '
    'During a k sweep with 22 k-values, this meant 22 separate train uploads per model. '
    'The fix: upload once in best_k_search and pass it as a parameter.'
)
code(
    'def best_k_search(val_mat, fit_mat, fit_labels, val_labels, k_list):\n'
    '    train_gpu = scipy_sparse_to_torch(fit_mat, DEVICE)  # upload ONCE\n'
    '    for k in k_list:\n'
    '        preds = knn_predict(..., _train_gpu=train_gpu)   # reuse across all k values\n'
    '    del train_gpu\n'
    '    return best_k, best_f1'
)

h2('5.7  The Real Fix: Sparse Non-Zero Transfer')
body(
    'The actual solution to the toarray() bottleneck: transfer only the non-zero values '
    'and reconstruct the dense matrix on the GPU itself, using its 1.7 TB/s memory bandwidth '
    'to fill the zeros — instead of the CPU doing it at ~50 GB/s and shipping 1.2 GB over PCIe.'
)
code(
    'def sparse_batch_to_gpu_dense(batch_csr, device):\n'
    '    n_rows, n_cols = batch_csr.shape\n'
    '    data    = torch.tensor(batch_csr.data,    dtype=torch.float32, device=device)  # non-zeros only\n'
    '    indices = torch.tensor(batch_csr.indices, dtype=torch.int64,   device=device)  # column indices\n'
    '    counts  = torch.tensor(np.diff(batch_csr.indptr), dtype=torch.int64, device=device)\n'
    '    row_ids = torch.repeat_interleave(\n'
    '        torch.arange(n_rows, dtype=torch.int64, device=device), counts)\n'
    '    dense = torch.zeros(n_rows, n_cols, dtype=torch.float32, device=device)  # GPU zero-fill\n'
    '    dense[row_ids, indices] = data  # scatter non-zeros into place\n'
    '    return dense'
)
body('The performance difference per batch:')
code(
    'Old way:  zero-fill 1.2 GB on CPU at ~50 GB/s   ≈ 24 ms\n'
    '          + transfer 1.2 GB over PCIe            ≈ 19 ms  → ~43 ms total\n'
    '\n'
    'New way:  transfer ~1.2 MB of non-zeros          ≈ 0.02 ms\n'
    '          + torch.zeros on GPU at ~1.7 TB/s      ≈ 0.7 ms  → ~0.7 ms total\n'
    '\n'
    '~40x less time per batch. 1000x less PCIe traffic. Same numerical results.'
)

h2('5.8  CuPy SpGEMM — Attempted, Blocked by CUDA Toolkit Version')
body(
    'The next logical step was CuPy SpGEMM: sparse×sparse GPU matrix multiplication '
    'via cuSPARSE. This would eliminate the dense matrix creation entirely — both train '
    'and test batch stay sparse on the GPU, and only the result (similarity scores) '
    'is materialized as dense for the topk operation.'
)
code(
    '# CuPy SpGEMM approach\n'
    'train_gpu = cupy_csr(...)          # sparse on GPU\n'
    'batch_gpu = cupy_csr(...)          # sparse on GPU — only non-zeros transferred\n'
    'sims_cp   = (train_gpu @ batch_gpu.T).toarray()  # sparse×sparse → dense result only\n'
    'sims      = torch.as_tensor(sims_cp, device=DEVICE).T  # zero-copy CuPy→PyTorch'
)
body(
    'CuPy confirmed working on RTX 5080 + CUDA 13 driver. However, when the transpose '
    '(.T) operation on a CuPy sparse matrix triggered an NVRTC kernel compilation to '
    'check canonical format, it failed with a compilation error. Root cause: CuPy\'s '
    'bundled libcudacxx headers reference __nv_fp8_e8m0 (an FP8 type added in CUDA 12.8+), '
    'but the installed CUDA toolkit on disk was v12.6 — incompatible. '
    'The CUDA driver version (13) sets a ceiling; it does not mean the toolkit is 13.'
)
body(
    'A workaround exists (setting _has_canonical_format = True on the CuPy CSR to skip '
    'the NVRTC check), but given that the PyTorch sparse×dense approach with '
    'sparse_batch_to_gpu_dense() already achieves ~1000x PCIe traffic reduction, '
    'the incremental gain from SpGEMM was not worth the added complexity. '
    'The PyTorch path remains the final implementation.'
)

h2('5.9  Cache-Based Validation — ~30x Speedup for k and Weight Sweeps')
body(
    'A key optimization added later in the project: instead of running a GPU k-NN pass '
    'for every k value in K_LIST, we run a SINGLE GPU pass that retrieves the top-K_MAX '
    'neighbors for each query and caches their labels and similarities. All k-sweep '
    'evaluations then run in pure numpy by slicing the cached top-K_MAX results:'
)
code(
    'def knn_cache_topk(train_mat, test_mat, train_labels, K_MAX):\n'
    '    # ONE GPU pass: retrieve top-K_MAX neighbors for all test docs\n'
    '    # Returns: topk_sims (n_test × K_MAX), topk_labels (n_test × K_MAX)\n'
    '\n'
    'def topk_to_scores(topk_sims, topk_labels, k, sim_power, class_labels):\n'
    '    # Pure numpy: slice top-k from cache, compute per-class score sums\n'
    '    # No GPU, no data transfer — just array indexing\n'
    '\n'
    'def best_k_search_cached(topk_sims, topk_labels, val_labels, k_list, sim_power):\n'
    '    # Sweep all k values and sim_power values using only the cache\n'
    '    # ~30x faster than running a GPU pass per k value'
)
body(
    'This optimization is critical for two reasons. First, it makes the sim_power sweep '
    '(K_MAX=101 × multiple sim_power values) computationally free after the initial GPU '
    'pass. Second, it makes the Powell optimizer fast enough to run: each optimizer '
    'evaluation is a numpy sum over cached score matrices, taking microseconds. '
    'The optimizer can evaluate thousands of weight combinations in seconds, thoroughly '
    'exploring the 7-parameter weight space.'
)
body(
    'The cache-based approach also enabled K_MAX=101 without cost — instead of running '
    '22 GPU passes (one per k value in K_LIST), we run 1 GPU pass with K_MAX=101 and '
    'sweep all k values in numpy. The k=51, 71, 101 values added to K_LIST were '
    'previously skipped because they were too slow; with caching they are free.'
)

# ══════════════════════════════════════════════════════════════════════════════
h1('PART 6 — THE FULL PROGRESSION: HOW WE GOT HERE')
# ══════════════════════════════════════════════════════════════════════════════

body('Every score listed is macro F1. Local = 80/20 validation split. LB = leaderboard.')

steps = [
    ('Step 1: First Submission — Baseline',
     '3 models (M1–M3), k=11, simple BM25 params',
     'Local: ~0.917 | LB: 0.4036',
     'Started with BM25+trigrams (M1), TF-IDF+bigrams (M2), BM25+unigrams (M3). '
     'k=11 fixed, basic weighted voting. Leaderboard score was 0.4036 — shockingly low. '
     'Cause: test.dat had no labels so the output format was wrong. Fixed and resubmitted.'),
    ('Step 2: BM25 Parameter Tuning',
     'Grid search over k1 ∈ {1.2, 1.5, 2.0} × b ∈ {0.5, 0.75, 1.0} on 12k sample',
     'LB: 0.9490',
     'Found k1=1.2, b=0.5 optimal. Also expanded vocabulary, extended k sweep to '
     'range(21,42). Jump from 0.4036 to 0.9490 was mostly the format fix + BM25 tuning.'),
    ('Step 3: k=11 Bug',
     'K_LIST accidentally set to only range(21,42)',
     'Local: 0.9186 (dropped)',
     'When extending the k range we dropped k=11 which had been the best for M1. '
     'Score dropped. Fixed by explicitly prepending 11: K_LIST = [11] + list(range(21,42)).'),
    ('Step 4: Expand to 7 Models',
     'Added M4 (BM25+bigrams Porter), M5 (lemma+BM25), M6 (raw words+BM25), M7 (lemma+TF-IDF)',
     'Local: 0.9201 | LB: 0.9570',
     'Each model brings a different vocabulary and/or scoring. M6 (raw words) was particularly '
     'valuable — it keeps word forms like "stocks" and "markets" that stemming conflates. '
     'Leaderboard reached 0.9570.'),
    ('Step 5: Char N-Gram M8 Added (Old Version)',
     'Character 3–4-grams with # boundary markers, BM25, chi2(80k)',
     'Local: 0.9212 | LB: 0.9567 (dropped)',
     'Best local score yet, but leaderboard dropped by 0.0003. The char n-gram features '
     'were overfit to the validation split. Removed permanently. '
     '(A refined version — M12 — later became one of the most important models.)'),
    ('Step 6: Revert to 7 Models, Test Without M3',
     'Removed char n-gram M8; tried dropping M3',
     'Local: 0.9204 (without M3) | LB: 0.9574 → 0.9573',
     'After reverting M8, leaderboard jumped to 0.9574 — best score so far. '
     'Then tried removing M3 (weakest model). Local improved slightly but leaderboard '
     'dropped to 0.9573. M3 was restored. Confirmed: M3 is necessary.'),
    ('Step 7: GPU Bottleneck Fixed',
     'sparse_batch_to_gpu_dense + pre-load train once per k sweep',
     'Same F1, ~40x faster per batch',
     'Diagnosed 20% GPU utilization from per-batch toarray() + PCIe transfer. '
     'Fixed by transferring only non-zeros (~1.2 MB vs 1.2 GB per batch) and filling '
     'zeros on the GPU. Also moved train upload outside the 22-iteration k loop.'),
    ('Step 8: Tried Extended k Sweep',
     'Added k=3,5,7,9 to K_LIST',
     'Local: marginal gain | LB: overfit risk',
     'Some models found k=3 or k=5 locally optimal but the extended sweep was tuning '
     'too tightly to the validation split. Reverted to K_LIST = [11] + range(21,42).'),
    ('Step 9: CuPy SpGEMM Attempt',
     'Sparse×sparse GPU matmul via cuSPARSE',
     'Blocked by CUDA toolkit 12.6 vs CuPy libcudacxx requiring 12.8+',
     'Reverted to PyTorch sparse×dense approach which already achieves 1000x PCIe reduction.'),
    ('Step 10: Expand M1 Features',
     'M1 vocab 200k→300k, chi2 150k→200k',
     'Local: 0.9203 (M1: 0.9186→0.9216) | LB: 0.9574 (unchanged)',
     'M1 individually improved significantly but the ensemble gain was marginal — the '
     'extra trigrams overlap with features already covered by other models. '
     'Leaderboard stayed at 0.9574, but the stronger M1 is kept for diversity.'),
    ('Step 11: Add M8 (TF-IDF + Trigrams)',
     'New M8: TF-IDF scoring on same trigram token set as M1, vocab 300k, chi2(200k)',
     'Local: 0.9211 (M8 individual: 0.9209) | LB: pending',
     'M8 is the TF-IDF counterpart to M1\'s BM25. Same features, different term saturation '
     'behavior. BM25 and TF-IDF disagree on documents where one class term appears '
     'many times — and those are exactly the borderline cases the ensemble needs to resolve. '
     'Ensemble improved from 0.9203 → 0.9211 — biggest single gain since M5.'),
    ('Step 12: Tried Squared F1 Weights',
     'w = best_f1**2 instead of best_f1',
     'Local: 0.9211 (identical)',
     'With all models scoring 0.913–0.922, squaring the weights barely shifts their '
     'relative influence. No votes changed. Reverted to linear weights.'),
    ('Step 13: Added M9 — LSI on TF-IDF',
     'TruncatedSVD(n_components=300, n_iter=5) on train_m8_full, dense cosine k-NN',
     'Local: 0.9219 (M9 solo: 0.8813) | LB: pending',
     'Introduced latent-semantic space as a fundamentally different view of the '
     'documents. M9 solo is lower than any sparse model because 300 latent dims cannot '
     'match 200k explicit n-grams for this classification task, but it disagrees with '
     'M1–M8 in useful ways. New dense-matrix helpers (knn_predict_dense, '
     'best_k_search_dense) added. K_LIST_DENSE = [11,21,31,41,51,71,101] — LSI favors '
     'larger k because dense similarity is smoother.'),
    ('Step 14: Added M10 — LSI on BM25',
     'Second TruncatedSVD(300) on train_m1_full (BM25 + trigrams)',
     'Local: 0.9220 (M10 solo: 0.8860) | LB: pending',
     'Tried a second LSI view for more diversity. Gain was marginal (+0.0001) because '
     'the two SVD-300 models are correlated — they both compress similar underlying '
     'n-gram structure. Kept M10 for the small lift, but noted that further LSI '
     'variants would be redundant.'),
    ('Step 15: Score-Level Ensembling',
     'Replaced hard-vote with L1-normalized per-class probability sums',
     'Local: 0.9240 | LB: pending',
     'Biggest gain of the second phase (+0.0020). Hard voting collapsed each model\'s '
     'confidence into a single label; soft voting preserves the full probability '
     'distribution. Added knn_predict_scores and knn_predict_dense_scores that return '
     'per-class similarity sums instead of hard labels. Ensemble now weight-sums soft '
     'probabilities across all 10 models.'),
    ('Step 16: Temperature Sharpening Sweep',
     'Swept T ∈ {1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0}',
     'Local: 0.9240 (best T=1 or 2 — no change)',
     'Tested whether sharpening per-model probabilities before summing would help. '
     'Null result: T=1 and T=2 tied at 0.9240. Higher T hurt (T=10 dropped to 0.9223). '
     'Interpretation: F1 weights are already doing the right job; sharpening just '
     'trades diversity for confidence without net benefit at this model mix.'),
    ('Step 17: Added M11 — PRF (Aggressive Tuning)',
     'alpha=0.7, beta=0.3, N=20 on M1 (BM25 + trigrams)',
     'Local: 0.9236 (M11 solo: 0.9182) | LB: pending',
     'First PRF attempt hurt the ensemble. Expanding queries toward the weighted '
     'centroid of their 20 nearest neighbors pulled borderline queries into '
     'wrong-class territory. M11 solo fell well below M1 (0.9216). Retuned rather '
     'than removed — the mechanism is orthogonal to LSI and could still help with '
     'the right drift magnitude.'),
    ('Step 18: PRF — Conservative Re-Tuning',
     'alpha=0.9, beta=0.1, N=10',
     'Local: 0.9245 (M11 solo: 0.9202) | LB: 0.9587',
     'Conservative tuning: original query dominates (90%), only 10 closest neighbors '
     'contribute a small 10% nudge. M11 solo rose to 0.9202. Ensemble climbed to 0.9245. '
     'T=1.5 became the optimal temperature with 11 models. '
     'Leaderboard submission: 0.9587 — best so far at this point. '
     'Confirms M9/M10/M11 and score-level ensembling transfer to held-out test.'),
    ('Step 19: Added M12 — Char 3-5-gram TF-IDF + Chi²',
     'Character 3–5-gram TF-IDF with boundary markers, chi2 feature selection',
     'Local: improved | LB: contribution to 0.9607',
     'Refined version of the old char n-gram M8 (which was BM25 + 3-4-grams and hurt LB). '
     'M12 uses TF-IDF instead of BM25, extends to 5-grams, and applies chi2 properly. '
     'Standalone F1 ~0.9111 — below word models, but highly complementary. '
     'The optimizer later assigned it weight 0.1364, the second highest after M1.'),
    ('Step 20: Added M13 — LSI (SVD-300) on Char N-gram Matrix',
     'TruncatedSVD(300) applied to the M12 character n-gram matrix',
     'Local: improved | LB: contribution to 0.9607',
     'Applied LSI to the character feature space, creating latent morphological topics. '
     'Standalone F1 ~0.8902 — similar to M9/M10. But the optimizer assigned it the '
     'highest weight of all models except M1 (0.1410), because char-space LSI is more '
     'decorrelated from word-level models than word-level LSI. '
     'Key insight: ensemble weight >> standalone F1 for complementary models.'),
    ('Step 21: Powell Optimizer for Ensemble Weights',
     'scipy.optimize.minimize(method="Powell") on 13-param weight vector',
     'Local: 0.9292 | LB: 0.9607',
     'Replaced F1-based weights with Powell-optimized weights. The optimizer '
     'discovered that M2/M4/M5/M6/M7/M8 are redundant and drove their weights to zero. '
     'Active models: M1(0.3433), M9(0.1355), M12(0.1336), M3(0.1247), M13(0.1062), '
     'M11(0.0490), M10(0.0463). Local F1: 0.9292. LB: 0.9607 — new best.'),
    ('Step 22: Failed — M15/M16/M17 (SVD-600 Upgrades)',
     'SVD-600 variants of M10/M9/M13 added to ensemble',
     'Local: 0.9290 (dropped) | LB: 0.9596 (dropped)',
     'Attempted to improve LSI models by doubling SVD dimensions to 600. '
     'Both local and LB dropped. SVD-600 adds noise along with signal at this corpus size. '
     'Removed permanently. All LSI models stay at SVD-300. Key lesson: more LSI dimensions '
     'consistently hurt, and adding more models gives optimizer more parameters to overfit.'),
    ('Step 23: Failed — Centroid Models (C-M1/C-M8/C-M12)',
     'Per-class centroid k-NN added to ensemble',
     'Local: 0.9294 (marginal gain) | LB: 0.9599 (dropped)',
     'Built per-class centroid vectors and added them as additional ensemble models. '
     'Local improved by 0.0002 but LB dropped by 0.0008. Classic overfitting from '
     'adding more optimizer parameters. Removed permanently. Key lesson confirmed: '
     'more models consistently hurt LB even when local val improves.'),
    ('Step 24: K_MAX=101 + Cache-Based Validation',
     'Extended K_LIST to include k=51,71,101; knn_cache_topk for ONE GPU pass',
     'Same local F1 | ~30x faster sweeps',
     'Added k=51, 71, 101 to K_LIST — previously infeasible due to GPU runtime cost. '
     'With cache-based validation (single GPU pass → numpy sweeps), extending k range '
     'is free. Also enables the Powell optimizer to run thousands of evaluations quickly. '
     'No F1 change from k range alone, but enables the next step.'),
    ('Step 25: Regularized Optimizer (7 Free Params)',
     'Fixed M2/M4/M5/M6/M7/M8 weights to zero; optimized only M1/M3/M9/M10/M11/M12/M13',
     'Local: 0.9295 | LB: 0.9608 — CURRENT BEST',
     'Since the 13-param optimizer always zeroed M2/M4/M5/M6/M7/M8, we fixed them to zero '
     'and ran the optimizer on only 7 free parameters. Fewer parameters = less overfitting. '
     'Local F1 stayed at 0.9295. LB improved +0.0001 to 0.9608. '
     'New weights: M1=0.3832, M3=0.1393, M10=0.1335, M12=0.1364, M13=0.1410, M11=0.0382, M9=0.0284. '
     'Note: M10 jumped significantly (0.0463→0.1335) and M9 dropped (0.1355→0.0284).'),
    ('Step 26: Failed — BM25 Grid Search on 12k Subset',
     'Expanded k1/b grid search on 12,000-sample subset to find better BM25 params',
     'Local: 0.9280 (dropped) | LB: not submitted',
     'Attempted to fine-tune k1/b beyond the initial k1=1.2, b=0.5 setting. '
     'The subset search found params that hurt full validation F1 (0.9295→0.9280) '
     'and caused M12 to collapse to weight=0 in the optimizer. '
     'Root cause: 12k-sample search is too noisy — subset F1 landscape does not match '
     'full-data landscape. Hardcoded k1=1.2, b=0.5 permanently. Do not run grid searches on subsets.'),
    ('Step 27: Failed — 5-Fold Cross-Validation Weights',
     '5-fold CV to average optimizer weights across splits',
     'Local: 0.9290 mean CV | LB: 0.9601 (dropped)',
     'Attempted to reduce optimizer overfitting by averaging weights across 5 CV folds. '
     'CV mean F1 was 0.9290 (below single-split 0.9295). LB dropped from 0.9608 to 0.9601. '
     'CV averaging makes weights more conservative and loses the sharp signal the '
     'single-split optimizer finds. Single-split 7-param Powell on 80/20 is the right approach. '
     'Reverted all CV changes including SVD-500 M9/M10 that were tested in this run.'),
    ('Step 28: Failed — M18 Asymmetric BM25',
     'Raw-TF query vector vs BM25 document vector (asymmetric scoring)',
     'Standalone: 0.9208 | Not added to ensemble',
     'Asymmetric BM25 uses raw term frequencies for query vectors but BM25 '
     'weights for document vectors. Theory: asymmetric scoring avoids double-applying '
     'the BM25 saturation to short queries. Practice: standalone F1=0.9208 — below M1\'s 0.9216. '
     'Shares the same chi2 feature space as M1, adding no new signal. '
     'Not worth the additional optimizer parameter cost.'),
    ('Step 29: Failed — M19 3vs4 Discriminative Chi2',
     'Chi2 vocabulary optimized for class 3 (Business) vs class 4 (Sci/Tech) discrimination',
     'Standalone: 0.3628 — complete failure',
     'The dominant confusion (43.5% of errors) is Business vs Sci/Tech. Attempted to build '
     'a specialized model by computing chi2 on a binary class-3-vs-4 subset. '
     'Complete failure: when projecting all 4-class documents to the business/tech feature space, '
     'class 1/2 documents scatter randomly because they have incidental business/tech vocabulary '
     'that gets amplified by L2 normalization. Model predicted only class 3/4 for everything. '
     'Do NOT try pairwise-focused feature selection for a 4-class k-NN model.'),
]

for title_s, change, scores, explanation in steps:
    p = doc.add_paragraph()
    r = p.add_run(f'{title_s}')
    r.bold = True
    r.font.size = Pt(11)
    body(f'    Change: {change}')
    body(f'    Scores: {scores}')
    body(f'    {explanation}')
    doc.add_paragraph()

h2('Leaderboard Milestones')
leaderboard = [
    ('0.4036', 'First submission — output format error (later fixed).'),
    ('0.9490', 'After BM25 parameter tuning, extended k sweep, larger vocab.'),
    ('0.9570', '7-model ensemble (M1–M7).'),
    ('0.9567', 'After adding char n-gram M8 (old version) — went DOWN. Reverted.'),
    ('0.9574', '7-model ensemble with M3 restored, M1 at 200k features.'),
    ('0.9573', 'After removing M3 — went DOWN. M3 restored.'),
    ('0.9587', '11-model ensemble (M1–M11 incl. M9 LSI-TFIDF, M10 LSI-BM25, M11 PRF) '
               'with score-level soft voting and T=1.5 temperature. Local: 0.9245.'),
    ('0.9607', '13-model ensemble after adding M12 (char 3-5-grams) and M13 (LSI char n-grams). '
               'Powell-optimized 13-param weights. Local: 0.9292.'),
    ('0.9596', 'After adding M15/M16/M17 (SVD-600 upgrades) — went DOWN. Reverted.'),
    ('0.9599', 'After adding centroid models (C-M1/C-M8/C-M12) — went DOWN. Reverted.'),
    ('0.9601', 'After 5-fold CV weights — went DOWN. Reverted to single-split optimizer.'),
    ('0.9608', 'FINAL BEST — K_MAX=101 + regularized 7-param Powell optimizer '
               '(M2/M4/M5/M6/M7/M8 fixed to zero). Local: 0.9295. '
               'Confirmed: regularization improved LB generalization by +0.0001 over 13-param run.'),
]
for score, note in leaderboard:
    bullet(f'{score} — {note}')

# ══════════════════════════════════════════════════════════════════════════════
h1('PART 7 — CODE STRUCTURE OVERVIEW')
# ══════════════════════════════════════════════════════════════════════════════

cells = [
    ('Cell 1', 'Imports',
     'Loads libraries: numpy, scipy, torch, sklearn, nltk. Detects GPU.'),
    ('Cell 2', 'Load Data',
     'Reads train.dat (label + text per line) and test.dat (text only). '
     '102,080 training samples, 25,520 test samples, 4 classes.'),
    ('Cell 3', 'Preprocess (Porter + trigrams)',
     'Defines preprocess() using Porter stemmer + unigrams + bigrams + trigrams. '
     'Used by M1, M2, M3, M4, M8.'),
    ('Cell 4', 'Apply Preprocess',
     'Converts all train and test texts to token lists. Run once — slow step.'),
    ('Cell 5', 'Helper Functions',
     'Core reusable functions: renormalize, build_vocab_idf, build_bm25_matrix, '
     'build_tfidf_matrix, apply_chi2, scipy_sparse_to_torch, '
     'sparse_batch_to_gpu_dense (the key GPU optimization), '
     'knn_predict (GPU-accelerated with distance-weighted voting), best_k_search.'),
    ('Cell 6', 'BM25 Param Search',
     'Grid search over k1 ∈ {1.2,1.5,2.0} × b ∈ {0.5,0.75,1.0} on 12k-sample subset. '
     'Sets BM25_K1=1.2 and BM25_B=0.5. (These are hardcoded — do not re-run grid search.)'),
    ('Cell 7 / 9a', 'Build M1',
     'BM25 + trigrams + chi2(200k), vocab 300k. Shape: (102080, 200000).'),
    ('Cell 8 / 9b', 'Build M2',
     'TF-IDF + bigrams + chi2(120k). Shape: (102080, 120000).'),
    ('Cell 9 / 9c', 'Build M3',
     'BM25 + unigrams only + chi2(50k). Shape: (102080, ~35566).'),
    ('Cell 9d', 'Build M4',
     'BM25 + bigrams (Porter) + chi2(120k). Shape: (102080, 120000).'),
    ('Cell 9e', 'Build M5',
     'Lemma + BM25 + bigrams + chi2(120k). Defines preprocess_lemma() using WordNet.'),
    ('Cell 9f', 'Build M6',
     'Raw words + BM25 + bigrams + chi2(120k). Defines preprocess_raw() — no stemming.'),
    ('Cell 9g', 'Build M7',
     'Lemma + TF-IDF + bigrams + chi2(120k). Reuses M5 token lists.'),
    ('Cell 9h', 'Build M8',
     'TF-IDF + trigrams + chi2(200k), vocab 300k. Reuses train_tokens from Cell 4. '
     'Shape: (102080, 200000). The TF-IDF counterpart to M1.'),
    ('Cell 9i', 'Build M9 (LSI TF-IDF)',
     'TruncatedSVD(n_components=300, n_iter=5) on train_m8_full. Produces dense '
     '(102080, 300) matrix. L2-normalized for cosine via dot product.'),
    ('Cell 9j', 'Build M10 (LSI BM25)',
     'Same as 9i but operating on train_m1_full (BM25 + trigrams) with a different '
     'random seed. Shape: (102080, 300).'),
    ('Cell 9k', 'Build M11 (PRF on M1)',
     'Defines build_prf_queries. For each query, runs 1st-pass k-NN, takes top-N '
     'neighbors weighted by similarity, computes sparse centroid, '
     'then builds expanded_query = 0.9*original + 0.1*centroid (N=10).'),
    ('Cell 9l', 'Build M12 (Char 3-5-gram TF-IDF)',
     'Extracts character n-grams (length 3,4,5) with boundary markers from stemmed tokens. '
     'Builds TF-IDF weighted sparse matrix, applies chi2 feature selection. '
     'Shape: (102080, ~150000).'),
    ('Cell 9m', 'Build M13 (LSI on Char N-grams)',
     'TruncatedSVD(n_components=300) on the M12 character n-gram matrix. '
     'Produces dense (102080, 300) latent morphological topic matrix. L2-normalized.'),
    ('Score Helpers (cell 3461e281)', 'Core scoring and caching functions',
     'knn_predict_scores, knn_predict_dense_scores (return per-class score matrices), '
     'knn_cache_topk, knn_cache_topk_dense (ONE GPU pass → cached top-K_MAX), '
     'topk_to_scores, best_k_search_cached (pure numpy k-sweeps on cache), '
     'l1_normalize_rows, build_centroid_matrix.'),
    ('Validation (cell 71cdabb0)', 'Cache all models, sweep k and sim_power',
     'Caches top-K_MAX neighbors for each of the 13 models (ONE GPU pass per model). '
     'Then sweeps k_list and sim_power in pure numpy via best_k_search_cached. '
     'Stores score_list_v (16 entries: 13 k-NN + 3 centroid; optimizer uses first 13). '
     '~30x faster than per-k GPU approach.'),
    ('Optimizer (cell-11)', 'Powell optimizer for ensemble weights',
     'Uses score_list_v[:13] and scipy.optimize.minimize(method="Powell") to find '
     'optimal_weights_13. 7 free parameters (M1/M3/M9/M10/M11/M12/M13); '
     'M2/M4/M5/M6/M7/M8 fixed to zero. Also prints confusion matrix + per-class error rates.'),
    ('Visualization (cell-viz)', '2×2 matplotlib figure',
     'Standalone F1 bar chart, ensemble weights bar chart, '
     'confusion matrix heatmap, per-class error breakdown. Uses %matplotlib inline.'),
    ('Prediction (cell-12)', 'Final test predictions',
     'Uses optimal_weights_13, 13 models, best k per model from validation. '
     'Builds test_m11_prf via build_prf_queries before M11 inference. '
     'Writes predictions.dat.'),
]

for cell, name, desc in cells:
    p = doc.add_paragraph()
    r = p.add_run(f'{cell} — {name}:  ')
    r.bold = True
    p.add_run(desc)
    p.paragraph_format.space_after = Pt(4)

# ══════════════════════════════════════════════════════════════════════════════
h1('PART 8 — KEY TAKEAWAYS')
# ══════════════════════════════════════════════════════════════════════════════

bullet('k-NN accuracy depends almost entirely on the quality of the feature representation — the algorithm itself is simple, the features do all the work.')
bullet('BM25 outperforms raw TF-IDF individually, but TF-IDF adds ensemble diversity because the two functions saturate high-frequency terms differently and disagree on borderline documents.')
bullet('Chi-squared feature selection removes noise and reduces memory — both improve accuracy. k-NN is especially sensitive to irrelevant dimensions because they dilute cosine similarity.')
bullet('Ensembles work because each model makes different errors. Even a weak model (M3, F1=0.913) can help the ensemble if it captures something the other models miss — we confirmed this empirically by observing a leaderboard drop when M3 was removed.')
bullet('Local validation F1 is not always a reliable proxy for leaderboard score. We had multiple cases where local improvements hurt the leaderboard (char n-grams v1, M15/M16/M17, centroid models) and a case where a local regression helped (keeping M3). With only 5 submissions per day, being conservative and testing carefully matters.')
bullet('GPU utilization is not automatic — the bottleneck is almost always data transfer, not computation. Transferring only non-zero values instead of dense arrays gave ~40x speedup per batch with 1000x less PCIe traffic.')
bullet('Diagnosing bottlenecks requires observation, not guessing. Watching GPU utilization in Task Manager revealed the sawtooth idle pattern immediately. The fix followed directly from understanding the cause.')
bullet('CUDA driver version ≠ CUDA toolkit version. The driver sets a ceiling for what is supported; the toolkit actually installed on disk determines what compilers and headers are available. A mismatch can silently cause library incompatibilities.')
bullet('Trial and error is a valid methodology — but keep good records of what you tried and what happened. Several of our improvements came from reverting failed experiments, which is only possible if you tracked what changed.')
bullet('A weak individual model can still help an ensemble. M9 (0.8813) and M10 (0.8876) both score lower than every sparse model, yet including them lifts the ensemble because their mistakes are different — they disagree with M1–M8 in useful places. Ensemble strength comes from decorrelated errors, not just individually strong models.')
bullet('Character n-grams can be more complementary than their standalone F1 suggests. M12 (char n-grams, F1=0.9111) and M13 (char-space LSI, F1=0.8902) received the 2nd and 1st highest optimizer weights respectively (after M1), because their character-level features are decorrelated from all word-level models. Ensemble weight is the true measure of a model\'s value, not standalone performance.')
bullet('Hard-vote ensembles discard per-query confidence; score-level ensembling preserves it. Switching from "each model votes for one class weighted by F1" to "each model contributes a probability distribution" gave the single biggest gain of the second phase (+0.002). Whenever possible, combine at the score level, not the label level.')
bullet('Powell-optimized weights outperform F1-based weights by finding complementarity, not just strength. The optimizer zeroed out M2/M4/M5/M6/M7/M8 — strong models whose signal is fully subsumed by the remaining 7. F1 weights would never make this discovery.')
bullet('Regularize the optimizer: fewer free parameters reduces overfitting to the validation split. Fixing the zeroed-out weights at zero and rerunning the optimizer on 7 parameters improved LB by +0.0001 with the same local F1. The gain is small but the principle is important — match optimizer complexity to the effective dimensionality of the problem.')
bullet('Adding more models to the ensemble consistently hurt LB even when local val improved. Each additional model adds an optimizer parameter, increasing the risk of overfitting the validation set. Never add a model without a strategy to prevent optimizer overfitting — and if unsure, don\'t add it.')
bullet('Cache-based validation transforms optimization feasibility. By caching top-K_MAX neighbors from a single GPU pass, the Powell optimizer can evaluate thousands of weight combinations in seconds (pure numpy), making thorough weight search practical without re-running GPU inference.')
bullet('BM25 parameter search on subsets is too noisy. A 12k-sample grid search produced parameters that hurt full validation F1 and caused a key model to collapse to weight=0. Use the full training set for parameter evaluation, or hardcode known-good values (k1=1.2, b=0.5).')
bullet('5-fold CV makes ensemble weights too conservative. CV averaging loses the sharp signal that single-split optimization finds, leading to lower LB performance despite lower variance. For this task, single-split 7-param Powell on 80/20 is the right approach.')
bullet('Temperature sharpening is not always helpful. For well-calibrated Powell-optimized weights, raising probabilities to a power just trades diversity for confidence without net benefit. The optimal temperature returned to T=1.0 after the weights were optimized.')
bullet('PRF requires conservative tuning on clean corpora. Aggressive query expansion (alpha=0.7, beta=0.3) worked well in classic IR papers on noisy query logs, but modern cleanly-written text classification needs timid drift (alpha=0.9, beta=0.1) to avoid pulling queries into wrong-class clusters.')
bullet('Pairwise feature selection fails for multi-class k-NN. Building a feature space optimized to discriminate two specific classes (Business vs Sci/Tech) causes documents from the other classes to scatter randomly in that space — a fundamental geometric problem, not a hyperparameter issue.')
bullet('The dominant remaining error (43.5% of all errors) is Business vs Sci/Tech confusion. This is a semantic overlap problem — both classes discuss companies, products, and markets. It is not fixable with bag-of-words features alone, and represents the ceiling for this approach on this dataset.')

# ── Save ──────────────────────────────────────────────────────────────────────
out = r'c:\Users\zp123_2zkvvkz\projects\csen-140-sp26\CSEN140_PR1_Writeup.docx'
doc.save(out)
print(f'Saved: {out}')
