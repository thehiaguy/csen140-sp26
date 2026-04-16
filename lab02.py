import string
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def process_data(fpath, idx=None, is_training=True):
    """
    This function processes the input data and returns the result.
    
    :param fpath: Input data to be processed
    :param idx: Optional dictionary of tokens
    :param is_training: Boolean flag indicating if the data is for training
    :return: if is_training=True, returns (csr_matrix, labels); otherwise returns csr_matrix.
             csr_matrix contains term-frequency values; labels is a list of int class labels.
    """
    if idx is None:
        idx = {}

    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    rows, cols, data_vals = [], [], []
    labels = []
    doc_idx = 0

    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_training:
                # First whitespace-delimited token is the class label
                parts = line.split(' ', 1)
                labels.append(int(parts[0]))
                text = parts[1] if len(parts) > 1 else ''
            else:
                text = line

            # Lowercase, strip punctuation, tokenize
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            tokens = text.split()

            # Remove stopwords / non-alpha tokens, then stem
            tokens = [stemmer.stem(t) for t in tokens if t.isalpha() and t not in stop_words]

            # Count term frequencies for this document
            token_counts = {}
            for token in tokens:
                if is_training and token not in idx:
                    idx[token] = len(idx)   # grow vocabulary only during training
                if token in idx:
                    token_counts[token] = token_counts.get(token, 0) + 1

            for token, count in token_counts.items():
                rows.append(doc_idx)
                cols.append(idx[token])
                data_vals.append(count)

            doc_idx += 1

    n_cols = len(idx)
    matrix = csr_matrix((data_vals, (rows, cols)), shape=(doc_idx, n_cols), dtype=int)

    return (matrix, labels) if is_training else matrix


# Example usage
idx = {}
train_data, train_labels = process_data('train.dat', idx=idx, is_training=True)
test_data = process_data('test.dat', idx=idx, is_training=False)
assert train_data.shape[1] == test_data.shape[1], "Train and test data do not have the same number of features"
print(f"Train: {train_data.shape}, Test: {test_data.shape}, Vocab: {len(idx)}")






def proximity(train_data, x):
    """
    Computes cosine similarity between query x and every row of train_data.
    Only rows with non-zero dot product are returned.

    All intermediate computations stay in sparse form; only scalar/1-D norm
    vectors (not the data matrices) are ever dense.

    :param train_data: csr_matrix of shape (n_train, n_features)
    :param x: csr_matrix of shape (1, n_features)
    :return: list of (row_index, cosine_similarity) pairs for non-zero matches
    """
    # Step 1: vectorized dot products — sparse matrix multiply
    # Result is sparse (n_train, 1); zero rows are simply absent.
    dots = train_data.dot(x.T)          # csr_matrix (n_train, 1)

    if dots.nnz == 0:
        return []

    # Step 2: locate non-zero entries
    dots_coo     = dots.tocoo()
    nonzero_rows = dots_coo.row                     # 1-D array of row indices
    dot_vals     = dots_coo.data.astype(float)      # matching dot-product values

    # Step 3: L2 norms using sparse element-wise ops (no dense conversion)
    # .multiply() is csr_matrix element-wise product — stays sparse.
    # .sum(axis=1) produces a small dense (k,1) result — not the data matrix.
    train_sub   = train_data[nonzero_rows]
    train_norms = np.asarray(
        train_sub.multiply(train_sub).sum(axis=1)
    ).flatten() ** 0.5

    x_norm = x.multiply(x).sum() ** 0.5
    if x_norm == 0:
        return []

    # Step 4: cosine similarity = dot / (||train_row|| * ||x||)
    cosine_sims = dot_vals / (train_norms * x_norm)

    return list(zip(nonzero_rows.tolist(), cosine_sims.tolist()))


# Example usage
x = test_data[0]
proximities = proximity(train_data, x)
proximities_sorted = sorted(proximities, key=lambda p: p[1], reverse=True)
print(f"Top 3 proximities out of {len(proximities)} using cosine similarity:", proximities_sorted[:3])
