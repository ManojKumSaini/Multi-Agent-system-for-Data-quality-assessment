# BERTopic Parameter Tuning Guide

---

### 1. Structural Subspace (UMAP)

---

**`UMAP_N_NEIGHBORS`** (Range: 10–50)
Controls the balance between local and global data structure in the manifold.

| Value | Effect |
|---|---|
| Low | Preserves local structure — fine-grained, specific groupings |
| High | Emphasizes global structure — broader, more general groupings |

---

**`UMAP_N_COMPONENTS`** (Range: 3–10)
The number of dimensions UMAP reduces embeddings to before clustering. Higher values preserve more semantic information but increase clustering complexity and memory usage.

---

**`UMAP_MIN_DIST`** (Range: 0.0–0.1)
Controls how tightly points are packed in the reduced space.

| Value | Effect |
|---|---|
| Near 0.0 | Maximum packing — tightest cluster density |
| Near 0.1 | More spread — physically separates correlated concepts |

---

**`UMAP_METRIC`** (Options: `cosine`, `euclidean`)
Distance metric used in the original embedding space.

- `cosine` — measures angular similarity; correct for sentence embeddings which are directional
- `euclidean` — measures absolute distance; less suitable for high-dimensional embedding spaces

---

### 2. Density Clustering (HDBSCAN)

---

**`HDBSCAN_MIN_CLUSTER_SIZE`** (Range: 50–500)
Minimum number of documents required to form a topic. The most impactful HDBSCAN parameter.

| Value | Effect |
|---|---|
| Low | Many small, granular topics — risk of micro-topic fragmentation |
| High | Few broad macro-themes — risk of collapsing distinct sub-topics |

---

**`HDBSCAN_MIN_SAMPLES`** (Range: 5–50)
Controls clustering conservatism. Higher values push more documents into the outlier class (-1). Typically set to 10–20% of `HDBSCAN_MIN_CLUSTER_SIZE`.

| Value | Effect |
|---|---|
| Low | Lenient boundaries — fewer outliers |
| High | Strict boundaries — more outliers, requires more aggressive outlier reduction |

---

**`HDBSCAN_METRIC`** (Options: `euclidean`, `cosine`)
Distance metric used in the UMAP-reduced space (not the original embedding space).

- `euclidean` — standard and appropriate for low-dimensional UMAP output
- `cosine` — less meaningful after dimensionality reduction

---

**`prediction_data`** (Options: `True`, `False`)
Must be set to `True` to enable `reduce_outliers()` in post-processing. Without this, outlier reduction will fail at runtime.

---

### 3. Vocabulary and Topic Representation (CountVectorizer)

---

**`VECTORIZER_MIN_DF`** (Range: 5–50)
Minimum document frequency for a word to appear in topic label representations. Filters the c-TF-IDF vocabulary only — does not affect sentence embeddings.

| Value | Effect |
|---|---|
| Low | Retains rare terms — risk of noise in topic labels |
| High | Only common terms survive — topic labels become generic |

---

**`VECTORIZER_NGRAM_RANGE`** (Options: `(1,1)`, `(1,2)`, `(1,3)`)
Controls whether topic keywords can be single words or multi-word phrases.

| Value | Effect |
|---|---|
| `(1,1)` | Unigrams only |
| `(1,2)` | Unigrams and bigrams — captures two-word phrases |
| `(1,3)` | Up to trigrams — captures longer phrases at higher compute cost |

---

### 4. Topic Merging (BERTopic)

---

**`NR_TOPICS`** (Options: `None`, `"auto"`, or integer)
Controls post-fit topic merging based on c-TF-IDF cosine similarity.

| Value | Effect |
|---|---|
| `None` | No merging — all raw HDBSCAN clusters preserved |
| `"auto"` | Automatically merges highly similar topics |
| Integer | Forces reduction to exactly N topics |

---

### 5. Embedding Model

---

**`EMBEDDING_MODEL`**
The sentence transformer model used to produce document embeddings before UMAP reduction. This is the most impactful single choice for embedding quality.

| Model | Speed | Domain |
|---|---|---|
| `all-MiniLM-L6-v2` | Fast | General English |
| `mukaj/fin-mpnet-base` | Medium | Financial text |

---

### 6. Outlier Reduction

---

**`OUTLIER_STRATEGY`** (Options: `"embeddings"`, `"c-tf-idf"`, `"probabilities"`)
Method used to reassign outlier documents (-1) back to real topics after fitting.

| Strategy | Method | Speed | Notes |
|---|---|---|---|
| `"embeddings"` | Nearest topic centroid in embedding space | Slow | Handles semantic paraphrasing |
| `"c-tf-idf"` | Word overlap with topic representations | Fast | Requires word overlap |
| `"probabilities"` | HDBSCAN soft probabilities | Medium | Requires `calculate_probabilities=True` |