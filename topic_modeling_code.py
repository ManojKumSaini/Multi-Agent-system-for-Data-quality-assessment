
# -------------------------------------------------------------------------------------------------------------------------------
# Stage 2: Topic Modelling (BERTopic Pipeline)                                                                                  
# -------------------------------------------------------------------------------------------------------------------------------
# End-to-end BERTopic pipeline for unsupervised topic discovery on financial news headlines.                                     
# Parameters are selected by the Doer agent (Stage 2 config.json) and loaded at runtime.                                        
# This script was written manually; the Doer selects parameters, not code (see Section 5.4.3).                                  
#                                                                                                                               
# Execution steps:                                                                                                              
#   Section 4  — Utilities (JSON writing, config loading from Doer output)                                                      
#   Section 5  — Load preprocessed documents from Stage 1 output                                                                
#   Section 6  — Compute/load sentence embeddings (mukaj/fin-mpnet-base, GPU)                                                   
#   Section 7  — Build BERTopic model (UMAP + HDBSCAN + MMR representation)                                                     
#   Section 8  — Fit model on embeddings                                                                                        
#   Section 9  — Outlier reduction (c-TF-IDF strategy) + topic merging (graph-based)                                            
#   Section 10 — Save outputs (documents_with_topics.csv, topic_info.csv, summary JSON)                                         
#   Section 11 — Generate word clouds for top-30 topics                                                                         
#   Section 12 — Main runner (orchestrates all steps sequentially)                                                              
#                                                                                                                               
# Input: outputs/preprocessing/NIFTY_preprocessed_final.csv (from Stage 1)                                                     
# Output: outputs/topic_modeling/ (model, embeddings, CSVs, word clouds, summary)                                               
#                                                                                                                               
# Config: Parameters loaded from outputs/topic_modelling/config.json (Doer-generated)                                           
#         Falls back to dataclass defaults if config.json absent                                                                 
# -------------------------------------------------------------------------------------------------------------------------------



# ── SECTION 1: CONFIG ────────────────────────────────────────
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import random
import numpy as np
from typing import List



@dataclass
class TopicModelConfig:
    """Configuration parameters for the BERTopic pipeline."""
    INPUT_PATH: Path = Path("outputs/preprocessing/NIFTY_preprocessed_final.csv")
    OUTPUT_DIR: Path = Path("outputs/topic_modelling")
    
    # Use field(init=False) so these are set dynamically after creation
    EMBEDDINGS_PATH: Path = field(init=False)
    MODEL_PATH: Path = field(init=False)

    EMBEDDING_MODEL: str = "mukaj/fin-mpnet-base"

    # UMAP hyper-parameters
    UMAP_N_NEIGHBORS: int = 15
    UMAP_N_COMPONENTS: int = 5
    UMAP_MIN_DIST: float = 0.05
    UMAP_METRIC: str = "cosine"

    # HDBSCAN hyper-parameters
    HDBSCAN_MIN_CLUSTER_SIZE: int = 120
    HDBSCAN_MIN_SAMPLES: int = 15
    HDBSCAN_METRIC: str = "euclidean"

    # Vectorizer / topic-reduction options
    VECTORIZER_MIN_DF: int = 15
    VECTORIZER_NGRAM_RANGE: Tuple[int, int] = (1, 2)
    TOP_N_WORDS: int = 10
    OUTLIER_REDUCTION_THRESHOLD: float = 0.1
    OUTLIER_STRATEGY: str = "c-tf-idf"

    # If None -> let custom Section 9 graph code decide the optimal number of topics
    NR_TOPICS: Optional[int] = None

    COHERENCE_METRIC: str = "c_v"
    INTER_TOPIC_SIM_THRESHOLD: float = 0.84

    ENABLE_WORDCLOUDS: bool = True

    def __post_init__(self):
        """Fixes NameError by building sub-paths after main properties exist."""
        self.EMBEDDINGS_PATH = self.OUTPUT_DIR / "embeddings.npy"
        self.MODEL_PATH = self.OUTPUT_DIR / "bertopic_model"


config = TopicModelConfig()


# ── SECTION 2: IMPORTS ───────────────────────────────────────
import json
import logging
import os
import time
import warnings

import networkx as nx          
import numpy as np
import pandas as pd

from bertopic import BERTopic
from graph.validator import build_topic_modeling_summary, compute_intertopic_similarity

# FIXED: Modern, cross-platform import that BERTopic supports natively
from sklearn.cluster import HDBSCAN

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP

warnings.filterwarnings("ignore", category=UserWarning)


# ── SECTION 3: LOGGING ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ── SECTION 4: UTILITIES ─────────────────────────────────────
def ensure_output_dir():
    """Safely creates the output folder using modern pathlib features."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def convert_numpy(obj):
    """Encodes custom NumPy array elements into plain Python types for JSON."""
    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Not serializable: {type(obj)}")


def write_json(path, data):
    """Writes a clean JSON file using a modern Path object interface."""
    from pathlib import Path
    target_path = Path(path)
    
    with target_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, default=convert_numpy, indent=2)


def load_runtime_config_from_doer():
    """Apply the Doer-selected BERTopic config from outputs/topic_modelling/config.json if present."""
    config_path = Path("outputs/topic_modelling/config.json")
    if not config_path.exists():
        return

    try:
        with config_path.open("r", encoding="utf-8") as f:
            overrides = json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read runtime config: {e}")
        return

    for key, value in overrides.items():
        if not hasattr(config, key):
            continue

        if key == "VECTORIZER_NGRAM_RANGE" and isinstance(value, str):
            try:
                value = tuple(int(part.strip()) for part in value.strip("()[]").split(","))
            except Exception:
                pass
        elif key in {"INPUT_PATH", "OUTPUT_DIR", "EMBEDDINGS_PATH", "MODEL_PATH"}:
            value = Path(value)

        setattr(config, key, value)

    config.__post_init__()
    logger.info(f"Loaded runtime config from {config_path}")


# ── SECTION 5: LOAD DATA ─────────────────────────────────────
def load_documents():
    """Loads text records safely and alerts you if file targets are missing."""
    if not config.INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessed input data: {config.INPUT_PATH}")

    df = pd.read_csv(config.INPUT_PATH)

    # Convert text column to clean strings to prevent downstream float/NaN issues
    documents = df["preprocessed_text"].astype(str).tolist()

    logger.info(f"Loaded source records dataset. Total rows: {len(df)}")

    return df, documents


# ── SECTION 6: EMBEDDINGS ────────────────────────────────────
def compute_embeddings(documents):
    """Generates mathematical vector maps from text documents using the GPU."""
    model = SentenceTransformer(
        config.EMBEDDING_MODEL,
        device="cuda"
    )

    # KEPT: Large batch_size optimized for high-VRAM university servers
    embeddings = model.encode(
        documents,
        show_progress_bar=True,
        batch_size=2048
    )

    # Cast to float32 to keep matrix sizes small and manageable
    embeddings = embeddings.astype(np.float32)

    # Save vectors to cache disk space using the path target directly
    np.save(config.EMBEDDINGS_PATH, embeddings)

    logger.info("New text embedding maps computed and cached successfully.")

    return embeddings


def load_or_create_embeddings(documents):
    """Loads pre-calculated vectors from disk, or builds them if out of sync."""
    if config.EMBEDDINGS_PATH.exists():
        try:
            embeddings = np.load(config.EMBEDDINGS_PATH)

            # Confirm row count matches current document count perfectly
            if embeddings.shape[0] == len(documents):
                logger.info("Found valid cached embeddings matrix. Loading directly.")
                return embeddings

            logger.warning("Cached vector layout mismatch. Recomputing matrix.")

        except Exception as e:
            logger.warning(f"Failed to read existing cache vectors: {e}")

    return compute_embeddings(documents)

# ── SECTION 7: BUILD MODELS ──────────────────────────────────
def build_vectorizer():
    """Builds a custom CountVectorizer with single-word stop word filters."""
    # FIXED: Split multi-word phrases into individual words so the vectorizer catches them
    custom_stopwords = list(set([
        *list(ENGLISH_STOP_WORDS),
        "markets", "research", "news", "update", "text",
        "plc", "form", "st", "say", "says", "said",
        "reuters", "report", "reported", "reports",
        "reg", "eptri", "llp",
        "forexdollar", "forexeuro", "forexyen",
        "forexpound", "stockstsx", "stockswall",
        "snapshotwall", "stxnews", "1australia",
        "1russia", "1china", "cas", "briefing", "wall",
        "83", "1japan", "1canada",
        "publication", "interim", "prospectus",
        "annual", "financial", "levi", "korsinsky",
        "zacks", "blog", "highlights", "industrial", "info",
        "morning", "ends", "flat", "premarket",
        "briefmoody", "digest", "brief", "inc", "co", "corp", "limited",
        "announces", "upcoming", "reminds",
    ]))

    return CountVectorizer(
        min_df=config.VECTORIZER_MIN_DF,
        max_df=1.0,
        ngram_range=config.VECTORIZER_NGRAM_RANGE,
        stop_words=custom_stopwords
    )


def build_topic_model():
    """Initializes the BERTopic pipeline with modern cluster configurations."""
    vectorizer_model = build_vectorizer()

    umap_model = UMAP(
        n_neighbors=config.UMAP_N_NEIGHBORS,
        n_components=config.UMAP_N_COMPONENTS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        low_memory=True,
        random_state=42
    )

    # FIXED: Dropped prediction_data=True to remain fully compatible with sklearn's HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=config.HDBSCAN_MIN_SAMPLES,
        metric=config.HDBSCAN_METRIC
    )

    representation_model = MaximalMarginalRelevance(diversity=0.4)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=config.NR_TOPICS,
        top_n_words=config.TOP_N_WORDS,
        calculate_probabilities=False,
        verbose=True
    )

    return topic_model, vectorizer_model


# ── SECTION 8: TRAIN MODEL ───────────────────────────────────
def train_model(topic_model, documents, embeddings):
    """Fits the BERTopic model on pre-computed text embeddings."""
    start = time.time()

    topics, _ = topic_model.fit_transform(documents, embeddings)
    topics_arr = np.array(topics)

    logger.info(f"Training completed successfully in {(time.time() - start):.2f}s")

    return topics_arr


# ── SECTION 9: OUTLIER REDUCTION & MERGING ─────────────────
def reduce_outliers(topic_model, documents, topics, vectorizer_model):
    """
    Reassigns unclustered outlier texts to their closest valid topic.
    Safeguards downstream matrix indexing in case all outliers are successfully reassigned.
    """
    topics_before = len(set(topics[topics != -1]))
    outliers_before = int((topics == -1).sum())

    logger.info(f"Topics before reduction: {topics_before}")
    logger.info(f"Outliers before reduction: {outliers_before}")

    # Generate the reassigned topic list based on c-TF-IDF text similarity
    new_topics = topic_model.reduce_outliers(
        documents,
        topics,
        strategy=config.OUTLIER_STRATEGY,
        threshold=config.OUTLIER_REDUCTION_THRESHOLD
    )

    # Force the core model to update its cluster word dictionaries
    topic_model.update_topics(
        documents,
        topics=new_topics,
        vectorizer_model=vectorizer_model
    )

    # FIX: If the outlier bucket drops to zero, BERTopic deletes the internal tracking attributes.
    # Forcing this flag to stay active ensures Section 10's matrix row mapping doesn't crash.
    if -1 not in set(new_topics):
        topic_model._outliers = 0  # Tells the internal structure to handle the empty state gracefully without dropping rows

    topics_arr = np.array(new_topics)

    topics_after = len(set(topics_arr[topics_arr != -1]))
    outliers_after = int((topics_arr == -1).sum())

    logger.info(f"Topics after reduction: {topics_after}")
    logger.info(f"Outliers after reduction: {outliers_after}")

    return {
        "topics": topics_arr,
        "topics_before": topics_before,
        "topics_after": topics_after,
        "outliers_before": outliers_before,
        "outliers_after": outliers_after,
        "outliers_recovered": outliers_before - outliers_after
    }

def merge_similar_topics(topic_model, documents, topics, vectorizer_model):
    """Collapses overlapping similar pairs all at once to keep IDs intact."""
    _, top_pairs = compute_intertopic_similarity(topic_model, config.INTER_TOPIC_SIM_THRESHOLD)

    if not top_pairs:
        logger.info("No topics above similarity threshold — skipping merge")
        return np.array(topics)

    G = nx.Graph()
    G.add_edges_from([(pair[0], pair[1]) for pair in top_pairs])
    merge_groups = [
        sorted(list(component))
        for component in nx.connected_components(G)
        if len(component) > 1
    ]

    logger.info(
        f"Merging {len(merge_groups)} topic group(s) "
        f"covering {sum(len(g) for g in merge_groups)} topics"
    )

    # FIXED: Run as a single batch operation so BERTopic adjusts internal IDs safely
    topic_model.merge_topics(documents, merge_groups)

    # Update representations after the batch merge
    topic_model.update_topics(
        documents,
        vectorizer_model=vectorizer_model
    )

    new_topics = np.array(topic_model.topics_)

    logger.info(f"Topics after merge: {len(set(t for t in new_topics if t != -1))}")

    return new_topics

# ── SECTION 10: OUTPUTS ──────────────────────────────────────
def save_outputs(df: pd.DataFrame, topic_model, topics: np.ndarray, summary: dict):
    """
    Saves generated datasets, reports, and execution telemetry to disk.
    Defensively guards missing dataframe columns and short word distributions.
    """
    df_output = df.copy()
    df_output["topic_id"] = topics

    active_tids = list(topic_model.get_topics().keys())
    
    label_map = {}
    for tid in active_tids:
        if tid == -1:
            label_map[-1] = "outlier"
            continue
            
        topic_words = topic_model.get_topic(tid)
        # Fallback to general topic ID format if the word representation is empty
        if not topic_words:
            label_map[tid] = f"topic_{tid}"
        else:
            label_map[tid] = "_".join([w for w, _ in topic_words[:3]])

    df_output["topic_label"] = df_output["topic_id"].map(label_map)

    # Defensively filter and include columns that actually exist in your dataset
    target_columns = ["id", "date", "news", "label", "pct_change", "preprocessed_text", "topic_id", "topic_label"]
    output_columns = [col for col in target_columns if col in df_output.columns]
    
    missing_columns = set(target_columns) - set(output_columns)
    if missing_columns:
        logger.warning(f"Skipping absent database source tracking fields: {missing_columns}")

    # Explicit Path object operators (/) used for directory joining
    docs_csv_path = config.OUTPUT_DIR / "documents_with_topics.csv"
    df_output[output_columns].to_csv(docs_csv_path, index=False)

    info_csv_path = config.OUTPUT_DIR / "topic_info.csv"
    topic_info = topic_model.get_topic_info()
    topic_info[topic_info["Topic"] != -1].to_csv(info_csv_path, index=False)

    # Use the global write_json utility that safely handles Path objects
    json_path = config.OUTPUT_DIR / "phase_2_summary.json"
    write_json(json_path, summary)

    logger.info(f" Primary pipeline asset sheets written to target: {config.OUTPUT_DIR.resolve()}")
# ── SECTION 11: WORDCLOUDS ──────────────────────────
def generate_wordclouds(topic_model, topic_counts: pd.Series):
    """
    Generates visual word clouds for your top-30 most frequent topics.
    Skips empty stop-word data groups and logs output directories.
    """
    if not config.ENABLE_WORDCLOUDS:
        return

    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")  # Prevent GUI window popups on headless servers
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("The 'wordcloud' library is absent — skipping canvas visualizations.")
        return

    wc_dir = config.OUTPUT_DIR / "wordclouds"
    wc_dir.mkdir(parents=True, exist_ok=True)

    # Optimization win: extract and match valid IDs once outside the loop
    available_ids = set(topic_model.get_topics().keys())
    top_30_ids = [tid for tid in topic_counts.head(30).index.tolist() if tid in available_ids]

    for tid in top_30_ids:
        word_scores = {w: round(float(s), 6) for w, s in topic_model.get_topic(tid)}

        # Guard check: prevent the wordcloud library from throwing errors on empty dicts
        if not word_scores:
            logger.warning(f"Topic {tid} is empty after word filtering. Skipping word cloud.")
            continue

        wc = WordCloud(
            width=600, height=400,
            background_color="white", max_words=30
        ).generate_from_frequencies(word_scores)

        plt.figure(figsize=(8, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()

        # Path objects are supported natively by modern Matplotlib engines
        file_path = wc_dir / f"topic_{tid:03d}.png"
        plt.savefig(file_path, dpi=100)
        plt.close()

    logger.info(f" Graphic word cloud files exported to: {wc_dir.resolve()}")
# ── SECTION 12: MAIN RUNNER ──────────────────────────────────
def main():
    """Executes the complete end-to-end BERTopic modeling pipeline workflow."""
    # Ensure root output folders exist before performing file write executions
    ensure_output_dir()
    load_runtime_config_from_doer()
    start_total = time.time()

    # Step 1: Read source document sets
    df, documents = load_documents()

    # Step 2: Extract embedded map vectors
    embeddings = load_or_create_embeddings(documents)

    # Step 3: Instantiate model properties
    topic_model, vectorizer_model = build_topic_model()

    # Step 4: Fit model space on university server systems
    topics = train_model(topic_model, documents, embeddings)

    # Step 5: Assign and reduce unassigned outlier records
    stats = reduce_outliers(topic_model, documents, topics, vectorizer_model)
    topics = stats["topics"]

    # Step 6: Connect and collapse highly overlapping communities
    topics = merge_similar_topics(topic_model, documents, topics, vectorizer_model)

    # Step 7: Save core model binary artifacts to disk safely
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(
        str(config.MODEL_PATH),
        serialization="safetensors",
        save_ctfidf=True
    )
    logger.info(" Final model state successfully stored on disk.")

    # Step 8: Build analytical metrics across entire dataset fields
    summary = build_topic_modeling_summary(
        df=df,
        topic_model=topic_model,
        topics=topics,
        stats=stats,
        documents=documents,
        top_n_words=config.TOP_N_WORDS,
        inter_topic_similarity_threshold=config.INTER_TOPIC_SIM_THRESHOLD,
    )

    # Step 9: Write final dataset sheets and diagnostic summaries
    save_outputs(df, topic_model, topics, summary)

    # Step 10: Process graphics visualizations
    topic_counts = pd.Series(topics)
    topic_counts = topic_counts[topic_counts != -1].value_counts()
    generate_wordclouds(topic_model, topic_counts)

    logger.info(f" Pipeline execution completed in {(time.time() - start_total):.2f}s")
    logger.info("Phase 2 Topic Modelling Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
