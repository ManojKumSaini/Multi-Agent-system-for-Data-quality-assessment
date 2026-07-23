
# -------------------------------------------------------------------------------------------------------------------------------
# Output Validator                                                                           
# -------------------------------------------------------------------------------------------------------------------------------
# Generates structured summaries after the Doer's code executes successfully.                                                   
# Produces statistics, samples, and quality metrics — passed to the Evaluator (not raw data).                                   
#                                                                                                                               
# Preprocessing validator: row counts, word distributions, non-alpha density, sample rows                                       
#                                                                                                                               
# Topic Modelling metrics:                                                                                                      
#   - document_coverage_pct: % of documents assigned to a topic (not outliers)                                                  
#   - topic_diversity: ratio of unique words across top-N words per topic                                                       
#   - mean_inter_topic_similarity: average cosine similarity between topic embeddings                                           
#   - topic_size_distribution: min, max, mean, std of documents per topic                                                       
#                                                                                                                               
# Coherence score (dropped from final evaluation):                                                                              
#   Technique: Extracts top-N words from each topic (excluding outlier topic -1),                                               
#   tokenises all documents using BERTopic's vectorizer, builds a gensim Dictionary,                                            
#   and computes c_v coherence via CoherenceModel across all topic word lists.                                                  
#   Sampled to 50,000 documents max for computational feasibility.                                                              
#   Dropped: thresholds (>0.4 good, 0.34-0.38 warning) proved unreliable for                                                   
#   financial headline data where short documents reduce score validity.                                                        
# -------------------------------------------------------------------------------------------------------------------------------

"""
graph/validator.py — Generates structured summaries for pipeline phases.
It centralizes file-based preprocessing validation and topic-modeling
metric/summary assembly so callers do not duplicate reporting logic.
"""

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from sklearn.metrics.pairwise import cosine_similarity


def generate_phase_summary(phase_name, output_dir="outputs"):
    """
    Generate a structured summary for a completed phase.
    Routes to the appropriate phase-specific validator.

    Returns:
        dict with statistics, samples, and validation checks.
        Returns None if validation cannot run.
    """
    validators = {
        "preprocessing": _validate_preprocessing,
        "topic_modelling": _validate_topic_modelling,
    }

    validator_fn = validators.get(phase_name)
    if not validator_fn:
        return None

    try:
        return validator_fn(output_dir)
    except Exception as e:
        return {"error": f"Validator failed: {str(e)}"}


def _validate_preprocessing(output_dir):
    """
    Validate preprocessing output.
    Reads the output CSV and generates comprehensive statistics.
    """
    csv_path = os.path.join(output_dir, "preprocessing", "NIFTY_preprocessed.csv")

    if not os.path.exists(csv_path):
        return {"error": f"Output file not found: {csv_path}"}

    # Load output
    df = pd.read_csv(csv_path)

    # Load original for comparison
    original_path = "data/NIFTY.xlsx"
    if os.path.exists(original_path):
        df_original = pd.read_excel(original_path)
        total_rows_in_file = len(df_original)
    else:
        total_rows_in_file = None

    # --- Statistics ---
    word_counts = df["preprocessed_text"].astype(str).str.split().str.len()

    # Non-alphabetic density: ratio of non-alpha chars to total chars
    total_chars = df["preprocessed_text"].astype(str).str.len().sum()
    alpha_chars = df["preprocessed_text"].astype(str).str.count(r"[a-z]").sum()
    non_alpha_density = round(1 - (alpha_chars / total_chars), 4) if total_chars > 0 else 0

    # Top repeated texts
    text_counts = df["preprocessed_text"].value_counts()
    top_5_repeated = [
        {"text_preview": text, "count": int(count)}
        for text, count in text_counts.head(5).items()
    ]

    # Random sample (20 rows, before/after)
    sample_size = min(20, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    sample_random = [
        {
            "id": row.get("id", ""),
            "date": str(row.get("date", "")),
            "news": str(row.get("news", ""))[:200],
            "preprocessed_text": str(row.get("preprocessed_text", ""))[:200],
        }
        for _, row in sample_df.iterrows()
    ]

    # --- Validation Checks ---
    expected_columns = ["id", "date", "news", "label", "pct_change", "preprocessed_text"]
    has_all_columns = all(col in df.columns for col in expected_columns)
    no_empty_text = (df["preprocessed_text"].astype(str).str.strip() == "").sum() == 0
    no_null_text = df["preprocessed_text"].isna().sum() == 0

    summary = {
        "execution_status": "Success",
        "statistics": {
            "total_rows_in_file": total_rows_in_file,
            "rows_after_cleaning": len(df),
            "rows_removed": (total_rows_in_file - len(df)) if total_rows_in_file else None,
            "removal_percentage": round(((total_rows_in_file - len(df)) / total_rows_in_file) * 100, 2) if total_rows_in_file else None,
            "word_count_dist": {
                "mean": round(word_counts.mean(), 2),
                "std": round(word_counts.std(), 2),
                "min": int(word_counts.min()),
                "p25": round(word_counts.quantile(0.25), 1),
                "p50": round(word_counts.quantile(0.50), 1),
                "p75": round(word_counts.quantile(0.75), 1),
                "max": int(word_counts.max()),
            },
            "non_alphabetic_density": non_alpha_density,
            "top_5_repeated": top_5_repeated,
        },
        "sample_random_20": sample_random,
        "validation_checks": {
            "has_all_expected_columns": has_all_columns,
            "no_empty_preprocessed_text": no_empty_text,
            "no_null_preprocessed_text": no_null_text,
            "columns_found": list(df.columns),
        },
    }

    return summary


def _validate_topic_modelling(output_dir):
    """
    Validate topic-modeling output.
    Prefers the generated phase summary if it already exists, and falls back to
    the saved topic modeling artifacts.
    """
    summary_path = os.path.join(output_dir, "topic_modeling", "phase_2_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    docs_path = os.path.join(output_dir, "topic_modeling", "documents_with_topics.csv")
    info_path = os.path.join(output_dir, "topic_modeling", "topic_info.csv")

    if not os.path.exists(docs_path):
        return {"error": f"Output file not found: {docs_path}"}

    df = pd.read_csv(docs_path)
    topic_info = pd.read_csv(info_path) if os.path.exists(info_path) else pd.DataFrame()

    topic_count = int(df["topic_id"].nunique()) if "topic_id" in df.columns else 0
    outlier_count = int((df["topic_id"] == -1).sum()) if "topic_id" in df.columns else 0

    return {
        "execution_status": "Success",
        "statistics": {
            "rows": int(len(df)),
            "topic_count": topic_count,
            "outlier_count": outlier_count,
            "topic_info_rows": int(len(topic_info)),
            "columns": list(df.columns),
        },
    }


def _topic_dict(model) -> Dict[int, List[tuple]]:
    return model.get_topics()


def compute_topic_diversity(topic_model, top_n: int) -> float:
    topics = _topic_dict(topic_model)
    words = [
        w
        for tid, ws in topics.items()
        if tid != -1
        for w, _ in ws[:top_n]
    ]
    return 0.0 if not words else round(len(set(words)) / len(words), 4)


def compute_intertopic_similarity(topic_model, threshold: float) -> Tuple[float, List[Tuple[int, int, float]]]:
    topics = _topic_dict(topic_model)
    valid_ids = sorted([tid for tid in topics if tid != -1])
    if len(valid_ids) < 2:
        return 0.0, []

    ctfidf_matrix = topic_model.c_tf_idf_
    all_matrix_topics = topic_model.get_topic_info()["Topic"].tolist()
    topic_id_to_row = {int(t): i for i, t in enumerate(all_matrix_topics)}
    rows = [topic_id_to_row[tid] for tid in valid_ids]
    sub_matrix = ctfidf_matrix[rows]
    sim_mat = cosine_similarity(sub_matrix)

    iu, ju = np.triu_indices_from(sim_mat, k=1)
    upper = sim_mat[iu, ju]
    mean_sim = round(float(upper.mean()), 4) if upper.size else 0.0

    pairs = [
        (valid_ids[iu[k]], valid_ids[ju[k]], round(float(upper[k]), 4))
        for k in range(len(iu))
        if upper[k] > threshold
    ]
    pairs.sort(key=lambda x: x[2], reverse=True)
    return mean_sim, pairs[:5]


def build_top_topics_report(topic_model, top_n: int = 30) -> List[Dict[str, Any]]:
    info = topic_model.get_topic_info()
    top = (
        info[info["Topic"] != -1]
        .sort_values("Count", ascending=False)
        .head(top_n)
    )
    return [
        {
            "topic_id": int(row["Topic"]),
            "count": int(row["Count"]),
            "name": row["Name"],
            "top_words": row["Representation"],
            "representative_docs": row["Representative_Docs"],
        }
        for _, row in top.iterrows()
    ]


def validate_topic_modeling_results(metrics: dict, modeling_rows: int) -> dict:
    required_keys = {"document_coverage_pct", "topics_after_outlier_reduction", "outlier_count_after"}
    missing_keys = required_keys - metrics.keys()
    assert not missing_keys, f"Missing critical metrics keys in validation: {missing_keys}"

    min_coverage = 50.0
    min_topics = 5

    checks = {
        "coverage_gte_threshold": metrics["document_coverage_pct"] >= min_coverage,
        "topics_gte_threshold": metrics["topics_after_outlier_reduction"] >= min_topics,
        "not_all_outliers": metrics["outlier_count_after"] < modeling_rows,
    }

    failed_checks = [key for key, passed in checks.items() if not passed]

    return {
        "all_checks_passed": len(failed_checks) == 0,
        "failed_checks": failed_checks,
        "check_details": checks,
    }


def build_topic_modeling_summary(
    df: pd.DataFrame,
    topic_model,
    topics: Sequence[int],
    stats: Dict[str, int],
    documents: Sequence[str],
    top_n_words: int,
    inter_topic_similarity_threshold: float,
) -> Dict[str, Any]:
    topics_arr = np.asarray(topics)
    mask = topics_arr != -1
    uniq, counts = np.unique(topics_arr[mask], return_counts=True)

    if counts.size:
        size_distribution = {
            "mean": round(float(counts.mean()), 2),
            "std": round(float(counts.std()), 2),
            "min": int(counts.min()),
            "max": int(counts.max()),
        }
    else:
        size_distribution = {"mean": 0.0, "std": 0.0, "min": 0, "max": 0}

    outliers = int(np.sum(~mask))
    coverage = round(((len(documents) - outliers) / max(len(documents), 1)) * 100, 2)

    mean_sim, top_pairs = compute_intertopic_similarity(
        topic_model,
        inter_topic_similarity_threshold,
    )

    metrics = {
        "topics_before_outlier_reduction": stats.get("topics_before", 0),
        "topics_after_outlier_reduction": int(len(uniq)),
        "outlier_count_before": stats.get("outliers_before", 0),
        "outlier_count_after": outliers,
        "outliers_recovered": stats.get("outliers_before", 0) - outliers,
        "document_coverage_pct": coverage,
        "topic_diversity": compute_topic_diversity(topic_model, top_n_words),
        "mean_inter_topic_similarity": mean_sim,
        "top_similar_topic_pairs": top_pairs,
        "topic_size_distribution": size_distribution,
    }

    validation = validate_topic_modeling_results(metrics, len(documents))

    return {
        "execution_status": "Success" if validation["all_checks_passed"] else "Error",
        "statistics": metrics,
        "top_topics": build_top_topics_report(topic_model, top_n=30),
        "validation": validation,
    }


