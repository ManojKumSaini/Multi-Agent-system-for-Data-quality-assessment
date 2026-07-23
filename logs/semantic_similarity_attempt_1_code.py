import pandas as pd
import spacy
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import sys

# ========== CONFIGURATION LOADING ==========
def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if not line.startswith('-'):
                    continue
                line = line[2:].strip()
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    config[key] = value
            return config
    except Exception as e:
        print(f"STATUS: ERROR - Failed to load config file: {str(e)}")
        sys.exit(1)

# ========== TEXT TRUNCATION ==========
def truncate_text(text, max_chars):
    return text[:max_chars] if len(text) > max_chars else text

# ========== NER EXTRACTION ==========
def extract_entities(text, ner_model, allowed_labels):
    doc = ner_model(text)
    return [ent.text for ent in doc.ents if ent.label_ in allowed_labels]

# ========== MAIN PIPILINE ==========
def main():
    config_path = 'logs/sbert_pair_guide.txt'
    config = load_config(config_path)

    # 1. Parse configuration values
    try:
        TARGET_TOPIC_FILTER = [int(x.strip()) for x in config['TARGET_TOPIC_FILTER'].split(',')]
        EMBEDDING_MODEL = config['EMBEDDING_MODEL']
        SBERT_BATCH_SIZE = int(config['SBERT_ENCODE_BATCH_SIZE'])
        MATRIX_BLOCK_SIZE = int(config['MATRIX_BLOCK_SIZE'])
        CANDIDATE_THRESHOLD = float(config['CANDIDATE_SIMILARITY_THRESHOLD'])
        SOFT_SIMILARITY_THRESHOLD = float(config['SOFT_SIMILARITY_THRESHOLD'])
        PRIMARY_NER_MODEL = config['PRIMARY_NER_MODEL']
        SPACY_BATCH_SIZE = int(config['SPACY_BATCH_SIZE'])
        GLOBAL_NOISE_EXCLUSIONS = set(config['GLOBAL_NOISE_EXCLUSIONS'].split(','))
        MAX_NEWS_CHARACTERS = int(config['MAX_NEWS_CHARACTERS'])
        OUTPUT_ROOT = config['OUTPUT_ROOT']
        TELEMETRY_LABELS = config.get('TELEMETRY_LABELS', '').split(',') if 'TELEMETRY_LABELS' in config else []
        ALLOWED_ENTITY_LABELS = config.get('ALLOWED_ENTITY_LABELS', '').split(',') if 'ALLOWED_ENTITY_LABELS' in config else ['PERSON', 'ORG', 'GPE', 'LOC']
    except KeyError as e:
        print(f"STATUS: ERROR - Missing configuration parameter: {str(e)}")
        sys.exit(1)

    # 2. Load source datas
    source_path = 'outputs/topic_modeling/documents_with_topics.csv'
    if not os.path.exists(source_path):
        print(f"STATUS: ERROR - Source file not found: {source_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(source_path)
        required_columns = ['news', 'date', 'topic_id', 'id']
        if not all(col in df.columns for col in required_columns):
            print(f"STATUS: ERROR - Missing required columns in source file")
            sys.exit(1)
    except Exception as e:
        print(f"STATUS: ERROR - Failed to load source file: {str(e)}")
        sys.exit(1)

    # 3. Filter by target topics
    df = df[df['topic_id'].isin(TARGET_TOPIC_FILTER)]
    if df.empty:
        print("STATUS: INFO - No documents matched target topics")
        return

    # 4. Load NER model
    try:
        ner_model = spacy.load(PRIMARY_NER_MODEL)
        ner_model.max_length = 2000000
    except Exception as e:
        print(f"STATUS: ERROR - Failed to load NER model: {str(e)}")
        sys.exit(1)

    # 5. Batch NER with row-wise fallback and row_results list
    def process_ner(texts, batch_size):
        row_results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_entities = [extract_entities(t, ner_model, ALLOWED_ENTITY_LABELS) for t in batch]
            for idx, entities in enumerate(batch_entities):
                if not entities:
                    # Apply regex fallback to single text
                    text = batch[idx]
                    tokens = re.findall(r'\b[A-Z][a-zA-Z0-9\-&_]*\b', text)
                    cleaned = [t.strip('.,;:!?()[]{}') for t in tokens]
                    row_results.append([t for t in cleaned if len(t) > 1 and t.lower() not in GLOBAL_NOISE_EXCLUSIONS])
                else:
                    row_results.append(entities)
        return row_results

    # Ensure df['news_ner'] matches df length
    df['news_ner'] = process_ner(df['news'].tolist(), SPACY_BATCH_SIZE)
    if len(df['news_ner']) != len(df):
        print(f"STATUS: ERROR - NER results length mismatch: {len(df['news_ner'])} vs {len(df)}")
        sys.exit(1)

    # 6. Truncate text before encoding
    def truncate_texts(texts, max_chars):
        return [truncate_text(t, max_chars) for t in texts]

    df['news'] = truncate_texts(df['news'].tolist(), MAX_NEWS_CHARACTERS)

    # 7. Generate SBERT embeddings (isolated in numpy)
    try:
        sbert_model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"STATUS: ERROR - Failed to load SBERT model: {str(e)}")
        sys.exit(1)

    def encode_texts(texts, batch_size):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = sbert_model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    text_embeddings = encode_texts(df['news'].tolist(), SBERT_BATCH_SIZE)

    # 8. Block-wise similarity with global index mask
    results = []
    topic_groups = df.groupby('topic_id')

    total_candidates = 0
    total_drops = 0
    total_saved = 0

    for topic_id, group in topic_groups:
        print(f"\nProcessing topic {topic_id} ({len(group)} documents)")

        # 9. Block processing: only use block.index.tolist()
        block_size = MATRIX_BLOCK_SIZE
        blocks = [group[i:i+block_size] for i in range(0, len(group), block_size)]

        for block in blocks:
            block_indices = block.index.tolist()
            block_indices = [i for i in block_indices if i < len(text_embeddings)]
            block_embeddings = text_embeddings[block_indices]
            # 11. Full embeddings of entire topic group (not just current block)
            full_group_indices = [j for j in group.index.tolist() if j < len(text_embeddings)]
            full_embeddings = text_embeddings[full_group_indices]

            # --- ADD THIS SAFETY CHECK HERE ---
            if len(block_embeddings) == 0 or len(full_embeddings) == 0:
                continue  # Skip this block if there is no data to compare
            # ----------------------------------

            # 12. Compute similarities between block and full topic
            similarity_matrix = cosine_similarity(block_embeddings, full_embeddings)
            similarity_matrix = similarity_matrix.astype(np.float32)  # Optimize memory
import pandas as pd
import spacy
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import sys

# ========== CONFIGURATION LOADING ==========
def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if not line.startswith('-'):
                    continue
                line = line[2:].strip()
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    config[key] = value
            return config
    except Exception as e:
        print(f"STATUS: ERROR - Failed to load config file: {str(e)}")
        sys.exit(1)

# ========== TEXT TRUNCATION ==========
def truncate_text(text, max_chars):
    return text[:max_chars] if len(text) > max_chars else text

# ========== NER EXTRACTION ==========
def extract_entities(text, ner_model, allowed_labels):
    doc = ner_model(text)
    return [ent.text for ent in doc.ents if ent.label_ in allowed_labels]

# ========== MAIN PIPELINE ==========
def main():
    config_path = 'outputs/sbert_pair_guide.txt'
    config = load_config(config_path)

    # 1. Parse configuration values
    try:
        TARGET_TOPIC_FILTER = [int(x.strip()) for x in config['TARGET_TOPIC_FILTER'].split(',')]
        EMBEDDING_MODEL = config['EMBEDDING_MODEL']
        SBERT_BATCH_SIZE = int(config['SBERT_ENCODE_BATCH_SIZE'])
        MATRIX_BLOCK_SIZE = int(config['MATRIX_BLOCK_SIZE'])
        CANDIDATE_THRESHOLD = float(config['CANDIDATE_SIMILARITY_THRESHOLD'])
        SOFT_SIMILARITY_THRESHOLD = float(config['SOFT_SIMILARITY_THRESHOLD'])
        PRIMARY_NER_MODEL = config['PRIMARY_NER_MODEL']
        SPACY_BATCH_SIZE = int(config['SPACY_BATCH_SIZE'])
        GLOBAL_NOISE_EXCLUSIONS = set(config['GLOBAL_NOISE_EXCLUSIONS'].split(','))
        MAX_NEWS_CHARACTERS = int(config['MAX_NEWS_CHARACTERS'])
        OUTPUT_ROOT = config['OUTPUT_ROOT']
        TELEMETRY_LABELS = [x.strip() for x in config['TELEMETRY_LABELS'].split(',')] if 'TELEMETRY_LABELS' in config else []
        ALLOWED_ENTITY_LABELS = config.get('ALLOWED_ENTITY_LABELS', '').split(',') if 'ALLOWED_ENTITY_LABELS' in config else ['PERSON', 'ORG', 'GPE', 'LOC']
    except KeyError as e:
        print(f"STATUS: ERROR - Missing configuration parameter: {str(e)}")
        sys.exit(1)

    # 2. Load source data
    source_path = 'outputs/topic_modeling/documents_with_topics.csv'
    if not os.path.exists(source_path):
        print(f"STATUS: ERROR - Source file not found: {source_path}")
        sys.exit(1)

    try:
        df = pd.read_csv(source_path)
        required_columns = ['news', 'date', 'topic_id', 'id']
        if not all(col in df.columns for col in required_columns):
            print(f"STATUS: ERROR - Missing required columns in source file")
            sys.exit(1)
    except Exception as e:
        print(f"STATUS: ERROR - Failed to load source file: {str(e)}")
        sys.exit(1)

    # 3. Filter by target topics and RESET index to align with raw numpy matrices
    df = df[df['topic_id'].isin(TARGET_TOPIC_FILTER)]
    df = df.reset_index(drop=True)
    
    if df.empty:
        print("STATUS: INFO - No documents matched target topics")
        return

    # 4. Load NER model
    try:
        ner_model = spacy.load(PRIMARY_NER_MODEL)
        ner_model.max_length = 2000000
    except Exception as e:
        print(f"STATUS: ERROR - Failed to load NER model: {str(e)}")
        sys.exit(1)

    # 5. Batch NER with noun chunk fallback
    def process_ner(texts, batch_size):
        """
        Extract named entities using spaCy.
        If no named entities are found, fall back to noun chunks.

        Returns:
            List[List[str]]
        """
        row_results = []

        for doc in ner_model.pipe(texts, batch_size=batch_size):

            # -----------------------------
            # Primary: Named Entity Recognition
            # -----------------------------
            entities = []

            for ent in doc.ents:
                if ent.label_ in ALLOWED_ENTITY_LABELS:
                    entity = ent.text.strip().lower()

                    if entity not in GLOBAL_NOISE_EXCLUSIONS:
                        entities.append(entity)

            # Remove duplicates while preserving order
            entities = list(dict.fromkeys(entities))

            # -----------------------------
            # Fallback: Noun Chunks
            # -----------------------------
            if not entities:

                noun_chunks = []

                for chunk in doc.noun_chunks:

                    text = chunk.lemma_.strip().lower()

                    # Remove punctuation
                    text = re.sub(r"[^\w\s-]", "", text)

                    if len(text) < 3:
                        continue

                    if text in GLOBAL_NOISE_EXCLUSIONS:
                        continue

                    # Ignore chunks made entirely of stop words
                    if all(token.is_stop for token in chunk):
                        continue

                    noun_chunks.append(text)

                # Remove duplicates
                entities = list(dict.fromkeys(noun_chunks))

            row_results.append(entities)

        return row_results


    df["news_ner"] = process_ner(df["news"].tolist(), SPACY_BATCH_SIZE)

    if len(df["news_ner"]) != len(df):
        print(
            f"STATUS: ERROR - NER results length mismatch: "
            f"{len(df['news_ner'])} vs {len(df)}"
        )
        sys.exit(1)

    # 6. Truncate text before encoding
    def truncate_texts(texts, max_chars):
        return [truncate_text(t, max_chars) for t in texts]

    df['news'] = truncate_texts(df['news'].tolist(), MAX_NEWS_CHARACTERS)

    # 7. Generate SBERT embeddings
    try:
        sbert_model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"STATUS: ERROR - Failed to load SBERT model: {str(e)}")
        sys.exit(1)

    def encode_texts(texts, batch_size):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = sbert_model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    text_embeddings = encode_texts(df['news'].tolist(), SBERT_BATCH_SIZE)

    # 8. Block-wise similarity with global index mask
    results = []
    topic_groups = df.groupby('topic_id')

    total_candidates = 0
    total_drops = 0
    total_saved = 0

    for topic_id, group in topic_groups:
        print(f"\nProcessing topic {topic_id} ({len(group)} documents)")

        block_size = MATRIX_BLOCK_SIZE
        group_indices = group.index.tolist()

        for i in range(0, len(group_indices), block_size):
            block_indices = group_indices[i:i+block_size]

            # Grab accurate embedding slices using the perfectly aligned indices
            block_embeddings = text_embeddings[block_indices]
            full_embeddings = text_embeddings[group_indices]

            # 12. Compute similarities between block and full topic
            similarity_matrix = cosine_similarity(block_embeddings, full_embeddings)
            similarity_matrix = similarity_matrix.astype(np.float32)

            # 13. Soft similarity filtering
            num_rows, num_cols = similarity_matrix.shape
            for r in range(num_rows):
                for c in range(num_cols):
                    global_i = block_indices[r]
                    global_j = group_indices[c]
                    
                    if global_i < global_j:  # Global forward triangular mask
                        score = similarity_matrix[r][c]

                        if score < SOFT_SIMILARITY_THRESHOLD:
                            continue
                        if score >= CANDIDATE_THRESHOLD:
                            # 14. Get text and entities using .loc
                            text1 = df.loc[global_i, 'news']
                            entities1 = df.loc[global_i, 'news_ner']
                            text2 = df.loc[global_j, 'news']
                            entities2 = df.loc[global_j, 'news_ner']

                            # 15. Entity Jaccard score
                            if entities1 and entities2:
                                entities1 = {e.lower().strip() for e in entities1}
                                entities2 = {e.lower().strip() for e in entities2}

                                intersection = entities1 & entities2
                                union = entities1 | entities2

                                entity_score = len(intersection) / len(union) if union else 0                           
                            else:
                                tokens1 = [t for t in re.split(r'\W+', text1.lower()) if t and t not in GLOBAL_NOISE_EXCLUSIONS]
                                tokens2 = [t for t in re.split(r'\W+', text2.lower()) if t and t not in GLOBAL_NOISE_EXCLUSIONS]
                                intersection = set(tokens1) & set(tokens2)
                                union = set(tokens1) | set(tokens2)
                                entity_score = len(intersection) / len(union) if union else 0

                            # 16. Apply rejection rules
                            if (score < 0.70 and entity_score < 0.90) or (score < 0.80 and entity_score < 0.50):
                                total_drops += 1
                                continue

                            # 17. Date difference in days
                            date1 = pd.to_datetime(df.loc[global_i, 'date'])
                            date2 = pd.to_datetime(df.loc[global_j, 'date'])
                            date_diff = abs((date2 - date1).days)

                            # 18. Append to results
                            results.append({
                                'id_left': df.loc[global_i, 'id'],
                                'news_left': text1,
                                'news_ner_left': '|'.join(entities1) if isinstance(entities1, list) else '',
                                'id_right': df.loc[global_j, 'id'],
                                'news_right': text2,
                                'news_ner_right': '|'.join(entities2) if isinstance(entities2, list) else '',
                                'news_similarity_score': score,
                                'entity_jaccard_score': entity_score,
                                'date_diff_days': date_diff
                            })
                            total_candidates += 1

        # 19. Export results per-topic
        if results:
            results_df = pd.DataFrame(results)
            results_df.rename(columns={
                'id_left': 'id1',
                'news_left': 'text1',
                'news_ner_left': 'entity1',
                'id_right': 'id2',
                'news_right': 'text2',
                'news_ner_right': 'entity2',
                'news_similarity_score': 'text_similarity_score',
                'entity_jaccard_score': 'entity_similarity_score',
                'date_diff_days': 'date_difference'
            }, inplace=True)
            
            results_df.sort_values(
                by=['text_similarity_score', 'entity_similarity_score'], 
                ascending=[False, False], 
                inplace=True
            )
            # ----------------------------------

            output_dir = os.path.join(OUTPUT_ROOT, f'topic_{topic_id}')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'topic.csv')
            results_df.to_csv(output_path, index=False)
            total_saved += len(results_df)
            results.clear()

    # 20. Final telemetry
    print("\nSTATUS: SUCCESS")
    telemetry = {
        'TOTAL_CANDIDATES_GENERATED': total_candidates,
        'TOTAL_RECORDS_DELETED': total_drops,
        'TOTAL_CLEAN_RECORDS_SAVED': total_saved
    }

    # 21. Use TELEMETRY_LABELS safely with a clean strip to avoid space-mismatch KeyErrors
    if TELEMETRY_LABELS:
        telemetry = {
            label.strip(): telemetry.get(label.strip(), 0) 
            for label in TELEMETRY_LABELS 
            if label.strip() in telemetry
        }

    for label, count in telemetry.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    main()
