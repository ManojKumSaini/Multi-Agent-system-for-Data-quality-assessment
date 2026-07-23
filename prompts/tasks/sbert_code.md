# Phase 3: SBERT Similarity Task Instructions

## ROLE
You are an expert Data Scientist. Your task is to generate one single executable Python script that uses SBERT similarity checks, runs NER, and filters data based on an external configuration guide.

---

## CONTEXTUAL INFRASTRUCTURE
- Source File Path: `topic_modelling/documents_with_topics.csv`
- Main Text Column: `news`
- Main Date Column: `date`
- Target Schema Columns: `id1`, `text1`, `entity1`, `id2`, `text2`, `entity2`, `text_similarity_score`, `entity_similarity_score`, `date_difference`

---

## TASK EXECUTION STEPS

### 1. Ingest Configuration & Filter Data Upfront
- Read the external file `sbert_pair_guide.md`. Parse out all configuration values, including the `TARGET_TOPIC_FILTER` numerical IDs list, model names, thresholds, and batch sizes.
- Load the source CSV from `topic_modelling/documents_with_topics.csv`.
- If the file is missing or required fields are absent, trigger the Error Handling protocol immediately.
- Filter the DataFrame upfront, keeping only rows where `topic_id` matches the list parsed from the guide.

### 2. Single-Pass Named Entity Recognition & Regex Repair
- Load the primary NER model specified in the guide via spaCy. Disable the parser and lemmatizer.
- Set up a batch stream processing loop matching the parsed batch size parameter.
- For each row, run entity extraction. If spaCy returns 0 entities, trigger the fallback engine using Capitalization Regex phrase matching (`\b[A-Z][a-zA-Z0-9\-&_]*\b`) to grab proper nouns.
- Ensure hyphens and ampersands are strictly kept inside the regex tokens to protect complex names, and strip any trailing symbols or punctuation from the edges. Only keep tokens greater than 1 character.
- Clean the strings by discarding any tokens that match the parsed noise exclusions junk words list. Save these clean lists into an intermediate column `news_ner`.

### 3. Bulk SBERT Embedding Generation
- Initialize SentenceTransformer using the model engine specified in the configuration guide.
- Generate text embeddings upfront for your filtered headlines using the encode batch size extracted from the guide.

### 4. Memory-Safe Block Similarity and Deduplication Loop
- Group the dataset by `topic_id`. For each valid topic grouping:
  - Implement a row-by-row block chunking slice loop using the block size parameter parsed from the guide.
  - Calculate cosine similarities (`sbert_model.similarity`) for the active block against all document vector spaces inside that topic.
  - Find all candidate pair mappings where similarity meets the threshold parsed from the guide.
  - Apply a strict triangular index sort mask (`global_row_idx`) 
TOTAL_CLEAN_RECORDS_SAVED: <Insert integer value of all final rows written to local topic files here>
```

---

## ERROR HANDLING
If the source file is missing, a runtime exception occurs, or required input columns are not found, your script must explicitly print: `STATUS: ERROR — <Insert descriptive reason here>` and call `sys.exit(1)`.
