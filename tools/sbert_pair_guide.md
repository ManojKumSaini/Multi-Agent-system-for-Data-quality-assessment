# SBERT Pair Generation & Strict Rejection Configuration Guide

## 1. Upfront Text Vectorization
* **EMBEDDING_MODEL**: `"all-MiniLM-L6-v2"`
* **SBERT_ENCODE_BATCH_SIZE**: `64`

---

## 2. Matrix Similarity, Chunking, and Deduplication
* **CANDIDATE_SIMILARITY_THRESHOLD**: Set `np.where` to `>= 0.0` to pass all raw candidate rows to the validation gate.
* **MATRIX_BLOCK_SIZE**: `1000`  
  *Note: Process vector similarities in blocks of 1,000 rows to prevent RAM memory crashes while maintaining maximum speed.*
* **DEDUPLICATION_MASK**: Force strictly `(global_row_idx < valid_cols)`. This removes self-comparisons and permanently drops backward duplicates (from `id2` to `id1`).

---

## 3. Entity Extraction & Fallback Repair
* **PRIMARY_NER_MODEL**: `en_core_web_trf`
* **SPACY_BATCH_SIZE**: `256` *(Processes text in batches of 256 to maximize GPU/CPU efficiency and prevent pipeline bottlenecks).*
* **FALLBACK_REPAIR_ENGINE**: Capitalization Regex phrase matching. Triggered during post-processing if a row has 0 entities.
  * **Pattern**: `\b[A-Z][a-zA-Z0-9\-&_]*\b`
  * **Character Preservation**: Hyphens (`-`) and ampersands (`&`) are strictly kept inside the regex pattern to protect complex names like "S&P" and hyphenated words. Trailing punctuation is stripped.
  * **Length Rule**: Token must be greater than 1 character.
* **GLOBAL_NOISE_EXCLUSIONS**: Delete these words if caught by the fallback engine:  
  `reg`, `director`, `pdmr`, `shareholding`, `holding`, `shares`, `ceo`, `conference`, `conferences`, `investor`, `investors`, `upcoming`, `present`

---

## 4. Dual-Gate Jaccard Calculation & Production Rejection

### Dual-Gate Jaccard Logic
Calculate the `entity_jaccard_score` using two priority gates:
* **Gate A (Entities Exist)**: If entities exist on BOTH sides, calculate Jaccard similarity using ONLY the extracted entity lists.
* **Gate B (Fallback Text)**: If entities are missing on either side, split raw text into tokens and lowercase them. Drop background filler words (`reg`, `director`, `pdmr`, `shareholding`, `to`, `at`, `in`, `for`, `the`, `a`, `and`, `is`, `on`). Calculate Jaccard similarity on the remaining tokens.  
  *Note: Hyphens are preserved in raw text split logic to stay consistent with the regex engine.*

### Production Rejection Rules
Rows are permanently dropped and deleted from production if they meet either condition below:

```python
# REJECTION_RULE_A
if (news_similarity_score < 0.70) and (entity_jaccard_score < 0.90):
    drop_row()

# REJECTION_RULE_B
if (news_similarity_score < 0.80) and (entity_jaccard_score < 0.50):
    drop_row()
```

---

## 5. Feature Engineering, Filtering & Export Specs
* **TARGET_TOPIC_FILTER**: Process ONLY rows belonging to these specific topic IDs. Drop all other topic IDs upfront:  
  `0, 1, 2, 3, 4, 5, 6, 7, 15, 17`
* **ENGINEERED_FEATURE**: Calculate the absolute day gap between dates as `date_diff_days`.
* **STORAGE_SPEC**: Group data by `topic_id` and automatically export a brand new, separate CSV file for every unique topic found.
