# Phase 1: Data Preprocessing

Return one executable Python script inside a single ```python code block. Do not add any explanation outside the code block.

---

## ROLE
You are an expert Data Preprocessing Engineer. Your task is to generate a Python script that cleans text data for downstream Transformer-based models. Refer to the DOMAIN CONTEXT section for domain-specific preservation rules.

---

## TASK
Preprocess `data/NIFTY.xlsx` and export a cleaned CSV. Inspect the data first, then apply appropriate cleaning operations.

---

## INPUT
- File: `data/NIFTY.xlsx`
- Text column: `news`
- Date column: `date`
- Preserve columns: `id`, `label`, `pct_change`

<DYNAMIC_DOMAIN_CONTEXT>
{domain_context}
</DYNAMIC_DOMAIN_CONTEXT>

---

## INSTRUCTIONS

1. Load the data. Store raw row count in variable `loaded`.
2. Create a new column `preprocessed_text` from the `news` column. NEVER modify the original `news` column.
3. Apply mandatory operations 1–7 from the Reference Guide (in the exact order specified). Do NOT apply operation 8 (whitespace collapse) yet — it MUST come after all optional operations.
4. Decide which optional operations to apply. For EACH optional operation:
   - Check `preprocessed_text` (NOT the original `news` column) for evidence that the operation is needed.
   - Evidence check must be column-wide (e.g., `.str.contains(..., regex=True, na=False).any()`), never single-row checks such as `iloc[0]`.
   - If applied, add an inline code comment explaining WHY based on what you observed in the data.
5. After ALL optional operations are complete, apply mandatory operation 8 (collapse whitespace and strip). This MUST be the absolute final text modification step. No other text changes after this.
6. Apply post-cleaning filters:
   - Remove rows where `preprocessed_text` is empty string. Store count as `after_empty_removal`.
   - Remove rows where `preprocessed_text` has fewer than a minimum word count (you decide threshold — justify in comment). Store count as `after_min_word_filter`.
   - Remove duplicate rows based on `preprocessed_text` AND `date` (same text on same date = true duplicate). Store count as `after_dedup`.
   - If any exact `preprocessed_text` appears more than 50 times, create a boolean helper column named `is_boilerplate` for those rows and include it in the final CSV export.
   - Count tracking is strict and sequential: `loaded >= after_empty_removal >= after_min_word_filter >= after_dedup`. Assign each variable immediately after its step and never reassign these count variables later (for example, inside boilerplate logic).
7. Before saving, cast: `df['preprocessed_text'] = df['preprocessed_text'].astype(str)`.
8. Drop any helper columns that are NOT needed in output (e.g., temporary working columns like `word_count`) — they must NOT appear in the final CSV. Keep `is_boilerplate` if it was created.

---

## OUTPUT
- Create directory `outputs/preprocessing/` if it does not exist.
- Save to `outputs/preprocessing/NIFTY_preprocessed.csv`.
- Columns in output CSV must be EXACTLY: `id`, `date`, `news`, `label`, `pct_change`, `preprocessed_text`, `is_boilerplate` (if boilerplate rows exist). No other columns.
- Use `columns=['id', 'date', 'news', 'label', 'pct_change', 'preprocessed_text', 'is_boilerplate']` in the `.to_csv()` call (include `is_boilerplate` if created).
- `index=False`.

---

## STDOUT — Print exactly these lines:

loaded: {loaded}
after_empty_removal: {after_empty_removal}
after_min_word_filter: {after_min_word_filter}
after_dedup: {after_dedup}
STATUS: Phase 1 Script Completed.

---

## ERROR HANDLING
- If file not found or required columns missing: print `STATUS: ERROR — <reason>` and `sys.exit(1)`.

---

## ALLOWED LIBRARIES
`pandas`, `numpy`, `re`, `os`, `sys`, `html` only. Do NOT use `nltk`, `spacy`, or any external NLP library.

## CODE CONSTRAINTS
- Do NOT use `.apply()`, `.applymap()`, or row-by-row `lambda` for regex operations. All regex must use vectorized `.str.replace()`.
- Only exception: `.apply(html.unescape)` is permitted for HTML entity decoding (no vectorized alternative exists).
- Do NOT modify original data columns — always create new columns.
- Do NOT leave helper columns (like `word_count`) in the final output.
