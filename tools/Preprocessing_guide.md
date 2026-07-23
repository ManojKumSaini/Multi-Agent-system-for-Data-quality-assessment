# Preprocessing Guide — Operations Reference

## Purpose
This guide defines all available text cleaning operations for the preprocessing phase. The target downstream models are SBERT and BERTopic (Transformer-based). Operations are divided into mandatory (always apply) and optional (apply based on data inspection).

---

## MANDATORY OPERATIONS (Apply ALL, in this exact order)

| # | Operation | Method / Pattern | Notes |
|---|-----------|-----------------|-------|
| 1 | Handle nulls | Replace NaN with `""`, cast to `str`, then `.str.strip()` | Always first |
| 2 | Remove URLs | `.str.replace(r"https?://\S+\|www\.\S+", " ", regex=True)` | Apply while text is still raw (before lowercase) |
| 3 | Remove emails | `.str.replace(r"\S+@\S+", " ", regex=True)` | Apply while text is still raw |
| 4 | Remove HTML tags | `.str.replace(r"<[^>]*>", " ", regex=True)` | |
| 5 | Fix encoding artifacts | `.str.replace(r"â€™\|Ã©\|Â\|\x00", "", regex=True)` | Use alternation. Do NOT use square character class brackets for multi-char sequences |
| 6 | Unescape HTML entities | `.apply(html.unescape)` | e.g., `&amp;` → `&`. This is the ONLY permitted use of `.apply()` |
| 7 | Lowercase | `.str.lower()` | |
| 8 | Collapse whitespace and strip | `.str.replace(r"\s+", " ", regex=True)` then `.str.strip()` | **MUST be the absolute LAST text modification step. Apply ONLY after ALL optional operations are complete. Do NOT apply before optional operations. This ensures that any spaces introduced by optional operations (replacing characters with " ") are properly collapsed.** |

---

## OPTIONAL OPERATIONS (Apply only if data evidence supports it)

Cross-reference the requirements of the active dataset against the properties provided in the `<DYNAMIC_DOMAIN_CONTEXT>` block before deciding. 
| # | Operation | Pattern | When to Use | Impact |
|---|-----------|---------|-------------|--------|
| 1 | Remove possessives | `r"['\u2019]s\b"` | `preprocessed_text` contains possessive forms (e.g., "company's", "market's") | Reduces vocabulary without losing core meaning |
| 2 | Remove plural indicators | `r"\(s\)"` | `preprocessed_text` contains constructions like "item(s)", "report(s)" | Normalizes plural/singular ambiguity |
| 3 | Remove remaining apostrophes | `r"['\u2019]"` | After possessive removal, if remaining apostrophes add noise | Normalizes contractions (e.g., "don't" → "dont") |
| 4 | Remove ellipses | `r"\.{3,}\|\u2026"` (replace with space) | `preprocessed_text` contains trailing dots or ellipsis characters | Removes incomplete sentence markers |
| 5 | Character filter | `r"[^a-z0-9\s\.\$&\u20ac%,-]"` (replace with space) | `preprocessed_text` contains special characters not meaningful for the domain | Keeps: lowercase letters, numbers, whitespace, periods, currency signs ($, €), percent (%), ampersand (&), commas, hyphens |
| 6 | Remove gibberish/corrupted words | `r"\b\w{25,}\b"` (replace with space) | Data contains corrupted tokens or unrealistically long strings | Removes corrupted tokens. Do NOT drop rows — replace matched words with a single space |

---

## DECISION GUIDELINES

- **Domain signals:** Refer to the DOMAIN CONTEXT section for symbols and tokens that must be preserved. Do NOT remove domain-relevant signals.
- **For Transformer models (SBERT/BERTopic):** These models use subword tokenization. Aggressive lemmatisation or stemming destroys semantic information. Lowercasing is sufficient normalization.
- **Do NOT apply:** Lemmatisation, stemming, or stopword removal.
- **Apostrophe handling:** Handle BOTH straight (`'`) and curly (`\u2019`) apostrophe variants in your patterns.
- **Inspection rule:** Always check `preprocessed_text` (the column being modified), NOT the original text column. The text is already lowercased by the time you decide on optional operations.
- **Replacement rule:** When removing characters/patterns, replace with a single space `" "`. The final whitespace collapse (mandatory operation 8) will clean up any resulting double spaces.
- **Optional operation comments:** For each optional operation you decide to apply, include an inline code comment explaining WHY based on evidence found in the data (e.g., "// Justified: dataset contains many possessive forms like 'company's'").

---

## ORDER OF OPERATIONS (Critical — Follow Exactly)

- Step 1: Mandatory operations 1–7 (in order listed above)
- Step 2: Optional operations (in order: possessives → plural indicators → apostrophes → ellipses → character filter → gibberish)
- Step 3: Mandatory operation 8 (collapse whitespace and strip)
- **DONE** — No further text modifications allowed after Step 3

**WARNING:** If you apply whitespace collapse (operation 8) before optional operations, the optional operations will introduce new multi-spaces that are never cleaned. This is the most common ordering mistake.

---

## POST-CLEANING FILTERS (After all text modifications)

Apply these AFTER mandatory operation 8:

| # | Filter | Method | Notes |
|---|--------|--------|-------|
| 1 | Remove empty rows | `df = df[df['preprocessed_text'] != ""]` | Rows that became empty after cleaning |
| 2 | Minimum word count | `df = df[df['preprocessed_text'].str.split().str.len() >= threshold]` | You decide threshold (justify in comment). For short-text datasets like headlines, 2–3 is typical |
| 3 | Deduplication | `df = df.drop_duplicates(subset=['preprocessed_text', 'date'])` | Same text on same date = true duplicate. Do NOT deduplicate on `id` alone or `preprocessed_text` alone |

---

## OUTPUT REQUIREMENTS

- Save to: `outputs/preprocessing/NIFTY_preprocessed.csv`
- Columns EXACTLY: `id`, `date`, `news`, `label`, `pct_change`, `preprocessed_text`
- Use `columns=['id', 'date', 'news', 'label', 'pct_change', 'preprocessed_text']` in `.to_csv()`
- `index=False`
- `.astype(str)` on `preprocessed_text` before saving
- Drop ALL helper columns (e.g., `word_count`) before saving

---

## CODE CONSTRAINTS

- Use vectorized Pandas `.str.replace(r"pattern", "replacement", regex=True)` for ALL regex operations.
- Do NOT use `.apply()`, `.applymap()`, or row-by-row `lambda` for regex. The only permitted `.apply()` is for `html.unescape`.
- Do NOT modify original data columns — always create a new column (`preprocessed_text`).
- ALLOWED LIBRARIES: `pandas`, `numpy`, `re`, `os`, `sys`, `html` only. No `nltk`, `spacy`, or external NLP libraries.

---