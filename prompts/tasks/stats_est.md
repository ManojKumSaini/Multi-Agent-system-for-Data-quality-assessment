
# Phase 5: Survival Analysis & Currency Modelling

## Objective
Fit survival distributions to labelled article pairs. Find the best distribution per topic that models how matching probability decays over time.

## Input
File: `topic_labeled_updated.csv`

Columns: `id1, text1, id2, text2, text_similarity_score, entity_similarity_score, date_difference, pair_id, label, confidence, needs_human_review, evaluator_verdict, final_label`

Key columns for this phase:
- `final_label` — the binary label (True/False)
- `entity_similarity_score` — NER-based similarity between article pairs
- `date_difference` — days between the two articles
- `topic` — topic assignment (if present; otherwise process all data as one group)

## Required Libraries
```
numpy, pandas, scipy.optimize.curve_fit, scipy.special.gammainc, scipy.stats.norm, sklearn.metrics.log_loss, sklearn.metrics.roc_auc_score, matplotlib.pyplot
```

## Instructions

Process each topic separately. For each topic:

### Retry Behavior

- If this phase is a retry, treat the previous attempt as the base implementation.
- Make the smallest possible edits needed to address the critique.
- Preserve any functions, imports, metrics, plotting, and output code that already work.
- Do NOT rewrite the script from scratch unless the critique explicitly says the structure is unusable.

### Step 1: Data Preparation

1. Where `final_label` is NaN AND `entity_similarity_score == 1.0` → set `final_label = False`
2. Keep only rows where:
   - (`final_label == False` AND `entity_similarity_score == 1.0`) OR (`final_label == True`)
3. Convert `final_label` to int (True→1, False→0)
4. Find max `date_difference` where `final_label == 1`
5. Remove all rows with `date_difference` above that max
6. Print: row count, count of 1s, count of 0s
7. If fewer than 50 rows remain, skip this topic with a warning

### Step 2: Exhaustive Search

```
FOR bin_size in range(1, 200):
    - Apply pd.qcut(date_difference, q=bin_size, duplicates='drop') → time_bin
    - Immediately store the bin labels in the dataframe: topic_df['time_bin'] = time_bin
    - Calculate mid_bin = midpoint of each bin interval, cast to int
    - Use topic_df['time_bin'] for all bin counts and stratified sampling
    
    - Train/test split:
        - n_samples = int(0.2 * min(topic_df['time_bin'].value_counts()))
        - If n_samples < 1, skip this bin_size
        - Test = sample n_samples per bin (random_state=42)
        - Train = remaining rows
        - If test has only 1 class, skip this bin_size
    
    FOR each distribution in registry (see Guide):
        - Get bounds (use get_piecewise_bounds(t_min, t_max) for Piecewise_Exp)
        - Try: curve_fit(func, train['mid_bin'], train['final_label'], bounds=bounds, maxfev=10000)
        - Except (RuntimeError, ValueError, RuntimeWarning): skip, continue
        - Predict on test: p = func(test['mid_bin'], *params)
        - Clip p to [1e-15, 1-1e-15]
        - Compute:
            - log_loss_val = log_loss(test['final_label'], p)
            - auc_val = roc_auc_score(test['final_label'], p)
            - aic_val = 2*n_params + 2*log_loss_val
            - bic_val = np.log(len(train))*n_params + 2*log_loss_val
        - Store: {distribution, bin_size, log_loss, auc, aic, bic, params}
```

### Step 3: Rank Results

1. Rank by: log_loss (ascending), AUC (descending), AIC (ascending), BIC (ascending)
2. Compute: `avg_rank = (log_loss_rank + AUC_rank) / 2`
3. Sort by avg_rank ascending → row 1 = best model
4. Save to: `results_{topic}.csv`

### Step 4: Output

For the best model per topic:
- Print: topic, distribution name, bin_size, parameters, log_loss, AUC
- Plot:
    - Scatter: empirical bin probabilities (group by mid_bin, mean of final_label)
    - Line: fitted curve using best params
    - Title: "Topic {name}: {distribution} (bins={n})"
    - X-label: "Time (days)", Y-label: "P(matching)"
- Plotting is optional if `matplotlib` is unavailable in the runtime environment; in that case, wrap the import in `try/except ImportError`, skip plotting, and still save all CSV outputs and the summary file.

### Step 5: Summary

After all topics:
- Print summary table: Topic | Best Distribution | Params | Log-Loss | AUC
- Save to: `currency_model_summary.csv`

## Error Handling Rules
- Catch fitting errors silently (skip that combination)
- If ALL distributions fail for a bin_size → skip it
- If a topic has < 50 rows after prep → skip with warning
- Never let a single error crash the full loop
