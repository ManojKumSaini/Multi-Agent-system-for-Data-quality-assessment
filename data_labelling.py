
# -------------------------------------------------------------------------------------------------------------------------------
# Stage 4: Data Labelling Pipeline
# -------------------------------------------------------------------------------------------------------------------------------
# Batch-based binary labelling of semantic text pairs using Doer + Evaluator agents.                                             
# Unlike other stages, this runs as a self-contained LangGraph subgraph with its own state.                                     
#                                                                                                                               
# Process:                                                                                                                      
#   Step 0 — Generate deterministic pair_id (MD5 hash of text1 + text2) for deduplication                                       
#   Step 1 — SBERT auto-pass: pairs with similarity > 0.9999 labelled True automatically                                       
#   Step 2 — Batch loop (40 unique pairs per batch):                                                                            
#            Doer labels pairs → low-confidence (≤3) flagged directly for human review                                          
#            → 10% sample of high-confidence (≥4) sent to Evaluator for verification                                            
#            → Evaluator disagreements also flagged for human review                                                             
#            → results propagated to all rows sharing same pair_id (duplicate handling)                                          
#            → 2 consecutive all-false batches triggers short-circuit (remaining = False)                                        
#   Step 3 — Save human review file (low-confidence + Evaluator-disagreed items)                                                
#                                                                                                                               
# Agents: Doer (Qwen 3 8B via OpenRouter), Evaluator (GPT OSS 120B via OpenRouter)                                             
# Input: outputs/sbert_output/{topic}/topic.csv (from Stage 3)                                                                  
# Output: outputs/data_labelling/{topic}/topic_labeled.csv + human_review.csv                                                   
# -------------------------------------------------------------------------------------------------------------------------------

import os
import json
import random
import hashlib
from pathlib import Path
import pandas as pd
from langgraph.graph import StateGraph, END
from graph.state import PipelineState
from agents.api import call_agent

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PROMPTS LOADED ONCE AT MODULE LEVEL (persistent agent configuration)
_DOER_PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "tasks", "data_labelling.md")
_EVAL_PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "tasks", "data_labelling_eval.md")

try:
    with open(_DOER_PROMPT_PATH, "r", encoding="utf-8") as f:
        DOER_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    DOER_SYSTEM_PROMPT = (
        "You are a data labeling agent. Compare text pairs. "
        "Return a raw JSON array with keys: 'id', 'label' (true/false), 'confidence' (1-5)."
    )

try:
    with open(_EVAL_PROMPT_PATH, "r", encoding="utf-8") as f:
        EVAL_SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    EVAL_SYSTEM_PROMPT = (
        "You are an expert evaluator. Review each labeled text pair. "
        "Return ONLY a raw JSON array with keys: 'id', 'verdict' ('correct' or 'incorrect')."
    )

class BatchLabelingState(PipelineState):
    batch_pairs: list              # Unique pairs sent to doer agent this batch
    batch_results: list            # Parsed JSON array from doer
    evaluator_items: list          # Items selected for evaluator review
    evaluator_results: list        # Parsed JSON array from evaluator
    all_false_batches_count: int   # Consecutive all-false batch counter
    human_review_items: list       # Items flagged for human review after topic finishes


# UTILITY: Generate pair_id from text1 + text2
def make_pair_id(text1: str, text2: str) -> str:
    """
    Deterministic hash of text1 + text2 (exact string, no flipping).
    Two rows get the same pair_id only if both text1 AND text2 are identical.
    NaN values are normalised to empty string to avoid false collisions.
    """
    t1 = "" if pd.isna(text1) else str(text1).strip()
    t2 = "" if pd.isna(text2) else str(text2).strip()
    raw = f"{t1}|||{t2}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


# NODE 1: DOER AGENT — labels all unique pairs in the batch
def batch_data_labeling_node(state: BatchLabelingState) -> dict:
    """Sends batch of unique pairs to Qwen-8B doer agent, returns labeled results."""
    if not state["batch_pairs"]:
        return {
            "batch_results": [],
            "all_false_batches_count": 0,
            "evaluator_verdict": "PASS"
        }

    print(f"\n[DOER] Sending {len(state['batch_pairs'])} unique pairs to Qwen-8B...")

    # Load doer system prompt
    system_prompt_content = DOER_SYSTEM_PROMPT

    formatted_pairs = []
    for item in state["batch_pairs"]:
        formatted_pairs.append(
            f"ID: {item['row_idx']}\nText 1: {item['text1']}\nText 2: {item['text2']}\n---"
        )

    task_prompt_content = (
        "Task: Classify semantic text similarity based on system guidelines.\n"
        "Return ONLY a clean JSON array matching the data item IDs. "
        "No markdown wrapper, no backticks, no explanation.\n\n"
        "Data Batch:\n" + "\n".join(formatted_pairs)
    )

    raw_response = call_agent(
        role="doer",
        system_prompt=system_prompt_content,
        user_prompt=task_prompt_content
    )

    print("\n" + "=" * 50)
    print("[RAW DOER OUTPUT]")
    print(raw_response)
    print("=" * 50 + "\n")

    # Clean markdown fences if present
    if raw_response and "```" in raw_response:
        segments = raw_response.split("```")
        for segment in segments:
            cleaned = segment.strip()
            if cleaned.startswith("json"):
                raw_response = cleaned[4:].strip()
                break
            elif cleaned.startswith("["):
                raw_response = cleaned
                break

    # Parse JSON
    try:
        llm_output = json.loads(raw_response)
        print(f"Doer response parsed successfully ({len(llm_output)} items).")
    except Exception as e:
        print(f" Doer JSON Parsing Error: {e}")
        print(f"Raw output was:\n{raw_response}")
        llm_output = []

    normalized_output = []
    for res in llm_output:
        if not isinstance(res, dict):
            continue
        normalized_item = {}
        
        # Pull ID safely regardless of capitalization
        normalized_item["id"] = res.get("id", res.get("ID", res.get("Id")))
        
        # Pull Label safely
        normalized_item["label"] = res.get("label", res.get("Label", res.get("LABEL", False)))
        
        # Pull Confidence or alternative naming configurations
        normalized_item["confidence"] = res.get(
            "confidence", 
            res.get("confidence_score", 
                    res.get("Confidence Score", res.get("Confidence", 5)))
        )
        
        normalized_output.append(normalized_item)
    
    llm_output = normalized_output

    # Normalise label strings to booleans
    for res in llm_output:
        raw_label = res.get("label")
        if isinstance(raw_label, str):
            res["label"] = raw_label.strip().lower() == "true"

    # All-false batch detection
    
    all_false = all(res.get("label") == False for res in llm_output) if llm_output else False
    new_false_count = state["all_false_batches_count"] + 1 if all_false else 0
    if all_false:
        print(f"Batch is 100% FALSE. Consecutive all-false count: {new_false_count}")

    # Build evaluator queue safely
    # Separate by confidence
    low_conf = [r for r in llm_output if int(r.get("confidence", 5)) <= 3]
    high_conf = [r for r in llm_output if int(r.get("confidence", 5)) >= 4]

    # Low-confidence items go directly to human review (not Evaluator)
    human_flagged = []
    for res in low_conf:
        if res.get("id") is None:
            continue
        try:
            matching_input = next(
                p for p in state["batch_pairs"] if int(p["row_idx"]) == int(res["id"])
            )
            human_flagged.append({
                "row_idx": res["id"],
                "text1": matching_input["text1"],
                "text2": matching_input["text2"],
                "doer_label": res.get("label", False),
                "confidence": res.get("confidence", 5),
                "reason": "low_confidence"
            })
        except StopIteration:
            pass

    # Only 10% sample of high-confidence items goes to Evaluator
    high_conf_sample = []
    if high_conf:
        sample_size = max(1, round(len(high_conf) * 0.10))
        high_conf_sample = random.sample(high_conf, min(sample_size, len(high_conf)))

    evaluator_queue = []
    for res in high_conf_sample:
        if res.get("id") is None:
            continue
        try:
            matching_input = next(
                p for p in state["batch_pairs"] if int(p["row_idx"]) == int(res["id"])
            )
            evaluator_queue.append({
                "row_idx": res["id"],
                "text1": matching_input["text1"],
                "text2": matching_input["text2"],
                "doer_label": res.get("label", False),
                "confidence": res.get("confidence", 5)
            })
        except StopIteration:
            pass

    print(
        f"Evaluator queue: {len(high_conf_sample)} sampled high-conf items | "
        f"Human review: {len(human_flagged)} low-conf items"
    )


    return {
        "batch_results": llm_output,
        "evaluator_items": evaluator_queue,
        "human_review_items": human_flagged,
        "all_false_batches_count": new_false_count,
        "evaluator_verdict": "PASS"}

# NODE 2: EVALUATOR AGENT — reviews selected items, says correct/incorrect
def evaluator_node(state: BatchLabelingState) -> dict:
    """Sends filtered items to GPT-o3 evaluator. Returns correct/incorrect per item."""
    if not state["evaluator_items"]:
        print("\n[EVALUATOR] Nothing to review this batch.")
        return {
            "evaluator_results": [],
            "evaluator_verdict": "PASS"
        }

    print(f"\n[EVALUATOR] Reviewing {len(state['evaluator_items'])} items with GPT-o3...")

    # Load evaluator system prompt
    eval_system_prompt = EVAL_SYSTEM_PROMPT

    formatted_items = []
    for item in state["evaluator_items"]:
        formatted_items.append(
            f"ID: {item['row_idx']}\n"
            f"Text 1: {item['text1']}\n"
            f"Text 2: {item['text2']}\n"
            f"Assigned Label: {item['doer_label']}\n"
            f"Confidence: {item['confidence']}\n---"
        )

    eval_task_prompt = (
        "Review each labeled pair below. For each, return ONLY 'correct' or 'incorrect'.\n"
        "Return ONLY a clean JSON array. No markdown, no backticks, no explanation.\n\n"
        "Items:\n" + "\n".join(formatted_items)
    )

    raw_eval_response = call_agent(
        role="evaluator",
        system_prompt=eval_system_prompt,
        user_prompt=eval_task_prompt
    )

    print("\n" + "=" * 50)
    print("[RAW EVALUATOR OUTPUT]")
    print(raw_eval_response)
    print("=" * 50 + "\n")

    # Clean markdown fences if present
    if raw_eval_response and "```" in raw_eval_response:
        segments = raw_eval_response.split("```")
        for segment in segments:
            cleaned = segment.strip()
            if cleaned.startswith("json"):
                raw_eval_response = cleaned[4:].strip()
                break
            elif cleaned.startswith("["):
                raw_eval_response = cleaned
                break

    # Parse JSON
    try:
        eval_output = json.loads(raw_eval_response)
        print(f"Evaluator response parsed successfully ({len(eval_output)} items).")
    except Exception as e:
        print(f"Evaluator JSON Parsing Error: {e}")
        print(f"Raw output was:\n{raw_eval_response}")
        eval_output = []

    normalized_eval_output = []
    for res in eval_output:
        if not isinstance(res, dict):
            continue
        normalized_res = {}
        
        # Pull ID safely regardless of capitalization
        normalized_res["id"] = res.get("id", res.get("ID", res.get("Id")))
        
        # Pull Verdict safely and force to a lowercase string
        raw_verdict = res.get("verdict", res.get("Verdict", res.get("VERDICT", "correct")))
        if isinstance(raw_verdict, str):
            normalized_res["verdict"] = raw_verdict.strip().lower()
        else:
            normalized_res["verdict"] = "correct"
            
        normalized_eval_output.append(normalized_res)
        
    eval_output = normalized_eval_output

    return {
        "evaluator_results": eval_output,
        "evaluator_verdict": "PASS"
    }
# GRAPH DEFINITION
def build_pipeline():
    graph = StateGraph(BatchLabelingState)

    graph.add_node("batch_labeler", batch_data_labeling_node)
    graph.add_node("evaluator",     evaluator_node)

    graph.set_entry_point("batch_labeler")
    graph.add_edge("batch_labeler", "evaluator")
    graph.add_edge("evaluator",     END)

    return graph.compile()

# MAIN RUNNER
def run_batch_experiment(topic_name: str = "topic_17"):
    print("=" * 60)
    print("  PHASE 4: LIVE SBERT + QWEN-8B + GPT-o3 LABELING PIPELINE")
    print("=" * 60)

    pipeline = build_pipeline()

    input_csv_path   = os.path.join("outputs", "sbert_output",   topic_name, "topic.csv")
    output_dir       = os.path.join("outputs", "data_labelling", topic_name)
    output_save_path = os.path.join(output_dir, "topic_labeled.csv")
    human_review_path = os.path.join(output_dir, "human_review.csv")

    if not os.path.exists(input_csv_path):
        print(f"Error: Input path '{input_csv_path}' not found.")
        return

    df = pd.read_csv(input_csv_path)

    required_cols = ['text1', 'text2', 'text_similarity_score']
    if not all(c in df.columns for c in required_cols):
        print(f"Error: CSV must contain columns: {required_cols}")
        return

    # STEP 0: GENERATE pair_id FOR EVERY ROW (once, on full dataset)
    # pair_id = md5(text1 + "|||" + text2) — exact strings, no flipping
    # Two rows share a pair_id only if both text1 AND text2 are identical.
    print("\n Step 0: Generating pair_ids for full dataset...")
    df['pair_id'] = df.apply(
        lambda row: make_pair_id(row['text1'], row['text2']), axis=1
    )
    total_rows        = len(df)
    unique_pair_count = df['pair_id'].nunique()
    duplicate_count   = total_rows - unique_pair_count
    print(f"pair_ids generated. Total rows: {total_rows} | "
          f"Unique pairs: {unique_pair_count} | Duplicates: {duplicate_count}")

    # Drop entity columns — not needed in labeled output
    df.drop(columns=[c for c in ['entity1', 'entity2'] if c in df.columns], inplace=True)

    # Initialise output columns if not present
    for col, default in [('label', None), ('confidence', None),
                         ('needs_human_review', False), ('evaluator_verdict', None)]:
        if col not in df.columns:
            df[col] = default

    # STEP 1: SBERT AUTO-PASS (score > 0.9999 → True, confidence 5)
    # Applied to all rows; duplicates are covered here too via pair_id
    # propagation in Step 2.
    print("\n Step 1: SBERT auto-pass scan (score > 0.9999)...")
    auto_pass_mask = df['text_similarity_score'].astype(float) > 0.9999
    df.loc[auto_pass_mask, 'label']     = True
    df.loc[auto_pass_mask, 'confidence'] = 5
    print(f"Auto-labeled {auto_pass_mask.sum()} rows as TRUE.")

    # STEP 2: BATCH LOOP
    # Each iteration collects the next 40 UNIQUE pair_ids that are still
    # unlabeled, sends only those to the doer, then copy-pastes results
    # to ALL rows sharing the same pair_id in the full DataFrame.
    print("\n Step 2: Batch processing (unique pair_id batches of 40)...")
    consecutive_false_batches = 0
    batch_size                = 40
    master_human_review_log   = []
    labeled_pair_ids          = set()   # Tracks pair_ids labeled so far this run
    batch_number              = 0

    while True:
        # Find all rows still unlabeled
        unlabeled_df = df[df['label'].isna()]
        if unlabeled_df.empty:
            print("\n All rows labeled. Exiting loop.")
            break

        # Short-circuit: 2 consecutive all-false batches
        if consecutive_false_batches >= 2:
            print("[SHORT-CIRCUIT] 2 consecutive all-false batches.")
            print("Auto-labeling all remaining unlabeled rows as FALSE.")
            remaining_mask = df['label'].isna()
            df.loc[remaining_mask, 'label']      = False
            df.loc[remaining_mask, 'confidence'] = 5
            break

        # Collect next 40 unique unseen pair_ids from unlabeled rows
        unique_batch_pairs = []
        seen_in_this_batch = set()
        for idx, row in unlabeled_df.iterrows():
            pid = row['pair_id']
            if pid not in labeled_pair_ids and pid not in seen_in_this_batch:
                unique_batch_pairs.append({
                    "text1":    str(row['text1']),
                    "text2":    str(row['text2']),
                    "row_idx":  int(idx),
                    "pair_id":  pid
                })
                seen_in_this_batch.add(pid)
            if len(unique_batch_pairs) == batch_size:
                break

        if not unique_batch_pairs:
            print("\n No more unique unlabeled pairs. Exiting loop.")
            break

        batch_number += 1
        print(f"\n{'─'*60}")
        print(f"Batch #{batch_number} — {len(unique_batch_pairs)} unique pairs "
              f"(covering {len(unlabeled_df)} unlabeled rows remaining)")
        print(f"{'─'*60}")

        initial_state = {
            "current_phase":           4,
            "retry_count":             0,
            "batch_pairs":             unique_batch_pairs,
            "batch_results":           [],
            "evaluator_items":         [],
            "evaluator_results":       [],
            "all_false_batches_count": consecutive_false_batches,
            "human_review_items":      [],
            "feedback_history":        [],
            "phase_outputs":           {},
            "verdict_history":         []
        }

        final_state = pipeline.invoke(initial_state)

        consecutive_false_batches = final_state.get("all_false_batches_count", 0)
        doer_results              = final_state.get("batch_results",     [])
        evaluator_results         = final_state.get("evaluator_results", [])
        evaluator_items           = final_state.get("evaluator_items",   [])
        human_flagged_items       = final_state.get("human_review_items", [])
        master_human_review_log.extend(human_flagged_items)


        # Build evaluator verdict lookup: row_idx → verdict
        eval_verdict_map = {
            res.get("id"): res.get("verdict", "correct")
            for res in evaluator_results
        }

        # Build doer result lookup: row_idx → {label, confidence}
        doer_result_map = {
            res.get("id"): res for res in doer_results
        }

        # ── Propagate results to ALL rows sharing each pair_id ─────────
        returned_pair_ids = set()
        for item in unique_batch_pairs:
            row_idx = item["row_idx"]
            pid     = item["pair_id"]
            res     = doer_result_map.get(row_idx)

            if res is None:
                # Doer silently dropped this pair — default + flag
                all_rows_with_pid = df[df['pair_id'] == pid].index
                df.loc[all_rows_with_pid, 'label']             = False
                df.loc[all_rows_with_pid, 'confidence']         = 1
                df.loc[all_rows_with_pid, 'needs_human_review'] = True
                df.loc[all_rows_with_pid, 'evaluator_verdict']  = "not_returned"
                print(f"pair_id {pid[:8]}… not returned by doer — defaulting False, flagged.")
                labeled_pair_ids.add(pid)
                continue

            label      = res.get("label", False)
            confidence = res.get("confidence", 5)
            verdict    = eval_verdict_map.get(row_idx)
            ev_label   = verdict if verdict else "not_reviewed"

            # Copy-paste to ALL rows with this pair_id
            all_rows_with_pid = df[df['pair_id'] == pid].index
            df.loc[all_rows_with_pid, 'label']             = label
            df.loc[all_rows_with_pid, 'confidence']         = confidence
            df.loc[all_rows_with_pid, 'evaluator_verdict']  = ev_label

            dup_count = len(all_rows_with_pid) - 1
            if dup_count > 0:
                print(f"pair_id {pid[:8]}… → copied result to {dup_count} duplicate row(s).")

            # Flag for human review if evaluator said INCORRECT
            if verdict == "incorrect":
                df.loc[all_rows_with_pid, 'needs_human_review'] = True
                matching = next(
                    (i for i in evaluator_items if i["row_idx"] == row_idx), {}
                )
                master_human_review_log.append({
                    "row_idx":           row_idx,
                    "pair_id":           pid,
                    "text1":             matching.get("text1", ""),
                    "text2":             matching.get("text2", ""),
                    "doer_label":        label,
                    "confidence":        confidence,
                    "evaluator_verdict": "incorrect"
                })
                print(f"pair_id {pid[:8]}… flagged for human review (evaluator: incorrect).")

            returned_pair_ids.add(pid)
            labeled_pair_ids.add(pid)

        # Periodic save after every batch
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_save_path, index=False)
        print(f"Batch #{batch_number} saved. "
              f"Human review count so far: {len(master_human_review_log)}")

    # STEP 3: SAVE HUMAN REVIEW FILE (human fills in after topic finishes)
    if master_human_review_log:
        human_df = pd.DataFrame(master_human_review_log)
        human_df['human_label'] = None   # Human fills this in
        human_df['human_notes'] = None
        human_df.to_csv(human_review_path, index=False)
        print(f"\n Human review file saved to: {human_review_path}")
        print(f"   → {len(master_human_review_log)} items require human correction.")
    else:
        print("\nNo items flagged for human review.")

    print(f"\nTopic labeling complete. Final output: {output_save_path}")


if __name__ == "__main__":
    run_batch_experiment()
