
# -------------------------------------------------------------------------------------------------------------------------------
# Routing and State Transitions                                                                                    
# -------------------------------------------------------------------------------------------------------------------------------
# Deterministic routing logic after Evaluator verdict. Controls pipeline flow:                                                   
#                                                                                                                               
#   PASS → advance_phase: reset state, move to next stage                                                                       
#   FAIL → retry_phase: append feedback, increment retry count (max 3 attempts)                                                 
#   FAIL (after 3 attempts) → escalate_to_human: generate escalation report                                                     
#   PASS_WITH_WARNING → human review: accept as-is or manually edit script and re-run                                           
#                                                                                                                               
# Also contains check_pipeline_complete to determine if all stages are done.                                                    
# MAX_RETRIES = 3 (bounded retry logic before escalation)                                                                       
# -------------------------------------------------------------------------------------------------------------------------------


import json
from graph.state import PipelineState, PHASES

MAX_RETRIES = 3


def route_after_evaluation(state: PipelineState):
    """
    Deterministic routing based on Evaluator's verdict.
    Returns: "advance", "retry", or "escalate"
    """
    phase_name = PHASES[state["current_phase"]]
    verdict = state["evaluator_verdict"]

    if verdict == "PASS":
        return "advance"

    elif verdict == "PASS_WITH_WARNING":
        print("" + "=" * 60)
        print("[HUMAN REVIEW] Evaluator passed with warnings.")
        print("=" * 60)

        # Print recommendations
        try:
            eval_data = json.loads(state.get("evaluator_feedback", "{}"))
            if eval_data.get("required_actions"):
                print("Recommended actions:")
                for action in eval_data["required_actions"]:
                    print(f"  {action}")
            print(f"Generated script: logs/{phase_name}_attempt_{state['retry_count'] + 1}_code.py")
        except:
            pass

        print("Options:")
        print("   Accept as-is — advance to next phase")
        print("   I'll edit the script manually — then re-run validation")
        choice = input("Your choice (1 or 2): ").strip()

        if choice == "2":
            input("Edit the script, then press Enter to re-run validation...")
            # Re-run executor + validator on the edited script file.
            from graph.executor import execute_code
            from graph.validator import generate_phase_summary
            script_path = f"outputs/doer_raw/{phase_name}_attempt_{state['retry_count'] + 1}.py"

            with open(script_path, "r", encoding="utf-8") as f:
                edited_code = f.read()

            result = execute_code(edited_code, phase_name, state["retry_count"] + 1)
            if result["success"]:
                summary = generate_phase_summary(phase_name)
                print("[HUMAN] Script re-executed successfully. Advancing.")
            state["current_phase"] += 1
        else:
            # Accept as-is, advance
            state["current_phase"] += 1

        state["retry_count"] = 0


    elif verdict == "FAIL":
        if state["retry_count"] >= MAX_RETRIES - 1:
            return "escalate"
        else:
            return "retry"

    # Unknown verdict — escalate to be safe
    return "escalate"


def advance_phase(state: PipelineState) -> PipelineState:
    """Move to next phase after PASS verdict."""
    phase_name = PHASES[state["current_phase"]]
    print(f"[ADVANCE] Phase '{phase_name}' PASSED (attempts: {state['retry_count'] + 1})")

    # Store result
    state["phase_outputs"][phase_name] = {
        "verdict": "PASS",
        "attempts": state["retry_count"] + 1,
    }

    # Reset for next phase
    state["current_phase"] += 1
    state["retry_count"] = 0
    state["feedback_history"] = []
    state["doer_task_prompt"] = ""
    state["doer_output"] = ""
    state["evaluator_verdict"] = ""
    state["evaluator_feedback"] = ""

    return state


def retry_phase(state: PipelineState) -> PipelineState:
    """Retry current phase after FAIL verdict."""
    phase_name = PHASES[state["current_phase"]]
    print(f"[RETRY] Phase '{phase_name}' FAILED — attempt {state['retry_count'] + 2} of {MAX_RETRIES}")

    # Accumulate feedback
    state["feedback_history"].append(state["evaluator_feedback"])
    state["retry_count"] += 1


    return state


def escalate_to_human(state: PipelineState) -> PipelineState:
    """Escalate to human after 3 failures or PASS_WITH_WARNING."""
    phase_name = PHASES[state["current_phase"]]
    verdict = state["evaluator_verdict"]

    if verdict == "PASS_WITH_WARNING":
        reason = "PASS WITH WARNING — subjective concern requires human review"
    else:
        reason = f"{MAX_RETRIES} consecutive failures — automated resolution exhausted"

    print(f"{'!'*60}")
    print(f"[ESCALATE] Phase: {phase_name} | Reason: {reason}")
    print(f"{'!'*60}")

    # Build report
    report = f"""{'='*60} ESCALATION REPORT {'='*60}

Phase: {phase_name}
Reason: {reason}
Total Attempts: {state['retry_count'] + 1}

--- VERDICT HISTORY ---
"""
    for entry in state["verdict_history"]:
        if entry["phase"] == phase_name:
            report += f"  Attempt {entry['attempt']}: {entry['verdict']}"
            report += f"  Summary: {entry['summary']}"

    report += f"""
--- LATEST FEEDBACK ---
{state['evaluator_feedback']}
"""

    state["needs_human"] = True
    state["human_report"] = report

    return state


def check_pipeline_complete(state: PipelineState):
    """Check if all phases are done. Returns 'continue' or 'complete'."""
    if state["current_phase"] >= len(PHASES):
        print(f"{'='*60}")
        print("[COMPLETE] All phases finished successfully!")
        print(f"{'='*60}")
        return "complete"
    return "continue"

