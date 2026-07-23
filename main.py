
from langgraph.graph import StateGraph, END

from graph.state import PipelineState, PHASES
from graph.nodes import manager_node, doer_node, evaluator_node
from graph.routing import (
    route_after_evaluation,
    advance_phase,
    retry_phase,
    escalate_to_human,
    check_pipeline_complete,
)


def build_pipeline():
    """Construct and compile the LangGraph pipeline."""

    graph = StateGraph(PipelineState)

    # --- Add nodes ---
    graph.add_node("manager", manager_node)
    graph.add_node("doer", doer_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("advance", advance_phase)
    graph.add_node("retry", retry_phase)
    graph.add_node("escalate", escalate_to_human)

    # --- Set entry point ---
    graph.set_entry_point("manager")

    # --- Define edges ---

    # Manager always sends to Doer
    graph.add_edge("manager", "doer")

    # Doer always sends to Evaluator
    graph.add_edge("doer", "evaluator")

    # Evaluator → conditional routing based on verdict
    graph.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "advance": "advance",
            "retry": "retry",
            "escalate": "escalate",
        },
    )

    # After advance → check if all phases done
    graph.add_conditional_edges(
        "advance",
        check_pipeline_complete,
        {
            "continue": "manager",
            "complete": END,
        },
    )

    # After retry → back to Manager (with accumulated feedback)
    graph.add_edge("retry", "manager")

    # Escalate → END (human takes over)
    graph.add_edge("escalate", END)

    return graph.compile()


def run():
    """Execute the pipeline."""

    print("" + "=" * 60)
    print("  MULTI-AGENT CURRENCY ASSESSMENT PIPELINE")
    print("=" * 60)
    print(f"  Phases: {' -> '.join(PHASES)}")
    print(f"  Max retries per phase: 3")
    print("=" * 60 + "")

    # Initial state
    initial_state = {
        "domain": "",
        "domain_signals": "",
        "current_phase": 0,
        "retry_count": 0,
        "doer_task_prompt": "",
        "dataset_stats": "",
        "doer_output": "",
        "execution_result": "",
        "evaluator_verdict": "",
        "evaluator_feedback": "",
        "feedback_history": [],
        "phase_outputs": {},
        "verdict_history": [],
        "needs_human": False,
        "human_report": "",
    }

    # Build and run
    pipeline = build_pipeline()
    final_state = pipeline.invoke(initial_state)

    # --- Print final report ---
    print("" + "=" * 60)
    print("  PIPELINE EXECUTION SUMMARY")
    print("=" * 60)

    if final_state["needs_human"]:
        print("Status: HALTED — Human intervention required")
        print(final_state["human_report"])
    else:
        print("Status: COMPLETED SUCCESSFULLY")
        print("Results per phase:")
        for phase, result in final_state["phase_outputs"].items():
            print(f"    {phase}: {result['verdict']} (attempts: {result['attempts']})")

    print("Full verdict history:")
    for entry in final_state["verdict_history"]:
        print(f"    [{entry['phase']}] Attempt {entry['attempt']}: {entry['verdict']}")

    print("" + "=" * 60)

    return final_state


if __name__ == "__main__":
    run()

