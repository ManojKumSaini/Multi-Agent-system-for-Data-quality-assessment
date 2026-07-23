
# -------------------------------------------------------------------------------------------------------------------------------
# Pipeline State Definition                                                                                                     
# -------------------------------------------------------------------------------------------------------------------------------
# Defines the execution sequence (PHASES) and shared state (PipelineState) for the LangGraph pipeline.                          
# State persists in memory across all node transitions within a single run.                                                     
#                                                                                                                               
# PHASES list determines execution order — index = current_phase in state.                                                      
# Stages: preprocessing (0&1), topic_modelling (2), semantic_similarity (3), data_labelling (4), stats_est (5)                  
#                                                                                                                               
# PipelineState carries: phase tracking, agent I/O, feedback accumulation,                                                      
# per-phase outputs, audit trail (verdict_history), and escalation flags.                                                       
# -------------------------------------------------------------------------------------------------------------------------------

from typing import TypedDict


# Phase names — order matters (pipeline executes in this sequence)
PHASES = [
    "preprocessing",        # Stage 0 & 1 (domain discovery happens here too)
    "topic_modelling",      # Stage 2
    "semantic_similarity",  # Stage 3
    "data_labelling",       # Stage 4
    "stats_est",            # Stage 5 (survival analysis)
]

class PipelineState(TypedDict):
    """All data that flows between nodes in the graph."""

    domain: str                 
    domain_signals: str        

    # --- Phase tracking ---
    current_phase: int          # Index into PHASES list (0-5)
    retry_count: int            # Retries at current phase (0, 1, 2)

    # --- Agent I/O ---
    doer_task_prompt: str      
    dataset_stats: str          
    doer_output: str            
    execution_result: str       
    evaluator_verdict: str      
    evaluator_feedback: str     

    # --- Feedback accumulation ---
    feedback_history: list     

    # --- Outputs per phase ---
    phase_outputs: dict        

    # --- Audit trail ---
    verdict_history: list      

    # --- Escalation ---
    needs_human: bool
    human_report: str

