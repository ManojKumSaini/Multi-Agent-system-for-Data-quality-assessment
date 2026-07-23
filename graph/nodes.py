
# -------------------------------------------------------------------------------------------------------------------------------
# Pipeline Nodes: Manager + Doer                                                                                 
# -------------------------------------------------------------------------------------------------------------------------------
# Two LangGraph nodes that handle task preparation and code generation/execution.                                               
#                                                                                                                               
# manager_node:                                                                                                                 
#   - First attempt: loads task prompt + guide, injects dynamic dataset stats (computed via pandas)                              
#   - Domain discovery (Stage 0, first attempt only): Manager LLM classifies dataset domain                                     
#   - Retry: appends feedback history + previous code to prompt                                                                 
#   - After 2 failures: Manager LLM generates adjusted guidance (escalation)                                                    
#                                                                                                                               
# doer_node:                                                                                                                    
#   - Calls Doer LLM with assembled prompt from manager_node                                                                    
#   - Stage 2 (topic modelling): Doer returns JSON config (not executable code)                                                 
#   - All other stages: extracts code → executes via subprocess → stores result in state                                        
#   - On success: runs validator to generate structured summary for Evaluator                                                   
#   - Logs raw responses and generated code to logs/ directory                                                                  
#                                                                                                                               
# State flow: manager_node → doer_node → evaluator_node                                                       
#
# Evaluator Node (LangGraph)                                                                                                    
# -------------------------------------------------------------------------------------------------------------------------------
# Independently verifies the Doer's output using phase-specific evaluation criteria.                                            
# Separation from Doer prevents self-evaluation bias (the Evaluator never sees the task prompt).                                
#                                                                                                                               
# Verdicts: PASS | FAIL | PASS_WITH_WARNING                                                                                    
#   - PASS: all criteria satisfied, pipeline advances to next stage                                                             
#   - FAIL: criteria not met, triggers retry (feedback appended to next Doer attempt)                                           
#   - PASS_WITH_WARNING: technically met but subjective concerns flagged for human review                                       
#                                                                                                                               
# Stage 2 (topic modelling): evaluates parameter selection against dataset stats                                                
# All other stages: evaluates code + execution result + validation summary                                                      
#                                                                                                                               
# Response parsing: JSON extraction first, keyword fallback if JSON malformed                                                   
# All verdicts logged to logs/{phase}_attempt_{n}_eval.json                                                                     
# -------------------------------------------------------------------------------------------------------------------------------



import json
import os
import re
from pathlib import Path
from signal import Signals
from graph.state import PipelineState, PHASES
from graph.prompts import (
    load_file,
    load_system_prompt,
    load_task_prompt,
    load_evaluation_criteria,
    load_guide,
)
from graph.executor import extract_code_from_response, execute_code
from agents.api import call_agent


def _json_safe(value):
    """Recursively convert common non-JSON-native values into plain Python types."""
    try:
        import numpy as np
    except Exception:
        np = None

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]
    return value



def manager_node(state: PipelineState) -> PipelineState:
    """
    Manager prepares the task for the Doer.
    - First attempt: loads task prompt + guide
    - Retry: appends accumulated feedback
    - After 2 failures: calls Manager LLM for adjusted guidance
    """
    phase_name = PHASES[state["current_phase"]]
    print(f"{'='*60}")
    print(f"[MANAGER] Phase: {phase_name} | Attempt: {state['retry_count'] + 1}")
    print(f"{'='*60}")

    # Domain discovery — only for preprocessing on first attempt
    if (
        phase_name == "preprocessing"
        and state["current_phase"] == 0
        and state["retry_count"] == 0
        and not state.get("domain")
    ):
        print("[MANAGER] Identifying dataset domain...")
        try:
            import pandas as pd

            df_sample = pd.read_excel("data/NIFTY.xlsx",nrows=50)
            sample_rows = df_sample["news"].dropna().head(20).tolist()
            sample_text = "\n".join(f"- {row[:300]}" for row in sample_rows)
        except Exception as e:
            sample_text = f"[Could not load sample: {e}]"

        domain_prompt = f"""
        You are a dataset classification expert.

        Analyze the following dataset samples and classify the dataset into EXACTLY ONE of the following domains:

        - Financial News
        - Medical Records
        - Legal Documents
        - Scientific Literature
        - Social Media Content
        - General News
        - Technical Documentation

        Do not create a new domain.

        Also identify domain-specific signals that should be preserved during preprocessing.

        Examples:
        Financial News -> %, $, ₹, stock tickers, market indices
        Medical Records -> ICD codes, dosage units, measurements
        Legal Documents -> section numbers, article references
        Social Media Content -> hashtags, mentions, emojis

        Dataset samples:

        {sample_text}

        Reply exactly in this format:

        DOMAIN: <one domain from the list>
        CONFIDENCE: <0-100>
        SIGNALS: <comma-separated list>
        """

        manager_system = load_system_prompt("manager")
        domain_response = call_agent("manager", manager_system, domain_prompt)

        if domain_response:
            for line in domain_response.strip().split("\n"):

                if line.upper().startswith("DOMAIN:"):
                    state["domain"] = line.split(":", 1)[1].strip()

                elif line.upper().startswith("CONFIDENCE:"):
                    try:
                        state["domain_confidence"] = int(
                            line.split(":", 1)[1].strip()
                        )
                    except:
                        state["domain_confidence"] = 0

                elif line.upper().startswith("SIGNALS:"):
                    state["domain_signals"] = line.split(":", 1)[1].strip()

            print(f"[MANAGER] Domain: {state['domain']}")
            print(f"[MANAGER] Confidence: {state['domain_confidence']}")
            print(f"[MANAGER] Signals: {state['domain_signals']}")

    # Load the base task prompt for this phase
    task_prompt = load_task_prompt(phase_name)

    if not task_prompt:
        state["needs_human"] = True
        state["human_report"] = f"Missing task prompt for phase: {phase_name}"
        return state

    if phase_name == "preprocessing":
        domain_context = (
            f"Domain: {state.get('domain', 'unknown')}\n"
            f"Signals to preserve: {state.get('domain_signals', 'none identified')}"
        )
        task_prompt = task_prompt.replace("{domain_context}", domain_context)

    if phase_name == "topic_modelling":
        import pandas as pd

        csv_path = "outputs/preprocessing/NIFTY_preprocessed.csv"
        if not os.path.exists(csv_path):
            print(f"[ERROR] Preprocessed file not found: {csv_path}")
            state["doer_task_prompt"] = "ERROR: Preprocessed file missing."
            return state

        df = pd.read_csv(csv_path)
        word_counts = df["preprocessed_text"].astype(str).str.split().str.len()

        dataset_stats = (
            f"- Total documents: {len(df)}\n"
            f"- Unique texts: {df['preprocessed_text'].nunique()}\n"
            f"- Average word count: {word_counts.mean():.1f}\n"
            f"- Median word count: {word_counts.median():.1f}\n"
            f"- Min word count: {word_counts.min()}\n"
            f"- Max word count: {word_counts.max()}\n"
            f"- Date range: {df['date'].min()} to {df['date'].max()}\n"
        )

        state["dataset_stats"] = dataset_stats
        task_prompt += f"\n\n## DATASET STATS\n{dataset_stats}"

    # Load phase-specific guide content and attach it to the Doer prompt.
    guide_content = load_guide(phase_name)

    if guide_content:
        task_prompt += f"\n\n## REFERENCE GUIDE\n{guide_content}"

    # If this is a retry, append feedback history
    if state["retry_count"] > 0 and state["feedback_history"]:
        previous_code_section = ""
        try:
            previous_output = json.loads(state.get("doer_output", "{}"))
            previous_code = previous_output.get("code", "")
            previous_execution = previous_output.get("execution", {})
            if previous_code:
                previous_code_section = (
                    "## PREVIOUS ATTEMPT CODE\n"
                    "Use this as the base and make the smallest possible edits to fix the listed issues.\n"
                    "Do not rewrite working parts from scratch.\n\n"
                    f"```python\n{previous_code}\n```\n"
                )
                if previous_execution:
                    previous_code_section += (
                        "## PREVIOUS EXECUTION CONTEXT\n"
                        f"- success: {previous_execution.get('success', False)}\n"
                        f"- return_code: {previous_execution.get('return_code', 'unknown')}\n"
                        f"- stderr: {str(previous_execution.get('stderr', ''))[:1500]}\n"
                    )
        except Exception:
            previous_code_section = ""

        feedback_section = "<CRITIQUE>"

        for i, fb in enumerate(state["feedback_history"], 1):
            feedback_section += f"### Attempt {i} Feedback:\n{fb}\n"

        # After 2 failed attempts, Manager provides adjusted guidance
        if state["retry_count"] >= 2:
            print("[MANAGER] 2 failures detected — generating adjusted guidance...")

            manager_system = load_system_prompt("manager")
            manager_input = (
                f"The Doer has failed {state['retry_count']} times at phase '{phase_name}'."
                f"Accumulated feedback:{feedback_section}"
                f"Provide concise, adjusted instructions to help the Doer succeed. "
                f"If the Evaluator's thresholds seem unrealistic for this data, "
                f"suggest relaxed but acceptable alternatives."
            )

            manager_guidance = call_agent("manager", manager_system, manager_input)

            if manager_guidance:
                feedback_section += f"### Manager Guidance (PRIORITY — follow this):\n{manager_guidance}\n"
                print(f"[MANAGER] Guidance provided ({len(manager_guidance)} chars)")

                os.makedirs("logs", exist_ok=True)
                manager_log_path = f"logs/{phase_name}_attempt_{state['retry_count'] + 1}_manager.txt"
                with open(manager_log_path, "w", encoding="utf-8") as f:
                    f.write("=== MANAGER INPUT ===\n")
                    f.write(manager_input)
                    f.write("\n\n=== MANAGER GUIDANCE ===\n")
                    f.write(manager_guidance)
                print(f"[MANAGER] Guidance saved: {manager_log_path}")

        feedback_section += "</CRITIQUE>"
        task_prompt += f"\n\n{previous_code_section}{feedback_section}"

    # Store prepared prompt in state for Doer
    state["doer_task_prompt"] = task_prompt

    return state

def doer_node(state: PipelineState) -> PipelineState:
    """
    Doer generates code and executes it.
    1. Calls LLM with task prompt
    2. Extracts code from response
    3. Executes code via subprocess
    4. Stores everything in state for Evaluator
    """
    phase_name = PHASES[state["current_phase"]]
    print(f"[DOER] Generating code for: {phase_name}")

    # Load system prompt
    if phase_name == "data_labelling":
        script_path = Path("data_labelling.py")
        if not script_path.exists():
            state["execution_result"] = json.dumps({"success": False, "error": "data_labelling.py not found"})
            state["doer_output"] = json.dumps({
                "error": "Standalone data_labelling.py script not found",
                "code": "",
                "execution": {"success": False, "stdout": "", "stderr": "Script missing"},
            })
            return state

        with script_path.open("r", encoding="utf-8") as f:
            code = f.read()

        execution_result = execute_code(
            code=code,
            phase_name=phase_name,
            attempt=state["retry_count"] + 1,
            timeout=120,
        )
        state["execution_result"] = json.dumps(_json_safe(execution_result), indent=2)

        doer_output = {
            "code": code,
            "execution": {
                "success": execution_result["success"],
                "stdout": execution_result["stdout"],
                "stderr": execution_result["stderr"],
                "return_code": execution_result["return_code"],
                "duration": execution_result["duration"],
                "script_path": execution_result["script_path"],
            },
        }

        if execution_result["success"]:
            from graph.validator import generate_phase_summary

            summary = generate_phase_summary(phase_name)
            if summary:
                doer_output["summary"] = summary
                print(f"[DOER] Validation summary generated for {phase_name}")

        state["doer_output"] = json.dumps(_json_safe(doer_output), indent=2)
        status = "SUCCESS" if execution_result["success"] else "FAILED"
        print(f"[DOER] Code: {len(code)} chars | Execution: {status}")

        os.makedirs("logs", exist_ok=True)
        code_log_path = f"logs/{phase_name}_attempt_{state['retry_count'] + 1}_code.py"
        with open(code_log_path, "w") as f:
            f.write(code)
        print(f"[DOER] Code saved: {code_log_path}")

        return state

    system_prompt = load_system_prompt("doer")

    # Get task prompt (prepared by Manager)
    task_prompt = state["doer_task_prompt"]

    # Call Doer LLM
    response = call_agent("doer", system_prompt, task_prompt)
    print(f"[DEBUG DOER RAW] First 1000 chars: {response[:1000] if response else 'NONE'}")

    if not response:
        state["execution_result"] = json.dumps({"success": False, "error": "No response from model"})
        state["doer_output"] = json.dumps({
            "error": "Doer LLM returned no response",
            "code": "",
            "execution": {"success": False, "stdout": "", "stderr": "No response from model"},
        })
        return state

    # Phase 2: Doer returns JSON config, not executable code.
    if phase_name == "topic_modelling":
        clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

        config = None
        try:
            config = json.loads(clean_response)
        except json.JSONDecodeError:
            json_match = re.search(r"```json\s*(.*?)```", clean_response, flags=re.DOTALL | re.IGNORECASE)
            if json_match:
                try:
                    config = json.loads(json_match.group(1).strip())
                except json.JSONDecodeError:
                    config = None

        if config is not None:
            os.makedirs("outputs/topic_modelling", exist_ok=True)
            config_path = "outputs/topic_modelling/config.json"
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            print(f"[DOER] Config saved: {config_path}")

            script_path = Path("topic_modeling_code.py")
            if script_path.exists():
                with script_path.open("r", encoding="utf-8") as f:
                    script_code = f.read()

                execution_result = execute_code(
                    code=script_code,
                    phase_name=phase_name,
                    attempt=state["retry_count"] + 1,
                    timeout=120,
                )
                state["execution_result"] = json.dumps(_json_safe(execution_result), indent=2)

                doer_output = {
                    "config": _json_safe(config),
                    "execution": {
                        "success": execution_result["success"],
                        "stdout": execution_result["stdout"],
                        "stderr": execution_result["stderr"],
                        "return_code": execution_result["return_code"],
                        "duration": execution_result["duration"],
                        "script_path": execution_result["script_path"],
                    },
                }

                if execution_result["success"]:
                    from graph.validator import generate_phase_summary

                    summary = generate_phase_summary(phase_name)
                    if summary:
                        doer_output["summary"] = summary
                        print(f"[DOER] Validation summary generated for {phase_name}")
            else:
                doer_output = {
                    "config": _json_safe(config),
                    "execution": {"success": True, "type": "config"},
                }

                state["execution_result"] = json.dumps({"success": True, "type": "config"})

            state["doer_output"] = json.dumps(_json_safe(doer_output), indent=2)
        else:
            print("[WARNING] Could not parse JSON from Doer response")
            state["doer_output"] = json.dumps({
                "error": "No valid JSON",
                "raw": clean_response[:2000],
            })
            state["execution_result"] = json.dumps({"success": False})

        os.makedirs("logs", exist_ok=True)
        raw_path = f"logs/{phase_name}_attempt_{state['retry_count'] + 1}_raw_response.txt"
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"[DOER] Raw response saved: {raw_path}")

        return state

    # Extract code from response
    code = extract_code_from_response(response)

    if not code:
        state["execution_result"] = json.dumps({"success": False, "error": "No executable code generated"})
        state["doer_output"] = json.dumps({
            "error": "No code block found in response",
            "raw_response": response[:3000],
            "code": "",
            "execution": {"success": False, "stdout": "", "stderr": "No executable code generated"},
        })
        return state

    # Execute the code
    execution_result = execute_code(
        code=code,
        phase_name=phase_name,
        attempt=state["retry_count"] + 1,
        timeout=120,
    )
    state["execution_result"] = json.dumps(_json_safe(execution_result), indent=2)

    # Package everything for the Evaluator
    doer_output = {
        "code": code,
        "execution": {
            "success": execution_result["success"],
            "stdout": execution_result["stdout"],
            "stderr": execution_result["stderr"],
            "return_code": execution_result["return_code"],
            "duration": execution_result["duration"],
            "script_path": execution_result["script_path"],
        },
    }

    # If execution succeeded, generate validation summary
    if execution_result["success"]:
        from graph.validator import generate_phase_summary
        summary = generate_phase_summary(phase_name)
        if summary:
            doer_output["summary"] = summary
            print(f"[DOER] Validation summary generated for {phase_name}")


    state["doer_output"] = json.dumps(_json_safe(doer_output), indent=2)
    status = "SUCCESS" if execution_result["success"] else "FAILED"
    print(f"[DOER] Code: {len(code)} chars | Execution: {status}")

    # Save doer's generated code to logs
    os.makedirs("logs", exist_ok=True)
    code_log_path = f"logs/{phase_name}_attempt_{state['retry_count'] + 1}_code.py"
    with open(code_log_path, "w") as f:
        f.write(code)
    print(f"[DOER] Code saved: {code_log_path}")


    return state


# ============================================================
# EVALUATOR NODE
# ============================================================

def evaluator_node(state: PipelineState) -> PipelineState:
    """
    Evaluator independently verifies the Doer's output.
    - Receives: code + execution results
    - Applies: phase-specific evaluation criteria
    - Returns: PASS / FAIL / PASS_WITH_WARNING + structured feedback
    """
    phase_name = PHASES[state["current_phase"]]
    print(f"[EVALUATOR] Evaluating phase: {phase_name}")

    # Load evaluator system prompt
    system_prompt = load_system_prompt("evaluator")

    if phase_name == "topic_modelling":
        eval_task = load_file(os.path.join("prompts", "tasks", "topic_modelling_eval_post.md"))
        if not eval_task:
            eval_task = load_evaluation_criteria(phase_name)

        eval_request = (
            f"{eval_task}\n\n"
            f"## DOER'S PARAMETER SELECTION:\n"
            f"{state['doer_output']}\n"
        )
        # Include the same dataset statistics the Manager sent to the Doer
        if state.get("dataset_stats"):
            eval_request += f"\n## DATASET STATS\n{state['dataset_stats']}\n"
    else:
        # Existing evaluator flow for execution-based phases
        eval_criteria = load_evaluation_criteria(phase_name)

        if not eval_criteria:
            eval_criteria = (
                "Evaluate the code for:"
                "1. Correctness — does it run without errors?"
                "2. Completeness — does it produce the expected output files?"
                "3. Quality — are the results reasonable?"
            )

        eval_request = (
            f"## DOER OUTPUT TO EVALUATE:"
            f"{state['doer_output']}"
            f"\n---\n"
            f"## EXECUTION RESULT:\n"
            f"{state.get('execution_result', '')}"
            f"\n---\n"
            f"## EVALUATION CRITERIA FOR PHASE: {phase_name.upper()}"
            f"{eval_criteria}"
            f"\n---\n"
            f"## INSTRUCTIONS:"
            f"Evaluate the Doer's output against the criteria above."
            f"Return your verdict in this exact JSON format:"
            f"```json"
            f"{{"
            f'    "verdict": "PASS" or "FAIL" or "PASS_WITH_WARNING",'
            f'    "checks": ['
            f'        {{"criterion": "description", "status": "pass" or "fail", "observed": "what you found", "expected": "what was required"}}'
            f"    ],"
            f'    "issues": "Summary of problems (empty string if PASS)",'
            f'    "recommendation": "What to fix (empty string if PASS)",'
            f'    "required_actions": "Specific changes needed (empty string if PASS)"'
            f"}}"
            f"```"
            f"VERDICT RULES:"
            f"- PASS: All criteria satisfied."
            f"- FAIL: One or more criteria not met, but fixable."
            f"- PASS_WITH_WARNING: Criteria technically met, but subjective concerns require human review."
        )

    if state.get("domain"):
        eval_request += (
            f"\n- Domain: {state['domain']}"
            f"\n- Signals to preserve: {state['domain_signals']}"
        )

    # Call Evaluator LLM
    response = call_agent("evaluator", system_prompt, eval_request)

    if not response:
        state["evaluator_verdict"] = "FAIL"
        state["evaluator_feedback"] = "Evaluator returned no response."
        state["verdict_history"].append({
            "phase": phase_name,
            "attempt": state["retry_count"] + 1,
            "verdict": "FAIL",
            "summary": "Evaluator returned no response",
        })

        return state

    # Parse verdict from response
    verdict, feedback = _parse_evaluator_response(response)

    state["evaluator_verdict"] = verdict
    state["evaluator_feedback"] = feedback

    # Record in audit trail
    state["verdict_history"].append({
        "phase": phase_name,
        "attempt": state["retry_count"] + 1,
        "verdict": verdict,
        "summary": feedback[:200],
    })


    print(f"[EVALUATOR] Verdict: {verdict}")

    # Save full evaluator response to log file
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/{phase_name}_attempt_{state['retry_count'] + 1}_eval.json"
    with open(log_path, "w") as f:
        f.write(feedback)
    print(f"[EVALUATOR] Response saved: {log_path}")

    # Print brief summary in terminal
    try:
        data = json.loads(feedback)
        if data.get("issues"):
            print(f"[EVALUATOR] Issues: {data['issues'][:200]}")
        if data.get("required_actions"):
            print(f"[EVALUATOR] Actions needed: {len(data['required_actions'])}")
    except:
        pass

    return state

# ============================================================
# HELPER: Parse Evaluator Response
# ============================================================

def _parse_evaluator_response(response):
    """
    Extract verdict and feedback from Evaluator's response.
    Tries JSON parsing first, falls back to keyword detection.
    Returns: (verdict_string, feedback_string)
    """
    # Try to extract JSON
    try:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1

        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            verdict = data.get("verdict", "FAIL").upper().strip()

            # Normalize verdict
            if "WARNING" in verdict:
                verdict = "PASS_WITH_WARNING"
            elif "PASS" in verdict and "FAIL" not in verdict:
                verdict = "PASS"
            else:
                verdict = "FAIL"

            feedback = json.dumps(data, indent=2)
            return verdict, feedback

    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: keyword detection
    response_upper = response.upper()

    if "PASS_WITH_WARNING" in response_upper or "PASS WITH WARNING" in response_upper:
        verdict = "PASS_WITH_WARNING"
    elif "VERDICT: PASS" in response_upper or '"PASS"' in response_upper:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return verdict, response