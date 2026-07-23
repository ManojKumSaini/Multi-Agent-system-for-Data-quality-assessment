
# -------------------------------------------------------------------------------------------------------------------------------
# Code Executor                                                                                                                 
# -------------------------------------------------------------------------------------------------------------------------------
# Extracts Python code from Doer agent responses and executes it in a sandboxed subprocess.                                     
# Used by all code-generation stages (1, 3, 5) after the Doer produces a response.                                          
#                                                                                                                               
# Flow: Doer response → extract code (strips <think> blocks, finds fenced/unfenced code) →                                     
#       save to outputs/doer_raw/{phase}_attempt_{n}.py → execute via venv Python → return result                               
#                                                                                                                               
# Timeout: 120s default (configurable per stage)                                                                                
# Python: Uses local venv (venv/Scripts/python.exe)                                                                             
# -------------------------------------------------------------------------------------------------------------------------------


import os
import re
import subprocess
import time


PYTHON_PATH = os.path.join("venv", "Scripts", "python.exe")


def extract_code_from_response(response):
    """Extract Python code from the model's response."""
    if not response:
        return None

    # Strip <think>...</think> blocks (Qwen 3 reasoning)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Attempt 1: Standard ```python ... ``` fenced block
    pattern = r"```[Pp]ython\s*(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Attempt 2: "python" at the start without backticks
    pattern_no_fence = r"^[Pp]ython\s*(.*)"
    match = re.search(pattern_no_fence, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Attempt 3: Any ``` ... ``` block (no language specified)
    pattern_generic = r"```\s*(.*?)```"
    match = re.search(pattern_generic, response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Attempt 4: Response IS the code (starts with import/from/#)
    stripped = response.strip()
    if stripped.startswith(("import ", "from ", "# ")):
        return stripped

    return None


def execute_code(code, phase_name, attempt, timeout=120):
    """
    Save and execute the Doer's generated code.

    Args:
        code: Python code string
        phase_name: Current phase (for file naming)
        attempt: Attempt number (1, 2, or 3)
        timeout: Max execution time in seconds

    Returns:
        dict with: success, stdout, stderr, return_code, script_path, duration
    """
    # Create output directory for raw scripts
    script_dir = os.path.join("outputs", "doer_raw")
    os.makedirs(script_dir, exist_ok=True)

    # Save script
    script_name = f"{phase_name}_attempt_{attempt}.py"
    script_path = os.path.join(script_dir, script_name)

    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    print(f"[EXECUTOR] Saved script: {script_path}")
    print(f"[EXECUTOR] Running with timeout={timeout}s...")

    # Check venv Python exists
    if not os.path.exists(PYTHON_PATH):
        print(f"[ERROR] Python not found at: {PYTHON_PATH}")
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Python not found at {PYTHON_PATH}",
            "return_code": -1,
            "script_path": script_path,
            "duration": 0,
        }

    # Execute
    start_time = time.time()

    try:
        result = subprocess.run(
            [PYTHON_PATH, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=".",
        )
        duration = time.time() - start_time

        execution_result = {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "script_path": script_path,
            "duration": round(duration, 2),
        }

        if result.returncode == 0:
            print(f"[EXECUTOR] Success in {duration:.1f}s")
        else:
            print(f"[EXECUTOR] Failed (return code: {result.returncode})")
            print(f"[EXECUTOR] stderr: {result.stderr[:300]}")

        return execution_result

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"[EXECUTOR] Timeout after {timeout}s")
        return {
            "success": False,
            "stdout": "",
            "stderr": f"TIMEOUT: Script exceeded {timeout} seconds.",
            "return_code": -1,
            "script_path": script_path,
            "duration": round(duration, 2),
        }

    except Exception as e:
        duration = time.time() - start_time
        print(f"[EXECUTOR] Error: {e}")
        return {
            "success": False,
            "stdout": "",
            "stderr": f"EXECUTION ERROR: {str(e)}",
            "return_code": -1,
            "script_path": script_path,
            "duration": round(duration, 2),
        }

