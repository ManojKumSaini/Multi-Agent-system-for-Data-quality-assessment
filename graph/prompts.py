# Loading prompts and guides for agents and tasks.
# Each agent role has a system prompt, and each task phase has a task prompt and evaluation criteria.
# all files are stored in the prompts/ directory, organized by type (system, tasks).
# Everything is loaded dynamically based on the role or phase name, with fallback warnings if files are missing.
# Prompts can be in .md or .txt format but only markdown files are used, and the code handles both. Guides for specific tools are also loaded from the tools/ directory.
# prompts/system/{role}_system.md for system prompts
# prompts/tasks/{phase_name}.md for task prompts
#
# Phase 3 is an exception: its prompt files are named "sbert_code" instead of # "semantic_similarity". 
# Therefore, the phase name is mapped to the existing # file names ("sbert_code" and "sbert_code_eval") during loading to maintain 
# compatibility with the prompt file structure.
import os

PROMPTS_DIR = "prompts"
TOOLS_DIR = "tools"


def load_file(path):
    """Safely load text from a file. Returns empty string if not found."""
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _find_file(directory, name_pattern):
    """
    Find a file matching a name pattern (ignores .md vs .txt extension).
    Returns full path or empty string.
    """
    if not os.path.exists(directory):
        return ""
    for filename in os.listdir(directory):
        stem = os.path.splitext(filename)[0].lower()
        if stem == name_pattern.lower():
            return os.path.join(directory, filename)
    return ""

def load_system_prompt(role):
    """
    Load system prompt for an agent role.
    Looks for: prompts/system/{role}_system.md or .txt
    """
    system_dir = os.path.join(PROMPTS_DIR, "system")
    # Try exact match first
    path = _find_file(system_dir, f"{role}_system")
    if path:
        return load_file(path)
    # Try capitalized (your Doer file is "Doer_system.txt")
    path = _find_file(system_dir, f"{role.capitalize()}_system")
    if path:
        return load_file(path)
    print(f"[WARNING] No system prompt found for role: {role}")
    return ""


def load_task_prompt(phase_name):
    """
    Load the Doer's task prompt for a specific phase.
    Looks for: prompts/tasks/{phase_name}.md
    """
    alias_map = {
        "semantic_similarity": "sbert_code",
    }
    phase_key = alias_map.get(phase_name, phase_name)
    path = _find_file(os.path.join(PROMPTS_DIR, "tasks"), phase_key)
    if path:
        return load_file(path)
    print(f"[WARNING] No task prompt found for phase: {phase_name}")
    return ""


def load_evaluation_criteria(phase_name):
    """
    Load evaluation criteria for a specific phase.
    Looks for: prompts/tasks/{phase_name}_eval.md
    """
    tasks_dir = os.path.join(PROMPTS_DIR, "tasks")
    alias_map = {
        "semantic_similarity": "sbert_code_eval",
    }
    phase_key = alias_map.get(phase_name, f"{phase_name}_eval")
    path = _find_file(tasks_dir, phase_key)
    if path:
        return load_file(path)
    print(f"[WARNING] No evaluation criteria found for phase: {phase_name}")
    return ""


def load_guide(phase_name):
    """
    Load reference guide for a phase.
    E.g., tools/Bertopic_guide.md for topic_modelling.
    """
    # Known mappings
    guide_map = {
        "preprocessing": "Preprocessing_guide",
        "topic_modelling": "Bertopic_guide",
        "semantic_similarity": "sbert_pair_guide",
        "stats_est": "Statiscal_estimation_guide",
    }

    name = guide_map.get(phase_name, f"{phase_name}_guide")
    path = _find_file(TOOLS_DIR, name)
    if path:
        return load_file(path)
    for root, _, files in os.walk("."):
        if "venv" in root.split(os.sep):
            continue
        for filename in files:
            stem = os.path.splitext(filename)[0].lower()
            if stem == name.lower():
                return load_file(os.path.join(root, filename))
    return ""