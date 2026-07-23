# -------------------------------------------------------------------------------------------------------------------------------
# Agent Configuration                                                                                                           
# -------------------------------------------------------------------------------------------------------------------------------
# API Endpoint: https://openrouter.ai/api/v1/chat/completions                                       
# API Endpoint: https://router.huggingface.co/v1              
# Both endpoints are used to route requests to the appropriate model providers (OpenRouter, HuggingFace, etc.) based on the agent configuration.                      
#                                                                                                                               
# Configures agent models and parameters used across all pipeline stages (0-5):                                                 
#   - Doer (qwen/qwen3-8b): Executes tasks, generates code | temp 0.1                                                         
#   - Evaluator (openai/gpt-oss-120b): Verifies output quality independently | temp 0.1                                        
#   - Manager (qwen/qwen3-14b): Provides corrective guidance on escalation | temp 0.3                                          
#                                                                                                                               
# -------------------------------------------------------------------------------------------------------------------------------


import os
import requests
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN")

AGENT_CONFIG = {
    "doer": {
        "model": "qwen/qwen3-8b",            # doer model
        "temperature": 0.1,
        "max_tokens": 4200,
        "provider": "openrouter",
    },
    "evaluator": {
        "model": "openai/gpt-oss-120b",      # evaluator model
        "temperature": 0.1,
        "max_tokens": 3000,
        "provider": "openrouter",
    },
    "manager": {
        "model": "qwen/qwen3-14b",           # manager model
        "temperature": 0.3,
        "max_tokens": 3000,
        "provider": "openrouter",
    },
}


def call_agent(role, system_prompt, user_prompt):
    """
    Call agents — all routing through OpenRouter.
    """
    agent = AGENT_CONFIG.get(role)

    if not agent:
        print(f"[ERROR] Unknown agent role: {role}")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENROUTER_TOKEN}",
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Thesis AI Project Pipeline",
    }

    payload = {
        "model": agent["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": agent["temperature"],
        "max_tokens": agent["max_tokens"],
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )

        if response.status_code != 200:
            print(
                f"[ERROR] {role} ({agent['model']}) returned "
                f"{response.status_code}: {response.text[:500]}"
            )
            return f"[AGENT_ERROR] status={response.status_code} text={response.text}"

        data = response.json()

        if (
            "choices" in data
            and len(data["choices"]) > 0
            and "message" in data["choices"][0]
        ):
            return data["choices"][0]["message"]["content"]

        print(f"[ERROR] Unexpected response format: {data}")
        return "[AGENT_ERROR] unexpected response format"

    except requests.exceptions.Timeout:
        print(f"[ERROR] {role} ({agent['model']}) timed out")
        return "[AGENT_ERROR] timeout"

    except Exception as e:
        print(f"[ERROR] {role} ({agent['model']}) failed: {e}")
        return f"[AGENT_ERROR] exception: {e}"