
# -------------------------------------------------------------------------------------------------------------------------------
# Stage 2: Post-Execution Evaluator (Topic Modelling)                                                                           
# -------------------------------------------------------------------------------------------------------------------------------
# Standalone script that runs the Evaluator agent AFTER topic modelling completes.                                               
# Sends execution statistics, top-30 topics, and config to GPT OSS 120B for quality review.                                     
#                                                                                                                               
# This is separate from the main pipeline graph because topic modelling code is manually written                                 
# (not Doer-generated), so the standard executor → validator → evaluator flow does not apply.                                   
# Instead: topic_modelling_code.py runs → outputs summary JSON → this script sends it for evaluation.                                
#                                                                                                                               
# Input: outputs/topic_modelling/phase_2_summary.json (execution stats)                                                         
#        outputs/topic_modelling/topic_info.csv (discovered topics)                                                              
#        outputs/topic_modelling/config.json (Doer-selected parameters)                                                          
# Output: logs/topic_modelling_post_eval.json (Evaluator verdict and feedback)                                                   
# -------------------------------------------------------------------------------------------------------------------------------


"""Run post-execution evaluation for topic modelling results."""
import json
import os
from agents.api import call_agent
from graph.prompts import load_system_prompt, load_task_prompt

# 1. Load results
stats_path = "outputs/topic_modelling/phase_2_summary.json"
topic_info_path = "outputs/topic_modelling/topic_info.csv"
config_path = "outputs/topic_modelling/config.json"

with open(stats_path, "r") as f:
    stats = json.load(f)
with open(config_path, "r") as f:
    config = json.load(f)

# Top 30 topics
import pandas as pd
df_topics = pd.read_csv(topic_info_path)
top30 = df_topics[df_topics['Topic'] >= 0].head(30)
top30_text = ""
for _, row in top30.iterrows():
    top30_text += f"Topic #{row['Topic']} | Size: {row['Count']} | Keywords: {row['Representation']}"

# 2. Build evaluator prompt
eval_task = load_task_prompt("topic_modelling_eval_post")

user_prompt = (
    f"{eval_task}" 
    f"## EXECUTION STATISTICS {json.dumps(stats, indent=2)}"
    f"## TOP 30 TOPICS {top30_text}"
    f"## CONFIGURATION USED {json.dumps(config, indent=2)}"
)

# 3. Call evaluator
print("=" * 60)
print("[EVALUATOR] Running post-execution evaluation for topic modelling...")
print("=" * 60)

system_prompt = load_system_prompt("evaluator")
feedback = call_agent("evaluator", system_prompt, user_prompt)

# 4. Save and display
os.makedirs("logs", exist_ok=True)
with open("logs/topic_modelling_post_eval.json", "w", encoding="utf-8") as f:
    f.write(feedback)

print(f"[EVALUATOR] Response: {feedback}")
