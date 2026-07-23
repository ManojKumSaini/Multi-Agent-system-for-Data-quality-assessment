# -------------------------------------------------------------------------------------------------------------------------------
# Agent testing script                                                                                                           
# -------------------------------------------------------------------------------------------------------------------------------
#  This script tests the agent API calls by invoking the call_agent function with a sample prompt.
#  All three agents (doer, evaluator, manager) can be tested by changing the role parameter in the call_agent function.
#  The script will print the response from the agent or an error message if the call fails.                                                                                                                         
# -------------------------------------------------------------------------------------------------------------------------------



from agents.api import call_agent

# Simple test 
response = call_agent(
    role="evaluator",
    system_prompt="You are a helpful assistant.",
    user_prompt="Say 'Hello, the API is working!' and nothing else.",
)

if response:
    print(f"✓ SUCCESS: {response}")
else:
    print("✗ FAILED: Check your OPENROUTER_TOKEN in .env")



