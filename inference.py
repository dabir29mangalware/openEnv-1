import os
import json
import logging
from openai import OpenAI
from src.envs.data_cleaner.client import DataCleanerClient
from src.envs.data_cleaner.models import ActionType

def main():
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    model_name = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile") # Using Groq's open-source model

    client = DataCleanerClient(base_url=api_base_url)
    
    # Configure OpenAI client to hit Groq
    llm = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    ) 

    print("[START] Starting Data Cleaner Episode")
    
    try:
        obs = client.reset()
    except Exception as e:
        print(f"Failed to connect to environment: {e}")
        return

    score = 0.0
    
    system_prompt = """
    You are an automated data cleaner RL Agent. You receive a messy dataset chunk.
    You must output exactly one JSON object representing your action. Do not add markdown or comments.
    Available action_type values: DROP_COLUMN, REMOVE_DUPLICATES, FORMAT_PHONE, FORMAT_DATE, IMPUTE_MEAN, SUBMIT_DATASET.
    Include "target_column" (string) if the action requires a column target (e.g., DROP_COLUMN, FORMAT_PHONE). For REMOVE_DUPLICATES or SUBMIT_DATASET, you can omit it.
    Example output format: {"action_type": "DROP_COLUMN", "target_column": "empty_col"}
    """
    
    messages = [
        {"role": "system", "content": system_prompt.strip()}
    ]

    while not obs.done:
        state_str = json.dumps({
            "metadata": obs.metadata,
            "current_view": obs.current_view,
            "feedback": obs.feedback
        }, indent=2)
        
        messages.append({"role": "user", "content": f"Current State: {state_str}\nProvide next action JSON."})
        
        try:
            completion = llm.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            raw_action = completion.choices[0].message.content
            action_dict = json.loads(raw_action)
            
            messages.append({"role": "assistant", "content": raw_action})
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            action_dict = {"action_type": "SUBMIT_DATASET"}
            raw_action = json.dumps(action_dict)

        try:
            obs = client.step(action_dict)
            score += obs.reward
            
            print(f"[STEP] Action: {raw_action} | Reward: {obs.reward}")
        except Exception as e:
            print(f"[STEP] Server error: {e}")
            break

    print(f"[END] Final Score: {score}")

if __name__ == "__main__":
    main()
