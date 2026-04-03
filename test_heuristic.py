import os
import json
import time
from src.envs.data_cleaner.client import DataCleanerClient

def main():
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    client = DataCleanerClient(base_url=api_base_url)

    print("[START] Starting Data Cleaner Episode (Heuristic Script)")
    try:
        obs = client.reset()
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    score = 0.0

    # A simple manual plan to show it works
    actions = [
        {"action_type": "DROP_COLUMN", "target_column": "empty_col"},
        {"action_type": "REMOVE_DUPLICATES"},
        {"action_type": "FORMAT_PHONE", "target_column": "phone"},
        {"action_type": "FORMAT_DATE", "target_column": "date"},
        {"action_type": "IMPUTE_MEAN", "target_column": "salary"},
        {"action_type": "SUBMIT_DATASET"}
    ]

    for action_dict in actions:
        if obs.done: break
        
        raw_action = json.dumps(action_dict)
        try:
            obs = client.step(action_dict)
            score += obs.reward
            print(f"[STEP] Applied Action: {raw_action} | Reward for this step: {obs.reward}")
            time.sleep(0.5)
        except Exception as e:
            print(f"[STEP] Server error: {e}")
            break

    print(f"[END] Final Score: {score}")

if __name__ == "__main__":
    main()
