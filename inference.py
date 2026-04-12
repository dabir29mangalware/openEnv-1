import os, json, requests, time
from openai import OpenAI

API_KEY = os.environ.get("API_KEY")
API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000").rstrip("/")

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

TASKS = [
    {"difficulty": "easy", "task_id": "data_cleaning_easy"},
    {"difficulty": "medium", "task_id": "data_cleaning_medium"},
    {"difficulty": "hard", "task_id": "data_cleaning_hard"},
]
BENCHMARK = "data_cleaner"

SYSTEM_PROMPT = """You are OpenEnv, an advanced reinforcement learning AI operating in a data-cleaning simulation built on the OpenEnv framework. Your core philosophy is to make complex data wrangling look effortless, breezy, and magical.

Objective:
Your primary goal is to interpret environmental state data (dataset metadata, null counts, duplicate counts, and dataframe previews) and output precise action commands (cleaning operations) to complete data standardization tasks of increasing difficulty.
- Level 1 (Easy): Clean missing values in a small dataset.
- Level 2 (Medium): Handle duplicates, messy text, and varying data types.
- Level 3 (Hard): Clean heavily corrupted datasets with aggressive noise.

Operational Rules:
- Data Integrity Reality: Every action modifies the underlying dataset. You must choose the correct columns to drop, format, or impute to perfectly match the target schema.
- Tone: Keep your reasoning clear, analytical, and highly structured. You are wrangling data, after all.

Available action_type values:
- IMPUTE_MEAN: Fill nulls in a numeric column with the column mean. Requires "target_column".
- IMPUTE_MEDIAN: Fill nulls in a numeric column with the column median. Requires "target_column".
- FILL_MODE: Fill nulls in any column with the most frequent value. Requires "target_column".
- REMOVE_DUPLICATES: Remove duplicate rows. No "target_column" needed.
- STANDARDIZE_TEXT: Strip whitespace and lowercase a text column. Requires "target_column".
- DROP_COLUMN: Drop a column (only if entirely empty). Requires "target_column".
- FORMAT_PHONE: Format phone numbers to +1-XXX-XXX-XXXX. Requires "target_column".
- FORMAT_DATE: Format dates to YYYY-MM-DD. Requires "target_column".
- SUBMIT_DATASET: Submit the cleaned dataset for scoring. Use when all cleaning is done.

Output Requirements:
Output exactly one JSON object per step. No markdown outside of the JSON block.
The JSON object must contain your requested 'action_type', a 'target_column' (if applicable), and a 'reasoning' field containing a brief, one-sentence explanation of your cleaning reasoning.

Example output: 
{
  "action_type": "IMPUTE_MEAN", 
  "target_column": "age",
  "reasoning": "Imputing the mean for the age column to resolve missing numeric values efficiently."
}"""

def log_start(task: str, env: str, model: str):
    payload = {"task": task, "env": env, "model": model}
    print(f"[START] {json.dumps(payload)}", flush=True)

def log_step(step: int, action: dict, reward: float, done: bool, error=None):
    payload = {
        "step": step,
        "action": action,
        "reward": round(max(0.001, min(0.999, float(reward))), 4),
        "done": done,
        "error": str(error) if error else None,
    }
    print(f"[STEP] {json.dumps(payload)}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    payload = {
        "success": success,
        "steps": steps,
        "score": round(max(0.001, min(0.999, float(score))), 4),
        "rewards": [round(max(0.001, min(0.999, float(r))), 4) for r in rewards],
    }
    print(f"[END] {json.dumps(payload)}", flush=True)

def main():
    print(f"[DEBUG] Validating server readiness at {ENV_URL}...", flush=True)
    for _ in range(5):
        try:
            if requests.get(f"{ENV_URL}/health").status_code == 200:
                print("[DEBUG] Server is healthy.", flush=True)
                break
        except:
            time.sleep(2)

    all_scores = []
    
    for task_info in TASKS:
        task_id = task_info["task_id"]
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        
        rewards = []
        steps_taken = 0
        score = 0.001
        success = False
        
        try:
            resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}).json()
            obs = resp.get("observation", {})
            done = resp.get("done", False)
            
            for step in range(1, 16):
                if done: break
                
                # MUST BE THE ONLY LLM CALL
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": str(obs)}
                    ],
                    temperature=0.0
                )
                action_str = completion.choices[0].message.content.strip()
                
                # Parse action and step environment
                resp = requests.post(f"{ENV_URL}/step", json={"action": {"command": action_str}}).json()
                obs = resp.get("observation", {})
                reward = resp.get("reward", 0.0) or 0.0
                done = resp.get("done", False)

                reward = max(0.001, min(0.999, float(reward)))
                rewards.append(reward)
                steps_taken = step
                
                log_step(step=step, action={"command": action_str}, reward=reward, done=done, error=None)
                
            score = rewards[-1] if rewards else 0.001
            score = max(0.001, min(0.999, float(score)))
            success = score >= 0.5
            
        except Exception as e:
            import traceback
            print(f"[ERROR] Task crashed: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            score = 0.001
            success = False
            
        finally:
            score = max(0.001, min(0.999, float(score)))
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
            all_scores.append(score)

if __name__ == "__main__":
    main()
