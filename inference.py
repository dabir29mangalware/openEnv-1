import os
import sys
import json
import time
import argparse
from openai import OpenAI

# Add 'src' to the module search path to satisfy static analyzers and local execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from envs.data_cleaner.client import DataCleanerClient

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN", os.getenv("GROQ_API_KEY", ""))

# Inference parameters
MAX_HISTORY_PAIRS = 6  # Sliding window: keep last N user/assistant pairs
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # seconds

TASKS = [
    {"difficulty": "easy", "name": "data_cleaning_easy"},
    {"difficulty": "medium", "name": "data_cleaning_medium"},
    {"difficulty": "hard", "name": "data_cleaning_hard"},
]

BENCHMARK = "data_cleaner"


# ---------------------------------------------------------------------------
# Structured logging helpers  (exact format required by hackathon)
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str):
    payload = {"task": task, "env": env, "model": model}
    print(f"[START] {json.dumps(payload)}", flush=True)


def log_step(step: int, action: dict, reward: float, done: bool, error=None):
    payload = {
        "step": step,
        "action": action,
        "reward": round(max(0.0001, min(0.9999, float(reward))), 4),
        "done": done,
        "error": str(error) if error else None,
    }
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    payload = {
        "success": success,
        "steps": steps,
        "score": round(max(0.0001, min(0.9999, float(score))), 4),
        "rewards": [round(max(0.0001, min(0.9999, float(r))), 4) for r in rewards],
    }
    print(f"[END] {json.dumps(payload)}", flush=True)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Compressed state builder (reduces token usage dramatically)
# ---------------------------------------------------------------------------
def build_compact_state(obs, step_num: int) -> str:
    m = obs.metadata
    lines = [
        f"Step {step_num}/{obs.max_steps} | Difficulty: {obs.difficulty}",
        f"Shape: {m.get('total_rows', '?')} rows × {m.get('total_columns', '?')} cols",
    ]

    null_counts = m.get("null_counts", {})
    if null_counts:
        null_str = ", ".join(f"{k}({v})" for k, v in null_counts.items())
        lines.append(f"Nulls: {null_str}")
    else:
        lines.append("Nulls: None")

    dup_count = m.get("duplicate_row_count", 0)
    if dup_count > 0:
        lines.append(f"Duplicate rows: {dup_count}")

    lines.append(f"Columns: {', '.join(m.get('columns', []))}")
    lines.append(f"Feedback: {obs.feedback}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call with retry
# ---------------------------------------------------------------------------
def call_llm(llm, model: str, messages: list) -> dict:
    for attempt in range(MAX_RETRIES):
        try:
            completion = llm.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=256,
            )
            raw = (completion.choices[0].message.content or "").strip()
            return json.loads(raw)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                print(f"[DEBUG] LLM retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}", flush=True)
                time.sleep(wait)
            else:
                print(f"[DEBUG] LLM failed after {MAX_RETRIES} retries: {e}", flush=True)
                return {"action_type": "SUBMIT_DATASET"}


# ---------------------------------------------------------------------------
# Run a single task (difficulty)
# ---------------------------------------------------------------------------
def run_task(client: DataCleanerClient, llm, task_name: str, difficulty: str, dataset_path: str = None):
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        obs = client.reset(difficulty=difficulty, dataset_path=dataset_path)
    except Exception as e:
        print(f"[DEBUG] Failed to connect to environment: {e}", flush=True)
        log_end(success=False, steps=0, score=0.0001, rewards=[])
        return 0.0001

    # Message history with sliding window
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    max_steps = obs.max_steps

    for step in range(1, max_steps + 1):
        if obs.done:
            break

        # Build compact state and add to messages
        state_str = build_compact_state(obs, step)
        messages.append(
            {"role": "user", "content": f"{state_str}\nProvide next action JSON."}
        )

        # Sliding window: keep only system + last N pairs
        if len(messages) > 1 + MAX_HISTORY_PAIRS * 2:
            messages = [messages[0]] + messages[-(MAX_HISTORY_PAIRS * 2):]

        # Get action from LLM
        action_dict = call_llm(llm, MODEL_NAME, messages)
        raw_action = json.dumps(action_dict)
        messages.append({"role": "assistant", "content": raw_action})

        # Execute action
        error = None
        try:
            obs, reward, done, info = client.step(action_dict)
        except Exception as e:
            error = str(e)
            reward = 0.0
            print(f"[DEBUG] Step error: {e}", flush=True)

        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=action_dict, reward=reward, done=obs.done, error=error)

        if obs.done:
            break

    # Calculate final score: use final reward
    score = rewards[-1] if rewards else 0.0
    score = max(0.0001, min(0.9999, float(score)))
    success = score >= 0.5

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run the Automated Data Cleaner agent.")
    parser.add_argument("--datasets", nargs="+", type=str, help="One or more paths to local CSV datasets to upload and clean", default=[None])
    args = parser.parse_args()

    client = DataCleanerClient(base_url=ENV_BASE_URL)

    llm = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    all_scores = []

    for ds_idx, dataset in enumerate(args.datasets):
        # If user provided a custom dataset, upload it via the API first
        server_dataset_path = None
        if dataset:
            if not os.path.exists(dataset):
                print(f"[ERROR] Could not find local dataset: {dataset}", flush=True)
                continue
            try:
                print(f"\n{'='*50}\n[DEBUG] Uploading {dataset} to environment server...\n{'='*50}", flush=True)
                server_dataset_path = client.upload(dataset)
            except Exception as e:
                print(f"[ERROR] Upload failed for {dataset}: {e}", flush=True)
                continue

        total_score = 0.0
        for task in TASKS:
            score = run_task(client, llm, task["name"], task["difficulty"], dataset_path=server_dataset_path)
            total_score += score
            all_scores.append(score)
            print(f"[DEBUG] Dataset {dataset or 'Random'} | Task '{task['name']}' score: {score:.4f}", flush=True)

    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print(f"\n[DEBUG] Overall average score across all executed tasks/datasets: {avg_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
