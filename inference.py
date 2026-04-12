import os
import sys
import json
import time
import argparse
import requests
from openai import OpenAI

# Add 'src' to the module search path to satisfy static analyzers and local execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from envs.data_cleaner.client import DataCleanerClient

# ---------------------------------------------------------------------------
# OpenAI client initialization (Literal platform compliance)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").rstrip("/")

if not os.environ.get("API_BASE_URL"):
    print("[FATAL] API_BASE_URL environment variable is MISSING or EMPTY.", flush=True)
    sys.exit(1)
if not os.environ.get("API_KEY"):
    print("[FATAL] API_KEY environment variable is MISSING or EMPTY.", flush=True)
    sys.exit(1)

api_base_url = os.environ["API_BASE_URL"]
if not api_base_url.startswith("http://") and not api_base_url.startswith("https://"):
    api_base_url = "http://" + api_base_url

if not api_base_url.endswith("/v1") and not api_base_url.endswith("/v1/"):
    api_base_url = api_base_url.rstrip("/") + "/v1"
os.environ["API_BASE_URL_V1"] = api_base_url

print(f"[DEBUG] API_BASE_URL = {os.environ['API_BASE_URL']}", flush=True)
print(f"[DEBUG] MODEL_NAME = {MODEL_NAME} | ENV_BASE_URL = {ENV_BASE_URL}", flush=True)


def discover_model(llm_client, preferred_model: str) -> str:
    """Try to discover the correct model name from the proxy's /models endpoint."""
    if preferred_model:
        # Many proxies don't support /models, so just return the requested MODEL_NAME if present
        return preferred_model
    
    url = f"{os.environ['API_BASE_URL_V1'].rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {os.environ['API_KEY']}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        available = [m["id"] for m in data.get("data", [])]
        print(f"[DEBUG] Available models on proxy: {available}", flush=True)
        # Use first available
        if available:
            chosen = available[0]
            print(f"[DEBUG] Returning model '{chosen}' from proxy list", flush=True)
            return chosen
    except Exception as e:
        print(f"[DEBUG] Could not list models from proxy: {e}. Using '{preferred_model}'.", flush=True)
    return preferred_model

MAX_HISTORY_PAIRS = 6  # Sliding window: keep last N user/assistant pairs
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]  # seconds

TASKS = [
    {"difficulty": "easy", "task_id": "data_cleaning_easy"},
    {"difficulty": "medium", "task_id": "data_cleaning_medium"},
    {"difficulty": "hard", "task_id": "data_cleaning_hard"},
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
        "reward": round(max(0.2222, min(0.8888, float(reward))), 4),
        "done": done,
        "error": str(error) if error else None,
    }
    print(f"[STEP] {json.dumps(payload)}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list):
    score = max(0.001, min(0.999, float(score)))
    rewards_payload = [round(max(0.2222, min(0.8888, float(r))), 4) for r in rewards]
    payload = {
        "success": success,
        "steps": steps,
        "score": float(f"{score:.3f}"),
        "rewards": rewards_payload,
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
    url = f"{os.environ['API_BASE_URL_V1'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['API_KEY']}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 256
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=45)
            resp.raise_for_status()
            
            # The proxy should return an OpenAI-compatible JSON response
            raw = resp.json()["choices"][0]["message"]["content"]
            if not raw:
                raw = ""
            raw = raw.strip()

            print(f"[DEBUG] LLM raw response (attempt {attempt+1}): {raw[:200]}", flush=True)
            # Handle responses wrapped in markdown code blocks
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
                
            return json.loads(raw)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[ERROR] LLM call attempt {attempt+1} failed: {e}", flush=True)
            if hasattr(e, 'response') and e.response is not None:
                print(f"[ERROR] Proxy response: {e.response.text}", flush=True)
            print(f"[ERROR] Traceback:\n{tb}", flush=True)
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF[attempt]
                print(f"[DEBUG] Retrying in {wait}s...", flush=True)
                time.sleep(wait)
            else:
                print(f"[WARN] LLM failed after {MAX_RETRIES} retries. Submitting dataset to register attempt.", flush=True)
                # Return SUBMIT_DATASET so the environment loop ends and the platform
                # records this as an attempted (but failing) interaction.
                return {"action_type": "SUBMIT_DATASET", "reasoning": "LLM proxy error, submitting to end episode."}


# ---------------------------------------------------------------------------
# Run a single task (difficulty)
# ---------------------------------------------------------------------------
def run_task(client: DataCleanerClient, llm, task_name: str, difficulty: str, model: str = None, dataset_path: str = None):
    model = model or MODEL_NAME
    log_start(task=task_name, env=BENCHMARK, model=model)

    rewards = []
    steps_taken = 0
    score = 0.2222
    success = False

    try:
        obs = client.reset(difficulty=difficulty, dataset_path=dataset_path)
    except Exception as e:
        print(f"[WARN] Failed to connect to environment server at {ENV_BASE_URL}: {e}", flush=True)
        log_end(success=False, steps=0, score=0.2222, rewards=[])
        return 0.2222

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
        action_dict = call_llm(llm, model, messages)
        raw_action = json.dumps(action_dict)
        messages.append({"role": "assistant", "content": raw_action})

        # Execute action
        error = None
        current_done = obs.done
        try:
            obs, reward, current_done, info = client.step(action_dict)
        except Exception as e:
            error = str(e)
            reward = 0.2222
            current_done = True
            obs.done = True
            print(f"[DEBUG] Step error: {e}", flush=True)

        reward = max(0.2222, min(0.8888, float(reward)))
        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=action_dict, reward=reward, done=current_done, error=error)

        if current_done:
            break

    # Calculate final score from environment grader metadata with strict (0,1) clamping
    metadata = obs.get("metadata", {}) if isinstance(obs, dict) else (getattr(obs, "metadata", None) or {})
    raw_score = metadata.get("grader_score", 0.0)
    if raw_score is None:
        raw_score = 0.0
    score = max(0.001, min(0.999, float(raw_score)))
    success = score >= 0.5
    
    # ALWAYS emit [END] log
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run the Automated Data Cleaner agent.")
    parser.add_argument("--datasets", nargs="+", type=str, help="One or more paths to local CSV datasets to upload and clean", default=[None])
    args = parser.parse_args()

    env_client = DataCleanerClient(base_url=ENV_BASE_URL)

    # Wait for the environment server to be ready (up to 300s for absolute reliability)
    print(f"[DEBUG] Waiting for environment server at {ENV_BASE_URL}...", flush=True)
    server_ready = False
    for attempt in range(60): # 60 * 5s = 300s
        try:
            if env_client.health():
                print(f"[DEBUG] Server healthy after {attempt + 1} attempts.", flush=True)
                server_ready = True
                break
        except Exception:
            pass
        time.sleep(5)
    
    if not server_ready:
        print(f"[FATAL] Server not healthy at {ENV_BASE_URL} after 300s. Exiting.", flush=True)
        sys.exit(1)

    llm = OpenAI(
        base_url=os.environ["API_BASE_URL_V1"],
        api_key=os.environ["API_KEY"],
    )
    print(f"[DEBUG] OpenAI client initialized with base_url={os.environ['API_BASE_URL_V1']}", flush=True)

    # Discover the correct model name from the proxy
    active_model = discover_model(llm, MODEL_NAME)
    print(f"[DEBUG] Using model: {active_model}", flush=True)

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
                server_dataset_path = env_client.upload(dataset)
            except Exception as e:
                print(f"[ERROR] Upload failed for {dataset}: {e}", flush=True)
                continue

        total_score = 0.2222
        for task in TASKS:
            try:
                score = run_task(env_client, llm, task["task_id"] if "task_id" in task else task["name"], task["difficulty"], active_model, dataset_path=server_dataset_path)
            except Exception as e:
                import traceback
                print(f"[ERROR] Task failed with exception: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                score = 0.001
            score = max(0.001, min(0.999, float(score)))
            total_score += score
            all_scores.append(score)
            print(f"[DEBUG] Dataset {dataset or 'Random'} | Task '{task['task_id']}' score: {score:.4f}", flush=True)

    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        avg_score = max(0.2222, min(0.8888, float(avg_score)))
        print(f"\n[DEBUG] Overall average score across all executed tasks/datasets: {avg_score:.4f}", flush=True)


if __name__ == "__main__":
    main()
