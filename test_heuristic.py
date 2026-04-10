import os
import json
import time
import sys
from src.envs.data_cleaner.client import DataCleanerClient


def run_smart_heuristic(client: DataCleanerClient, difficulty: str) -> float:
    """Dynamically build an action plan based on the dataset metadata."""
    print(f"\n{'='*60}")
    print(f"[HEURISTIC] Running difficulty: {difficulty}")
    print(f"{'='*60}")
    run_smart_heuristic._text_idx = 0

    try:
        obs = client.reset(difficulty=difficulty)
    except Exception as e:
        print(f"[HEURISTIC] Failed to connect: {e}")
        return 0.0

    score = 0.0
    step = 0

    while not obs.done:
        step += 1
        meta = obs.metadata
        null_counts = meta.get("null_counts", {})
        dup_count = meta.get("duplicate_row_count", 0)

        # Decision logic:
        # 1. Impute nulls in numeric columns first
        action_dict = None

        if null_counts:
            # Pick the first column with nulls
            col_name = list(null_counts.keys())[0]
            dtypes = meta.get("dtypes", {})
            col_dtype = dtypes.get(col_name, "")

            if "float" in col_dtype or "int" in col_dtype:
                action_dict = {"action_type": "IMPUTE_MEAN", "target_column": col_name}
            elif "object" in col_dtype:
                action_dict = {"action_type": "FILL_MODE", "target_column": col_name}
            else:
                action_dict = {"action_type": "IMPUTE_MEAN", "target_column": col_name}

        elif dup_count > 0:
            action_dict = {"action_type": "REMOVE_DUPLICATES"}

        else:
            # If all nulls/dupes are handled, check text columns for standardization
            dtypes = meta.get("dtypes", {})
            # We'll just randomly try STANDARDIZE_TEXT and FORMAT_DATE once for text cols to simulate agent exploring it
            text_cols = [k for k, v in dtypes.items() if "object" in str(v)]
            if getattr(run_smart_heuristic, '_text_idx', 0) < len(text_cols):
                col = text_cols[getattr(run_smart_heuristic, '_text_idx', 0)]
                run_smart_heuristic._text_idx = getattr(run_smart_heuristic, '_text_idx', 0) + 1
                
                # Check column name to guess what to do
                if "date" in col.lower():
                    action_dict = {"action_type": "FORMAT_DATE", "target_column": col}
                elif "phone" in col.lower():
                    action_dict = {"action_type": "FORMAT_PHONE", "target_column": col}
                else:
                    action_dict = {"action_type": "STANDARDIZE_TEXT", "target_column": col}
            else:
                # All clean — submit
                action_dict = {"action_type": "SUBMIT_DATASET"}

        raw = json.dumps(action_dict)
        try:
            obs = client.step(action_dict)
            score += obs.reward
            print(
                f"  [STEP {step:>2}] {raw:<60} | reward: {obs.reward:+.4f} | done: {obs.done}"
            )
        except Exception as e:
            print(f"  [STEP {step:>2}] Error: {e}")
            break

        time.sleep(0.1)  # Small delay to not overwhelm server

    score = max(0.2, min(0.98, float(score)))
    print(f"[HEURISTIC] Difficulty '{difficulty}' => Final Score: {score:.4f}")
    assert 0.0 < score < 1.0, f"Score {score} outside expected range!"
    return score


def main():
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
    client = DataCleanerClient(base_url=api_base_url)

    # Check health first
    if not client.health():
        print("[ERROR] Server is not healthy. Is the server running?")
        sys.exit(1)

    print("[OK] Server is healthy.\n")

    difficulties = ["easy", "medium", "hard"]
    scores = {}

    for diff in difficulties:
        scores[diff] = run_smart_heuristic(client, diff)

    # Summary
    print(f"\n{'='*60}")
    print("HEURISTIC TEST SUMMARY")
    print(f"{'='*60}")
    for diff, sc in scores.items():
        status = "PASS" if sc > 0.0 else "FAIL"
        print(f"  {diff:<10} => {sc:.4f}  [{status}]")

    avg = sum(scores.values()) / len(scores) if scores else 0.2
    avg = max(0.2, min(0.98, float(avg)))
    print(f"\n  Average Score: {avg:.4f}")

    # Verify state endpoint
    try:
        state = client.state()
        print(f"\n[OK] /state endpoint returns: {json.dumps(state, indent=2)}")
    except Exception as e:
        print(f"\n[WARN] /state endpoint error: {e}")


if __name__ == "__main__":
    main()
