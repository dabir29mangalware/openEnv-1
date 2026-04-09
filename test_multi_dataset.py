"""
Multi-Dataset Performance Test
Tests the Data Cleaner environment against multiple real-world datasets.
Profiles each dataset, injects messiness, runs the heuristic agent, and reports scores.

Usage:
    python test_multi_dataset.py
"""

import pandas as pd
import numpy as np
import time
import os
import sys
import traceback

import glob

# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Dataset Discovery
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_datasets():
    csv_files = glob.glob(os.path.join(BASE_DIR, "*.csv"))
    datasets = {}
    for filepath in csv_files:
        name = os.path.basename(filepath)
        datasets[name] = {
            "file": filepath,
            "description": f"Dynamic dataset target: {name}",
            "na_value": None,
        }
    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# Helper: profile a dataframe
# ─────────────────────────────────────────────────────────────────────────────
def profile_df(df: pd.DataFrame) -> dict:
    """Generate metadata for a dataframe (same format as the environment)."""
    null_counts = df.isnull().sum()
    null_dict = {col: int(v) for col, v in null_counts.items() if v > 0}
    dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "columns": list(df.columns),
        "null_counts": null_dict,
        "total_nulls": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": dtypes,
        "numeric_cols": numeric_cols,
        "text_cols": text_cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mess injection (mirrors environment.py logic)
# ─────────────────────────────────────────────────────────────────────────────
def inject_nulls(df: pd.DataFrame, frac: float = 0.08, seed: int = 42) -> pd.DataFrame:
    """Inject null values into ~frac of numeric cells."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mask = rng.rand(len(df)) < frac
        df.loc[mask, col] = np.nan
    return df


def inject_duplicates(df: pd.DataFrame, n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Inject n duplicate rows."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    dupes = df.iloc[idx].copy()
    return pd.concat([df, dupes], ignore_index=True)


def inject_whitespace_noise(df: pd.DataFrame, frac: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Add random whitespace to text columns."""
    rng = np.random.RandomState(seed)
    df = df.copy()
    text_cols = df.select_dtypes(include=["object"]).columns
    for col in text_cols:
        mask = rng.rand(len(df)) < frac
        df.loc[mask, col] = df.loc[mask, col].apply(
            lambda x: f"  {x}  " if isinstance(x, str) else x
        )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Cell-level similarity scoring (same as environment.py)
# ─────────────────────────────────────────────────────────────────────────────
def cell_similarity(current: pd.DataFrame, perfect: pd.DataFrame) -> float:
    """Calculate fraction of matching cells between current and perfect."""
    if current.shape != perfect.shape:
        # Handle row count mismatch (e.g. duplicates added)
        min_rows = min(len(current), len(perfect))
        current = current.iloc[:min_rows].reset_index(drop=True)
        perfect = perfect.iloc[:min_rows].reset_index(drop=True)

    common_cols = sorted(set(current.columns) & set(perfect.columns))
    total_cells = current.shape[0] * len(common_cols)
    if total_cells == 0:
        return 0.0

    matching = 0
    for col in common_cols:
        s_curr = current[col]
        s_perf = perfect[col]
        
        # Align NaNs
        both_na = s_curr.isna() & s_perf.isna()
        matching += both_na.sum()
        
        # Check valid items
        not_na = ~(s_curr.isna() | s_perf.isna())
        if not_na.any():
            if pd.api.types.is_numeric_dtype(s_curr) and pd.api.types.is_numeric_dtype(s_perf):
                matching += np.isclose(s_curr[not_na].astype(float), s_perf[not_na].astype(float), rtol=1e-5, atol=1e-8).sum()
            else:
                matching += (s_curr[not_na].astype(str).str.strip().str.lower() == s_perf[not_na].astype(str).str.strip().str.lower()).sum()

    return float(matching) / float(total_cells)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic cleaning agent
# ─────────────────────────────────────────────────────────────────────────────
def heuristic_clean(df: pd.DataFrame, max_steps: int = 40) -> tuple:
    """
    Run a heuristic cleaning strategy. Returns (cleaned_df, actions_taken, steps).
    Mirrors what the environment agent would do.
    """
    actions = []
    step = 0

    for _ in range(max_steps):
        step += 1
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        if len(cols_with_nulls) > 0:
            col = cols_with_nulls.index[0]
            n_nulls = int(cols_with_nulls.iloc[0])
            dtype = str(df[col].dtype)

            if "float" in dtype or "int" in dtype:
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                actions.append(f"IMPUTE_MEAN({col}) [{n_nulls} nulls]")
            else:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
                actions.append(f"FILL_MODE({col}) [{n_nulls} nulls]")
            continue

        dup_count = df.duplicated().sum()
        if dup_count > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            actions.append(f"REMOVE_DUPLICATES [{dup_count} rows]")
            continue

        # All clean
        break

    return df, actions, step


# ─────────────────────────────────────────────────────────────────────────────
# Main test runner
# ─────────────────────────────────────────────────────────────────────────────
def test_dataset(name: str, config: dict) -> dict:
    """Test one dataset and return results."""
    filepath = config["file"]

    if not os.path.exists(filepath):
        return {"name": name, "status": "SKIP", "error": f"File not found: {filepath}"}

    print(f"\n{'='*70}")
    print(f"  DATASET: {name}")
    print(f"  {config['description']}")
    print(f"{'='*70}")

    try:
        # Load
        t0 = time.time()
        if config.get("na_value"):
            df = pd.read_csv(filepath, na_values=[config["na_value"]])
        else:
            df = pd.read_csv(filepath)
        load_time = time.time() - t0

        # Profile original
        original_profile = profile_df(df)
        print(f"\n  📊 Original Profile:")
        print(f"     Shape: {original_profile['rows']} × {original_profile['cols']}")
        print(f"     Memory: {original_profile['memory_mb']} MB")
        print(f"     Existing Nulls: {original_profile['total_nulls']}")
        print(f"     Existing Duplicates: {original_profile['duplicate_rows']}")
        print(f"     Numeric columns: {len(original_profile['numeric_cols'])}")
        print(f"     Text columns: {len(original_profile['text_cols'])}")
        print(f"     Load time: {load_time*1000:.1f}ms")

        # Save perfect copy (before mess injection)
        perfect_df = df.copy()

        # Inject mess
        t0 = time.time()
        messy_df = inject_nulls(df.copy())
        messy_df = inject_duplicates(messy_df)
        messy_df = inject_whitespace_noise(messy_df)
        inject_time = time.time() - t0

        messy_profile = profile_df(messy_df)
        print(f"\n  🔧 After Mess Injection:")
        print(f"     Shape: {messy_profile['rows']} × {messy_profile['cols']}")
        print(f"     Injected Nulls: {messy_profile['total_nulls']}")
        print(f"     Injected Duplicates: {messy_profile['duplicate_rows']}")
        print(f"     Injection time: {inject_time*1000:.1f}ms")

        # Pre-cleaning similarity
        pre_similarity = cell_similarity(messy_df, perfect_df)
        print(f"     Pre-clean similarity: {pre_similarity:.4f}")

        # Run heuristic cleaning
        t0 = time.time()
        cleaned_df, actions, steps = heuristic_clean(messy_df.copy())
        clean_time = time.time() - t0

        # Post-cleaning similarity
        post_similarity = cell_similarity(cleaned_df, perfect_df)
        improvement = post_similarity - pre_similarity

        print(f"\n  🧹 Cleaning Results:")
        print(f"     Steps taken: {steps}")
        print(f"     Actions:")
        for i, act in enumerate(actions, 1):
            print(f"       {i:>2}. {act}")
        print(f"     Clean time: {clean_time*1000:.1f}ms")
        print(f"     Post-clean similarity: {post_similarity:.4f}")
        print(f"     Improvement: +{improvement:.4f}")

        # Final score (normalized)
        score = max(0.0001, min(0.9999, float(post_similarity)))
        status = "PASS" if score >= 0.5 else "FAIL"
        print(f"\n  ✅ Score: {score:.4f}  [{status}]")

        # Residual issues
        cleaned_profile = profile_df(cleaned_df)
        if cleaned_profile["total_nulls"] > 0:
            print(f"  ⚠️  Residual nulls: {cleaned_profile['total_nulls']}")
        if cleaned_profile["duplicate_rows"] > 0:
            print(f"  ⚠️  Residual duplicates: {cleaned_profile['duplicate_rows']}")

        return {
            "name": name,
            "status": status,
            "rows": original_profile["rows"],
            "cols": original_profile["cols"],
            "memory_mb": original_profile["memory_mb"],
            "pre_similarity": round(pre_similarity, 4),
            "post_similarity": round(post_similarity, 4),
            "improvement": round(improvement, 4),
            "score": round(score, 4),
            "steps": steps,
            "load_ms": round(load_time * 1000, 1),
            "clean_ms": round(clean_time * 1000, 1),
            "error": None,
        }

    except Exception as e:
        traceback.print_exc()
        return {"name": name, "status": "ERROR", "error": str(e)}


def main():
    print("=" * 70)
    print("  DATA CLEANER — MULTI-DATASET PERFORMANCE TEST")
    print("=" * 70)

    datasets = get_datasets()
    if not datasets:
        print("  No .csv datasets found in the directory!")
        return

    results = []
    for name, config in datasets.items():
        result = test_dataset(name, config)
        results.append(result)

    # Summary table
    print(f"\n\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Dataset':<15} {'Rows':>6} {'Cols':>5} {'MB':>6} {'Pre':>7} {'Post':>7} {'Score':>7} {'Steps':>6} {'Time':>8} {'Status':>7}")
    print(f"  {'-'*15} {'-'*6} {'-'*5} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*7}")

    total_score = 0
    passed = 0
    for r in results:
        if r.get("error") and r["status"] in ("SKIP", "ERROR"):
            print(f"  {r['name']:<15} {'—':>6} {'—':>5} {'—':>6} {'—':>7} {'—':>7} {'—':>7} {'—':>6} {'—':>8} {r['status']:>7}")
        else:
            total_score += r["score"]
            if r["status"] == "PASS":
                passed += 1
            print(
                f"  {r['name']:<15} {r['rows']:>6} {r['cols']:>5} {r['memory_mb']:>6} "
                f"{r['pre_similarity']:>7.4f} {r['post_similarity']:>7.4f} {r['score']:>7.4f} "
                f"{r['steps']:>6} {r['clean_ms']:>7.1f}ms {r['status']:>7}"
            )

    n = len([r for r in results if r["status"] not in ("SKIP", "ERROR")])
    avg_score = total_score / n if n > 0 else 0.0
    print(f"\n  Average Score: {avg_score:.4f} | Passed: {passed}/{n}")

    # Performance assessment
    print(f"\n  📈 Performance Assessment:")
    if avg_score >= 0.9:
        print("     🟢 EXCELLENT — Agent handles all datasets efficiently")
    elif avg_score >= 0.7:
        print("     🟡 GOOD — Agent works well but has room for improvement")
    elif avg_score >= 0.5:
        print("     🟠 ACCEPTABLE — Agent passes minimum threshold")
    else:
        print("     🔴 NEEDS WORK — Agent below minimum performance threshold")


if __name__ == "__main__":
    main()
