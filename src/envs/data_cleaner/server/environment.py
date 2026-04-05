import pandas as pd
import numpy as np
import re
import os
from uuid import uuid4
from typing import Tuple, Dict, Any, Optional

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        pass

from ..models import (
    DataCleanerAction,
    DataCleanerObservation,
    DataCleanerState,
    ActionType,
)

# Step limits per difficulty
STEP_LIMITS = {"easy": 15, "medium": 25, "hard": 40}




class DataCleanerEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self._state: Optional[DataCleanerState] = None
        self.df: Optional[pd.DataFrame] = None
        self.perfect_df: Optional[pd.DataFrame] = None
        self.done: bool = False
        self.difficulty: str = "easy"
        self._last_similarity: float = 0.0
        self._null_cache: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def _load_raw_csv(self, dataset_path: str = None) -> pd.DataFrame:
        """Load a CSV dataset. If dataset_path is provided, use it."""
        if dataset_path and os.path.exists(dataset_path):
            return pd.read_csv(dataset_path, na_values=["NA", "N/A", "null"])

        candidates = []
        # Search locally and in the parent directory
        search_dirs = [
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))),
            "."
        ]
        
        for d in search_dirs:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith('.csv'):
                        candidates.append(os.path.join(d, f))
        
        # Deduplicate paths and filter
        candidates = list(set(candidates))
        if not candidates:
            raise FileNotFoundError("No CSV files found in environment directories.")
            
        import random
        candidates.sort() # Ensure reproducible ordering
        random.seed(42)
        path = random.choice(candidates)
        # Read with common NA patterns
        return pd.read_csv(path, na_values=["NA", "N/A", "null"])

    def _generate_dataset(
        self, difficulty: str = "easy", dataset_path: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw = self._load_raw_csv(dataset_path)

        # Ensure id column
        if "id" not in raw.columns:
            raw.insert(0, "id", range(1, len(raw) + 1))

        # --- Build perfect answer key ---
        perfect_df = raw.copy()
        for col in perfect_df.columns:
            if col == "id":
                continue
            if perfect_df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(perfect_df[col]):
                    perfect_df[col] = perfect_df[col].fillna(perfect_df[col].mean())
                else:
                    mode_val = perfect_df[col].mode()
                    if not mode_val.empty:
                        perfect_df[col] = perfect_df[col].fillna(mode_val.iloc[0])

        # --- Subset columns dynamically by difficulty ---
        cols = list(raw.columns)
        if difficulty == "easy" and len(cols) > 6:
            cols = cols[:6]
        elif difficulty == "medium" and len(cols) > 10:
            cols = cols[:10]
            
        messy_df = raw[cols].copy()
        perfect_df = perfect_df[cols].copy()

        # --- Inject additional mess for medium/hard ---
        if difficulty in ("medium", "hard"):
            # Inject duplicate rows (copy first 5 rows to end)
            dup_count = min(5, len(messy_df))
            dup_rows = messy_df.head(dup_count).copy()
            messy_df = pd.concat([messy_df, dup_rows], ignore_index=True)

        if difficulty == "hard":
            # Add whitespace noise to any object columns
            for col in messy_df.select_dtypes(include=["object"]).columns:
                messy_df[col] = messy_df[col].apply(
                    lambda x: f"  {x}  " if pd.notna(x) else x
                )
                # Perfect df has clean strings
                perfect_df[col] = perfect_df[col].apply(
                    lambda x: str(x).strip().lower() if pd.notna(x) else x
                )

        return messy_df, perfect_df

    # ------------------------------------------------------------------
    # Similarity scoring (0.0 – 1.0)
    # ------------------------------------------------------------------

    def _compute_similarity(self) -> float:
        """Vectorized cell-level similarity between current df and perfect_df."""
        if self.df is None or self.perfect_df is None:
            return 0.0

        try:
            # Use views to avoid high memory usage
            current = self.df
            perfect = self.perfect_df

            # Match columns
            common_cols = sorted(set(current.columns) & set(perfect.columns))
            if not common_cols:
                return 0.0

            # Match rows by id if possible
            if "id" in common_cols:
                current = current.sort_values("id")
                perfect = perfect.sort_values("id")

                common_ids = set(current["id"]) & set(perfect["id"])
                # We do copy here to align exactly
                current = current[current["id"].isin(common_ids)].reset_index(drop=True)
                perfect = perfect[perfect["id"].isin(common_ids)].reset_index(drop=True)
            else:
                min_len = min(len(current), len(perfect))
                # Truncate
                current = current.head(min_len).reset_index(drop=True)
                perfect = perfect.head(min_len).reset_index(drop=True)

            if current.empty or perfect.empty:
                return 0.0

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

        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Null cache (avoid recomputing every observation)
    # ------------------------------------------------------------------

    def _invalidate_cache(self):
        self._null_cache = None

    def _get_null_counts(self) -> Dict[str, int]:
        if self._null_cache is None and self.df is not None:
            counts = self.df.isnull().sum().to_dict()
            # Only include columns with nulls (cleaner for LLM)
            self._null_cache = {k: v for k, v in counts.items() if v > 0}
        return self._null_cache or {}

    # ------------------------------------------------------------------
    # Core API: reset / step / state
    # ------------------------------------------------------------------

    def reset(self, difficulty: str = "easy", dataset_path: str = None) -> DataCleanerObservation:
        self.difficulty = difficulty if difficulty in STEP_LIMITS else "easy"
        self.df, self.perfect_df = self._generate_dataset(self.difficulty, dataset_path)
        self.done = False
        self._state = DataCleanerState(
            episode_id=str(uuid4()),
            step_count=0,
            total_reward=0.0,
            difficulty=self.difficulty,
        )
        self._invalidate_cache()
        self._last_similarity = self._compute_similarity()
        return self._get_observation(
            f"Environment reset. Difficulty: {self.difficulty}. "
            f"Dataset has {len(self.df)} rows and {len(self.df.columns)} columns.",
            0.0,
        )

    def state(self) -> dict:
        """Return current environment state as a dict."""
        if self._state is None:
            return {
                "episode_id": "",
                "step_count": 0,
                "total_reward": 0.0,
                "difficulty": "easy",
            }
        return self._state.model_dump()

    def _get_observation(self, feedback: str, reward: float) -> DataCleanerObservation:
        max_steps = STEP_LIMITS.get(self.difficulty, 50)
        step_count = self._state.step_count if self._state else 0

        if self.df is not None:
            metadata = {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "columns": list(self.df.columns),
                "null_counts": self._get_null_counts(),
                "duplicate_row_count": int(self.df.duplicated().sum()),
                "dtypes": {
                    col: str(dtype)
                    for col, dtype in self.df.dtypes.items()
                },
            }
            # Cap current_view columns at 10 for large datasets
            view_df = self.df.head(5)
            if len(view_df.columns) > 10:
                view_df = view_df[list(view_df.columns)[:10]]
            current_view = (
                view_df.replace({np.nan: None}).to_dict(orient="records")
            )
        else:
            metadata = {}
            current_view = []

        return DataCleanerObservation(
            metadata=metadata,
            current_view=current_view,
            feedback=feedback,
            done=self.done,
            reward=reward,
            step_count=step_count,
            max_steps=max_steps,
            difficulty=self.difficulty,
        )

    def step(self, action: DataCleanerAction) -> DataCleanerObservation:
        if self.done:
            return self._get_observation("Episode is already finished.", 0.0)

        max_steps = STEP_LIMITS.get(self.difficulty, 50)
        self._state.step_count += 1

        # Auto-submit if step limit exceeded
        if self._state.step_count > max_steps:
            return self._do_submit(
                f"Step limit ({max_steps}) exceeded. Auto-submitting dataset."
            )

        reward = 0.0
        feedback = ""

        old_similarity = self._last_similarity

        try:
            if action.action_type == ActionType.DROP_COLUMN:
                feedback, reward = self._action_drop_column(action.target_column)

            elif action.action_type == ActionType.REMOVE_DUPLICATES:
                feedback, reward = self._action_remove_duplicates()

            elif action.action_type == ActionType.FORMAT_PHONE:
                feedback, reward = self._action_format_phone(action.target_column)

            elif action.action_type == ActionType.FORMAT_DATE:
                feedback, reward = self._action_format_date(action.target_column)

            elif action.action_type == ActionType.IMPUTE_MEAN:
                feedback, reward = self._action_impute_mean(action.target_column)

            elif action.action_type == ActionType.IMPUTE_MEDIAN:
                feedback, reward = self._action_impute_median(action.target_column)

            elif action.action_type == ActionType.FILL_MODE:
                feedback, reward = self._action_fill_mode(action.target_column)

            elif action.action_type == ActionType.STANDARDIZE_TEXT:
                feedback, reward = self._action_standardize_text(action.target_column)

            elif action.action_type == ActionType.SUBMIT_DATASET:
                return self._do_submit("Agent submitted the dataset.")

        except Exception as e:
            feedback = f"Action generated an error: {str(e)}"
            reward = 0.0

        # Compute delta reward from similarity improvement
        if reward == 0.0 and feedback:
            # No explicit reward set — compute from similarity delta
            new_sim = self._compute_similarity()
            delta = new_sim - old_similarity
            reward = round(max(delta, 0.0), 4)  # Only positive progress
            self._last_similarity = new_sim

        self._state.total_reward += reward
        return self._get_observation(feedback, reward)

    # ------------------------------------------------------------------
    # Submit logic
    # ------------------------------------------------------------------

    def _do_submit(self, context_msg: str) -> DataCleanerObservation:
        similarity = self._compute_similarity()
        reward = round(similarity, 4)  # 0.0 – 1.0
        self.done = True
        self._state.total_reward += reward

        if similarity >= 0.99:
            feedback = f"{context_msg} Perfect match! Score: {similarity:.4f}"
        elif similarity >= 0.8:
            feedback = f"{context_msg} Good match. Score: {similarity:.4f}"
        else:
            feedback = f"{context_msg} Partial match. Score: {similarity:.4f}"

        return self._get_observation(feedback, reward)

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _action_drop_column(self, col: Optional[str]) -> Tuple[str, float]:
        if col is None or col not in self.df.columns:
            return f"Column '{col}' not found.", 0.0

        if self.df[col].isnull().all():
            self.df = self.df.drop(columns=[col])
            self._invalidate_cache()
            return f"Successfully dropped empty column: {col}", 0.0  # delta computed later
        else:
            self.df = self.df.drop(columns=[col])
            self._invalidate_cache()
            # Penalty for dropping non-empty column updated to 0.0 to prevent negative scores
            return f"Dropped non-empty column: {col} (penalty applied)", 0.0

    def _action_remove_duplicates(self) -> Tuple[str, float]:
        initial_len = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        final_len = len(self.df)
        self._invalidate_cache()
        if final_len < initial_len:
            removed = initial_len - final_len
            return f"Removed {removed} duplicate rows.", 0.0  # delta computed later
        return "No duplicates found.", 0.0

    def _action_format_phone(self, col: Optional[str]) -> Tuple[str, float]:
        if col is None or col not in self.df.columns:
            return f"Column '{col}' not found.", 0.0

        def fix_phone(x):
            if pd.isna(x):
                return x
            numbers = re.sub(r"\D", "", str(x))
            if len(numbers) == 10:
                return f"+1-{numbers[:3]}-{numbers[3:6]}-{numbers[6:]}"
            elif len(numbers) == 11 and numbers.startswith("1"):
                return f"+1-{numbers[1:4]}-{numbers[4:7]}-{numbers[7:]}"
            return x

        old = self.df[col].copy()
        self.df[col] = self.df[col].apply(fix_phone)
        self._invalidate_cache()
        if not self.df[col].equals(old):
            return f"Formatted phone numbers in column: {col}.", 0.0
        return "No phone numbers needed formatting.", 0.0

    def _action_format_date(self, col: Optional[str]) -> Tuple[str, float]:
        if col is None or col not in self.df.columns:
            return f"Column '{col}' not found.", 0.0
        try:
            old = self.df[col].copy()
            self.df[col] = pd.to_datetime(self.df[col]).dt.strftime("%Y-%m-%d")
            self._invalidate_cache()
            if not self.df[col].equals(old):
                return f"Formatted dates in column: {col}.", 0.0
            return "No dates needed formatting.", 0.0
        except Exception as e:
            return f"Error formatting dates: {str(e)}", 0.0

    def _action_impute_mean(self, col: Optional[str]) -> Tuple[str, float]:
        if col is None or col not in self.df.columns:
            return f"Column '{col}' not found.", 0.0
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            return f"Column '{col}' is not numeric.", 0.0
        if not self.df[col].isnull().any():
            return f"No nulls in column '{col}'.", 0.0

        mean_val = self.df[col].mean()
        self.df[col] = self.df[col].fillna(mean_val)
        self._invalidate_cache()
        return f"Imputed mean ({mean_val:.4f}) for nulls in column: {col}.", 0.0

    def _action_impute_median(self, col: Optional[str]) -> Tuple[str, float]:
        if col is None or col not in self.df.columns:
            return f"Column '{col}' not found.", 0.0
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            return f"Column '{col}' is not numeric.", 0.0
        if not self.df[col].isnull().any():
            return f"No nulls in column '{col}'.", 0.0

        median_val = self.df[col].median()
        self.df[col] = self.df[col].fillna(median_val)
        self._invalidate_cache()
        return f"Imputed median ({median_val:.4f}) for nulls in column: {col}.", 0.0

    def _action_fill_mode(self, col: Optional[str]) -> Tuple[str, float]:
        if col is None or col not in self.df.columns:
            return f"Column '{col}' not found.", 0.0
        if not self.df[col].isnull().any():
            return f"No nulls in column '{col}'.", 0.0

        mode_val = self.df[col].mode()
        if mode_val.empty:
            return f"Could not compute mode for column '{col}'.", 0.0
        self.df[col] = self.df[col].fillna(mode_val.iloc[0])
        self._invalidate_cache()
        return f"Filled nulls with mode ({mode_val.iloc[0]}) in column: {col}.", 0.0

    def _action_standardize_text(self, col: Optional[str]) -> Tuple[str, float]:
        if col is None or col not in self.df.columns:
            return f"Column '{col}' not found.", 0.0

        old = self.df[col].copy()
        self.df[col] = self.df[col].apply(
            lambda x: str(x).strip().lower() if pd.notna(x) else x
        )
        self._invalidate_cache()
        if not self.df[col].equals(old):
            return f"Standardized text in column: {col}.", 0.0
        return f"No text changes needed in column '{col}'.", 0.0


# ------------------------------------------------------------------
# Standalone Task Graders
# ------------------------------------------------------------------
def grade_data_cleaning_easy(*args, **kwargs) -> float:
    # Programmatic grader for easy task
    env = kwargs.get("env")
    if env and hasattr(env, "_compute_similarity"):
        return env._compute_similarity()
    return 0.0

def grade_data_cleaning_medium(*args, **kwargs) -> float:
    # Programmatic grader for medium task
    env = kwargs.get("env")
    if env and hasattr(env, "_compute_similarity"):
        return env._compute_similarity()
    return 0.0

def grade_data_cleaning_hard(*args, **kwargs) -> float:
    # Programmatic grader for hard task
    env = kwargs.get("env")
    if env and hasattr(env, "_compute_similarity"):
        return env._compute_similarity()
    return 0.0
