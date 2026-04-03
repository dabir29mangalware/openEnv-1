import pandas as pd
import numpy as np
import re
from uuid import uuid4
from typing import Tuple, Dict, Any

try:
    from core.env_server import Environment
except ImportError:
    class Environment:
        pass

from ..models import (
    DataCleanerAction, 
    DataCleanerObservation, 
    DataCleanerState, 
    ActionType
)

class DataCleanerEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.state = None
        self.df = None
        self.perfect_df = None
        self.done = False

    def _generate_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load the custom Housing Data CSV
        try:
            messy_df = pd.read_csv("HousingData (1).csv")
        except FileNotFoundError:
            # Fallback if run from a different directory
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
            messy_df = pd.read_csv(os.path.join(base_dir, "HousingData (1).csv"))
            
        # Add id column for consistent sorting during submission check
        if "id" not in messy_df.columns:
            messy_df.insert(0, "id", range(1, len(messy_df) + 1))
            
        perfect_df = messy_df.copy()
        
        # Build the answer key: all missing numeric values filled with their mean
        for col in perfect_df.columns:
            if perfect_df[col].isnull().any() and pd.api.types.is_numeric_dtype(perfect_df[col]):
                perfect_df[col] = perfect_df[col].fillna(perfect_df[col].mean())
                
        return messy_df, perfect_df

    def reset(self) -> DataCleanerObservation:
        self.df, self.perfect_df = self._generate_dataset()
        self.done = False
        self.state = DataCleanerState(
            episode_id=str(uuid4()),
            step_count=0,
            total_reward=0.0
        )
        return self._get_observation("Environment reset generated a new dataset.", 0.0)

    def _get_observation(self, feedback: str, reward: float) -> DataCleanerObservation:
        if self.df is not None:
            metadata = {
                "total_rows": len(self.df),
                "columns": list(self.df.columns),
                "null_counts": self.df.isnull().sum().to_dict()
            }
            # Handle potential NaNs in current_view
            current_view = self.df.head(5).replace({np.nan: None}).to_dict(orient="records")
        else:
            metadata = {}
            current_view = []

        return DataCleanerObservation(
            metadata=metadata,
            current_view=current_view,
            feedback=feedback,
            done=self.done,
            reward=reward
        )

    def step(self, action: DataCleanerAction) -> DataCleanerObservation:
        if self.done:
            return self._get_observation("Episode is already finished.", 0.0)
            
        self.state.step_count += 1
        reward = 0.0
        feedback = ""
        
        try:
            if action.action_type == ActionType.DROP_COLUMN:
                if action.target_column in self.df.columns:
                    if self.df[action.target_column].isnull().all():
                        self.df = self.df.drop(columns=[action.target_column])
                        reward = 0.2
                        feedback = f"Successfully dropped empty column: {action.target_column}"
                    else:
                        self.df = self.df.drop(columns=[action.target_column])
                        reward = -0.1
                        feedback = f"Dropped non-empty valid column: {action.target_column}"
                else:
                    feedback = f"Column {action.target_column} not found."
            
            elif action.action_type == ActionType.REMOVE_DUPLICATES:
                initial_len = len(self.df)
                self.df = self.df.drop_duplicates()
                final_len = len(self.df)
                if final_len < initial_len:
                    reward = 0.2
                    feedback = f"Removed {initial_len - final_len} duplicate rows."
                else:
                    feedback = "No duplicates found."
                    
            elif action.action_type == ActionType.FORMAT_PHONE:
                col = action.target_column
                if col in self.df.columns:
                    def fix_phone(x):
                        if pd.isna(x): return x
                        numbers = re.sub(r'\D', '', str(x))
                        if len(numbers) == 10:
                            return f"+1-{numbers[:3]}-{numbers[3:6]}-{numbers[6:]}"
                        elif len(numbers) == 11 and numbers.startswith('1'):
                            return f"+1-{numbers[1:4]}-{numbers[4:7]}-{numbers[7:]}"
                        return x

                    old_phones = self.df[col].copy()
                    self.df[col] = self.df[col].apply(fix_phone)
                    if not self.df[col].equals(old_phones):
                        reward = 0.2
                        feedback = f"Formatted phone numbers in column: {col}."
                    else:
                        feedback = "No phone numbers needed formatting."
                else:
                    feedback = f"Column {col} not found."
                    
            elif action.action_type == ActionType.FORMAT_DATE:
                col = action.target_column
                if col in self.df.columns:
                    old_dates = self.df[col].copy()
                    try:
                        self.df[col] = pd.to_datetime(self.df[col]).dt.strftime("%Y-%m-%d")
                        if not self.df[col].equals(old_dates):
                            reward = 0.2
                            feedback = f"Formatted dates in column: {col}."
                        else:
                            feedback = "No dates needed formatting."
                    except Exception as e:
                        feedback = f"Error formatting dates: {str(e)}"
                else:
                    feedback = f"Column {col} not found."
                    
            elif action.action_type == ActionType.IMPUTE_MEAN:
                col = action.target_column
                if col in self.df.columns and pd.api.types.is_numeric_dtype(self.df[col]):
                    if self.df[col].isnull().any():
                        mean_val = self.df[col].mean()
                        self.df[col] = self.df[col].fillna(mean_val)
                        reward = 0.2
                        feedback = f"Imputed mean for nulls in column: {col}."
                    else:
                        feedback = f"No nulls found in column: {col}."
                else:
                    feedback = f"Column {col} not found or not numeric."
            
            elif action.action_type == ActionType.SUBMIT_DATASET:
                self.df = self.df.sort_values('id').reset_index(drop=True)
                sorted_perfect = self.perfect_df.sort_values('id').reset_index(drop=True)
                
                # Check frame equal
                try:
                    pd.testing.assert_frame_equal(self.df, sorted_perfect, check_dtype=False)
                    reward = 1.0
                    feedback = "Perfect match! Data cleaning successful."
                except AssertionError:
                    reward = 0.0
                    feedback = "Dataset submitted but does not match the perfect dataset."
                
                self.done = True
                
        except Exception as e:
            feedback = f"Action generated an error: {str(e)}"
            reward = 0.0

        self.state.total_reward += reward
        return self._get_observation(feedback, reward)
