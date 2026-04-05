from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

# Import OpenEnv base classes (Pydantic-based)
try:
    from openenv.core.env_server import Action, Observation, State, Reward
except ImportError:
    # Fallback Pydantic base classes if openenv-core is not installed
    class Action(BaseModel):
        pass

    class Observation(BaseModel):
        pass

    class State(BaseModel):
        pass

    class Reward(BaseModel):
        pass


class ActionType(str, Enum):
    DROP_COLUMN = "DROP_COLUMN"
    REMOVE_DUPLICATES = "REMOVE_DUPLICATES"
    FORMAT_PHONE = "FORMAT_PHONE"
    FORMAT_DATE = "FORMAT_DATE"
    IMPUTE_MEAN = "IMPUTE_MEAN"
    IMPUTE_MEDIAN = "IMPUTE_MEDIAN"
    FILL_MODE = "FILL_MODE"
    STANDARDIZE_TEXT = "STANDARDIZE_TEXT"
    SUBMIT_DATASET = "SUBMIT_DATASET"


class DataCleanerAction(Action):
    action_type: ActionType
    target_column: Optional[str] = None


class DataCleanerObservation(Observation):
    metadata: Dict[str, Any]
    current_view: List[Dict[str, Any]]
    feedback: str
    done: bool
    reward: float
    step_count: int = 0
    max_steps: int = 50
    difficulty: str = "easy"


class DataCleanerState(State):
    episode_id: str
    step_count: int
    total_reward: float
    difficulty: str = "easy"


class DataCleanerReward(Reward):
    value: float
