from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    from core.env_server import Action, Observation, State
except ImportError:
    # Fallback generic base classes if core.env_server is not installed natively
    @dataclass
    class Action: pass
    @dataclass
    class Observation: pass
    @dataclass
    class State: pass

class ActionType(str, Enum):
    DROP_COLUMN = "DROP_COLUMN"
    REMOVE_DUPLICATES = "REMOVE_DUPLICATES"
    FORMAT_PHONE = "FORMAT_PHONE"
    FORMAT_DATE = "FORMAT_DATE"
    IMPUTE_MEAN = "IMPUTE_MEAN"
    SUBMIT_DATASET = "SUBMIT_DATASET"

@dataclass
class DataCleanerAction(Action):
    action_type: ActionType
    target_column: Optional[str] = None

@dataclass
class DataCleanerObservation(Observation):
    metadata: Dict[str, Any]
    current_view: List[Dict[str, Any]]
    feedback: str
    done: bool
    reward: float

@dataclass
class DataCleanerState(State):
    episode_id: str
    step_count: int
    total_reward: float
