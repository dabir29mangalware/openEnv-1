import requests
from typing import Dict, Any, Optional
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

try:
    from openenv.core.env_server import HTTPEnvClient
except ImportError:
    class HTTPEnvClient:
        def __init__(self, base_url: str):
            self.base_url = base_url

from .models import DataCleanerObservation


class DataCleanerClient(HTTPEnvClient):
    def __init__(self, base_url: str, timeout: int = 30):
        super().__init__(base_url)
        self.timeout = timeout

        # Connection pooling + retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def reset(self, difficulty: str = "easy", dataset_path: str = None) -> DataCleanerObservation:
        params = {"difficulty": difficulty}
        if dataset_path:
            params["dataset_path"] = dataset_path
            
        response = self.session.post(
            f"{self.base_url}/reset",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return DataCleanerObservation(**response.json())

    def upload(self, file_path: str) -> str:
        with open(file_path, "rb") as f:
            response = self.session.post(
                f"{self.base_url}/upload",
                files={"file": (file_path.split("/")[-1], f, "text/csv")},
                timeout=self.timeout
            )
        response.raise_for_status()
        return response.json()["dataset_path"]

    def step(self, action: dict) -> tuple:
        response = self.session.post(
            f"{self.base_url}/step",
            json=action,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        
        # Handle backward-compatibility for different response formats
        if "observation" in data:
            obs = DataCleanerObservation(**data["observation"])
            # Reward can be flat float or dict {"value": float}
            raw_reward = data["reward"]
            if isinstance(raw_reward, dict):
                reward = raw_reward["value"]
            else:
                reward = float(raw_reward)
            done = data["done"]
            info = data.get("info", {})
        else:
            obs = DataCleanerObservation(**data)
            reward = obs.reward
            done = obs.done
            info = {}
            
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/state",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> bool:
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=self.timeout,
            )
            return response.status_code == 200
        except Exception:
            return False
