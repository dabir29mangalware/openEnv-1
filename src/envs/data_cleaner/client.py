import requests
from typing import Dict, Any

try:
    from core.env_server import HTTPEnvClient
except ImportError:
    class HTTPEnvClient:
        def __init__(self, base_url: str):
            self.base_url = base_url

from .models import DataCleanerObservation, DataCleanerAction

class DataCleanerClient(HTTPEnvClient):
    def __init__(self, base_url: str):
        super().__init__(base_url)

    def reset(self) -> DataCleanerObservation:
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return DataCleanerObservation(**response.json())

    def step(self, action: dict) -> DataCleanerObservation:
        response = requests.post(f"{self.base_url}/step", json=action)
        response.raise_for_status()
        return DataCleanerObservation(**response.json())
