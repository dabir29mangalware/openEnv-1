from fastapi import FastAPI
from .environment import DataCleanerEnvironment
import dataclasses

def create_fastapi_app(env: DataCleanerEnvironment) -> FastAPI:
    app = FastAPI(title="The Automated Data Cleaner Environment")

    @app.post("/reset")
    def reset():
        obs = env.reset()
        return dataclasses.asdict(obs)

    @app.post("/step")
    def step(action: dict):
        from ..models import DataCleanerAction, ActionType
        act_type = ActionType(action.get("action_type"))
        target_col = action.get("target_column")
        act = DataCleanerAction(action_type=act_type, target_column=target_col)
        
        obs = env.step(act)
        return dataclasses.asdict(obs)

    return app

# Expose global app for uvicorn
env = DataCleanerEnvironment()
app = create_fastapi_app(env)
