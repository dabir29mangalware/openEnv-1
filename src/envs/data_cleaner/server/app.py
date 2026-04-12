from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import tempfile
from .environment import DataCleanerEnvironment
from ..models import DataCleanerAction, ActionType


def create_fastapi_app(env: DataCleanerEnvironment) -> FastAPI:
    app = FastAPI(title="The Automated Data Cleaner Environment", version="1.0.0")

    @app.get("/health")
    def health():
        """Health check endpoint — must return 200 with status healthy."""
        return {"status": "healthy"}

    @app.get("/metadata")
    def metadata():
        """Return environment metadata."""
        return {
            "name": "data_cleaner",
            "description": "Automated Data Cleaner - An RL environment for training AI agents to clean messy real-world datasets",
            "version": "2.0.0",
        }

    @app.get("/schema")
    def schema():
        """Return JSON schemas for action, observation, and state."""
        from ..models import DataCleanerObservation, DataCleanerState
        return {
            "action": DataCleanerAction.model_json_schema(),
            "observation": DataCleanerObservation.model_json_schema(),
            "state": DataCleanerState.model_json_schema(),
        }

    @app.post("/upload")
    def upload_dataset(file: UploadFile = File(...)):
        """Upload a CSV dataset to interact with in the environment."""
        if not file.filename.endswith('.csv'):
            return JSONResponse(status_code=400, content={"error": "Only CSV files are allowed"})

        # Save securely locally
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"message": "File uploaded successfully", "dataset_path": file_path}

    @app.post("/reset")
    def reset(
        difficulty: str = Query(default="easy", pattern="^(easy|medium|hard)$"),
        dataset_path: str = None
    ):
        """Reset the environment with a given difficulty level."""
        obs = env.reset(difficulty=difficulty, dataset_path=dataset_path)
        obs_dict = obs.model_dump()
        # Extract reward and done to top level (standard OpenEnv format)
        reward = obs_dict.pop("reward", 0.5)
        done = obs_dict.pop("done", False)
        # Clamp reward strictly between 0 and 1
        reward = max(0.2222, min(0.8888, float(reward)))
        return {
            "observation": obs_dict,
            "reward": reward,
            "done": done,
        }

    @app.post("/step")
    def step(action: dict):
        """Execute one action in the environment."""
        try:
            act_type = ActionType(action.get("action_type"))
        except (ValueError, KeyError):
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid action_type: {action.get('action_type')}"},
            )

        target_col = action.get("target_column")
        act = DataCleanerAction(action_type=act_type, target_column=target_col)
        obs = env.step(act)

        obs_dict = obs.model_dump()
        # Extract reward and done to top level (standard OpenEnv format)
        reward = obs_dict.pop("reward", 0.5)
        done = obs_dict.pop("done", False)
        # Clamp reward strictly between 0 and 1
        reward = max(0.2222, min(0.8888, float(reward)))

        return {
            "observation": obs_dict,
            "reward": reward,
            "done": done,
            "info": {}
        }

    @app.get("/state")
    def get_state():
        """Return the current environment state."""
        return env.state()

    return app


# Expose global app for uvicorn
env = DataCleanerEnvironment()
app = create_fastapi_app(env)
