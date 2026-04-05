from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import JSONResponse
import os
import shutil
import tempfile
from .environment import DataCleanerEnvironment
from ..models import DataCleanerAction, ActionType


def create_fastapi_app(env: DataCleanerEnvironment) -> FastAPI:
    app = FastAPI(title="The Automated Data Cleaner Environment")

    @app.get("/health")
    def health():
        """Health check endpoint — must return 200 for HF Spaces ping."""
        return {"status": "ok"}

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
        return obs.model_dump()

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
        
        from ..models import DataCleanerReward
        reward_obj = DataCleanerReward(value=obs.reward)

        return {
            "observation": obs.model_dump(),
            "reward": reward_obj.model_dump(),
            "done": obs.done,
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
