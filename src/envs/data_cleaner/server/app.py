from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from .environment import DataCleanerEnvironment
from ..models import DataCleanerAction, ActionType


def create_fastapi_app(env: DataCleanerEnvironment) -> FastAPI:
    app = FastAPI(title="The Automated Data Cleaner Environment")

    @app.get("/health")
    def health():
        """Health check endpoint — must return 200 for HF Spaces ping."""
        return {"status": "ok"}

    @app.post("/reset")
    def reset(difficulty: str = Query(default="easy", pattern="^(easy|medium|hard)$")):
        """Reset the environment with a given difficulty level."""
        obs = env.reset(difficulty=difficulty)
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
        return obs.model_dump()

    @app.get("/state")
    def get_state():
        """Return the current environment state."""
        return env.state()

    return app


# Expose global app for uvicorn
env = DataCleanerEnvironment()
app = create_fastapi_app(env)
