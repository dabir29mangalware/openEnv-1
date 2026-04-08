import uvicorn
from src.envs.data_cleaner.server.app import app

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
