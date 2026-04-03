FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy fastapi uvicorn requests pydantic openai

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.envs.data_cleaner.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
