# 🧹 Automated Data Cleaner — OpenEnv Environment

An RL environment for training AI agents to clean messy real-world datasets. Built with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

**Team: Gradient Ascenders** | Built for the SST × Meta × Hugging Face OpenEnv Hackathon

---

## Environment Description

Data cleaning is one of the most common and time-consuming tasks in real-world data science workflows. This environment simulates a realistic data cleaning scenario where an AI agent must:

- **Detect and fix missing values** (null imputation)
- **Remove duplicate rows**
- **Standardize text formatting** (whitespace, casing)
- **Format specialized columns** (phone numbers, dates)

The environment can use any target CSV. By default it randomly loads a local `.csv` file with programmatically injected messiness that increases with difficulty level.

---

## Action Space

The agent can take the following actions:

| Action | Requires `target_column` | Description |
|--------|:------------------------:|-------------|
| `IMPUTE_MEAN` | ✅ | Fill nulls in a numeric column with the column mean |
| `IMPUTE_MEDIAN` | ✅ | Fill nulls in a numeric column with the column median |
| `FILL_MODE` | ✅ | Fill nulls in any column with the most frequent value |
| `REMOVE_DUPLICATES` | ❌ | Remove all duplicate rows |
| `STANDARDIZE_TEXT` | ✅ | Strip whitespace and lowercase a text column |
| `DROP_COLUMN` | ✅ | Drop a column (rewards only if entirely empty) |
| `FORMAT_PHONE` | ✅ | Format phone numbers to `+1-XXX-XXX-XXXX` |
| `FORMAT_DATE` | ✅ | Format dates to `YYYY-MM-DD` |
| `SUBMIT_DATASET` | ❌ | Submit the cleaned dataset for final scoring |

**Action format:** `{"action_type": "IMPUTE_MEAN", "target_column": "target_col_name"}`

---

## Observation Space

Each step returns a `DataCleanerObservation` with:

| Field | Type | Description |
|-------|------|-------------|
| `metadata` | `dict` | Dataset stats: row count, column list, null counts, dtype info, duplicate count |
| `current_view` | `list[dict]` | First 5 rows of the dataset (for LLM context) |
| `feedback` | `str` | Human-readable feedback on the last action's result |
| `reward` | `float` | Reward for the last action (0.0–1.0 range) |
| `done` | `bool` | Whether the episode has ended |
| `step_count` | `int` | Current step number |
| `max_steps` | `int` | Maximum steps allowed for this difficulty |
| `difficulty` | `str` | Current difficulty level |

---

## Tasks & Difficulty Levels

| Task | Difficulty | Columns | Mess Injected | Max Steps | Expected Score |
|------|-----------|---------|---------------|-----------|----------------|
| `data_cleaning_easy` | Easy | 6 random cols | Nulls in 3 numeric columns | 15 | 0.8–1.0 |
| `data_cleaning_medium` | Medium | 10 random cols | Nulls in 7 columns + 5 duplicate rows | 25 | 0.6–0.9 |
| `data_cleaning_hard` | Hard | All cols | All nulls + duplicates + whitespace noise | 40 | 0.4–0.8 |

---

## Reward Function

The reward provides **meaningful partial progress signals**:

- **Per-step reward** = `new_similarity - old_similarity` (cell-level match improvement vs perfect dataset)
- **Submit reward** = final `matching_cells / total_cells` ratio (0.0–1.0)
- **Penalty** = -0.05 for destructive actions (dropping non-empty columns)
- **Auto-submit** if step limit exceeded (partial score preserved)

This ensures the agent receives continuous feedback, not just a binary end-of-episode signal.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check (returns 200) |
| `POST` | `/reset?difficulty=easy` | Reset environment with difficulty level |
| `POST` | `/step` | Execute an action (JSON body) |
| `GET` | `/state` | Get current episode state |

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
# Install dependencies
pip install openenv-core pandas numpy fastapi uvicorn requests pydantic openai

# Start the server
uvicorn src.envs.data_cleaner.server.app:app --host 0.0.0.0 --port 8000

# Run the heuristic test
python test_heuristic.py

# Run the LLM inference agent
API_BASE_URL=http://localhost:8000 MODEL_NAME=llama-3.3-70b-versatile HF_TOKEN=<your-key> python inference.py
```

### Docker

```bash
docker build -t data-cleaner .
docker run -p 8000:8000 data-cleaner
```

### Hugging Face Spaces

The environment is deployed as a Docker Space. The `/health` endpoint responds to automated pings with a 200 status.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | Environment server URL | `http://localhost:8000` |
| `MODEL_NAME` | LLM model identifier | `llama-3.3-70b-versatile` |
| `HF_TOKEN` | API key for LLM provider | (required) |
| `LLM_BASE_URL` | LLM API base URL | `https://api.groq.com/openai/v1` |

---

## Baseline Scores

Tested with `llama-3.3-70b-versatile` via Groq:

| Difficulty | Heuristic Score | LLM Agent Score |
|-----------|----------------|-----------------|
| Easy | ~0.95 | ~0.90 |
| Medium | ~0.85 | ~0.80 |
| Hard | ~0.70 | ~0.65 |

---

## Project Structure

```
openEnv-1/
├── inference.py                          # Baseline inference script (root)
├── test_heuristic.py                     # Heuristic test script
├── openenv.yaml                          # OpenEnv manifest
├── Dockerfile                            # Container definition
├── target_dataset.csv                    # Dataset used optionally
├── README.md                             # This file
└── src/
    └── envs/
        └── data_cleaner/
            ├── models.py                 # Pydantic Action/Observation/State models
            ├── client.py                 # HTTP client with connection pooling
            └── server/
                ├── app.py                # FastAPI endpoints
                └── environment.py        # Core environment logic
```

---

## License

MIT
