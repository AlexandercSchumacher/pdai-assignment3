# Personal Energy Forecast Planner (Assignment 3)

A Streamlit app that predicts your energy score over the next three days based on lifestyle habits, powered by a RandomForest model trained on Oura ring data. Uses LLM-powered features for natural language scenario building, agentic optimization, and RAG-based experiment analysis.

## What's new in Assignment 3

**Database layer (PostgreSQL / SQLite)**
- Replaced CSV-based experiment log with a proper relational database using SQLAlchemy
- Supports PostgreSQL (production on Railway) and SQLite (local development) via `DATABASE_URL`
- New tables: `experiments` (with predicted + actual energy tracking), `feedback`, `optimizer_runs`

**Validation tab**
- Predicted vs actual energy scatter plot with accuracy metrics (MAE, RMSE, bias)
- AI Optimizer convergence tracking: how many iterations to find the best scenario per run
- Aggregated user feedback dashboard across all AI features

**User feedback system**
- Thumbs up/down buttons on every AI feature output (Smart Scenario, Optimizer, RAG)
- Feedback stored in database, summary shown in the Validation tab

**UI/UX polish**
- Custom color theme via Streamlit config and CSS
- Styled metric cards, tabs, sidebar, and section headers
- Sample input/output examples in the Smart Scenario tab for documentation clarity

**Railway deployment**
- Procfile for one-click Railway deploy
- Auto-detects `DATABASE_URL` for Postgres or falls back to local SQLite

## Features

**Core forecast**
- 3-day energy score prediction with Monte Carlo uncertainty bands
- Baseline vs. scenario comparison across six lifestyle parameters

**Feature A - Smart Scenario Builder** (structured output + few-shot prompting)
- Describe your upcoming days in plain English, parameters extracted automatically

**Feature B - AI Energy Optimizer** (multi-call agentic tool use loop)
- LLM agent explores the parameter space to find the best scenario for your goal
- Convergence data saved to database for analysis

**Feature C - Experiment Log Analyst** (RAG pipeline)
- Semantic search over your experiment history using text-embedding-3-small

## Setup

**Requirements**
- Python 3.12+
- An OpenAI API key

**Installation**

```bash
git clone https://github.com/AlexandercSchumacher/pdai-assignment3.git
cd pdai-assignment3

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**Environment**

```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

**Run locally**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. On first run, it will create a local SQLite database at `data/energy_app.db` and offer to train the model if no trained model exists.

## Deploy to Railway

1. Push the repo to GitHub
2. Create a new Railway project and connect the GitHub repo
3. Add a **PostgreSQL** add-on (one click in the Railway dashboard)
4. Set `OPENAI_API_KEY` in the Railway environment variables
5. Railway auto-detects the `Procfile` and deploys

The `DATABASE_URL` is injected automatically by the Postgres add-on.

## Project structure

```
app.py                  # Streamlit UI (6 tabs)
src/
  database.py           # SQLAlchemy models + CRUD (NEW in A3)
  llm.py                # LLM features (scenario parsing, optimizer, RAG)
  forecast.py           # RandomForest model and Monte Carlo simulation
  feature_engineering.py
  data_load.py
  train.py
  viz.py
.streamlit/
  config.toml           # Custom theme (NEW in A3)
Procfile                # Railway deployment (NEW in A3)
.env.example
requirements.txt
```

## Notes

- `.env` and personal data files are excluded from the repository via `.gitignore`
- Model files are also excluded. Run `python -m src.train` to train locally
- All LLM calls use `gpt-4o-mini` and `text-embedding-3-small` via the OpenAI API
- The database schema is created automatically on first run
