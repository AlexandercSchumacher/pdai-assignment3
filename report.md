# Assignment 3 — Additions over Assignment 2

**Personal Energy Forecast Planner** · Alexander Schumacher · PDAI, ESADE MiBA 25/26

**Live demo:** https://web-production-41b97.up.railway.app
**Repository:** https://github.com/AlexandercSchumacher/pdai-assignment3
**DB browser:** https://pgweb-production-6c0b.up.railway.app (basic auth)

## 1. Context

Assignment 2 shipped a Streamlit app with a RandomForest energy forecast (Monte Carlo uncertainty bands) and three LLM features: a Smart Scenario Builder using structured output, an AI Energy Optimizer using an agentic tool-use loop, and an Experiment Log Analyst using RAG. Five tabs, deployed once to EC2. The prototype worked end-to-end but had four structural limits: state lived in a flat CSV that a second user would clobber; there was no way to tell whether the AI features were useful, nor whether the forecast matched reality over time; and the UI read as default Streamlit. Assignment 3 is one iteration that closes all four.

## 2. What's new

**Real database layer.** Three tables replace the flat CSV: experiments (scenario parameters and predicted energy per day, with an optional actual-energy outcome logged later), feedback (one row per thumbs-up / thumbs-down click on an AI output), and optimizer runs (iteration count, call history, and best scenario per run). The same code path supports a managed Postgres instance in production and a local SQLite file during development, so the choice of deployment target never leaks into the application logic.

**Validation tab.** Turns the app from "generates predictions" into "measures its own predictions". Three blocks on one page: predicted-vs-actual energy with error metrics that refresh as the user logs real outcomes; an optimizer-convergence view showing how many iterations each run took and what best score it reached; and an aggregated view of thumbs-up / thumbs-down rates per AI feature.

**Feedback on every AI output.** Thumbs-up / thumbs-down buttons attached to each Smart Scenario extraction, each Optimizer run, and each RAG answer. A click stores the feature name, the rating, and the AI output itself, so future work can surface failing prompts or seed few-shot examples from positive samples. The buttons collapse to a short confirmation after a click so a page rerun cannot double-count a rating.

**Managed deployment with a browsable database.** Three services in a single project: the app itself, a managed Postgres, and a browsable database viewer behind basic auth on a separate domain. On first boot the web service trains the model if the artifact is missing, does a health check against Postgres, and fails fast if the database is unreachable. The database viewer is strictly a validation affordance for the grader; a real user sees the data through the Experiment Log and Validation tabs, not the raw tables.

**UI polish.** A custom theme plus about sixty lines of CSS give the app a consistent identity: gradient header, colored active-tab states, a sidebar gradient, section headers. The four Day-1 metric cards on the Forecast tab are redesigned — the Scenario card overlays scenario (solid) vs baseline (dashed) as a small inline sparkline, risk renders as a color-coded pill, the error metrics on the Drivers tab show a residual histogram, and R² is a traffic-light chip. The Smart Scenario tab documents itself with a worked "sample input → extracted parameters" example, so the documentation lives inside the tool where it gets read.

**First-visit onboarding tour.** On every fresh page load the app dims the background and walks a new visitor through the sidebar, the tab strip, the forecast chart, and the two main AI inputs, one section at a time with a short tooltip next to each. The tour auto-switches tabs as it goes, and a single Dismiss click exits it for the session. It is a lightweight frontend component with no external libraries — the intent is that first-time users get a 30-second mental map of the app before they touch anything.

## 3. Worth calling out

Training stays offline and runs only when the trained artifact is missing, preserving the offline / online split from Assignment 2. Predicted energy is written into the experiment row at save time, so predicted-vs-actual comparisons stay stable even if the model is retrained later. The repository contains source only — the demo dataset and the trained artifact are regenerated on first boot, so a fresh clone just works without any prior state.

## 4. What's next

Make Validation a real feedback loop: use the accumulated predicted / actual pairs to retrain the model on the user's own outcomes rather than only the Oura history and a synthetic supplement, and use the thumbs-up / thumbs-down history to pick few-shot examples for the Smart Scenario prompt. That turns the app from a static forecaster into something that improves the more it is used.
