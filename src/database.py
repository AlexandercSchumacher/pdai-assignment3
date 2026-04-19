"""Database layer using SQLAlchemy.

Supports both SQLite (local development) and PostgreSQL (Railway deployment)
via the DATABASE_URL environment variable.

If DATABASE_URL is not set, falls back to a local SQLite file at data/energy_app.db.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------

def _get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        # Railway uses postgres:// but SQLAlchemy 2.x needs postgresql://
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url
    return "sqlite:///data/energy_app.db"


_engine = None
_SessionFactory = None


def get_engine():
    global _engine
    if _engine is None:
        url = _get_database_url()
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(url, connect_args=connect_args, echo=False)
    return _engine


def get_session() -> Session:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Experiment(Base):
    """Replaces the CSV experiment log with a proper database table."""

    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    scenario_name = Column(String(200), default="")
    tags = Column(Text, default="")
    day1_delta = Column(Float, default=0.0)
    notes = Column(Text, default="")

    # Scenario parameters stored as JSON string
    scenario_params = Column(Text, default="{}")

    # Predicted energy scores from the forecast
    predicted_energy_day1 = Column(Float, nullable=True)
    predicted_energy_day2 = Column(Float, nullable=True)
    predicted_energy_day3 = Column(Float, nullable=True)

    # Actual energy scores logged by the user for validation
    actual_energy_day1 = Column(Float, nullable=True)
    actual_energy_day2 = Column(Float, nullable=True)
    actual_energy_day3 = Column(Float, nullable=True)


class Feedback(Base):
    """User ratings on AI feature outputs (thumbs up/down)."""

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    feature = Column(String(50))  # 'smart_scenario', 'optimizer', 'rag'
    rating = Column(Integer)  # 1 = thumbs down, 5 = thumbs up
    comment = Column(Text, default="")
    context = Column(Text, default="{}")  # JSON: the AI output for reference


class OptimizerRun(Base):
    """Tracks each AI Optimizer run for convergence analysis."""

    __tablename__ = "optimizer_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    goal = Column(Text, default="")
    iterations = Column(Integer, default=0)
    best_avg_energy = Column(Float, default=0.0)
    best_scenario = Column(Text, default="{}")  # JSON
    call_history = Column(Text, default="[]")  # JSON
    final_recommendation = Column(Text, default="")


# ---------------------------------------------------------------------------
# Table creation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they don't exist."""
    Base.metadata.create_all(get_engine())


# ---------------------------------------------------------------------------
# Experiment CRUD
# ---------------------------------------------------------------------------

def add_experiment(
    scenario_name: str,
    tags: str,
    day1_delta: float,
    notes: str,
    scenario_params: dict | None = None,
    predicted_energy: dict[str, float] | None = None,
) -> Experiment:
    session = get_session()
    try:
        exp = Experiment(
            scenario_name=scenario_name,
            tags=tags,
            day1_delta=day1_delta,
            notes=notes,
            scenario_params=json.dumps(scenario_params or {}),
            predicted_energy_day1=predicted_energy.get("day1") if predicted_energy else None,
            predicted_energy_day2=predicted_energy.get("day2") if predicted_energy else None,
            predicted_energy_day3=predicted_energy.get("day3") if predicted_energy else None,
        )
        session.add(exp)
        session.commit()
        session.refresh(exp)
        return exp
    finally:
        session.close()


def get_all_experiments() -> pd.DataFrame:
    session = get_session()
    try:
        rows = session.query(Experiment).order_by(Experiment.created_at.desc()).all()
        if not rows:
            return pd.DataFrame(columns=[
                "id", "created_at", "scenario_name", "tags", "day1_delta", "notes",
                "predicted_energy_day1", "predicted_energy_day2", "predicted_energy_day3",
                "actual_energy_day1", "actual_energy_day2", "actual_energy_day3",
            ])
        data = []
        for r in rows:
            data.append({
                "id": r.id,
                "created_at": r.created_at,
                "scenario_name": r.scenario_name,
                "tags": r.tags,
                "day1_delta": r.day1_delta,
                "notes": r.notes,
                "predicted_energy_day1": r.predicted_energy_day1,
                "predicted_energy_day2": r.predicted_energy_day2,
                "predicted_energy_day3": r.predicted_energy_day3,
                "actual_energy_day1": r.actual_energy_day1,
                "actual_energy_day2": r.actual_energy_day2,
                "actual_energy_day3": r.actual_energy_day3,
            })
        return pd.DataFrame(data)
    finally:
        session.close()


def update_actual_energy(
    experiment_id: int,
    actual_day1: float | None = None,
    actual_day2: float | None = None,
    actual_day3: float | None = None,
) -> None:
    session = get_session()
    try:
        exp = session.query(Experiment).filter_by(id=experiment_id).first()
        if exp:
            if actual_day1 is not None:
                exp.actual_energy_day1 = actual_day1
            if actual_day2 is not None:
                exp.actual_energy_day2 = actual_day2
            if actual_day3 is not None:
                exp.actual_energy_day3 = actual_day3
            session.commit()
    finally:
        session.close()


def delete_experiment(experiment_id: int) -> None:
    session = get_session()
    try:
        exp = session.query(Experiment).filter_by(id=experiment_id).first()
        if exp:
            session.delete(exp)
            session.commit()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Feedback CRUD
# ---------------------------------------------------------------------------

def add_feedback(
    feature: str,
    rating: int,
    comment: str = "",
    context: dict | None = None,
) -> Feedback:
    session = get_session()
    try:
        fb = Feedback(
            feature=feature,
            rating=rating,
            comment=comment,
            context=json.dumps(context or {}),
        )
        session.add(fb)
        session.commit()
        session.refresh(fb)
        return fb
    finally:
        session.close()


def get_feedback_summary() -> dict[str, Any]:
    """Return aggregated feedback stats per feature."""
    session = get_session()
    try:
        rows = session.query(Feedback).all()
        if not rows:
            return {}

        by_feature: dict[str, list[int]] = {}
        for r in rows:
            by_feature.setdefault(r.feature, []).append(r.rating)

        summary = {}
        for feature, ratings in by_feature.items():
            positive = sum(1 for r in ratings if r >= 4)
            negative = sum(1 for r in ratings if r <= 2)
            total = len(ratings)
            summary[feature] = {
                "total": total,
                "positive": positive,
                "negative": negative,
                "approval_rate": round(positive / total * 100, 1) if total > 0 else 0,
            }
        return summary
    finally:
        session.close()


def get_all_feedback() -> pd.DataFrame:
    session = get_session()
    try:
        rows = session.query(Feedback).order_by(Feedback.created_at.desc()).all()
        if not rows:
            return pd.DataFrame(columns=["id", "created_at", "feature", "rating", "comment"])
        data = [
            {
                "id": r.id,
                "created_at": r.created_at,
                "feature": r.feature,
                "rating": r.rating,
                "comment": r.comment,
            }
            for r in rows
        ]
        return pd.DataFrame(data)
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Optimizer run CRUD
# ---------------------------------------------------------------------------

def save_optimizer_run(
    goal: str,
    iterations: int,
    best_avg_energy: float,
    best_scenario: dict,
    call_history: list[dict],
    final_recommendation: str,
) -> OptimizerRun:
    session = get_session()
    try:
        run = OptimizerRun(
            goal=goal,
            iterations=iterations,
            best_avg_energy=best_avg_energy,
            best_scenario=json.dumps(best_scenario),
            call_history=json.dumps(call_history),
            final_recommendation=final_recommendation,
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        return run
    finally:
        session.close()


def get_all_optimizer_runs() -> pd.DataFrame:
    session = get_session()
    try:
        rows = session.query(OptimizerRun).order_by(OptimizerRun.created_at.desc()).all()
        if not rows:
            return pd.DataFrame(columns=[
                "id", "created_at", "goal", "iterations",
                "best_avg_energy", "call_history",
            ])
        data = []
        for r in rows:
            history = json.loads(r.call_history) if r.call_history else []
            data.append({
                "id": r.id,
                "created_at": r.created_at,
                "goal": r.goal,
                "iterations": r.iterations,
                "best_avg_energy": r.best_avg_energy,
                "call_history": history,
            })
        return pd.DataFrame(data)
    finally:
        session.close()
