"""Startup check: connect to the database, create tables, and log counts."""

from __future__ import annotations

import os
import sys

from sqlalchemy import text

from src.database import Experiment, Feedback, OptimizerRun, get_engine, get_session, init_db


def main() -> int:
    url = os.environ.get("DATABASE_URL", "").strip()
    kind = "postgres" if url.startswith(("postgres://", "postgresql://")) else "sqlite"
    host = url.split("@")[-1].split("/")[0] if "@" in url else "(local)"
    raw_preview = url[:10] + "..." + url[-10:] if len(url) > 25 else url
    print(f"[db_check] backend={kind} host={host} raw_preview={raw_preview!r} env_len={len(url)}")

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("[db_check] SELECT 1 ok")

    init_db()
    print("[db_check] init_db ok, tables ensured")

    session = get_session()
    try:
        exp_count = session.query(Experiment).count()
        fb_count = session.query(Feedback).count()
        run_count = session.query(OptimizerRun).count()
    finally:
        session.close()

    print(
        f"[db_check] rows experiments={exp_count} feedback={fb_count} optimizer_runs={run_count}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
