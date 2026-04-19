from __future__ import annotations

import json
from datetime import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.data_load import load_personal_data
from src.database import (
    add_experiment,
    add_feedback,
    delete_experiment,
    get_all_experiments,
    get_all_feedback,
    get_all_optimizer_runs,
    get_feedback_summary,
    init_db,
    save_optimizer_run,
    update_actual_energy,
)
from src.feature_engineering import get_latest_state
from src.forecast import compare_forecasts, load_model_bundle, simulate_forecast
from src.train import train_model
from src.viz import build_forecast_figure, build_importance_figure

load_dotenv()

st.set_page_config(
    page_title="Energy Forecast Planner",
    page_icon="https://em-content.zobj.net/source/twitter/408/high-voltage_26a1.png",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Header gradient bar */
    .stApp > header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }

    /* Main title styling */
    h1 {
        color: #0f3460 !important;
        border-bottom: 3px solid #e94560;
        padding-bottom: 0.3rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f3460 !important;
        color: white !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #f8f9fc;
        border: 1px solid #e1e5eb;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #5a6577 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #eef1f6 100%);
    }
    [data-testid="stSidebar"] h2 {
        color: #0f3460;
    }

    /* Feedback buttons */
    .feedback-row {
        display: flex;
        gap: 8px;
        align-items: center;
        margin-top: 8px;
    }

    /* Success/info boxes */
    .stSuccess, .stInfo {
        border-radius: 8px;
    }

    /* Expander headers */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #0f3460;
    }

    /* Footer section */
    .footer-metrics {
        border-top: 2px solid #e1e5eb;
        padding-top: 1rem;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Initialize database
# ---------------------------------------------------------------------------

init_db()

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_personal_dataframe() -> tuple[pd.DataFrame, dict]:
    return load_personal_data("data/oura_personal.csv")


@st.cache_resource(show_spinner=False)
def get_model() -> dict | None:
    return load_model_bundle("models/energy_model.pkl")


def load_metadata(path: str = "models/metadata.json") -> dict:
    meta_path = Path(path)
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Time / hour helpers
# ---------------------------------------------------------------------------


def model_hour_to_time(hour_value: float) -> time:
    normalized = float(hour_value) % 24
    hour_int = int(normalized)
    minute_int = int(round((normalized - hour_int) * 60))
    if minute_int == 60:
        hour_int = (hour_int + 1) % 24
        minute_int = 0
    return time(hour=hour_int, minute=minute_int)


def model_hour_to_label(hour_value: float) -> str:
    t = model_hour_to_time(hour_value)
    return f"{t.hour:02d}:{t.minute:02d}"


def time_to_model_hour(value: time) -> float:
    hour_value = value.hour + value.minute / 60.0
    if hour_value < 12:
        hour_value += 24
    return hour_value


def circular_hour_diff(target_hour: float, baseline_hour: float) -> float:
    diff = (target_hour % 24) - (baseline_hour % 24)
    if diff > 12:
        diff -= 24
    if diff < -12:
        diff += 24
    return diff


def clock_duration_minutes(bed_time: time, wake_time: time) -> int:
    bed_min = bed_time.hour * 60 + bed_time.minute
    wake_min = wake_time.hour * 60 + wake_time.minute
    duration = wake_min - bed_min
    if duration <= 0:
        duration += 24 * 60
    return int(duration)


def duration_label(minutes: float) -> str:
    rounded = int(round(minutes))
    h = rounded // 60
    m = rounded % 60
    return f"{h}h {m:02d}m"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    if "scenario_params" not in st.session_state:
        st.session_state.scenario_params = {
            "name": "Current Scenario",
            "bedtime_target_hour": None,
            "wake_target_hour": None,
            "bedtime_shift_hours": 0.0,
            "sleep_delta_min": 0,
            "training_load_delta": 0,
            "caffeine_cutoff_hour": 15,
            "alcohol": False,
            "late_meal": False,
        }

    if "llm_parsed_params" not in st.session_state:
        st.session_state.llm_parsed_params = None
    if "optimizer_result" not in st.session_state:
        st.session_state.optimizer_result = None


# ---------------------------------------------------------------------------
# Apply LLM scenario to main session state
# ---------------------------------------------------------------------------


def apply_llm_scenario_to_state(
    params: dict,
    baseline_bedtime_hour: float,
    baseline_sleep_duration_min: float,
    name: str = "AI Scenario",
) -> None:
    shift = float(params.get("bedtime_shift_hours", 0.0))
    sleep_delta = int(params.get("sleep_delta_min", 0))
    baseline_wake_hour = baseline_bedtime_hour + baseline_sleep_duration_min / 60.0

    new_bedtime_hour = baseline_bedtime_hour + shift
    new_wake_hour = baseline_wake_hour + shift + sleep_delta / 60.0

    st.session_state.scenario_params = {
        "name": name,
        "bedtime_target_hour": new_bedtime_hour,
        "wake_target_hour": new_wake_hour % 24,
        "bedtime_shift_hours": shift,
        "sleep_delta_min": sleep_delta,
        "training_load_delta": int(params.get("training_load_delta", 0)),
        "caffeine_cutoff_hour": int(params.get("caffeine_cutoff_hour", 15)),
        "alcohol": bool(params.get("alcohol", False)),
        "late_meal": bool(params.get("late_meal", False)),
    }


# ---------------------------------------------------------------------------
# Training block
# ---------------------------------------------------------------------------


def training_block() -> dict | None:
    model_bundle = get_model()
    if model_bundle is not None:
        return model_bundle

    st.warning("No trained model found. Please run training once.")
    if st.button("Train model now", type="primary"):
        with st.spinner("Training in progress..."):
            train_model(
                personal_csv="data/oura_personal.csv",
                synthetic_csv="data/synthetic.csv",
                model_out="models/energy_model.pkl",
                metadata_out="models/metadata.json",
                n_synth_days=730,
                random_state=42,
            )
        st.cache_resource.clear()
        st.success("Training completed. Reloading app.")
        st.rerun()

    return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _render_param_grid(params: dict) -> None:
    c1, c2, c3 = st.columns(3)
    shift = params.get("bedtime_shift_hours", 0.0)
    c1.metric(
        "Bedtime shift",
        f"{shift:+.1f}h",
        delta=f"{'earlier' if shift < 0 else 'later' if shift > 0 else 'unchanged'}",
        delta_color="normal" if shift <= 0 else "inverse",
    )
    sleep_delta = params.get("sleep_delta_min", 0)
    c2.metric(
        "Sleep change",
        f"{sleep_delta:+d} min",
        delta=f"{abs(sleep_delta) // 60}h {abs(sleep_delta) % 60}m",
        delta_color="normal" if sleep_delta >= 0 else "inverse",
    )
    train_delta = params.get("training_load_delta", 0)
    c3.metric(
        "Training load",
        f"{train_delta:+d}",
        delta_color="off",
    )

    c4, c5, c6 = st.columns(3)
    c4.metric("Caffeine cutoff", f"{params.get('caffeine_cutoff_hour', 15):02d}:00")
    c5.metric("Alcohol", "Yes" if params.get("alcohol") else "No")
    c6.metric("Late meal", "Yes" if params.get("late_meal") else "No")


def _render_feedback_buttons(feature: str, context: dict, key_suffix: str) -> None:
    """Render thumbs up/down feedback buttons for an AI feature output."""
    fb_key = f"fb_{feature}_{key_suffix}"
    if fb_key in st.session_state:
        st.caption(f"Thanks for your feedback ({st.session_state[fb_key]})")
        return

    col_up, col_down, col_comment = st.columns([1, 1, 4])
    if col_up.button("👍", key=f"{fb_key}_up", help="This was helpful"):
        add_feedback(feature=feature, rating=5, context=context)
        st.session_state[fb_key] = "positive"
        st.rerun()
    if col_down.button("👎", key=f"{fb_key}_down", help="This was not helpful"):
        add_feedback(feature=feature, rating=1, context=context)
        st.session_state[fb_key] = "negative"
        st.rerun()


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------


def app() -> None:
    init_session_state()

    st.title("Personal Energy Forecast and Planning")
    st.caption(
        "3-day energy forecast based on Oura daily data. "
        "Compare baseline vs scenario including uncertainty bands."
    )

    personal_df, load_info = load_personal_dataframe()
    model_bundle = training_block()
    if model_bundle is None:
        st.stop()

    metadata = load_metadata()

    min_date = personal_df["date"].min().date()
    max_date = personal_df["date"].max().date()

    # -----------------------------------------------------------------------
    # Sidebar controls
    # -----------------------------------------------------------------------
    st.sidebar.header("Controls")

    selected_range = st.sidebar.date_input(
        "Data range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(selected_range, tuple):
        start_date, end_date = selected_range
    else:
        start_date, end_date = min_date, selected_range

    reference_date = st.sidebar.date_input(
        "Forecast starts from",
        value=end_date,
        min_value=start_date,
        max_value=end_date,
    )

    show_baseline = st.sidebar.toggle("Show baseline", value=True)
    show_scenario = st.sidebar.toggle("Show scenario", value=True)

    filtered = personal_df[
        (personal_df["date"].dt.date >= start_date)
        & (personal_df["date"].dt.date <= end_date)
    ].copy()
    if filtered.empty:
        filtered = personal_df.copy()

    state = get_latest_state(filtered, reference_date=pd.Timestamp(reference_date))
    baseline_bedtime_hour = float(state.get("bedtime_start_hour", 23.0))
    baseline_sleep_duration_min = float(state.get("total_sleep_duration_min", 420.0))
    baseline_wake_hour = baseline_bedtime_hour + baseline_sleep_duration_min / 60.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("Scenario Builder")
    st.sidebar.caption(
        "Baseline: "
        f"bedtime {model_hour_to_label(baseline_bedtime_hour)}, "
        f"wake {model_hour_to_label(baseline_wake_hour)}, "
        f"sleep {duration_label(baseline_sleep_duration_min)}"
    )

    current_params = st.session_state.scenario_params
    default_target_hour = current_params.get("bedtime_target_hour")
    if default_target_hour is None:
        default_target_hour = baseline_bedtime_hour + float(
            current_params.get("bedtime_shift_hours", 0.0)
        )
    default_wake_hour = current_params.get("wake_target_hour")
    if default_wake_hour is None:
        default_wake_hour = baseline_wake_hour

    with st.sidebar.form("scenario_form"):
        scenario_name = st.text_input(
            "Scenario name", value=current_params.get("name", "Current Scenario")
        )
        target_bedtime = st.time_input(
            "Target bedtime",
            value=model_hour_to_time(float(default_target_hour)),
            step=900,
        )
        target_wakeup = st.time_input(
            "Target wake-up time",
            value=model_hour_to_time(float(default_wake_hour)),
            step=900,
        )
        target_sleep_duration_min = clock_duration_minutes(target_bedtime, target_wakeup)
        preview_sleep_delta = int(round(target_sleep_duration_min - baseline_sleep_duration_min))
        st.caption(
            f"Target sleep: {duration_label(target_sleep_duration_min)} "
            f"({preview_sleep_delta:+d} min)"
        )
        training_load_delta = st.slider(
            "Training load delta",
            min_value=-40,
            max_value=40,
            value=int(current_params.get("training_load_delta", 0)),
            step=1,
        )
        caffeine_cutoff_hour = st.slider(
            "Caffeine cutoff hour",
            min_value=12,
            max_value=22,
            value=int(current_params.get("caffeine_cutoff_hour", 15)),
            step=1,
        )
        col_a, col_b = st.columns(2)
        alcohol = col_a.checkbox("Alcohol", value=bool(current_params.get("alcohol", False)))
        late_meal = col_b.checkbox("Late meal", value=bool(current_params.get("late_meal", False)))
        apply_clicked = st.form_submit_button("Apply scenario")

    if apply_clicked:
        bedtime_target_hour = time_to_model_hour(target_bedtime)
        wake_target_hour = target_wakeup.hour + target_wakeup.minute / 60.0
        sleep_delta_min = int(
            min(max(int(round(target_sleep_duration_min - baseline_sleep_duration_min)), -180), 180)
        )
        bedtime_shift_hours = min(
            max(circular_hour_diff(bedtime_target_hour, baseline_bedtime_hour), -4.0), 4.0
        )
        st.session_state.scenario_params = {
            "name": scenario_name or "Current Scenario",
            "bedtime_target_hour": bedtime_target_hour,
            "wake_target_hour": wake_target_hour,
            "bedtime_shift_hours": bedtime_shift_hours,
            "sleep_delta_min": sleep_delta_min,
            "training_load_delta": training_load_delta,
            "caffeine_cutoff_hour": caffeine_cutoff_hour,
            "alcohol": alcohol,
            "late_meal": late_meal,
        }

    # -----------------------------------------------------------------------
    # Compute forecasts
    # -----------------------------------------------------------------------
    scenario_for_run = dict(st.session_state.scenario_params)
    target_hour_for_run = scenario_for_run.get("bedtime_target_hour")
    if target_hour_for_run is not None:
        dynamic_shift = circular_hour_diff(float(target_hour_for_run), baseline_bedtime_hour)
        scenario_for_run["bedtime_shift_hours"] = min(max(dynamic_shift, -4.0), 4.0)
    wake_hour_for_run = scenario_for_run.get("wake_target_hour")
    if target_hour_for_run is not None and wake_hour_for_run is not None:
        recomputed_sleep_duration = clock_duration_minutes(
            model_hour_to_time(float(target_hour_for_run)),
            model_hour_to_time(float(wake_hour_for_run)),
        )
        scenario_for_run["sleep_delta_min"] = min(
            max(int(round(recomputed_sleep_duration - baseline_sleep_duration_min)), -180), 180
        )

    baseline_forecast = simulate_forecast(
        model_bundle, base_state=state, scenario=None,
        horizon_days=3, n_samples=1200, random_state=42,
    )
    scenario_forecast = simulate_forecast(
        model_bundle, base_state=state, scenario=scenario_for_run,
        horizon_days=3, n_samples=1200, random_state=123,
    )
    comparison = compare_forecasts(baseline_forecast, scenario_forecast)

    # -----------------------------------------------------------------------
    # Tabs (A3: added Validation tab)
    # -----------------------------------------------------------------------
    tab_forecast, tab_smart, tab_optimizer, tab_log, tab_validation, tab_drivers = st.tabs(
        ["Forecast", "Smart Scenario", "AI Optimizer", "Experiment Log", "Validation", "Drivers"]
    )

    # ---- Forecast tab ----
    with tab_forecast:
        fig = build_forecast_figure(
            baseline=baseline_forecast,
            scenario=scenario_forecast,
            show_baseline=show_baseline,
            show_scenario=show_scenario,
        )
        st.plotly_chart(fig, use_container_width=True)

        if not comparison.empty:
            day1 = comparison.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Day 1 Delta", f"{day1['delta_median']:+.1f}")
            c2.metric("Day 1 Baseline", f"{day1['median_baseline']:.1f}")
            c3.metric("Day 1 Scenario", f"{day1['median_scenario']:.1f}")
            c4.metric("Risk (Day 1)", f"{scenario_forecast.iloc[0]['risk']}")

        display_df = comparison[[
            "day", "date", "median_baseline", "median_scenario",
            "delta_median", "p10_baseline", "p10_scenario",
            "p90_baseline", "p90_scenario", "risk_scenario",
        ]].copy().rename(columns={
            "median_baseline": "baseline_median",
            "median_scenario": "scenario_median",
            "p10_baseline": "baseline_p10",
            "p10_scenario": "scenario_p10",
            "p90_baseline": "baseline_p90",
            "p90_scenario": "scenario_p90",
            "risk_scenario": "risk",
        })
        st.dataframe(display_df, use_container_width=True)

        with st.expander("Energy Score Formula and Model Assumptions"):
            st.markdown("""
**Energy Formula**
- `stress_component = clamp(100 * stress_high_min / (stress_high_min + recovery_min), 0, 100)`
- `energy_score = 0.5*readiness_score + 0.3*sleep_score + 0.2*clamp(100 - stress_component)`

**Forecast Logic**
- The model predicts `t+1` autoregressively over 3 days.
- Uncertainty uses Monte Carlo residual sampling with p10/p50/p90 bands.
            """)

    # ---- Smart Scenario tab (Feature A) ----
    with tab_smart:
        st.subheader("Smart Scenario Builder")
        st.caption(
            "Describe your upcoming situation in plain English and the AI will automatically "
            "fill in the scenario parameters and show you how it affects your energy forecast."
        )

        with st.expander("Sample input and output", expanded=False):
            st.markdown("""
**Input:** *"I'm flying to New York tomorrow, will sleep about 5 hours, and have a work dinner with drinks."*

**Extracted parameters:**
| Parameter | Value | Reasoning |
|---|---|---|
| Bedtime shift | +2.0h | Long-haul flight implies late bedtime |
| Sleep change | -120 min | Only 5 hours vs ~7h baseline |
| Training load | 0 | No exercise mentioned |
| Caffeine cutoff | 15:00 | Default, not mentioned |
| Alcohol | Yes | "drinks" at dinner |
| Late meal | Yes | Work dinner context |
            """)

        example_prompts = [
            "I'm flying to New York tomorrow, will sleep about 5 hours, and have a work dinner with drinks.",
            "I have a triathlon on Saturday so I'll do a hard training session tomorrow, skipping my evening coffee.",
            "Quiet weekend - planning to sleep in, no alcohol, going to bed early at 10pm.",
        ]
        st.markdown("**Example descriptions:**")
        for ex in example_prompts:
            st.markdown(f"- *{ex}*")

        user_description = st.text_area(
            "Your situation",
            placeholder="e.g. I'm flying overnight tomorrow, sleeping 5 hours, and having drinks at the company dinner.",
            height=100,
        )

        col_parse, col_clear = st.columns([1, 4])
        parse_clicked = col_parse.button("Parse with AI", type="primary", disabled=not user_description.strip())
        if col_clear.button("Clear"):
            st.session_state.llm_parsed_params = None
            st.rerun()

        if parse_clicked and user_description.strip():
            from src.llm import parse_scenario_from_text
            try:
                with st.spinner("Calling LLM to extract parameters..."):
                    parsed = parse_scenario_from_text(
                        user_text=user_description,
                        baseline_state=state.to_dict(),
                    )
                st.session_state.llm_parsed_params = parsed
            except ImportError as e:
                st.error(f"Missing dependency: {e}")
            except ValueError as e:
                st.error(f"Configuration error: {e}")
            except Exception as e:
                st.error(f"LLM call failed: {e}")

        parsed_params = st.session_state.llm_parsed_params
        if parsed_params:
            st.success("Parameters extracted successfully.")

            st.markdown("**AI Reasoning:**")
            st.markdown(f"> {parsed_params.get('reasoning', '')}")

            st.markdown("**Extracted parameters:**")
            _render_param_grid(parsed_params)

            llm_scenario = {
                "bedtime_shift_hours": parsed_params["bedtime_shift_hours"],
                "sleep_delta_min": parsed_params["sleep_delta_min"],
                "training_load_delta": parsed_params["training_load_delta"],
                "caffeine_cutoff_hour": parsed_params["caffeine_cutoff_hour"],
                "alcohol": parsed_params["alcohol"],
                "late_meal": parsed_params["late_meal"],
            }
            with st.spinner("Running forecast preview..."):
                llm_forecast = simulate_forecast(
                    model_bundle,
                    base_state=state,
                    scenario=llm_scenario,
                    horizon_days=3,
                    n_samples=800,
                    random_state=77,
                )
                llm_comparison = compare_forecasts(baseline_forecast, llm_forecast)

            st.markdown("**Forecast preview for this scenario:**")
            if not llm_comparison.empty:
                fc1, fc2, fc3 = st.columns(3)
                for i, col in enumerate([fc1, fc2, fc3]):
                    if i < len(llm_comparison):
                        row = llm_comparison.iloc[i]
                        col.metric(
                            f"Day {row['day']} ({row['date'].strftime('%a')})",
                            f"{row['median_scenario']:.1f}",
                            delta=f"{row['delta_median']:+.1f} vs baseline",
                            delta_color="normal" if row["delta_median"] >= 0 else "inverse",
                        )

            # A3: Feedback buttons
            _render_feedback_buttons(
                feature="smart_scenario",
                context={"input": user_description, "output": parsed_params},
                key_suffix="smart",
            )

            if st.button("Apply this scenario to main forecast", type="primary"):
                apply_llm_scenario_to_state(
                    parsed_params,
                    baseline_bedtime_hour,
                    baseline_sleep_duration_min,
                    name="AI: " + user_description[:40],
                )
                st.success("Scenario applied. Switch to the Forecast tab to see the results.")
                st.rerun()

    # ---- AI Optimizer tab (Feature B) ----
    with tab_optimizer:
        st.subheader("AI Energy Optimizer")
        st.caption(
            "Tell the AI what you want to achieve and it will explore different combinations "
            "of sleep, training, and caffeine adjustments to find what works best for your goal."
        )

        example_goals = [
            "Maximize my average energy over the next 3 days.",
            "I have an important presentation on Day 2 - optimize for peak energy that day.",
            "I need to recover from a hard week of training while keeping Day 3 energy high.",
        ]
        st.markdown("**Example goals:**")
        for g in example_goals:
            st.markdown(f"- *{g}*")

        goal_input = st.text_input(
            "Your energy goal",
            placeholder="e.g. Maximize my energy on Day 2 - I have a big presentation.",
        )
        constraint_input = st.text_input(
            "Any fixed constraints? (optional)",
            placeholder="e.g. I must go to the gym tomorrow, I can't avoid the work dinner.",
        )

        full_goal = goal_input.strip()
        if constraint_input.strip():
            full_goal += f" Constraints: {constraint_input.strip()}"

        col_run, col_clear2 = st.columns([1, 4])
        run_clicked = col_run.button(
            "Run AI Optimizer", type="primary", disabled=not goal_input.strip()
        )
        if col_clear2.button("Clear results"):
            st.session_state.optimizer_result = None
            st.rerun()

        if run_clicked and full_goal:
            from src.llm import optimize_scenario_with_agent

            iteration_log: list[tuple[int, dict, dict]] = []

            def progress_callback(iteration: int, params: dict, result: dict) -> None:
                iteration_log.append((iteration, params, result))

            try:
                with st.status("AI Optimizer running...", expanded=True) as status:
                    status.write("Starting agentic optimization loop...")
                    opt_result = optimize_scenario_with_agent(
                        goal=full_goal,
                        model_bundle=model_bundle,
                        base_state=state,
                        max_iterations=6,
                        on_tool_call=progress_callback,
                    )
                    for it, params, res in iteration_log:
                        status.write(
                            f"Iteration {it}: avg energy = **{res['average_energy']:.1f}** "
                            f"(D1={res['day1_energy']:.1f}, D2={res.get('day2_energy', '?'):.1f}, "
                            f"D3={res.get('day3_energy', '?'):.1f})"
                        )
                    status.update(
                        label=f"Done - {opt_result['iterations']} scenarios explored.",
                        state="complete",
                    )
                st.session_state.optimizer_result = opt_result

                # A3: Save optimizer run to database for convergence tracking
                save_optimizer_run(
                    goal=full_goal,
                    iterations=opt_result.get("iterations", 0),
                    best_avg_energy=opt_result.get("best_avg_energy", 0.0),
                    best_scenario=opt_result.get("best_scenario", {}),
                    call_history=opt_result.get("call_history", []),
                    final_recommendation=opt_result.get("final_recommendation", ""),
                )

            except ImportError as e:
                st.error(f"Missing dependency: {e}")
            except ValueError as e:
                st.error(f"Configuration error: {e}")
            except Exception as e:
                st.error(f"Optimizer failed: {e}")

        opt_result = st.session_state.optimizer_result
        if opt_result:
            st.markdown("---")
            best_avg = opt_result.get("best_avg_energy", 0.0)
            n_iter = opt_result.get("iterations", 0)
            st.success(f"Optimization complete - {n_iter} scenarios explored. Best average energy: **{best_avg:.1f}/100**")

            final_text = opt_result.get("final_recommendation", "")
            if final_text.strip():
                with st.expander("AI Recommendation (full text)", expanded=True):
                    st.markdown(final_text)

            best_scenario = opt_result.get("best_scenario")
            if best_scenario:
                st.markdown("**Best scenario parameters found:**")
                _render_param_grid(best_scenario)

                with st.spinner("Running forecast for best scenario..."):
                    best_forecast = simulate_forecast(
                        model_bundle,
                        base_state=state,
                        scenario=best_scenario,
                        horizon_days=3,
                        n_samples=800,
                        random_state=99,
                    )
                    best_comparison = compare_forecasts(baseline_forecast, best_forecast)

                st.markdown("**Projected energy with best scenario:**")
                if not best_comparison.empty:
                    bc1, bc2, bc3 = st.columns(3)
                    for i, col in enumerate([bc1, bc2, bc3]):
                        if i < len(best_comparison):
                            row = best_comparison.iloc[i]
                            col.metric(
                                f"Day {row['day']} ({row['date'].strftime('%a')})",
                                f"{row['median_scenario']:.1f}",
                                delta=f"{row['delta_median']:+.1f} vs baseline",
                                delta_color="normal" if row["delta_median"] >= 0 else "inverse",
                            )

                # A3: Feedback buttons
                _render_feedback_buttons(
                    feature="optimizer",
                    context={"goal": full_goal, "best_avg": best_avg, "iterations": n_iter},
                    key_suffix="opt",
                )

                if st.button("Apply best scenario to main forecast", type="primary"):
                    apply_llm_scenario_to_state(
                        best_scenario,
                        baseline_bedtime_hour,
                        baseline_sleep_duration_min,
                        name="AI Optimized",
                    )
                    st.success("Applied. Switch to the Forecast tab to see the full view.")
                    st.rerun()

            call_history = opt_result.get("call_history", [])
            if call_history:
                with st.expander(f"Optimization Journey ({len(call_history)} tool calls)"):
                    for record in call_history:
                        it = record["iteration"]
                        p = record["params"]
                        r = record["result"]
                        avg = r.get("average_energy", 0.0)
                        is_best = best_scenario and all(
                            abs(float(p.get(k, 0)) - float(best_scenario.get(k, 0))) < 0.01
                            for k in ["bedtime_shift_hours", "sleep_delta_min", "training_load_delta"]
                        )
                        label = f"**Iteration {it}** - avg energy {avg:.1f}" + (" (best)" if is_best else "")
                        st.markdown(label)
                        col_p, col_r = st.columns(2)
                        with col_p:
                            st.markdown("*Parameters tried:*")
                            st.json(p)
                        with col_r:
                            st.markdown("*Forecast result:*")
                            days = r.get("forecast_by_day", [])
                            if days:
                                st.dataframe(pd.DataFrame(days), use_container_width=True)
                        st.markdown("---")

    # ---- Experiment Log tab (A3: database-backed) ----
    with tab_log:
        st.subheader("Experiment Log")
        st.caption(
            "Document your experiments and track predicted vs actual energy. "
            "Data is stored in a database (PostgreSQL in production, SQLite locally)."
        )

        # Add new experiment
        with st.form("log_form"):
            log_tags = st.text_input(
                "Tags (comma-separated)",
                value=",".join(
                    t for t in [
                        "alcohol" if st.session_state.scenario_params.get("alcohol") else "",
                        "late_meal" if st.session_state.scenario_params.get("late_meal") else "",
                    ] if t
                ),
            )
            log_notes = st.text_input("Notes", value="")
            day1_delta = float(comparison.iloc[0]["delta_median"]) if not comparison.empty else 0.0

            # Predicted energy from current scenario
            predicted_energy = {}
            if not scenario_forecast.empty:
                for i, row in scenario_forecast.iterrows():
                    predicted_energy[f"day{row['day']}"] = round(float(row["median"]), 1)

            add_log = st.form_submit_button("Save experiment to database")

        if add_log:
            add_experiment(
                scenario_name=st.session_state.scenario_params.get("name", "Scenario"),
                tags=log_tags,
                day1_delta=round(day1_delta, 3),
                notes=log_notes,
                scenario_params=dict(st.session_state.scenario_params),
                predicted_energy=predicted_energy,
            )
            st.success("Experiment saved to database.")
            st.rerun()

        # Display experiments from database
        experiments_df = get_all_experiments()
        if not experiments_df.empty:
            st.markdown(f"**{len(experiments_df)} experiments in database**")

            display_cols = [
                "id", "created_at", "scenario_name", "tags", "day1_delta", "notes",
                "predicted_energy_day1", "actual_energy_day1",
            ]
            available_cols = [c for c in display_cols if c in experiments_df.columns]
            st.dataframe(
                experiments_df[available_cols],
                use_container_width=True,
                hide_index=True,
            )

            # Log actual energy for validation
            st.markdown("---")
            st.subheader("Log Actual Energy Outcome")
            st.caption(
                "After following a scenario, log your actual energy to validate the model's predictions."
            )

            exp_ids = experiments_df["id"].tolist()
            selected_exp_id = st.selectbox(
                "Select experiment to update",
                options=exp_ids,
                format_func=lambda x: f"#{x} - {experiments_df[experiments_df['id'] == x].iloc[0]['scenario_name']}",
            )

            with st.form("actual_energy_form"):
                act_col1, act_col2, act_col3 = st.columns(3)
                actual_d1 = act_col1.number_input("Actual Day 1 energy (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
                actual_d2 = act_col2.number_input("Actual Day 2 energy (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
                actual_d3 = act_col3.number_input("Actual Day 3 energy (0-100)", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
                submit_actual = st.form_submit_button("Save actual energy")

            if submit_actual and selected_exp_id:
                update_actual_energy(
                    experiment_id=selected_exp_id,
                    actual_day1=actual_d1,
                    actual_day2=actual_d2,
                    actual_day3=actual_d3,
                )
                st.success(f"Actual energy saved for experiment #{selected_exp_id}.")
                st.rerun()
        else:
            st.info("No experiments logged yet. Use the form above to save your first experiment.")

        # RAG: Ask questions about experiment history (Feature C)
        st.markdown("---")
        st.subheader("Ask About Your Experiments")
        st.caption(
            "Ask any natural language question about your experiment history. "
            "The AI finds the most relevant entries using semantic search and "
            "answers based strictly on your personal data."
        )

        # Build a log_df from DB for RAG
        rag_log_df = experiments_df.rename(columns={"created_at": "created_at"}) if not experiments_df.empty else pd.DataFrame()
        n_entries = len(rag_log_df)

        if n_entries < 2:
            st.warning(
                f"You have {n_entries} log {'entry' if n_entries == 1 else 'entries'}. "
                "Add at least 2 entries above before querying your history."
            )
        else:
            example_questions = [
                "When did going to bed earlier give me the biggest energy boost?",
                "What effect did alcohol have on my energy across all experiments?",
                "Which scenario produced the best Day 1 result?",
            ]
            st.markdown("**Example questions:**")
            for q in example_questions:
                st.markdown(f"- *{q}*")

            rag_question = st.text_input(
                "Your question",
                placeholder="e.g. What happened to my energy when I had alcohol?",
                key="rag_question_input",
            )

            if st.button(
                "Ask AI", type="primary",
                disabled=not rag_question.strip(),
                key="rag_ask_btn",
            ):
                from src.llm import answer_from_log
                try:
                    with st.spinner("Searching your experiment history..."):
                        rag_result = answer_from_log(
                            question=rag_question,
                            log_df=rag_log_df.copy(),
                            top_k=5,
                        )
                    st.session_state["rag_result"] = rag_result
                except ImportError as e:
                    st.error(f"Missing dependency: {e}")
                except ValueError as e:
                    st.error(f"Configuration error: {e}")
                except Exception as e:
                    st.error(f"RAG query failed: {e}")

            if st.session_state.get("rag_result"):
                rag = st.session_state["rag_result"]
                st.markdown("**Answer:**")
                st.markdown(rag["answer"])

                # A3: Feedback on RAG answer
                _render_feedback_buttons(
                    feature="rag",
                    context={"question": rag_question if rag_question else "", "answer": rag["answer"]},
                    key_suffix="rag",
                )

                sources = rag.get("sources", [])
                if sources:
                    with st.expander(f"Retrieved entries used as context ({len(sources)})"):
                        for i, src in enumerate(sources):
                            st.markdown(f"**Entry {i + 1}**")
                            st.code(src["text"], language=None)

    # -----------------------------------------------------------------------
    # Tab: Validation (A3 - new)
    # -----------------------------------------------------------------------
    with tab_validation:
        st.subheader("Model Validation Dashboard")
        st.caption(
            "Track how well the model's predictions match reality. "
            "Log actual energy outcomes in the Experiment Log tab, then view validation metrics here."
        )

        experiments_df = get_all_experiments()

        # Filter to experiments that have both predicted and actual values
        has_validation = experiments_df.dropna(
            subset=["predicted_energy_day1", "actual_energy_day1"]
        ) if not experiments_df.empty else pd.DataFrame()

        if has_validation.empty:
            st.info(
                "No validation data yet. To see validation charts:\n"
                "1. Save experiments in the Experiment Log tab\n"
                "2. After living through the scenario, log your actual energy scores\n"
                "3. Return here to see predicted vs actual comparison"
            )
        else:
            st.markdown(f"**{len(has_validation)} experiments with actual outcomes**")

            # --- Predicted vs Actual scatter plot ---
            st.markdown("### Predicted vs Actual Energy")

            pred_vals = []
            actual_vals = []
            labels = []
            for _, row in has_validation.iterrows():
                for day in [1, 2, 3]:
                    pred = row.get(f"predicted_energy_day{day}")
                    actual = row.get(f"actual_energy_day{day}")
                    if pred is not None and actual is not None and pd.notna(pred) and pd.notna(actual):
                        pred_vals.append(float(pred))
                        actual_vals.append(float(actual))
                        labels.append(f"#{row['id']} Day {day}")

            if pred_vals:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=pred_vals,
                    y=actual_vals,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    textfont=dict(size=9),
                    marker=dict(size=10, color="#0f3460", opacity=0.7),
                    name="Experiments",
                ))
                # Perfect prediction line
                min_val = min(min(pred_vals), min(actual_vals)) - 5
                max_val = max(max(pred_vals), max(actual_vals)) + 5
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    line=dict(color="#e94560", dash="dash", width=2),
                    name="Perfect prediction",
                ))
                fig_scatter.update_layout(
                    xaxis_title="Predicted Energy",
                    yaxis_title="Actual Energy",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Accuracy metrics
                import numpy as np
                pred_arr = np.array(pred_vals)
                actual_arr = np.array(actual_vals)
                mae = float(np.mean(np.abs(pred_arr - actual_arr)))
                rmse = float(np.sqrt(np.mean((pred_arr - actual_arr) ** 2)))
                bias = float(np.mean(pred_arr - actual_arr))

                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("MAE (validation)", f"{mae:.1f}")
                mc2.metric("RMSE (validation)", f"{rmse:.1f}")
                mc3.metric(
                    "Bias",
                    f"{bias:+.1f}",
                    delta="over-predicts" if bias > 0 else "under-predicts",
                    delta_color="inverse" if abs(bias) > 3 else "off",
                )
            else:
                st.warning("Log actual energy for at least one day to see the scatter plot.")

        # --- Agent Convergence ---
        st.markdown("---")
        st.markdown("### AI Optimizer Convergence")
        st.caption("How many iterations does the optimizer need to find the best scenario?")

        optimizer_runs_df = get_all_optimizer_runs()
        if optimizer_runs_df.empty:
            st.info("No optimizer runs recorded yet. Use the AI Optimizer tab to generate data.")
        else:
            st.markdown(f"**{len(optimizer_runs_df)} optimizer runs recorded**")

            # Convergence chart: for each run, plot energy over iterations
            fig_conv = go.Figure()
            for _, run in optimizer_runs_df.iterrows():
                history = run.get("call_history", [])
                if not history:
                    continue
                iters = [h["iteration"] for h in history]
                energies = [h["result"].get("average_energy", 0) for h in history]
                # Cumulative best
                cum_best = []
                best_so_far = 0
                for e in energies:
                    best_so_far = max(best_so_far, e)
                    cum_best.append(best_so_far)

                run_label = f"Run #{run['id']}: {run['goal'][:40]}"
                fig_conv.add_trace(go.Scatter(
                    x=iters,
                    y=cum_best,
                    mode="lines+markers",
                    name=run_label,
                    line=dict(width=2),
                    marker=dict(size=6),
                ))

            fig_conv.update_layout(
                xaxis_title="Iteration",
                yaxis_title="Best Average Energy Found",
                template="plotly_white",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_conv, use_container_width=True)

            # Summary table
            summary_data = []
            for _, run in optimizer_runs_df.iterrows():
                history = run.get("call_history", [])
                energies = [h["result"].get("average_energy", 0) for h in history] if history else []
                best_iter = 0
                if energies:
                    best_iter = energies.index(max(energies)) + 1
                summary_data.append({
                    "Run": f"#{run['id']}",
                    "Goal": run["goal"][:60],
                    "Iterations": run["iterations"],
                    "Best energy": run["best_avg_energy"],
                    "Best found at iteration": best_iter,
                })
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        # --- User Feedback Summary ---
        st.markdown("---")
        st.markdown("### User Feedback on AI Features")

        fb_summary = get_feedback_summary()
        if not fb_summary:
            st.info("No feedback collected yet. Use the thumbs up/down buttons on AI feature outputs.")
        else:
            fb_cols = st.columns(len(fb_summary))
            feature_labels = {
                "smart_scenario": "Smart Scenario",
                "optimizer": "AI Optimizer",
                "rag": "Experiment Q&A",
            }
            for col, (feature, stats) in zip(fb_cols, fb_summary.items()):
                label = feature_labels.get(feature, feature)
                col.metric(
                    label,
                    f"{stats['approval_rate']:.0f}% approval",
                    delta=f"{stats['positive']} positive, {stats['negative']} negative",
                    delta_color="off",
                )
                col.caption(f"{stats['total']} total ratings")

            all_feedback = get_all_feedback()
            if not all_feedback.empty:
                with st.expander("All feedback entries"):
                    st.dataframe(all_feedback, use_container_width=True, hide_index=True)

    # ---- Drivers tab ----
    with tab_drivers:
        st.subheader("Forecast Drivers")
        importance = model_bundle.get("feature_importance", {})
        imp_fig = build_importance_figure(importance, top_n=12)
        st.plotly_chart(imp_fig, use_container_width=True)

        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        if sorted_items:
            st.markdown("Top drivers in the current model:")
            for feature, score in sorted_items:
                st.write(f"- `{feature}`: {score:.4f}")

        if metadata.get("metrics"):
            metrics = metadata["metrics"]
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{metrics['mae']:.2f}")
            c2.metric("RMSE", f"{metrics['rmse']:.2f}")
            c3.metric("R2", f"{metrics['r2']:.3f}")

        with st.expander("Data and Mapping Assumptions"):
            if load_info.get("assumptions"):
                for line in load_info["assumptions"]:
                    st.write(f"- {line}")

    # -----------------------------------------------------------------------
    # Footer metrics
    # -----------------------------------------------------------------------
    st.markdown('<div class="footer-metrics">', unsafe_allow_html=True)
    footer_left, footer_mid, footer_right = st.columns(3)
    footer_left.metric("Personal data days", len(personal_df))
    footer_mid.metric("Scenario", st.session_state.scenario_params.get("name", "Current Scenario"))
    footer_right.metric("Demo data", "Yes" if bool(personal_df["is_demo"].any()) else "No")
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    app()
