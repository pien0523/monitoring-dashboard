"""Production Equipment Monitoring & Predictive Maintenance Dashboard.

Streamlit entry point.  Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st

from src.anomaly import create_alert_feed, detect_anomalies, generate_anomaly_explanation
from src.config import (
    SLIDER_STEP_MINUTES,
    AnomalyConfig,
    MaintenanceConfig,
    OEEConfig,
    SimulationConfig,
)
from src.data_loader import get_or_create_data
from src.data_quality import has_blocking_errors, prepare_uploaded_data, validate_data_quality
from src.kpi import (
    calculate_bottleneck_score,
    calculate_oee,
    calculate_oee_by_machine,
    calculate_oee_trend,
)
from src.logger import get_logger
from src.maintenance import generate_machine_health_summary, train_maintenance_model
from src.recommendations import generate_maintenance_recommendations
from src.visualization import (
    create_failure_probability_chart,
    create_health_score_chart,
    create_oee_factor_chart,
    create_oee_trend_chart,
    create_sensor_anomaly_chart,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Manufacturing Monitoring Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

_OEE_CFG = OEEConfig()
_MAINT_CFG = MaintenanceConfig()
_ANOMALY_CFG = AnomalyConfig()


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading sensor data…")
def load_data() -> pd.DataFrame:
    """Load or generate sensor dataset."""
    return get_or_create_data()


@st.cache_data(show_spinner="Training predictive model…")
def get_trained_model(data_hash: int, _df: pd.DataFrame) -> object:
    """Train and cache the Random Forest model.

    Args:
        data_hash: Stable integer key used by st.cache_data for invalidation.
        _df: Active sensor DataFrame (underscore prefix skips Streamlit hashing).
    """
    return train_maintenance_model(_df)


# ---------------------------------------------------------------------------
# Time window helpers
# ---------------------------------------------------------------------------

def get_time_windows(df: pd.DataFrame, selected_time: pd.Timestamp) -> dict:
    """Return pre-sliced DataFrames for each time-window context.

    Args:
        df: Full sensor DataFrame.
        selected_time: User-selected time reference.

    Returns:
        Dict of labelled DataFrames: current_snapshot, today_window, recent_window,
        trend_window, sensor_window.
    """
    today_start = selected_time.normalize()

    # current_snapshot: last row per machine at or before selected_time
    up_to_now = df[df["timestamp"] <= selected_time]
    if up_to_now.empty:
        current_snapshot = pd.DataFrame()
    else:
        current_snapshot = (
            up_to_now.sort_values("timestamp")
            .groupby("machine_id")
            .last()
            .reset_index()
        )

    today_window = df[(df["timestamp"] >= today_start) & (df["timestamp"] <= selected_time)]
    recent_window = df[
        (df["timestamp"] > selected_time - pd.Timedelta(hours=6))
        & (df["timestamp"] <= selected_time)
    ]
    trend_window = df[
        (df["timestamp"] > selected_time - pd.Timedelta(days=7))
        & (df["timestamp"] <= selected_time)
    ]
    sensor_window = df[
        (df["timestamp"] > selected_time - pd.Timedelta(hours=24))
        & (df["timestamp"] <= selected_time)
    ]

    return {
        "current_snapshot": current_snapshot,
        "today_window": today_window,
        "recent_window": recent_window,
        "trend_window": trend_window,
        "sensor_window": sensor_window,
    }


# ---------------------------------------------------------------------------
# Data source helpers
# ---------------------------------------------------------------------------

def _show_quality_report(report: dict) -> None:
    """Render a Data Quality Report expander in the current Streamlit context.

    Args:
        report: Dict returned by ``validate_data_quality``.
    """
    with st.expander("Data Quality Report", expanded=has_blocking_errors(report)):
        info = report.get("info", {})
        cols = st.columns(3)
        cols[0].metric("Rows", f"{info.get('row_count', 0):,}")
        cols[1].metric("Machines", info.get("machine_count", "—"))
        cols[2].metric("Columns", info.get("column_count", 0))

        if info.get("time_range"):
            st.caption(f"Time range: {info['time_range']}")

        errors = report.get("blocking_errors", [])
        warnings = report.get("warnings", [])

        if errors:
            st.markdown("**Blocking errors** — must be fixed before the dashboard can load:")
            for e in errors:
                st.error(e)
        if warnings:
            st.markdown("**Warnings** — dashboard will run, but review these:")
            for w in warnings:
                st.warning(w)
        if not errors and not warnings:
            st.success("All quality checks passed.")


def _resolve_data_source() -> pd.DataFrame:
    """Render the Data Source section in the sidebar and return the active DataFrame.

    Returns:
        Sample dataset or user-uploaded dataset after quality validation.
    """
    source = st.sidebar.radio(
        "Data Source",
        ["Use Sample Manufacturing Dataset", "Upload Company CSV"],
        index=0,
    )

    if source == "Use Sample Manufacturing Dataset":
        return load_data()

    # --- Upload path ---
    uploaded = st.sidebar.file_uploader(
        "Upload CSV file",
        type=["csv"],
        help="Required columns: timestamp, machine_id, status, utilization_pct, "
             "temperature_c, vibration_hz, output_count, defect_count, cycle_time_sec",
    )

    if uploaded is None:
        st.info("Upload a CSV file in the sidebar to use your own data, or switch to the sample dataset.")
        st.stop()

    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.stop()

    df = prepare_uploaded_data(raw_df)
    report = validate_data_quality(df)

    _show_quality_report(report)

    if has_blocking_errors(report):
        st.error(
            "The uploaded file has blocking errors that prevent the dashboard from loading. "
            "Fix the issues listed in the Data Quality Report above, then re-upload."
        )
        st.stop()

    if report.get("warnings"):
        st.warning("Data loaded with warnings. Review the Data Quality Report in the sidebar.")

    return df


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> pd.Timestamp:
    """Render the time control section of the sidebar and return the selected timestamp.

    Works with any active DataFrame — both the built-in sample dataset and a
    user-uploaded CSV — by deriving the time range from the passed DataFrame.

    Args:
        df: Full sensor DataFrame (sample or uploaded).

    Returns:
        Selected pd.Timestamp.
    """
    t_min: pd.Timestamp = df["timestamp"].min()
    t_max: pd.Timestamp = df["timestamp"].max()

    st.sidebar.markdown("**Time Control**")
    st.sidebar.caption(
        f"Data range: {t_min.strftime('%Y-%m-%d %H:%M')} → {t_max.strftime('%Y-%m-%d %H:%M')}"
    )

    time_mode = st.sidebar.radio(
        "Mode",
        ["Latest Snapshot", "Historical Replay"],
        index=0,
    )

    if time_mode == "Latest Snapshot":
        selected_time = t_max
        st.sidebar.info(f"Viewing: **{selected_time.strftime('%Y-%m-%d %H:%M')}**")
    else:
        # Datetime slider — thumb displays the actual timestamp, step = 15 min.
        # Works for any date range, including uploaded CSVs with different periods.
        selected_dt = st.sidebar.slider(
            "Select Time",
            min_value=t_min.to_pydatetime(),
            max_value=t_max.to_pydatetime(),
            value=t_max.to_pydatetime(),
            step=timedelta(minutes=SLIDER_STEP_MINUTES),
            format="YYYY-MM-DD HH:mm",
        )
        selected_time = pd.Timestamp(selected_dt)
        st.sidebar.info(f"Viewing: **{selected_time.strftime('%Y-%m-%d %H:%M')}**")

    st.sidebar.markdown("---")
    st.sidebar.caption("Manufacturing Monitoring Dashboard v1.0")
    return selected_time


# ---------------------------------------------------------------------------
# Page 1: Operations Command Center
# ---------------------------------------------------------------------------

def _oee_status_badge(oee: float) -> str:
    if oee >= _OEE_CFG.oee_good_threshold:
        return "🟢 Good"
    if oee >= _OEE_CFG.oee_watch_threshold:
        return "🟡 Watch"
    return "🔴 Critical"


def _machine_card_color(row: pd.Series, health_summary: pd.DataFrame) -> str:
    """Return a color label for the machine card based on status and health."""
    status = row.get("status", "running")
    machine_id = row.get("machine_id", "")

    health_row = health_summary[health_summary["machine_id"] == machine_id]
    health = float(health_row["health_score"].iloc[0]) if not health_row.empty else 80.0
    fail_prob = float(health_row["failure_probability_7d"].iloc[0]) if not health_row.empty else 0.0

    if status in ("down",) or health < _MAINT_CFG.health_score_critical or fail_prob > 0.70:
        return "red"
    if (
        status == "maintenance"
        or health < _MAINT_CFG.health_score_warning
        or fail_prob > 0.40
    ):
        return "orange"
    return "green"


def page_command_center(df: pd.DataFrame, windows: dict, selected_time: pd.Timestamp) -> None:
    """Render Operations Command Center page."""
    st.title("Production Equipment Monitoring & Predictive Maintenance Dashboard")
    st.subheader("Operations Command Center")
    st.caption(f"Last updated: **{selected_time.strftime('%Y-%m-%d %H:%M')}**")
    st.markdown("---")

    current_snapshot = windows["current_snapshot"]
    today_window = windows["today_window"]

    model = get_trained_model(hash((len(df), str(df["timestamp"].max()))), df)
    health_summary = generate_machine_health_summary(df, selected_time, model)

    # --- Top KPI cards ---
    oee_result = calculate_oee(today_window, _OEE_CFG)
    overall_oee = oee_result["oee"]

    online_count = 0
    offline_count = 0
    if not current_snapshot.empty:
        online_count = int(current_snapshot["status"].isin(["running", "idle"]).sum())
        offline_count = int(current_snapshot["status"].isin(["down", "maintenance"]).sum())

    # Anomaly count for today
    today_anomaly_df = today_window.copy()
    if len(today_anomaly_df) >= _ANOMALY_CFG.min_rows_for_model:
        today_anomaly_df = detect_anomalies(today_anomaly_df, contamination=_ANOMALY_CFG.default_contamination)
    today_anomalies = int(today_anomaly_df.get("is_anomaly", pd.Series(False)).sum())

    # Maintenance risk count
    maint_risk_count = 0
    if not health_summary.empty:
        maint_risk_count = int(
            (
                (health_summary["failure_probability_7d"] >= 0.70)
                | (health_summary["health_score"] < _MAINT_CFG.health_score_critical)
            ).sum()
        )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Overall OEE", f"{overall_oee * 100:.1f}%", delta=_oee_status_badge(overall_oee))
    with kpi2:
        st.metric("Online Machines", f"{online_count} / {online_count + offline_count}")
    with kpi3:
        st.metric("Today's Anomalies", today_anomalies)
    with kpi4:
        st.metric("Maintenance Risk", f"{maint_risk_count} machine(s)")

    st.markdown("---")

    # --- Machine Status Grid ---
    st.subheader("Machine Status")

    if current_snapshot.empty:
        st.warning("No snapshot data available for the selected time.")
    else:
        machine_ids = sorted(current_snapshot["machine_id"].unique())
        cols = st.columns(min(len(machine_ids), 4))
        for i, machine_id in enumerate(machine_ids):
            row = current_snapshot[current_snapshot["machine_id"] == machine_id].iloc[0]
            col = cols[i % 4]

            health_row = health_summary[health_summary["machine_id"] == machine_id]
            health_score = float(health_row["health_score"].iloc[0]) if not health_row.empty else 80.0
            fail_prob = float(health_row["failure_probability_7d"].iloc[0]) if not health_row.empty else 0.0

            machine_oee = calculate_oee(
                today_window[today_window["machine_id"] == machine_id], _OEE_CFG
            )["oee"]

            color = _machine_card_color(row, health_summary)
            status_icon = {"running": "🟢", "idle": "🟡", "down": "🔴", "maintenance": "🔵"}.get(
                row["status"], "⚪"
            )

            with col:
                with st.container(border=True):
                    st.markdown(f"**{machine_id}** {status_icon} `{row['status'].upper()}`")
                    st.markdown(
                        f"Health: **{health_score:.0f}** | OEE: **{machine_oee * 100:.1f}%**"
                    )
                    st.markdown(
                        f"Temp: {row['temperature_c']:.1f}°C | Vib: {row['vibration_hz']:.1f} Hz"
                    )
                    st.markdown(f"Util: {row['utilization_pct']:.0f}% | Risk: {fail_prob * 100:.0f}%")

    st.markdown("---")

    # --- Alert Feed + Recommendations ---
    col_alert, col_rec = st.columns([1, 1])

    with col_alert:
        st.subheader("Alert Feed")
        alert_df = create_alert_feed(df, selected_time, max_alerts=10)
        if alert_df.empty:
            st.info("No active alerts at this time.")
        else:
            for _, alert_row in alert_df.iterrows():
                icon = {"High": "🔴", "Medium": "🟡", "Low": "🔵"}.get(alert_row["severity"], "⚪")
                ts_str = pd.Timestamp(alert_row["timestamp"]).strftime("%H:%M")
                st.markdown(
                    f"{icon} `[{alert_row['severity']}]` **{ts_str}** | {alert_row['message']}"
                )

    with col_rec:
        st.subheader("Recommended Actions")
        if health_summary.empty:
            st.info("No recommendations available.")
        else:
            recs = generate_maintenance_recommendations(health_summary)
            for rank, (_, rec_row) in enumerate(recs.head(3).iterrows(), 1):
                machine_id = rec_row["machine_id"]
                rec_text = rec_row["recommendation"]
                savings = rec_row["expected_savings"]
                fail_prob = rec_row["failure_probability_7d"] * 100

                # Context hint per machine story
                hint = _recommendation_hint(machine_id, rec_row)

                with st.container(border=True):
                    st.markdown(f"**{rank}. {machine_id}** — {rec_text}")
                    st.caption(hint)
                    if savings > 0:
                        st.caption(f"Est. savings if maintained: **${savings:,.0f}**")


def _recommendation_hint(machine_id: str, row: pd.Series) -> str:
    """Generate a context-specific recommendation hint."""
    fail_prob = row.get("failure_probability_7d", 0.0)
    vib = row.get("avg_vibration_6h", 10.0)
    temp = row.get("avg_temperature_6h", 65.0)
    oee = row.get("oee", 1.0)

    if machine_id == "M-003":
        return "Vibration trend is increasing; 7-day failure probability is high. Schedule maintenance within 48 hours."
    if machine_id == "M-005":
        return "Repeated temperature anomalies during high-utilization periods. Inspect cooling system."
    if machine_id == "M-004":
        return "Low OEE due to slow cycle time. Investigate bottleneck — process optimization may reduce cycle time."
    if machine_id == "M-007":
        return "Frequent unplanned downtime events are reducing availability. Review maintenance schedule."
    if vib > 15:
        return f"Vibration is elevated ({vib:.1f} Hz). Monitor for mechanical wear."
    if temp > 78:
        return f"Average temperature is high ({temp:.1f}°C). Check cooling capacity."
    if fail_prob > 0.40:
        return f"7-day failure probability at {fail_prob * 100:.0f}%. Inspect at next available window."
    return "Continue monitoring. No immediate action required."


# ---------------------------------------------------------------------------
# Page 2: OEE & Bottleneck Analysis
# ---------------------------------------------------------------------------

def page_oee_analysis(df: pd.DataFrame, windows: dict, selected_time: pd.Timestamp) -> None:
    """Render OEE & Bottleneck Analysis page."""
    st.title("OEE & Bottleneck Analysis")
    st.caption(
        f"Data context: {(selected_time - pd.Timedelta(days=7)).strftime('%Y-%m-%d')} – "
        f"{selected_time.strftime('%Y-%m-%d %H:%M')}"
    )
    st.markdown("---")

    today_window = windows["today_window"]
    trend_window = windows["trend_window"]

    machine_ids = sorted(df["machine_id"].unique())
    selected_machines = st.multiselect(
        "Filter machines",
        options=machine_ids,
        default=machine_ids,
    )

    if not selected_machines:
        st.warning("Please select at least one machine.")
        return

    filtered_today = today_window[today_window["machine_id"].isin(selected_machines)]
    filtered_trend = trend_window[trend_window["machine_id"].isin(selected_machines)]

    # OEE by machine
    oee_by_machine = calculate_oee_by_machine(filtered_today, _OEE_CFG)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("OEE by Machine")
        if oee_by_machine.empty:
            st.info("No OEE data.")
        else:
            # Simple bar chart for OEE only
            import plotly.express as px
            plot_df = oee_by_machine.copy()
            plot_df["oee_pct"] = (plot_df["oee"] * 100).round(1)
            fig = px.bar(
                plot_df.sort_values("oee_pct"),
                x="oee_pct",
                y="machine_id",
                orientation="h",
                color="oee_pct",
                color_continuous_scale=["red", "orange", "green"],
                range_color=[0, 100],
                text="oee_pct",
                labels={"oee_pct": "OEE (%)", "machine_id": "Machine"},
                title="Today's OEE by Machine",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("OEE Factor Decomposition")
        st.plotly_chart(create_oee_factor_chart(oee_by_machine), use_container_width=True)

    st.subheader("7-Day OEE Trend")
    trend_df = calculate_oee_trend(filtered_trend, _OEE_CFG)
    st.plotly_chart(create_oee_trend_chart(trend_df), use_container_width=True)

    st.subheader("Bottleneck Ranking")
    if not oee_by_machine.empty:
        # Enrich with extra columns
        enriched = oee_by_machine.copy()

        for machine_id, grp in filtered_today.groupby("machine_id"):
            idx = enriched["machine_id"] == machine_id
            enriched.loc[idx, "downtime_minutes"] = float(len(grp[grp["status"] == "down"]))
            total_out = grp["output_count"].sum()
            total_def = grp["defect_count"].sum()
            enriched.loc[idx, "defect_rate"] = float(total_def / total_out) if total_out > 0 else 0.0
            running = grp[(grp["status"] == "running") & (grp["cycle_time_sec"] > 0)]
            enriched.loc[idx, "avg_cycle_time"] = running["cycle_time_sec"].mean() if not running.empty else 0.0
            enriched.loc[idx, "avg_utilization"] = grp["utilization_pct"].mean()

        bottleneck_df = calculate_bottleneck_score(enriched)

        display_cols = [
            "machine_id", "oee", "availability", "performance", "quality",
            "downtime_minutes", "defect_rate", "avg_cycle_time", "bottleneck_score", "main_issue"
        ]
        existing = [c for c in display_cols if c in bottleneck_df.columns]
        show = bottleneck_df[existing].copy()

        for pct_col in ["oee", "availability", "performance", "quality"]:
            if pct_col in show.columns:
                show[pct_col] = (show[pct_col] * 100).round(1).astype(str) + "%"
        if "defect_rate" in show.columns:
            show["defect_rate"] = (show["defect_rate"] * 100).round(2).astype(str) + "%"
        if "avg_cycle_time" in show.columns:
            show["avg_cycle_time"] = show["avg_cycle_time"].round(1).astype(str) + "s"

        st.dataframe(show, use_container_width=True)
    else:
        st.info("No data for bottleneck analysis.")


# ---------------------------------------------------------------------------
# Page 3: Anomaly Detection
# ---------------------------------------------------------------------------

def page_anomaly_detection(df: pd.DataFrame, windows: dict, selected_time: pd.Timestamp) -> None:
    """Render Anomaly Detection page."""
    st.title("Anomaly Detection")
    st.markdown("*Isolation Forest ML model with deviation-based explanation*")
    st.markdown("---")

    sensor_window = windows["sensor_window"]
    machine_ids = sorted(df["machine_id"].unique())

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        selected_machine = st.selectbox("Machine", options=machine_ids, index=2)
    with col_ctrl2:
        sensor = st.selectbox(
            "Sensor",
            options=["temperature_c", "vibration_hz", "utilization_pct", "cycle_time_sec"],
            index=0,
        )
    with col_ctrl3:
        contamination = st.slider(
            "Sensitivity",
            min_value=_ANOMALY_CFG.min_contamination,
            max_value=_ANOMALY_CFG.max_contamination,
            value=_ANOMALY_CFG.default_contamination,
            step=0.01,
            help="Higher sensitivity flags more points as anomalies.",
        )

    machine_sensor_df = sensor_window[sensor_window["machine_id"] == selected_machine].copy()

    if len(machine_sensor_df) < _ANOMALY_CFG.min_rows_for_model:
        st.warning(
            f"Only {len(machine_sensor_df)} rows in the 24-hour window for {selected_machine}. "
            "Anomaly detection requires at least "
            f"{_ANOMALY_CFG.min_rows_for_model} rows."
        )
        anomaly_df = machine_sensor_df.copy()
        anomaly_df["is_anomaly"] = False
        anomaly_df["anomaly_score"] = 0.0
        anomaly_df["anomaly_severity"] = "normal"
    else:
        anomaly_df = detect_anomalies(machine_sensor_df, contamination=contamination)

    # Summary KPIs
    total_anomalies = int(anomaly_df["is_anomaly"].sum())
    high_sev = int((anomaly_df.get("anomaly_severity", "") == "high").sum())
    latest_anomaly = (
        anomaly_df[anomaly_df["is_anomaly"]]["timestamp"].max()
        if total_anomalies > 0
        else None
    )
    sensor_std = anomaly_df.groupby("machine_id")[[
        "temperature_c", "vibration_hz", "cycle_time_sec"
    ]].std().max(axis=1)
    most_affected = sensor_std.idxmax() if not sensor_std.empty else "N/A"

    ka, kb, kc, kd = st.columns(4)
    with ka:
        st.metric("Total Anomalies (24h)", total_anomalies)
    with kb:
        st.metric("High Severity", high_sev)
    with kc:
        st.metric("Most Affected", selected_machine)
    with kd:
        st.metric(
            "Latest Anomaly",
            latest_anomaly.strftime("%H:%M") if latest_anomaly else "None",
        )

    st.markdown("---")
    st.subheader(f"{selected_machine} — Sensor Chart (24h)")
    st.plotly_chart(
        create_sensor_anomaly_chart(anomaly_df, sensor, selected_machine),
        use_container_width=True,
    )

    # Explanation panel
    st.subheader("Anomaly Explanation")
    anomalies_only = anomaly_df[anomaly_df["is_anomaly"]].sort_values("timestamp", ascending=False)

    if anomalies_only.empty:
        st.info("No anomalies detected in the current window.")
    else:
        latest_row = anomalies_only.iloc[0]
        explanation = generate_anomaly_explanation(
            df[df["machine_id"] == selected_machine], latest_row
        )
        with st.container(border=True):
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                st.markdown(f"**Machine:** {latest_row.get('machine_id', 'N/A')}")
                st.markdown(
                    f"**Timestamp:** {pd.Timestamp(latest_row['timestamp']).strftime('%Y-%m-%d %H:%M')}"
                )
                st.markdown(f"**Temperature:** {latest_row.get('temperature_c', 0):.1f}°C")
                st.markdown(f"**Vibration:** {latest_row.get('vibration_hz', 0):.2f} Hz")
            with col_e2:
                st.markdown(f"**Anomaly Score:** {latest_row.get('anomaly_score', 0):.3f}")
                st.markdown(f"**Severity:** {latest_row.get('anomaly_severity', 'N/A')}")
                st.markdown(f"**Cycle Time:** {latest_row.get('cycle_time_sec', 0):.1f}s")
                st.markdown(f"**Utilization:** {latest_row.get('utilization_pct', 0):.1f}%")
            st.markdown(f"**Interpretation:** {explanation}")

    # Recent anomaly table
    st.subheader("Recent Anomaly Log")
    if anomalies_only.empty:
        st.info("No anomaly events to display.")
    else:
        display = anomalies_only[
            ["timestamp", "machine_id", "temperature_c", "vibration_hz",
             "cycle_time_sec", "anomaly_score", "anomaly_severity"]
        ].head(20).copy()
        display["timestamp"] = display["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display.columns = ["Timestamp", "Machine", "Temp (°C)", "Vibration (Hz)",
                           "Cycle (s)", "Anomaly Score", "Severity"]
        st.dataframe(display, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4: Predictive Maintenance
# ---------------------------------------------------------------------------

def page_predictive_maintenance(df: pd.DataFrame, windows: dict, selected_time: pd.Timestamp) -> None:
    """Render Predictive Maintenance page."""
    st.title("Predictive Maintenance")
    st.markdown("*Random Forest model trained on rolling sensor features*")
    st.markdown("---")

    model = get_trained_model(hash((len(df), str(df["timestamp"].max()))), df)
    health_summary = generate_machine_health_summary(df, selected_time, model)
    recommendations = generate_maintenance_recommendations(health_summary)

    # Top KPI cards
    high_risk_count = 0
    avg_health = 0.0
    expected_savings_total = 0.0
    open_recs = 0

    if not health_summary.empty:
        high_risk_count = int((health_summary["failure_probability_7d"] >= 0.70).sum())
        avg_health = float(health_summary["health_score"].mean())

    if not recommendations.empty:
        expected_savings_total = float(recommendations["expected_savings"].clip(lower=0).sum())
        open_recs = int((recommendations["recommendation"] != "Continue normal monitoring").sum())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("High Risk Machines", high_risk_count)
    with k2:
        st.metric("Avg Health Score", f"{avg_health:.1f}")
    with k3:
        st.metric("Expected Savings", f"${expected_savings_total:,.0f}")
    with k4:
        st.metric("Open Recommendations", open_recs)

    st.markdown("---")

    col_health, col_prob = st.columns(2)
    with col_health:
        st.plotly_chart(create_health_score_chart(health_summary), use_container_width=True)
    with col_prob:
        st.plotly_chart(create_failure_probability_chart(health_summary), use_container_width=True)

    st.subheader("Maintenance Recommendation Table")
    if recommendations.empty:
        st.info("No recommendations available.")
    else:
        display = recommendations.copy()
        if "health_score" in display.columns:
            display["health_score"] = display["health_score"].round(1)
        if "failure_probability_7d" in display.columns:
            display["failure_probability_7d"] = (
                display["failure_probability_7d"] * 100
            ).round(1).astype(str) + "%"
        if "confidence" in display.columns:
            display["confidence"] = (display["confidence"] * 100).round(0).astype(str) + "%"
        if "expected_savings" in display.columns:
            display["expected_savings"] = display["expected_savings"].apply(
                lambda v: f"${v:,.0f}" if v > 0 else f"-${abs(v):,.0f}"
            )

        display.columns = [c.replace("_", " ").title() for c in display.columns]
        st.dataframe(display, use_container_width=True)

    # Cost-benefit explanation
    st.subheader("Cost-Benefit Logic")
    with st.expander("How expected savings are calculated"):
        st.markdown(
            f"""
| Parameter | Value |
|---|---|
| Planned maintenance cost | ${_MAINT_CFG.planned_maintenance_cost:,.0f} |
| Unplanned failure cost | ${_MAINT_CFG.unplanned_failure_cost:,.0f} |

**Formula:**
`Expected Savings = (Failure Probability × Unplanned Failure Cost) − Planned Maintenance Cost`

**Example:** M-003 at 80% failure probability:
`(0.80 × $8,000) − $1,500 = $4,900 in expected savings`

A positive savings value means scheduled maintenance is financially justified.
"""
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Dashboard entry point."""
    st.sidebar.title("⚙️ Dashboard Controls")
    st.sidebar.markdown("---")

    df = _resolve_data_source()

    if df.empty:
        st.error("No data available. Run `python -m src.simulator` first.")
        st.stop()

    st.sidebar.markdown("---")
    selected_time = render_sidebar(df)
    windows = get_time_windows(df, selected_time)

    pages = {
        "Operations Command Center": page_command_center,
        "OEE & Bottleneck Analysis": page_oee_analysis,
        "Anomaly Detection": page_anomaly_detection,
        "Predictive Maintenance": page_predictive_maintenance,
    }

    with st.sidebar:
        st.markdown("---")
        page_name = st.radio("Navigate", list(pages.keys()), index=0)

    pages[page_name](df, windows, selected_time)


if __name__ == "__main__":
    main()
