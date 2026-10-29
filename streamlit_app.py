import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
from baseline import run_baseline_analysis, load_prices, MAIN_DATASET, INFLATION_DATASET
import Scripts.spike_ml as sml

st.set_page_config(
    page_title="Co-Boost: Sri Lanka Food Security DSS",
    page_icon="🛒",
    layout="wide"
)

df = load_prices(MAIN_DATASET)
weekly_df = df[df['frequency'] == 'weekly']

# Cached baseline results (computed once per dataset change)
@st.cache_resource
def get_baseline_results():
    return run_baseline_analysis(MAIN_DATASET, INFLATION_DATASET)

result = get_baseline_results()

# Compute alerts - severity = predicted price RISK (expected move %),
# confidence = model uncertainty. Kept separate: low confidence means
# "noisy data", not "dangerous price" (fixed from the old inverted mapping).
alert_severities = {}
for comm in weekly_df['commodity'].unique():
    if pd.isna(comm):
        continue
    comm_results = result['baseline_results'].get(comm, {})
    if not comm_results:
        continue
    residual_data = comm_results.get('residual_based', {})
    if not residual_data:
        continue
    confidence = residual_data.get('confidence', 'Low')
    inflation = residual_data.get('inflation_rate', 0)
    forecast = residual_data.get('next_week_forecast', 0)
    last_price = residual_data.get('last_price', 0)

    expected_move = (abs(forecast - last_price) / last_price * 100
                     if last_price else 0.0)
    if expected_move >= 10:
        severity = 'High'      # big predicted move next week
    elif expected_move >= 5:
        severity = 'Medium'
    else:
        severity = 'Low'

    alert_severities[comm] = {
        'severity': severity,
        'confidence': confidence,
        'inflation': inflation,
        'expected_move': expected_move,
        'forecast': forecast,
        'last_price': last_price,
    }

# Categorize alerts
high_alerts = {k: v for k, v in alert_severities.items() if v['severity'] == 'High'}
medium_alerts = {k: v for k, v in alert_severities.items() if v['severity'] == 'Medium'}
low_alerts = {k: v for k, v in alert_severities.items() if v['severity'] == 'Low'}

# Parent-child mapping for commodity grouping
PARENT_CHILD_MAP = {
    'Raw Red - ( Average )': {'parent': 'Raw Red', 'type': 'average'},
    'Raw Red - No 1.': {'parent': 'Raw Red', 'type': 'no1'},
    'Red Onions - Average': {'parent': 'Red Onions', 'type': 'average'},
    'B.Onions': {'parent': 'B.Onions', 'type': 'base'},
    'B.Onions - Local': {'parent': 'B.Onions', 'type': 'local'},
    'Beans - Butter': {'parent': 'Beans', 'type': 'butter'},
    'Beans - Green': {'parent': 'Beans', 'type': 'green'},
    'Long Beans': {'parent': 'Beans', 'type': 'long'},
    'Beans': {'parent': 'Beans', 'type': 'base'},
    'Mullet': {'parent': 'Fish', 'type': 'mullet'},
    'Thalapath': {'parent': 'Fish', 'type': 'thalapath'},
    'Balaya': {'parent': 'Fish', 'type': 'balaya'},
    'Small Mullet': {'parent': 'Fish', 'type': 'small_mullet'},
}

# Page config
st.title("Co-Boost: Sri Lanka Food Security DSS")
st.markdown("**Colombo Market Price Spikes Alert System** | Weekly DCS Data | 2019-2026")

# Sidebar with user preferences
st.sidebar.header("📋 User Preferences")
selected_commodities = st.sidebar.multiselect(
    "Select commodities to monitor:",
    options=sorted(weekly_df['commodity'].dropna().unique(), key=str),
    default=[],
    help="Select up to 5 commodities for detailed tracking"
)
st.sidebar.caption(f"Selected: {len(selected_commodities)} commodities")

# Alert list area (main area - no sidebar clutter)
st.header("🚨 Price Alert List")

# High severity
if high_alerts:
    st.error(f"⚠️ **HIGH SEVERITY: {len(high_alerts)} alerts requiring immediate attention**")
    with st.expander(f"View {len(high_alerts)} High Severity Alerts"):
        high_data = []
        for comm, data in sorted(high_alerts.items(), key=lambda x: x[1]['forecast'], reverse=True):
            high_data.append({
                'Commodity': comm,
                'Forecast Price': f"Rs. {data['forecast']:.2f}",
                'Last Price': f"Rs. {data['last_price']:.2f}",
                'Expected Move': f"{data['expected_move']:.1f}%",
                'Confidence': data['confidence'],
                'Food Inflation': f"{data['inflation']:.1f}%",
                'Severity': 'HIGH'
            })
        st.dataframe(pd.DataFrame(high_data), height=300)
else:
    st.success("✅ **No High Severity Alerts** - Prices are stable across all commodities")

# Medium severity
if medium_alerts:
    st.warning(f"⚠️ **MEDIUM SEVERITY: {len(medium_alerts)} alerts monitoring required**")
    with st.expander(f"View {len(medium_alerts)} Medium Severity Alerts"):
        med_data = []
        for comm, data in sorted(medium_alerts.items(), key=lambda x: x[1]['forecast'], reverse=True):
            med_data.append({
                'Commodity': comm,
                'Forecast Price': f"Rs. {data['forecast']:.2f}",
                'Last Price': f"Rs. {data['last_price']:.2f}",
                'Expected Move': f"{data['expected_move']:.1f}%",
                'Confidence': data['confidence'],
                'Food Inflation': f"{data['inflation']:.1f}%",
                'Severity': 'MEDIUM'
            })
        st.dataframe(pd.DataFrame(med_data), height=300)
else:
    st.info("ℹ️ No Medium Severity Alerts")

# Low severity
if low_alerts:
    st.caption(f"ℹ️ **LOW SEVERITY: {len(low_alerts)} alerts** - Prices stable, monitoring only")
    with st.expander(f"View {len(low_alerts)} Low Severity Alerts"):
        low_data = []
        for comm, data in sorted(low_alerts.items()):
            low_data.append({
                'Commodity': comm,
                'Forecast Price': f"Rs. {data['forecast']:.2f}",
                'Last Price': f"Rs. {data['last_price']:.2f}",
                'Expected Move': f"{data['expected_move']:.1f}%",
                'Confidence': data['confidence'],
                'Food Inflation': f"{data['inflation']:.1f}%",
                'Severity': 'LOW'
            })
        st.dataframe(pd.DataFrame(low_data), height=200)
else:
    st.caption("ℹ️ No Low Severity Alerts")

# ------------------------------------------------- anomaly & spike watch
@st.cache_resource
def get_spike_bundle():
    return sml.load_bundle()

@st.cache_data(ttl=3600)
def get_anomaly_table():
    return sml.anomaly_table(load_prices(MAIN_DATASET), get_spike_bundle())

st.header("⚡ Anomaly & Spike Watch")
st.caption("Special alerts for statistically unusual behaviour - different symbols "
           "from the regular severity list above. "
           "⚡ spike predicted | 🔺 unusual high | 🔻 unusual low | "
           "🧊 stable as never | 🌀 unusual move")

anomaly = get_anomaly_table()
bundle = get_spike_bundle()
scored = anomaly[anomaly.spike_prob == anomaly.spike_prob]  # non-NaN probabilities

spike_mode = st.sidebar.radio(
    "⚡ Spike alert mode",
    ["Balanced (top ~10%)", "High-confidence"],
    help="Alert-fatigue rule: Balanced fires the top ~10% of monitored "
         "commodities by spike probability (self-calibrating after every "
         "model retrain). High-confidence fires only above a fixed "
         "probability for critical supply-chain decisions.")
if spike_mode.startswith("Balanced"):
    bal_thr = float(scored.spike_prob.quantile(0.90)) if len(scored) else bundle["threshold"]
    thr = max(bal_thr, 0.05)
    st.sidebar.caption(f"Balanced threshold: {thr:.2f} (auto from data)")
else:
    thr = 0.55
    st.sidebar.caption("Fixed threshold: ≥ 0.55 probability")

spikes = scored[scored.spike_prob >= thr] if len(scored) else scored.iloc[0:0]
stats_flags = anomaly[anomaly.flag.notna() & ~anomaly.index.isin(spikes.index)]

if len(anomaly) == 0:
    st.success("✅ No anomalies detected - all monitored prices look normal")
else:
    intro_cols = st.columns([1, 1, 1, 1])
    intro_cols[0].metric("⚡ Spike predicted", len(spikes),
                         delta=None if spike_mode.startswith("Balanced") else "≥0.55",
                         delta_color="off")
    intro_cols[1].metric("🔺 / 🔻 Unusual price",
                         int((stats_flags.flag.isin(["high", "low"])).sum()))
    intro_cols[2].metric("🧊 Stable as never", int((stats_flags.flag == "calm").sum()))
    intro_cols[3].metric("🌀 Unusual move", int((stats_flags.flag == "move").sum()))

    sp_view = spikes.assign(
        symbol="⚡", detail="⚡ Spike predicted (ML)",
        direction_lab=spikes.direction.map(
            lambda d: f"↑ Up ({d})" if d == "up" else
                      (f"↓ Down ({d})" if d == "down" else "→ Stable")))
    sf_view = stats_flags.assign(direction_lab=stats_flags.direction)
    alert_tbl = pd.concat([
        sp_view[["symbol", "detail", "spike_prob", "direction_lab"]],
        sf_view[["symbol", "detail", "spike_prob", "direction_lab"]],
    ]).reset_index()
    alert_tbl["spike_prob"] = alert_tbl["spike_prob"].map(
        lambda p: "-" if pd.isna(p) else f"{p:.0%}")
    alert_tbl.columns = ["", "Commodity", "Alert", "Spike ML probability",
                         "ML direction next week"]
    st.dataframe(alert_tbl, height=min(60 + 35 * len(alert_tbl), 420), hide_index=True)

    st.subheader("🔎 Anomaly drill-down (prediction direction vs past weeks)")
    drill = pd.concat([spikes, stats_flags]).sort_values(
        "spike_prob", ascending=False, na_position="last")
    for _, row in drill.head(8).iterrows():
        comm = row["commodity"]
        comm_df = weekly_df[weekly_df["commodity"] == comm].sort_values("observation_date")
        if comm_df.empty:
            continue
        prices = comm_df["price"].to_numpy(float)[-26:]
        dates = comm_df["observation_date"].to_numpy()[-26:]
        last_price, last_date = float(prices[-1]), pd.to_datetime(dates[-1])
        next_date = (last_date + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        pred_r1 = 0.0 if pd.isna(row["pred_r1"]) else float(row["pred_r1"])
        wk_moves = np.abs(np.diff(prices) / prices[:-1])
        typical_move = float(np.mean(wk_moves)) if len(wk_moves) else 0.0

        fig = px.line(pd.DataFrame({"Date": pd.to_datetime(dates),
                                    "Price": prices}),
                      x="Date", y="Price", template="plotly_white",
                      title=f"{row['symbol']} {comm} - {row['detail']}")
        fig.add_scatter(x=[last_date], y=[last_price], mode="markers",
                        marker=dict(color="red", size=10), name="Current")
        fig.add_vline(x=next_date, line_dash="dash", line_color="red")

        dir_word, dir_arrow = {"up": ("UPWARD", "↗"), "down": ("DOWNWARD", "↘"),
                               "stable": ("STABLE", "→")}[row["direction"]]
        band = 1.96 * np.std(np.diff(prices[-9:]) / prices[-9:-1]) if len(prices) > 10 else typical_move
        fc_price = last_price * (1 + pred_r1)
        y_hi, y_lo = fc_price * (1 + band), max(0.0, fc_price * (1 - band))
        fig.add_scatter(x=[pd.Timestamp(next_date)], y=[y_hi], mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=[pd.Timestamp(next_date)], y=[y_lo], mode="lines",
                        line=dict(width=0), showlegend=False, hoverinfo="skip",
                        fill="tonexty", fillcolor="rgba(255,80,0,0.15)")
        fig.add_scatter(x=[pd.Timestamp(next_date)], y=[fc_price], mode="markers",
                        marker=dict(color="orange", size=11, symbol="arrow-up"
                                    if row["direction"] == "up" else
                                    ("arrow-down" if row["direction"] == "down" else "circle")),
                        name="ML forecast")
        st.plotly_chart(fig, use_container_width=True)
        spike_txt = "-" if pd.isna(row["spike_prob"]) else f"{row['spike_prob']:.0%}"
        st.markdown(
            f"**ML expects next week: {dir_word} {dir_arrow} "
            f"({pred_r1:+.1%} → Rs. {fc_price:.2f})** "
            f"| typical past-week move: {typical_move:.1%} "
            f"| spike probability: {spike_txt}"
        )
        st.markdown("---")

# Commodity drill-down
if selected_commodities:
    st.header("📊 Price History & Forecast")
    
    # Show selected commodities (limit to 5 for performance)
    for comm in selected_commodities[:5]:
        comm_df = weekly_df[weekly_df['commodity'] == comm].sort_values('observation_date')
        if len(comm_df) > 0:
            prices = comm_df['price'].values
            dates = comm_df['observation_date'].values
            
            # Get baseline results
            baseline = result['baseline_results'].get(comm, {})
            
            # Create chart
            fig_data = pd.DataFrame({'Price': prices}, index=dates).reset_index()
            fig_data.columns = ['Date', 'Price']
            
            last_date = dates[-1]
            last_dt = datetime.strptime(last_date, '%Y-%m-%d')
            forecast_date = (last_dt + timedelta(days=7)).strftime('%Y-%m-%d')
            
            fig_px = px.line(fig_data, x='Date', y='Price', 
                           title=f"{comm} - Price History & Next Week Forecast",
                           template="plotly_white")
            
            fig_px.add_vline(x=forecast_date, line_dash='dash', line_color='red', 
                           annotation_text='Next Week Forecast', annotation_position='top right')
            
            # Add confidence annotation
            if comm in baseline:
                res_data = baseline.get('residual_based', {})
                conf = res_data.get('confidence', 'N/A')
                inf = res_data.get('inflation_rate', 0)
                forecast = res_data.get('next_week_forecast', 0)
                
                color_map = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
                fig_px.add_annotation(
                    x=0.02, y=0.98,
                    xref='paper', yref='paper',
                    text=f'Confidence: <b><span style="color:{color_map.get(conf, "gray")}">{conf}</span></b> | '
                         f'Food Inflation: <b>{inf:.1f}%</b> | '
                         f'Next Week Forecast: Rs.{forecast:.2f}',
                    showarrow=False,
                    bordercolor=color_map.get(conf, 'gray'),
                    borderwidth=1,
                    bgcolor='rgba(255,255,255,0.8)',
                    font=dict(size=11)
                )
            
            st.plotly_chart(fig_px, use_container_width=True)
            st.markdown("---")
else:
    # Show default commodity if none selected
    default_comm = weekly_df['commodity'].dropna().unique()[0] if len(weekly_df['commodity'].dropna().unique()) > 0 else None
    if default_comm:
        comm = default_comm
        comm_df = weekly_df[weekly_df['commodity'] == comm].sort_values('observation_date')
        prices = comm_df['price'].values
        dates = comm_df['observation_date'].values
        
        baseline = result['baseline_results'].get(comm, {})
        
        fig_data = pd.DataFrame({'Price': prices}, index=dates).reset_index()
        fig_data.columns = ['Date', 'Price']
        
        last_date = dates[-1]
        last_dt = datetime.strptime(last_date, '%Y-%m-%d')
        forecast_date = (last_dt + timedelta(days=7)).strftime('%Y-%m-%d')
        
        fig_px = px.line(fig_data, x='Date', y='Price', 
                       title=f"{comm} - Price History & Next Week Forecast",
                       template="plotly_white")
        
        fig_px.add_vline(x=forecast_date, line_dash='dash', line_color='red', 
                       annotation_text='Next Week Forecast', annotation_position='top right')
        
        if comm in baseline:
            res_data = baseline.get('residual_based', {})
            conf = res_data.get('confidence', 'N/A')
            inf = res_data.get('inflation_rate', 0)
            forecast = res_data.get('next_week_forecast', 0)
            
            color_map = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
            fig_px.add_annotation(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text=f'Confidence: <b><span style="color:{color_map.get(conf, "gray")}">{conf}</span></b> | '
                     f'Food Inflation: <b>{inf:.1f}%</b> | '
                     f'Next Week Forecast: Rs.{forecast:.2f}',
                showarrow=False,
                bordercolor=color_map.get(conf, 'gray'),
                borderwidth=1,
                bgcolor='rgba(255,255,255,0.8)',
                font=dict(size=11)
            )
        
        st.plotly_chart(fig_px, use_container_width=True)

# Help section in sidebar
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Help")
st.sidebar.info(
    "Co-Boost DSS helps food security officers monitor Colombo market price spikes,\n"
    "forecast next week's staple food prices, and provide confidence scores (High/Medium/Low)\n\n"
    "Data Sources: official DCS Weekly PDFs (2019-2026), filename-canonical weeks\n"
    "Price Forecasts: residual-based baseline with inflation-adjusted confidence\n"
    "Alert Severity = predicted price move next week (\u226510% High, \u22655% Medium)\n"
    "Spike Alerts: ML (HistGB/MLP) probability, Balanced ~top10% or \u22650.55 fixed"
)

# Footer
st.markdown('---')
st.markdown(
    '*Co-Boost: Food Security Officers - Colombo Market Price Monitoring | '
    'Baselines: Residual-Based | Median-Based | Seasonal-Based | '
    'Data: DCS Weekly (2019-2026)*'
)