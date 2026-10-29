import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
from baseline import run_baseline_analysis, load_prices, MAIN_DATASET, INFLATION_DATASET

df = load_prices(MAIN_DATASET)
weekly_df = df[df['frequency'] == 'weekly']

# Pre-compute baseline results WITH inflation data
baseline_results = run_baseline_analysis(MAIN_DATASET, INFLATION_DATASET)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Sri Lanka Food Security DSS - Colombo Market Prices", style={"textAlign": "center", "color": "#2c3e50"}),
    
    html.Div([
        html.Label("Select Commodity"),
        dcc.Dropdown(
            id="commodity-dropdown",
            options=[
                {"label": c, "value": c} for c in sorted(weekly_df["commodity"].unique())
            ],
            value="Anamalu",
            clearable=False,
        ),
    ], style={"padding": "20px", "width": "50%", "display": "inline-block"}),
    
    html.Div([
        html.Label("Select Baseline Method"),
        dcc.RadioItems(
            id="baseline-selector",
            options=[
                {"label": "Residual-Based (Trend + Uncertainty)", "value": "residual"},
                {"label": "Median-Based (Revert to Median)", "value": "median"},
                {"label": "Seasonal-Based (Change Pattern)", "value": "seasonal"},
            ],
            value="residual",
            inline=True,
        ),
    ], style={"padding": "20px", "width": "30%", "display": "inline-block", "verticalAlign": "top"}),
    
    html.Div([
        html.Div(id="confidence-display", style={"padding": "10px", "fontSize": "18px", "fontWeight": "bold"}),
    ], style={"padding": "20px"}),
    
    html.Div([
        dcc.Graph(id="price-history-chart"),
    ], style={"padding": "20px"}),
])


@app.callback(
    Output("price-history-chart", "figure"),
    Output("confidence-display", "children"),
    Input("commodity-dropdown", "value"),
    Input("baseline-selector", "value"),
)
def update_chart(commodity, baseline_method):
    comm_df = weekly_df[weekly_df['commodity'] == commodity].sort_values('observation_date')
    
    if len(comm_df) < 2:
        fig = go.Figure()
        fig.add_annotation(text="Not enough data for this commodity", showarrow=False)
        return fig, "No data available"
    
    prices = comm_df['price'].values
    dates = comm_df['observation_date'].values
    
    # Get baseline results for this commodity
    comm_results = baseline_results['baseline_results'].get(commodity, {})
    selected_result = comm_results.get(baseline_method, {})
    
    # Compute confidence display with inflation context
    confidence = selected_result.get('confidence', 'Low')
    color_map = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
    confidence_color = color_map.get(confidence, 'gray')
    
    # Get inflation rate for display
    inflation_rate = selected_result.get('inflation_rate')
    if inflation_rate is not None:
        inflation_display = f"| Current Food Inflation: <b><span style='color:{confidence_color}'>{inflation_rate:.1f}%</span></b>"
    else:
        inflation_display = ""
    
    confidence_display = f"Confidence: <b><span style='color:{confidence_color}'>{confidence}</span></b>{inflation_display} (based on {baseline_method} baseline)"
    
    # Build figure
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines+markers',
        name='Actual Price',
        line=dict(color='royalblue', width=2),
        marker=dict(size=8),
    ))
    
    # Next week forecast based on selected baseline
    if baseline_method == 'residual' and 'next_week_forecast' in selected_result:
        forecast = selected_result['next_week_forecast']
        pi_width = selected_result.get('prediction_interval_width', 0)
        # Add forecast point
        last_date = dates[-1]
        # Parse date and add 7 days
        from datetime import datetime, timedelta
        last_dt = datetime.strptime(last_date, '%Y-%m-%d')
        forecast_date = (last_dt + timedelta(days=7)).strftime('%Y-%m-%d')
        
        fig.add_trace(go.Scatter(
            x=[forecast_date, forecast_date],
            y=[prices[0], forecast],
            mode='lines+markers',
            name='Next Week Forecast',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=10, color='red'),
        ))
        
        # Add prediction interval shading concept
        if pi_width > 0:
            pass  # Interval concept shown via confidence color
    
    elif baseline_method == 'median' and 'next_week_forecast' in selected_result:
        forecast = selected_result['next_week_forecast']
        last_date = dates[-1]
        from datetime import datetime, timedelta
        last_dt = datetime.strptime(last_date, '%Y-%m-%d')
        forecast_date = (last_dt + timedelta(days=7)).strftime('%Y-%m-%d')
        
        fig.add_trace(go.Scatter(
            x=[forecast_date],
            y=[forecast],
            mode='markers',
            name='Forecast (Median Baseline)',
            marker=dict(size=12, color='orange', symbol='triangle-up'),
        ))
    
    elif baseline_method == 'seasonal' and 'next_week_forecast' in selected_result:
        forecast = selected_result['next_week_forecast']
        last_date = dates[-1]
        from datetime import datetime, timedelta
        last_dt = datetime.strptime(last_date, '%Y-%m-%d')
        forecast_date = (last_dt + timedelta(days=7)).strftime('%Y-%m-%d')
        
        fig.add_trace(go.Scatter(
            x=[forecast_date],
            y=[forecast],
            mode='markers',
            name='Forecast (Seasonal Baseline)',
            marker=dict(size=12, color='purple', symbol='diamond'),
        ))
    
    # Layout updates
    fig.update_layout(
        title=f"{commodity} - Price History & Forecast",
        xaxis_title="Date",
        yaxis_title="Price (Rs./kg)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )
    
    # Add annotation with baseline info and inflation
    info_text = f"<b>Baseline: {baseline_method}</b><br>" + confidence_display
    fig.add_annotation(
        x=0.01, y=0.99,
        xref="paper", yref="paper",
        text=info_text,
        showarrow=False,
        bordercolor=confidence_color,
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.8)",
        font=dict(size=11),
    )
    
    return fig, confidence_display


if __name__ == "__main__":
    app.run(debug=True)