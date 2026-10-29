import calendar

import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd as MonthEndOffset
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta

MAIN_DATASET = "outputs/official_prices_wrp.csv"
INFLATION_DATASET = "Dataset/food_inflation.csv"


def _normalize_name(name) -> str:
    """Canonical commodity name: unify dash glyphs (PDFs mix U+2010 '‐' with
    ASCII '-'), collapse whitespace. Prevents split series like
    'Beans - Butter' vs 'Beans ‐ Butter'."""
    s = str(name)
    for dash in ("\u2010", "\u2011", "\u2013", "\u2014"):
        s = s.replace(dash, "-")
    return " ".join(s.split()).strip()


def _glued_price_name(name) -> bool:
    """True if a printed price got glued into the commodity name during
    extraction (e.g. 'Salt 1Kg.Pkt. 60.00'); those rows are corrupt."""
    import re
    return bool(re.search(r"\d+Kg\.[A-Za-z().]*\s+\d+\.\d", str(name)))


def _exclusion_mask(commodity: pd.Series) -> pd.Series:
    """Return True for commodity names excluded from analytics (see
    outputs/commodity_exclusions.csv - e.g. non-food betel leaves/arecanuts)."""
    try:
        fx = pd.read_csv("outputs/commodity_exclusions.csv")
    except FileNotFoundError:
        return pd.Series(False, index=commodity.index)
    drop = pd.Series(False, index=commodity.index)
    names = commodity.fillna("").astype(str)
    for pat in fx["pattern"].astype(str):
        root = pat.rstrip("*").replace(" ", "")
        drop |= names.str.replace(" ", "").str.contains(root, case=False, regex=True)
    return drop


def load_prices(csv_path: str) -> pd.DataFrame:
    """Load prices in canonical form for the baseline pipeline.

    Supports the official wrp schema (year/month/week_of_month, price_rs,
    extracted by Scripts/parse_wrp.py) and the legacy canonical schema
    (observation_date, price, frequency, source).
    See outputs/DATASET_NOTES.md for dataset decisions, including the
    exclusion of non-food commodities (betel leaves, arecanuts).
    """
    df = pd.read_csv(csv_path)
    if "price_rs" in df.columns:
        day = [min(int(w) * 7, calendar.monthrange(int(y), int(m))[1])
               for y, m, w in zip(df["year"], df["month"], df["week_of_month"])]
        out = pd.DataFrame({
            "observation_date": pd.to_datetime(
                dict(year=df["year"], month=df["month"], day=day)).dt.strftime("%Y-%m-%d"),
            "frequency": "weekly",
            "source": "DCS",
            "commodity": df["commodity"],
            "parent": df["parent"],
            "child": df["child"],
            "unit": df["unit"],
            "price": df["price_rs"],
            "source_document": df["source_file"],
        })
    else:
        out = df
    out = out.copy()
    for col in ("commodity", "parent", "child"):
        if col in out.columns:
            out[col] = out[col].map(_normalize_name)
    out = out[~out["commodity"].map(_glued_price_name)]
    out = out[~_exclusion_mask(out["commodity"])]
    out = out.drop_duplicates(subset=["source_document", "commodity"], keep="first") \
        if "source_document" in out.columns else out.drop_duplicates(
            subset=["observation_date", "commodity"], keep="first")
    return out.reset_index(drop=True)


class PriceBaseline:
    """Baseline models for Colombo market price time series with confidence scoring."""

    def __init__(self, df: pd.DataFrame, inflation_df: pd.DataFrame = None):
        self.df = df
        self.inflation_df = inflation_df
        if inflation_df is not None:
            self.inflation_df['Date'] = pd.to_datetime(self.inflation_df['Date'], format='%d/%m/%Y')
            self.inflation_df = self.inflation_df.sort_values('Date')

    def _get_inflation_for_date(self, date) -> float:
        """Get food inflation rate for a specific date (monthly)."""
        if self.inflation_df is None:
            return None
        target_date = pd.to_datetime(date, format='%Y-%m-%d')
        # Find the closest month (end of month)
        month_end = target_date.replace(day=1) + MonthEndOffset(1) - timedelta(days=1)
        month_key = month_end.strftime('%d/%m/%Y')
        match = self.inflation_df[self.inflation_df['Date'].astype(str) == month_key]
        if len(match) > 0:
            return float(match['Value'].iloc[0])
        # Try year-only match for early data
        match = self.inflation_df[self.inflation_df['Date'].dt.year == target_date.year]
        if len(match) > 0:
            # Return the most recent month available in that year
            return float(match['Value'].iloc[-1])
        return None

    def _compute_confidence(self, cv: float, inflation_rate: float = None) -> tuple:
        """Compute confidence score based on coefficient of variation, adjusted by inflation rate.
        
        Args:
            cv: Coefficient of variation (std/mean * 100)
            inflation_rate: Current food inflation rate percentage, or None if unavailable
        """
        if cv < 0:
            return 'Low', cv
        
        # Adjust confidence thresholds based on national food inflation rate
        # Crisis period (inflation > 20%): wider thresholds due to overall market instability
        # Elevated inflation (10-20%): moderate adjustments
        # Stable period (< 10%): standard thresholds
        if inflation_rate is not None:
            if inflation_rate > 20:
                # Crisis: High < 15%, Medium 15-30%, Low > 30%
                if cv < 15:
                    confidence = 'High'
                    color = 'green'
                elif cv < 30:
                    confidence = 'Medium'
                    color = 'orange'
                else:
                    confidence = 'Low'
                    color = 'red'
            elif inflation_rate >= 10:
                # Elevated: High < 12%, Medium 12-25%, Low > 25%
                if cv < 12:
                    confidence = 'High'
                    color = 'green'
                elif cv < 25:
                    confidence = 'Medium'
                    color = 'orange'
                else:
                    confidence = 'Low'
                    color = 'red'
            else:
                # Stable: Standard thresholds High < 10%, Medium 10-20%, Low > 20%
                if cv < 10:
                    confidence = 'High'
                    color = 'green'
                elif cv < 20:
                    confidence = 'Medium'
                    color = 'orange'
                else:
                    confidence = 'Low'
                    color = 'red'
        else:
            # No inflation data: use standard thresholds
            if cv < 10:
                confidence = 'High'
                color = 'green'
            elif cv < 20:
                confidence = 'Medium'
                color = 'orange'
            else:
                confidence = 'Low'
                color = 'red'
        
        return confidence, cv

    def residual_based(self, commodity: str) -> dict:
        """Residual-based baseline: use last model residuals to estimate prediction interval."""
        comm_df = self.df[self.df['commodity'] == commodity].sort_values('observation_date')
        if len(comm_df) < 3:
            return None

        prices = comm_df['price'].values
        dates = pd.to_datetime(comm_df['observation_date'])

        # Get inflation rate for the last observation date
        inflation_rate = self._get_inflation_for_date(dates.iloc[-1])

        # Fit linear trend
        X = (dates - pd.Timestamp('2022-01-01')).dt.days.values.reshape(-1, 1)
        y = prices
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Compute residuals
        residuals = y - y_pred
        residual_std = np.std(residuals)

        # Last observation forecast
        last_price = prices[-1]
        # Forecast next week: trend + residual-based uncertainty
        X_next = (pd.Timestamp(comm_df['observation_date'].iloc[-1]) + timedelta(days=7) - pd.Timestamp('2022-01-01')).days
        next_price_trend = model.predict([[X_next]])[0]
        next_price = next_price_trend

        # Confidence based on residual standard deviation
        if residual_std > 0:
            # Prediction interval width
            pi_width = 1.96 * residual_std  # 95% PI
            # Confidence: narrower interval = higher confidence
            cv = (residual_std / last_price) * 100
            # Use inflation-adjusted confidence scoring
            confidence, adj_cv = self._compute_confidence(cv, inflation_rate)
            # Map confidence to color
            color_map = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
            color = color_map.get(confidence, 'gray')
        else:
            pi_width = 0
            confidence = 'High'
            color = 'green'

        return {
            'commodity': commodity,
            'method': 'residual_based',
            'last_price': float(last_price),
            'trend_slope': float(model.coef_[0]),
            'next_week_forecast': float(next_price),
            'prediction_interval_width': float(pi_width),
            'confidence': confidence,
            'confidence_color': color,
            'residual_std': float(residual_std),
            'historical_rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
            'inflation_rate': inflation_rate,
        }

    def median_based(self, commodity: str) -> dict:
        """Median-based baseline: use historical median as forecast reference."""
        comm_df = self.df[self.df['commodity'] == commodity].sort_values('observation_date')
        if len(comm_df) < 3:
            return None

        prices = comm_df['price'].values
        median_price = np.median(prices)

        # Get inflation rate for this commodity's last observation date
        inflation_rate = self._get_inflation_for_date(comm_df['observation_date'].iloc[-1])

        # Use last price as forecast, compare to median
        last_price = prices[-1]
        change_from_median = last_price - median_price

# Confidence based on how far last price is from median
        mad = np.median(np.abs(prices - median_price))  # median absolute deviation
        if mad > 0:
            cv = (mad / median_price) * 100
        else:
            cv = 0
        
        # Use inflation-adjusted confidence scoring
        confidence, adj_cv = self._compute_confidence(cv, inflation_rate)
        # Map confidence to color
        color_map = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
        color = color_map.get(confidence, 'gray')
        
        # Simple forecast: revert to median
        forecast = median_price

        # Get inflation rate
        inflation_rate = self._get_inflation_for_date(comm_df['observation_date'].iloc[-1])

        return {
            'commodity': commodity,
            'method': 'median_based',
            'last_price': float(last_price),
            'median_price': float(median_price),
            'change_from_median': float(change_from_median),
            'next_week_forecast': float(forecast),
            'confidence': confidence,
            'confidence_color': color,
            'mad': float(mad),
            'inflation_rate': inflation_rate,
        }

    def seasonal_based(self, commodity: str) -> dict:
        """Seasonal-based baseline: use historical seasonality patterns for forecast."""
        comm_df = self.df[self.df['commodity'] == commodity].sort_values('observation_date')
        if len(comm_df) < 4:
            return None

        prices = comm_df['price'].values

        # Get inflation rate for this commodity's last observation date
        inflation_rate = self._get_inflation_for_date(comm_df['observation_date'].iloc[-1])

        # Calculate week-over-week changes
        changes = np.diff(prices)

        # If we have enough data, use average seasonal pattern
        if len(changes) >= 3:
            # Average of recent changes as forecast adjustment
            avg_change = np.mean(changes[-3:])
            last_price = prices[-1]
            forecast = last_price + avg_change
        else:
            # Not enough data for seasonal, use simple mean reversion
            # Use the first price as "seasonal baseline" and revert
            forecast = np.mean(prices)

# Confidence based on number of data points and consistency
        price_std = np.std(prices)
        cv = (price_std / np.mean(prices)) * 100 if np.mean(prices) > 0 else 100
        
        # Use inflation-adjusted confidence scoring
        confidence, adj_cv = self._compute_confidence(cv, inflation_rate)
        # Map confidence to color
        color_map = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}
        color = color_map.get(confidence, 'gray')
        
        # Get inflation rate
        inflation_rate = self._get_inflation_for_date(comm_df['observation_date'].iloc[-1])

        return {
            'commodity': commodity,
            'method': 'seasonal_based',
            'last_price': float(last_price),
            'next_week_forecast': float(forecast),
            'confidence': confidence,
            'confidence_color': color,
            'avg_recent_change': float(np.mean(changes[-3:])) if len(changes) >= 3 else 0,
            'inflation_rate': inflation_rate,
        }

    def run_all_baselines(self, commodity: str) -> dict:
        """Run all three baseline methods for a commodity."""
        return {
            'residual_based': self.residual_based(commodity),
            'median_based': self.median_based(commodity),
            'seasonal_based': self.seasonal_based(commodity),
        }


def run_baseline_analysis(csv_path: str, inflation_path: str = None) -> dict:
    """Run complete baseline analysis on extracted prices."""
    df = load_prices(csv_path)
    inflation_df = None
    if inflation_path:
        inflation_df = pd.read_csv(inflation_path)
    ba = PriceBaseline(df, inflation_df)

    # Get commodities with enough data
    commodities = df['commodity'].unique()
    results = {}

    for comm in commodities:
        baseline_results = ba.run_all_baselines(comm)
        # Filter out None results (not enough data)
        valid_results = {k: v for k, v in baseline_results.items() if v is not None}
        if valid_results:
            results[comm] = valid_results

    return {
        'data_info': {
            'total_rows': len(df),
            'date_range': f"{df['observation_date'].min()} to {df['observation_date'].max()}",
            'commodities': df['commodity'].nunique(),
            'sources': df['source'].unique().tolist(),
            'inflation_data': inflation_path is not None,
        },
        'baseline_results': results,
    }