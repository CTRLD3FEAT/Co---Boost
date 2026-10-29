"""Data layer for the Food Price Early Warning dashboard.

Source: official DCS weekly retail price reports (Colombo district),
extracted into outputs/official_prices_wrp.csv by Scripts/parse_wrp.py.
Cleaning/normalization/exclusions happen in baseline.load_prices - never
duplicated here. No imputation: missing prices are missing.
"""
import streamlit as st
import pandas as pd

WEEKLY_CSV = "outputs/official_prices_wrp.csv"

_CACHED_LOADER = None


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def load_weekly(path: str = WEEKLY_CSV) -> pd.DataFrame:
    from baseline import load_prices, MAIN_DATASET
    d = load_prices(path if path else MAIN_DATASET)
    d = d.rename(columns={"observation_date": "week"})
    d["week"] = pd.to_datetime(d["week"], errors="coerce")
    d = d[d["week"].notna() & (d["price"] > 0)]
    return d


@st.cache_data(show_spinner=False)
def latest_week(dy: pd.DataFrame):
    return dy["week"].max()


@st.cache_data(show_spinner=False)
def items(dy: pd.DataFrame) -> list[str]:
    return sorted(dy["commodity"].dropna().unique())


@st.cache_data(show_spinner=False)
def categories(dy: pd.DataFrame) -> pd.Series:
    cat = dy.groupby("commodity")["parent"].agg(
        lambda s: s.mode().iat[0] if len(s) else "")
    return cat


@st.cache_data(show_spinner=False)
def source_files(dy: pd.DataFrame) -> pd.Series:
    """Most recent source PDF per commodity (for the next-step box)."""
    last = dy.sort_values("week").groupby("commodity").tail(1)
    return last.set_index("commodity")["source_document"]
