"""Streamlit home page for Gooberberg Terminal."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Gooberberg Terminal", page_icon="📈", layout="wide")

st.title("Gooberberg Terminal")
st.caption("Quant research, data engineering, and model operations workspace.")

st.markdown(
    """
    Use the recommended quick-start order:

    1. Copy `.env.example` to `.env` and fill local values.
    2. Install dependencies with `uv sync --dev`.
    3. Start Redis and the app services with `docker compose up --build`.
    4. Open the API at <http://localhost:8000/docs>.
    5. Open the Streamlit UI at <http://localhost:8501>.
    """
)
