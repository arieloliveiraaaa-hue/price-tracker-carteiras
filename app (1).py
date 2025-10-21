
import streamlit as st
import pandas as pd
from datetime import date, timedelta
import numpy as np

# Primary provider
import yfinance as yf
# Fallback provider
from pandas_datareader.stooq import StooqDailyReader

st.set_page_config(page_title="Price Tracker - Carteiras (Robusto)", layout="wide")
st.title("📈 Price Tracker — Carteiras Recomendadas (com fallback)")
st.caption("Se o Yahoo/YFinance falhar, o app usa o Stooq automaticamente.")

st.markdown("**B3:** use `.SA` (ex.: `VIVT3.SA`, `TOTS3.SA`, `ELET3.SA`, `TAEE11.SA`, `EGIE3.SA`, `CPLE6.SA`, `IVVB11.SA`).  **EUA:** `AAPL`, `MSFT`.")

# ---- Helpers ----
def normalize_for_stooq(ticker: str) -> str:
    # Stooq usa tickers da B3 sem sufixo; EUA em maiúsculo funciona.
    # Ex.: VIVT3.SA -> VIVT3, IVVB11.SA -> IVVB11
    t = ticker.strip().upper()
    if t.endswith(".SA"):
        t = t.replace(".SA", "")
    return t

def fetch_history_yf(ticker: str, start: date) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            start=start,
            end=date.today() + timedelta(days=1),
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,  # evita race conditions em alguns hosts
            timeout=15,
        )
        if df is None or df.empty:
            return None
        # Padroniza colunas
        out = df.rename(columns={"Close": "close"})
        out = out[["close"]].dropna()
        return out
    except Exception:
        return None

def fetch_history_stooq(ticker: str, start: date) -> pd.DataFrame | None:
    try:
        stooq_sym = normalize_for_stooq(ticker)
        rdr = StooqDailyReader(symbols=stooq_sym, start=start, end=date.today() + timedelta(days=1))
        df = rdr.read()
        if df is None or df.empty:
            return None
        # Se o Stooq devolver MultiIndex (símbolo, data), flatten
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel(0)
        df = df.sort_index()
        out = df.rename(columns={"Close": "close"})
        out = out[["close"]].dropna()
        return out
    except Exception:
        return None

def get_history(ticker: str, start: date) -> pd.DataFrame | None:
    # Tenta Yahoo -> fallback Stooq
    df = fetch_history_yf(ticker, start)
    if df is not None and not df.empty:
        return df
    return fetch_history_stooq(ticker, start)

def pct_change(curr: float | None, base: float | None):
    if curr is None or base is None or base == 0:
        return None
    return (curr / base - 1.0) * 100.0

def first_value_on_or_after(df: pd.DataFrame, ref_date: date):
    if df is None or df.empty:
        return None
    # encontra o primeiro índice >= ref_date
    idx = df.index.searchsorted(pd.to_datetime(ref_date))
    if idx >= len(df.index):
        return None
    return float(df.iloc[idx]["close"])

def last_two(df: pd.DataFrame):
    if df is None or df.empty:
        return None, None
    closes = df["close"].dropna()
    if closes.empty:
        return None, None
    curr = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) >= 2 else None
    return curr, prev

def compute_row(ticker: str):
    today = date.today()
    start = date(today.year, 1, 1)
    df = get_history(ticker, start=start)

    curr, prev = last_two(df)
    mtd_first = first_value_on_or_after(df, date(today.year, today.month, 1))
    ytd_first = first_value_on_or_after(df, date(today.year, 1, 1))

    return {
        "Ticker": ticker,
        "Preço atual": curr,
        "Dia (%)": pct_change(curr, prev),
        "MTD (%)": pct_change(curr, mtd_first),
        "YTD (%)": pct_change(curr, ytd_first),
        "Fonte": "Yahoo" if fetch_history_yf(ticker, start) is not None else "Stooq"
    }

# ---- UI ----
st.sidebar.header("Entrada de Tickers")
method = st.sidebar.radio("Como quer fornecer os tickers?", ["Colar manualmente", "Upload de CSV"])

default_list = "VIVT3.SA\nTOTS3.SA\nELET3.SA\nTAEE11.SA\nEGIE3.SA\nCPLE6.SA\nIVVB11.SA\nAAPL\nMSFT"

tickers = []
if method == "Colar manualmente":
    raw = st.sidebar.text_area("Cole um por linha", value=default_list, height=200)
    tickers = [t.strip() for t in raw.splitlines() if t.strip()]
else:
    up = st.sidebar.file_uploader("Envie um CSV com a coluna 'ticker'", type=["csv"])
    if up:
        try:
            df_up = pd.read_csv(up)
            if "ticker" in df_up.columns:
                tickers = df_up["ticker"].dropna().astype(str).str.strip().tolist()
            else:
                st.sidebar.error("CSV precisa ter a coluna 'ticker'.")
        except Exception as e:
            st.sidebar.error(f"Erro ao ler CSV: {e}")

if st.sidebar.button("🔄 Atualizar"):
    rows = []
    for t in tickers:
        try:
            rows.append(compute_row(t))
        except Exception as e:
            rows.append({"Ticker": t, "Preço atual": None, "Dia (%)": None, "MTD (%)": None, "YTD (%)": None, "Fonte": "—"})
    df = pd.DataFrame(rows)

    # Formatação
    fmt = df.copy()
    fmt["Preço atual"] = fmt["Preço atual"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "—")
    for col in ["Dia (%)", "MTD (%)", "YTD (%)"]:
        fmt[col] = fmt[col].map(lambda x: f"{x:,.2f}%" if pd.notnull(x) else "—")

    st.dataframe(fmt, use_container_width=True)
    st.download_button("⬇️ Baixar CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="price_tracker.csv", mime="text/csv")
else:
    st.info("Adicione tickers e clique em **Atualizar** para ver os dados.")
    st.code(default_list, language="text")
