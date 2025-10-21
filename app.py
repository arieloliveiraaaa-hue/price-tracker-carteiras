
import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta, date
import calendar

st.set_page_config(page_title="Price Tracker - Carteiras", layout="wide")

st.title("📈 Price Tracker — Carteiras Recomendadas")
st.caption("Ultra simples: cole os tickers, clique em **Atualizar** e compartilhe o link do app quando estiver publicado.")

st.markdown("**Dicas de ticker:** B3 use sufixo `.SA` (ex.: `VIVT3.SA`, `TOTS3.SA`, `ELET3.SA`, `TAEE11.SA`, `EGIE3.SA`, `CPLE6.SA`). EUA use o ticker puro (ex.: `AAPL`, `MSFT`).")

# Sidebar: input method
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

refresh = st.sidebar.button("🔄 Atualizar")

# Utility functions
def first_trading_close(ticker, start_date):
    # Download from start_date to today to get first available close >= start_date
    df = yf.download(ticker, start=start_date, end=date.today() + timedelta(days=1), progress=False, auto_adjust=False)
    if df.empty or "Close" not in df.columns:
        return None
    first_close = df["Close"].iloc[0]
    last_close = df["Close"].iloc[-1]
    return first_close, last_close

def get_current_and_prev_close(ticker):
    df = yf.download(ticker, period="10d", interval="1d", progress=False, auto_adjust=False)
    if df.empty or "Close" not in df.columns:
        return None, None
    # Last two closes (accounting for weekends/holidays)
    closes = df["Close"].dropna()
    curr = closes.iloc[-1]
    prev = closes.iloc[-2] if len(closes) >= 2 else None
    return curr, prev

def safe_pct(curr, base):
    if curr is None or base is None or base == 0:
        return None
    return (curr / base - 1.0) * 100.0

def month_start(d: date):
    return date(d.year, d.month, 1)

def ytd_start(d: date):
    return date(d.year, 1, 1)

def compute_row(ticker):
    today = date.today()

    # Current & daily
    curr, prev = get_current_and_prev_close(ticker)

    # MTD
    mstart = month_start(today)
    mtd_first, mtd_last = first_trading_close(ticker, mstart)
    # If mtd_last is None, fallback to curr
    if mtd_last is None and curr is not None:
        mtd_last = curr

    # YTD
    ystart = ytd_start(today)
    ytd_first, ytd_last = first_trading_close(ticker, ystart)
    if ytd_last is None and curr is not None:
        ytd_last = curr

    return {
        "Ticker": ticker,
        "Preço atual": curr,
        "Dia (%)": safe_pct(curr, prev),
        "MTD (%)": safe_pct(mtd_last, mtd_first),
        "YTD (%)": safe_pct(ytd_last, ytd_first),
    }

if refresh and tickers:
    rows = []
    for t in tickers:
        try:
            rows.append(compute_row(t))
        except Exception as e:
            rows.append({"Ticker": t, "Preço atual": None, "Dia (%)": None, "MTD (%)": None, "YTD (%)": None})
    df = pd.DataFrame(rows)
    # Format nicely
    fmt_df = df.copy()
    fmt_df["Preço atual"] = fmt_df["Preço atual"].map(lambda x: f"{x:,.2f}" if pd.notnull(x) else "—")
    for col in ["Dia (%)", "MTD (%)", "YTD (%)"]:
        fmt_df[col] = fmt_df[col].map(lambda x: f"{x:,.2f}%" if pd.notnull(x) else "—")
    st.dataframe(fmt_df, use_container_width=True)
    st.download_button("⬇️ Baixar CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="price_tracker.csv", mime="text/csv")
else:
    st.info("Adicione tickers e clique em **Atualizar** para ver os dados.")
    st.code(default_list, language="text")
