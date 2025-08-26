# app_rsi_bybit_screener.py
# Krypto RSI Screener – Bybit Swap only (4h & 15m)
# - RSI: Länge 14, Quelle (O+H+L+C)/4
# - RSI-Spalten: Label + Wert in Klammern (ohne Dezimalstellen)
# - RSI-Cross 4h: bullish/bearish, falls innerhalb der letzten X Stunden (Slider) ein Schwellen- oder Signal-Cross passierte; sonst "Nein"
# - EMA-Checks:
#   • EMA 5/10/20/30 (4h): Oberhalb / Unterhalb / Neutral
#   • EMA 5/10/20/30 (15m): Bullisch / Bärisch / Neutral
#   • EMA-Cross 15m: bullish / bearish, falls EMA(5) die EMAs (10/20/30) innerhalb der letzten X Stunden kreuzte; sonst "Nein"
# - Anzeige: Coin | Preis | RSI 4h | RSI 15m | RSI-Cross 4h | EMA 5/10/20/30 (4h) | EMA 5/10/20/30 (15m) | EMA-Cross 15m
# - Zeilenfärbung: beide TF überkauft = rot, beide TF überverkauft = grün

import os, time, requests
from datetime import datetime, timezone
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import ccxt

# ---------------------------
# Parameter
# ---------------------------
RSI_LEN = 14
RSI_SIG = 14
OB_DEFAULT = 70
OS_DEFAULT = 30
TF_LIST = ["4h", "15m"]
LIMITS = {"4h": 600, "15m": 1000}
TOP_N_DEFAULT = 250

# ---------------------------
# Utils
# ---------------------------
def ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat()

def rsi_series_typical(o, h, l, c, length=14) -> pd.Series:
    tp = (o + h + l + c) / 4.0
    delta = tp.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ewm = pd.Series(gain, index=tp.index).ewm(alpha=1/length, adjust=False).mean()
    loss_ewm = pd.Series(loss, index=tp.index).ewm(alpha=1/length, adjust=False).mean()
    rs = gain_ewm / loss_ewm.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi_state_and_label(v: float, ob: float, os_: float) -> str:
    vv = int(round(v))
    if v >= ob: return f"Überkauft ({vv})"
    if v <= os_: return f"Überverkauft ({vv})"
    return ""

def ema_position_label(df: pd.DataFrame, tf: str) -> str:
    """Staffelung EMAs 5/10/20/30: Oberhalb/Unterhalb (4h) bzw. Bullisch/Bärisch (15m) / Neutral"""
    if df is None or len(df) < 35:
        return "Neutral"
    e5, e10, e20, e30 = ema(df["close"], 5), ema(df["close"], 10), ema(df["close"], 20), ema(df["close"], 30)
    close = df["close"].iloc[-1]
    if close > e5.iloc[-1] and close > e10.iloc[-1] and close > e20.iloc[-1] and close > e30.iloc[-1]:
        return "Oberhalb" if tf == "4h" else "Bullisch"
    if close < e5.iloc[-1] and close < e10.iloc[-1] and close < e20.iloc[-1] and close < e30.iloc[-1]:
        return "Unterhalb" if tf == "4h" else "Bärisch"
    return "Neutral"

def ema5_crosses_all_within_hours(df_15m: pd.DataFrame, hours_back: int) -> str:
    """
    Prüft, ob EMA(5) innerhalb der letzten 'hours_back' Stunden ALLE (10/20/30) gekreuzt hat.
    - bullish: vorher unter allen, danach über allen
    - bearish: vorher über allen, danach unter allen
    - sonst: "Nein"
    """
    if df_15m is None or len(df_15m) < 50:
        return "Nein"

    threshold_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - hours_back * 3600_000

    e5  = ema(df_15m["close"], 5)
    e10 = ema(df_15m["close"], 10)
    e20 = ema(df_15m["close"], 20)
    e30 = ema(df_15m["close"], 30)

    idx = df_15m.index[df_15m["ts"] >= threshold_ms]
    if len(idx) < 2:
        return "Nein"

    def state(i: int) -> str:
        if e5.iloc[i] > e10.iloc[i] and e5.iloc[i] > e20.iloc[i] and e5.iloc[i] > e30.iloc[i]:
            return "above_all"
        if e5.iloc[i] < e10.iloc[i] and e5.iloc[i] < e20.iloc[i] and e5.iloc[i] < e30.iloc[i]:
            return "below_all"
        return "mixed"

    last_signal = "Nein"
    for i in range(idx[0]+1, len(df_15m)):
        prev, now = state(i-1), state(i)
        if prev == "below_all" and now == "above_all":
            last_signal = "bullish"
        elif prev == "above_all" and now == "below_all":
            last_signal = "bearish"
    return last_signal

def rsi_cross_4h_within_hours(df_4h: pd.DataFrame, ob: float, os_: float, sig_len: int, hours_back: int) -> str:
    """
    Prüft, ob binnen 'hours_back' Stunden ein RSI-Cross (4h) passierte:
      - Schwellen-Cross: über OS (bullish) / unter OB (bearish)
      - Signal-Cross: RSI kreuzt seine SMA(sig_len) auf/abwärts
    Gibt 'bullish' / 'bearish' / 'Nein' zurück.
    """
    if df_4h is None or len(df_4h) < max(3, sig_len + 1):
        return "Nein"

    threshold_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - hours_back * 3600_000
    start_idx = 1
    idx = df_4h.index[df_4h["ts"] >= threshold_ms]
    if len(idx) > 0:
        start_idx = max(int(idx[0]), 1)

    rsi = df_4h["rsi"]
    sma = rsi.rolling(sig_len).mean()
    last_signal = "Nein"

    for i in range(start_idx, len(df_4h)):
        # Schwellen-Cross
        if rsi.iloc[i-1] < os_ and rsi.iloc[i] >= os_:
            last_signal = "bullish"
        if rsi.iloc[i-1] > ob and rsi.iloc[i] <= ob:
            last_signal = "bearish"
        # Signal-Cross
        if not np.isnan(sma.iloc[i-1]) and not np.isnan(sma.iloc[i]):
            if rsi.iloc[i-1] < sma.iloc[i-1] and rsi.iloc[i] >= sma.iloc[i]:
                last_signal = "bullish"
            if rsi.iloc[i-1] > sma.iloc[i-1] and rsi.iloc[i] <= sma.iloc[i]:
                last_signal = "bearish"

    return last_signal

# ---------------------------
# Datenquellen
# ---------------------------
load_dotenv()
CMC_KEY = os.getenv("CMC_API_KEY", "").strip()

@st.cache_data(ttl=3600)
def fetch_top_symbols(limit: int = 250) -> List[str]:
    try:
        if CMC_KEY:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            params = {"start": 1, "limit": limit, "convert": "USD"}
            headers = {"X-CMC_PRO_API_KEY": CMC_KEY}
            r = requests.get(url, params=params, headers=headers, timeout=30)
            return [x["symbol"].upper() for x in r.json().get("data", [])][:limit]
        else:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": min(limit, 250), "page": 1}
            r = requests.get(url, params=params, timeout=30)
            data = r.json()
            out, seen = [], set()
            for x in data:
                s = str(x.get("symbol","")).upper()
                if s and s not in seen:
                    seen.add(s); out.append(s)
            return out[:limit]
    except Exception:
        return []

@st.cache_resource
def load_bybit_swap():
    ex = ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    markets = ex.load_markets()
    return ex, markets

def resolve_on_bybit_swap(base: str, markets: Dict) -> Optional[str]:
    for c in (f"{base}/USDT:USDT", f"{base}/USDC:USDC"):
        if c in markets: return c
    return None

def fetch_ohlcv_safe(ex, symbol: str, tf: str, limit: int) -> Optional[pd.DataFrame]:
    try:
        raw = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)
        if not raw or len(raw) < 20: return None
        df = pd.DataFrame(raw, columns=["ts","open","high","low","close","volume"])
        df["iso"] = df["ts"].apply(ts_to_iso)
        return df
    except Exception:
        return None

# ---------------------------
# Screening
# ---------------------------
def analyze_base(base: str, ex, markets, ob: int, os_: int, rsi_cross_4h_hours: int, ema_cross_15m_hours: int):
    symbol = resolve_on_bybit_swap(base, markets)
    if not symbol: return None

    out = {
        "Coin": base, "Preis": None,
        "RSI 4h": "", "RSI 15m": "",
        "RSI-Cross 4h": "Nein",
        "EMA 5/10/20/30 (4h)": "Neutral",
        "EMA 5/10/20/30 (15m)": "Neutral",
        "EMA-Cross 15m": "Nein"
    }

    df_4h, df_15m = None, None

    for tf in TF_LIST:
        df = fetch_ohlcv_safe(ex, symbol, tf, LIMITS[tf])
        if df is None:
            continue
        if out["Preis"] is None:
            out["Preis"] = float(df["close"].iloc[-1])
        # RSI
        df["rsi"] = rsi_series_typical(df["open"], df["high"], df["low"], df["close"], RSI_LEN)
        last = float(df["rsi"].iloc[-1])
        out[f"RSI {tf}"] = rsi_state_and_label(last, ob, os_)
        if tf == "4h":  df_4h = df
        if tf == "15m": df_15m = df
        time.sleep(ex.rateLimit/1000.0)

    if df_4h is not None:
        out["EMA 5/10/20/30 (4h)"] = ema_position_label(df_4h, "4h")
        out["RSI-Cross 4h"] = rsi_cross_4h_within_hours(df_4h, ob, os_, RSI_SIG, rsi_cross_4h_hours)

    if df_15m is not None:
        out["EMA 5/10/20/30 (15m)"] = ema_position_label(df_15m, "15m")
        out["EMA-Cross 15m"] = ema5_crosses_all_within_hours(df_15m, ema_cross_15m_hours)

    # Nur listen, wenn 4h ODER 15m im OB/OS-Bereich
    if out["RSI 4h"] or out["RSI 15m"]:
        return out
    return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Krypto RSI Screener – Bybit Swap only (4h & 15m)", layout="wide")
st.title("Krypto RSI Screener – Bybit Swap only (4h & 15m)")

with st.sidebar:
    top_n = st.slider("Top-Coins (CMC/Gecko)", 50, 250, TOP_N_DEFAULT, step=50)
    ob = st.number_input("Überkauft (OB)", 50, 100, OB_DEFAULT)
    os_ = st.number_input("Überverkauft (OS)", 0, 50, OS_DEFAULT)
    rsi_cross_4h_hours = st.slider("RSI-Cross 4h – Fenster (Stunden)", 2, 24, 8, step=1)
    ema_cross_15m_hours = st.slider("EMA-Cross 15m – Fenster (Stunden)", 1, 24, 8, step=1)
    show_all = st.checkbox("Alle Treffer zeigen (sonst Top 100)", True)
    st.caption("Quelle: Bybit Swap • Universe: Top-Coins")
    run_btn = st.button("Jetzt scannen ✅")

if run_btn:
    ex, markets = load_bybit_swap()
    bases = fetch_top_symbols(top_n)
    rows = []
    prog = st.progress(0.0)
    for i, b in enumerate(bases, 1):
        res = analyze_base(b, ex, markets, ob, os_, rsi_cross_4h_hours, ema_cross_15m_hours)
        if res: rows.append(res)
        prog.progress(i/len(bases))
    if not rows:
        st.warning("Keine Treffer (OB/OS) gefunden."); st.stop()

    df = pd.DataFrame(rows)

    # Confluence (nur für Sortierung)
    same = ((df["RSI 4h"].str.startswith("Überkauft")) & (df["RSI 15m"].str.startswith("Überkauft"))) | \
           ((df["RSI 4h"].str.startswith("Überverkauft")) & (df["RSI 15m"].str.startswith("Überverkauft")))

    df = df.assign(_conf=same.astype(int)).sort_values(by=["_conf"], ascending=False)

    if not show_all and len(df) > 100:
        df = df.head(100)

    df = df[[
        "Coin","Preis","RSI 4h","RSI 15m","RSI-Cross 4h",
        "EMA 5/10/20/30 (4h)","EMA 5/10/20/30 (15m)","EMA-Cross 15m"
    ]]

    def row_bg(row):
        if row["RSI 4h"].startswith("Überkauft") and row["RSI 15m"].startswith("Überkauft"):
            return ['background-color: #ffe6e6']*len(row)
        if row["RSI 4h"].startswith("Überverkauft") and row["RSI 15m"].startswith("Überverkauft"):
            return ['background-color: #e6ffea']*len(row)
        return ['']*len(row)

    def val_color(v):
        if isinstance(v, str) and v.startswith("Überkauft"): return "color: #b00020; font-weight: 700;"
        if isinstance(v, str) and v.startswith("Überverkauft"): return "color: #087f23; font-weight: 700;"
        return ""

    st.dataframe(
        df.style.apply(row_bg, axis=1).applymap(val_color, subset=["RSI 4h","RSI 15m"]),
        use_container_width=True, hide_index=True
    )
