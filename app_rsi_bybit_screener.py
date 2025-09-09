# app_rsi_bybit_screener.py — Bybit v5 REST, schnelle Symbol-Liste & Cache

import os, time, random, requests
from datetime import datetime, timezone
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

RSI_LEN = 14
RSI_SIG = 14
OB_DEFAULT = 70
OS_DEFAULT = 30
TOP_N_DEFAULT = 250

INTERVAL_MAP = {"4h": "240", "15m": "15"}
LIMITS = {"4h": 600, "15m": 1000}
BYBIT_BASE = "https://api.bybit.com"

# ---------- math utils ----------
def ts_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms/1000, tz=timezone.utc).isoformat()

def rsi_series_typical(o, h, l, c, length=14) -> pd.Series:
    tp = (o + h + l + c) / 4.0
    delta = tp.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ewm = pd.Series(gain, index=tp.index).ewm(alpha=1/length, adjust=False).mean()
    loss_ewm = pd.Series(loss, index=tp.index).ewm(alpha=1/length, adjust=False).mean()
    rs = gain_ewm / pd.Series(loss_ewm).replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def rsi_state_and_label(v: float, ob: float, os_: float) -> str:
    vv = int(round(v))
    if v >= ob: return f"Überkauft ({vv})"
    if v <= os_: return f"Überverkauft ({vv})"
    return ""

def ema_position_label(df: pd.DataFrame, tf: str) -> str:
    if df is None or len(df) < 35: return "Neutral"
    e5, e10, e20, e30 = ema(df["close"], 5), ema(df["close"], 10), ema(df["close"], 20), ema(df["close"], 30)
    close = df["close"].iloc[-1]
    if close > e5.iloc[-1] and close > e10.iloc[-1] and close > e20.iloc[-1] and close > e30.iloc[-1]:
        return "Oberhalb" if tf == "4h" else "Bullisch"
    if close < e5.iloc[-1] and close < e10.iloc[-1] and close < e20.iloc[-1] and close < e30.iloc[-1]:
        return "Unterhalb" if tf == "4h" else "Bärisch"
    return "Neutral"

def ema5_crosses_all_within_hours(df_15m: pd.DataFrame, hours_back: int) -> str:
    if df_15m is None or len(df_15m) < 50: return "Nein"
    threshold_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - hours_back * 3600_000
    e5, e10, e20, e30 = ema(df_15m["close"],5), ema(df_15m["close"],10), ema(df_15m["close"],20), ema(df_15m["close"],30)
    idx = df_15m.index[df_15m["ts"] >= threshold_ms]
    if len(idx) < 2: return "Nein"
    def state(i):
        if e5.iloc[i] > e10.iloc[i] and e5.iloc[i] > e20.iloc[i] and e5.iloc[i] > e30.iloc[i]: return "above_all"
        if e5.iloc[i] < e10.iloc[i] and e5.iloc[i] < e20.iloc[i] and e5.iloc[i] < e30.iloc[i]: return "below_all"
        return "mixed"
    last = "Nein"
    for i in range(idx[0]+1, len(df_15m)):
        prev, now = state(i-1), state(i)
        if prev == "below_all" and now == "above_all": last = "bullish"
        elif prev == "above_all" and now == "below_all": last = "bearish"
    return last

def rsi_cross_4h_within_hours(df_4h: pd.DataFrame, ob: float, os_: float, sig_len: int, hours_back: int) -> str:
    if df_4h is None or len(df_4h) < max(3, sig_len+1): return "Nein"
    threshold_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - hours_back * 3600_000
    idx = df_4h.index[df_4h["ts"] >= threshold_ms]
    start = max(int(idx[0]) if len(idx) else 1, 1)
    rsi = df_4h["rsi"]; sma = rsi.rolling(sig_len).mean()
    last = "Nein"
    for i in range(start, len(df_4h)):
        if rsi.iloc[i-1] < os_ and rsi.iloc[i] >= os_: last = "bullish"
        if rsi.iloc[i-1] > ob and rsi.iloc[i] <= ob:   last = "bearish"
        if not np.isnan(sma.iloc[i-1]) and not np.isnan(sma.iloc[i]):
            if rsi.iloc[i-1] < sma.iloc[i-1] and rsi.iloc[i] >= sma.iloc[i]: last = "bullish"
            if rsi.iloc[i-1] > sma.iloc[i-1] and rsi.iloc[i] <= sma.iloc[i]: last = "bearish"
    return last

# ---------- config / secrets ----------
load_dotenv()
CMC_KEY = os.getenv("CMC_API_KEY", "") or st.secrets.get("CMC_API_KEY", "")
TG_BOT_TOKEN_DEFAULT = os.getenv("TELEGRAM_BOT_TOKEN", "") or st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID_DEFAULT   = os.getenv("TELEGRAM_CHAT_ID", "")   or st.secrets.get("TELEGRAM_CHAT_ID", "")

@st.cache_resource(show_spinner=False)
def get_http_session():
    s = requests.Session()
    s.headers.update({"User-Agent": "krypto-rsi-screener/1.1"})
    return s

# ---------- Bybit REST helpers ----------
@st.cache_data(ttl=3600, show_spinner=False)
def bybit_linear_symbol_map() -> Dict[str, str]:
    """
    Holt EINMAL alle linearen Perps von Bybit (USDT/USDC) und baut:
      map[BASE] = SYMBOL (z.B. BTC -> BTCUSDT)
    Damit sparen wir tausende Probier-Calls.
    """
    session = get_http_session()
    params = {"category": "linear", "limit": 1000}
    delay = 0.7
    for _ in range(10):
        try:
            r = session.get(f"{BYBIT_BASE}/v5/market/instruments-info", params=params, timeout=15)
            js = r.json()
            if str(js.get("retCode")) == "0":
                res = js.get("result", {})
                lst = res.get("list", []) or []
                mp = {}
                for x in lst:
                    sym = str(x.get("symbol","")).upper()
                    # symbol endet meist auf USDT/USDC → BASE ist Prefix
                    base = sym.replace("USDT","").replace("USDC","")
                    if base and (sym.endswith("USDT") or sym.endswith("USDC")):
                        mp[base] = sym
                return mp
        except requests.RequestException:
            time.sleep(delay + random.random()*0.3); delay = min(delay*1.8, 6.0)
    return {}

def bybit_get_kline(symbol: str, tf: str, limit: int = 500, max_tries: int = 8) -> Optional[pd.DataFrame]:
    session = get_http_session()
    params = {"category":"linear","symbol":symbol,"interval":INTERVAL_MAP[tf],"limit":min(limit,1000)}
    delay = 0.5
    for _ in range(max_tries):
        try:
            r = session.get(f"{BYBIT_BASE}/v5/market/kline", params=params, timeout=15)
            js = r.json()
            if str(js.get("retCode")) == "0":
                lst = js.get("result", {}).get("list", [])
                if not lst: return None
                rows = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in reversed(lst)]
                df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
                df["iso"] = df["ts"].apply(ts_to_iso)
                return df
            return None
        except requests.RequestException:
            time.sleep(delay + random.random()*0.2); delay = min(delay*1.8, 5.0)
    return None

# ---------- Universe ----------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_top_symbols(limit: int = 250) -> List[str]:
    try:
        if CMC_KEY:
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
            params = {"start":1,"limit":limit,"convert":"USD"}
            headers = {"X-CMC_PRO_API_KEY": CMC_KEY}
            js = requests.get(url, params=params, headers=headers, timeout=30).json()
            return [x["symbol"].upper() for x in js.get("data", [])][:limit]
        else:
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {"vs_currency":"usd","order":"market_cap_desc","per_page":min(limit,250),"page":1}
            data = requests.get(url, params=params, timeout=30).json()
            out, seen = [], set()
            for x in data:
                s = str(x.get("symbol","")).upper()
                if s and s not in seen: seen.add(s); out.append(s)
            return out[:limit]
    except Exception:
        return []

# ---------- Alerts ----------
def send_telegram(bot_token: str, chat_id: str, text: str):
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
    except Exception:
        pass

# ---------- Screening ----------
def analyze_base(base: str, sym_map: Dict[str,str], ob: int, os_: int, rsi_h: int, ema_h: int) -> Optional[Dict]:
    sym = sym_map.get(base)
    if not sym: 
        return None

    out = {"Coin": base, "Preis": None,
           "RSI 4h":"", "RSI 15m":"",
           "RSI-Cross 4h":"Nein",
           "EMA 5/10/20/30 (4h)":"Neutral",
           "EMA 5/10/20/30 (15m)":"Neutral",
           "EMA-Cross 15m":"Nein"}

    df4 = bybit_get_kline(sym, "4h", LIMITS["4h"])
    if df4 is None: return None
    out["Preis"] = float(df4["close"].iloc[-1])
    df4["rsi"] = rsi_series_typical(df4["open"], df4["high"], df4["low"], df4["close"], RSI_LEN)
    out["RSI 4h"] = rsi_state_and_label(float(df4["rsi"].iloc[-1]), ob, os_)
    out["EMA 5/10/20/30 (4h)"] = ema_position_label(df4, "4h")
    out["RSI-Cross 4h"] = rsi_cross_4h_within_hours(df4, ob, os_, RSI_SIG, rsi_h)

    df15 = bybit_get_kline(sym, "15m", LIMITS["15m"])
    if df15 is not None:
        df15["rsi"] = rsi_series_typical(df15["open"], df15["high"], df15["low"], df15["close"], RSI_LEN)
        out["RSI 15m"] = rsi_state_and_label(float(df15["rsi"].iloc[-1]), ob, os_)
        out["EMA 5/10/20/30 (15m)"] = ema_position_label(df15, "15m")
        out["EMA-Cross 15m"] = ema5_crosses_all_within_hours(df15, ema_h)

    if out["RSI 4h"] or out["RSI 15m"]:
        return out
    return None

# ---------- UI ----------
st.set_page_config(page_title="Krypto RSI Screener – Bybit Swap only (4h & 15m)", layout="wide")
st.title("Krypto RSI Screener – Bybit Swap only (4h & 15m)")
st.caption("RSI auf (O+H+L+C)/4, Länge 14, SMA(14)-Signal. Zeilenfärbung: beide TF überkauft = rot, beide TF überverkauft = grün.")

with st.sidebar:
    st.subheader("Filter")
    top_n = st.slider("Top-Coins (CMC/Gecko)", 50, 250, TOP_N_DEFAULT, step=50)
    extra_symbols = st.text_input("Extra Coins (Whitelist, Komma-getrennt)", "").upper()
    ob = st.number_input("Überkauft (OB)", 50, 100, OB_DEFAULT)
    os_ = st.number_input("Überverkauft (OS)", 0, 50, OS_DEFAULT)
    rsi_cross_4h_hours = st.slider("RSI-Cross 4h – Fenster (Stunden)", 2, 24, 8, step=1)
    ema_cross_15m_hours = st.slider("EMA-Cross 15m – Fenster (Stunden)", 1, 24, 8, step=1)
    show_all = st.checkbox("Alle Treffer zeigen (sonst Top 100)", True)
    st.caption("Quelle: Bybit Swap • Universe: Top-Coins + Whitelist")

    st.markdown("---")
    st.subheader("Telegram-Alerts (optional)")
    default_info = []
    if TG_BOT_TOKEN_DEFAULT: default_info.append("BotToken via Secrets")
    if TG_CHAT_ID_DEFAULT:   default_info.append("ChatID via Secrets")
    if default_info: st.info("Vorkonfiguriert: " + ", ".join(default_info))
    tg_enable = st.checkbox("Alerts senden", False)
    tg_token = st.text_input("Bot Token", type="password", value=TG_BOT_TOKEN_DEFAULT if tg_enable else "")
    tg_chat = st.text_input("Chat ID / @Channel", value=TG_CHAT_ID_DEFAULT if tg_enable else "")

    run_btn = st.button("Jetzt scannen ✅")

if run_btn:
    # 1) Universum bauen
    bases = fetch_top_symbols(top_n)
    if extra_symbols:
        for sym in [s.strip() for s in extra_symbols.split(",") if s.strip()]:
            if sym not in bases: bases.append(sym)

    # 2) Einmalige Symbol-Map von Bybit (schnell & gecacht)
    sym_map = bybit_linear_symbol_map()
    if not sym_map:
        st.error("Konnte Bybit-Instrumente nicht laden. Bitte später erneut versuchen.")
        st.stop()

    # nur Coins scannen, die Bybit wirklich hat
    candidates = [b for b in bases if b in sym_map]
    if not candidates:
        st.warning("Keine der ausgewählten Top-Coins sind als Bybit-Perps verfügbar.")
        st.stop()

    rows, alert_msgs = [], []
    prog = st.progress(0.0)
    for i, b in enumerate(candidates, 1):
        res = analyze_base(b, sym_map, ob, os_, rsi_cross_4h_hours, ema_cross_15m_hours)
        if res:
            rows.append(res)
            if tg_enable and tg_token and tg_chat:
                if res["RSI-Cross 4h"] != "Nein":
                    alert_msgs.append(f"RSI-Cross 4h {res['RSI-Cross 4h'].upper()} – {b} @ {res['Preis']}")
                if res["EMA-Cross 15m"] != "Nein":
                    alert_msgs.append(f"EMA-Cross 15m {res['EMA-Cross 15m'].upper()} – {b} @ {res['Preis']}")
        prog.progress(i/len(candidates))
        time.sleep(0.08)  # ganz kleine Pause, sehr konservativ

    if not rows:
        st.info("Es gab diesmal keine OB/OS-Treffer unter den verfügbaren Bybit-Coins.")
        st.stop()

    if tg_enable and tg_token and tg_chat and alert_msgs:
        send_telegram(tg_token, tg_chat, "📣 RSI/EMA Alerts:\n" + "\n".join(f"• {m}" for m in alert_msgs))

    df = pd.DataFrame(rows)
    same = ((df["RSI 4h"].str.startswith("Überkauft")) & (df["RSI 15m"].str.startswith("Überkauft"))) | \
           ((df["RSI 4h"].str.startswith("Überverkauft")) & (df["RSI 15m"].str.startswith("Überverkauft")))
    df = df.assign(_conf=same.astype(int)).sort_values(by=["_conf"], ascending=False)
    if not show_all and len(df) > 100: df = df.head(100)

    df = df[["Coin","Preis","RSI 4h","RSI 15m","RSI-Cross 4h","EMA 5/10/20/30 (4h)","EMA 5/10/20/30 (15m)","EMA-Cross 15m"]]

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

    st.success(f"{len(df)} Treffer")
    st.dataframe(df.style.apply(row_bg, axis=1).applymap(val_color, subset=["RSI 4h","RSI 15m"]),
                 use_container_width=True, hide_index=True)
else:
    st.info("Konfiguriere links die Filter & klicke **Jetzt scannen**.")
