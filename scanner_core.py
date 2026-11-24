"""Reine Analysefunktionen des SignalBot; keine Broker-Aufträge, keine Prognose."""
from __future__ import annotations

import re
from typing import Optional
import pandas as pd

# Bei Ausfall der Online-Indexliste nur ein begrenztes, unverbindliches Beispieluniversum.
FALLBACK_TICKERS = """
AAPL MSFT NVDA AMZN GOOGL GOOG META TSLA BRK-B JPM V MA COST WMT HD PG
KO PEP ORCL CRM ADBE AMD INTC AVGO NFLX CSCO QCOM TXN IBM MU AMAT
LRCX KLAC PANW CRWD NOW PLTR UBER ABNB BKNG DIS SBUX MCD NKE LOW
TGT TJX ROST DG DLTR CMG YUM DHI LEN CAT DE GE GEHC HON BA LMT
RTX GD NOC UNP UPS FDX WM MMM ETN EMR PH ITW JCI CARR OTIS CMI
PWR VRT URI TT IR FAST GWW ADP PAYX INTU ACN CDNS SNPS ANET
DELL HPQ HPE APH TEL KEYS CI ELV UNH JNJ PFE MRK ABBV LLY TMO
DHR ABT SYK MDT BSX ISRG GILD AMGN REGN VRTX BMY ZTS CVS HCA
HUM CHTR CMCSA T VZ TMUS AXP GS MS BAC WFC C SCHW BLK BX KKR
USB PNC SPGI MCO ICE CME CB MMC AON TRV PGR ALL MET PRU AFL AIG
XOM CVX COP EOG SLB MPC PSX VLO KMI WMB OKE NEE DUK SO AEP
EXC XEL D SRE ED EIX PEG ATO ETR VST CEG EQIX AMT PLD DLR PSA
WELL O SPG VICI CBRE CCI EQR AVB WY INVH RCL CCL NCLH DAL
UAL LUV MAR HLT MGM LVS WYNN EXPE EBAY PYPL
""".split()


def tidy_tickers(raw: str, max_count: int = 500) -> list[str]:
    """Säubert eine per Leerzeichen/Komma/Zeilenumbruch getrennte Tickerliste."""
    seen = set()
    result = []
    for item in re.split(r"[\s,;]+", raw.upper().strip()):
        item = item.strip()
        # Nur US-Aktiengattungen umformen; Börsensuffix wie SAP.DE beibehalten.
        if re.fullmatch(r"[A-Z]{1,5}\.[A-Z]", item):
            item = item.replace(".", "-")
        if not item or not re.fullmatch(r"[A-Z0-9^=.\-]{1,20}", item):
            continue
        if item not in seen:
            seen.add(item)
            result.append(item)
        if len(result) >= max_count:
            break
    return result


def normalized_close(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty or "Close" not in df.columns:
        return pd.Series(dtype="float64")
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    close = close[close > 0]
    return close.sort_index()


def analyze_ticker(ticker: str, dataframe: pd.DataFrame) -> Optional[dict]:
    """Analyse ohne Zukunftsdaten. Score = heuristische Sortierung, keine Gewinnchance."""
    close = normalized_close(dataframe)
    if len(close) < 65:
        return None

    sma20 = close.rolling(20, min_periods=20).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    delta = close.diff()
    gains = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    losses = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gains / losses.replace(0, float("nan"))
    rsi = 100 - 100 / (1 + rs)
    if gains.iloc[-1] > 0 and losses.iloc[-1] == 0:
        rsi.iloc[-1] = 100.0
    elif gains.iloc[-1] == 0 and losses.iloc[-1] == 0:
        rsi.iloc[-1] = 50.0

    above = sma20 > sma50
    crosses = above & ~above.shift(1, fill_value=False)
    # Erst nach min. 50 gültigen Handelstagen zählt eine Kreuzung.
    crosses.iloc[:50] = False
    recent_cross = bool(crosses.iloc[-5:].any())
    new_cross_today = bool(crosses.iloc[-1])
    uptrend = bool(above.iloc[-1])
    price_above_sma = bool(close.iloc[-1] > sma20.iloc[-1])
    positive_10d = bool(close.iloc[-1] > close.iloc[-11])
    sma_rising = bool(sma20.iloc[-1] > sma20.iloc[-6])
    rsi_value = float(rsi.iloc[-1])

    # Bewusst einfache, transparente Regel. Keine trainierte KI, keine Wahrscheinlichkeit.
    score = (
        (40 if uptrend else 0)
        + (20 if recent_cross and uptrend else 0)
        + (15 if price_above_sma else 0)
        + (10 if 45 <= rsi_value <= 65 else 0)
        + (10 if positive_10d else 0)
        + (5 if sma_rising else 0)
    )

    if new_cross_today:
        status = "🟢 Neue SMA-Kreuzung"
    elif recent_cross and uptrend:
        status = "🟢 Kreuzung letzte 5 Tage"
    elif uptrend:
        status = "🔵 Aufwärtstrend"
    else:
        status = "⚪ Kein Kaufsignal"

    try:
        day_str = pd.Timestamp(close.index[-1]).strftime("%d.%m.%Y")
    except Exception:
        day_str = str(close.index[-1])

    return {
        "Ticker": ticker,
        "Technischer Zustand": status,
        "Score (0–100)": int(score),
        "Letzter Schlusskurs": round(float(close.iloc[-1]), 2),
        "SMA20": round(float(sma20.iloc[-1]), 2),
        "SMA50": round(float(sma50.iloc[-1]), 2),
        "RSI14": round(rsi_value, 1),
        "10-Tage-Veränderung %": round((float(close.iloc[-1] / close.iloc[-11]) - 1) * 100, 2),
        "Schlusskurstag": day_str,
        "Positiver Trend": uptrend,
        "Kreuzung letzte 5 Tage": bool(recent_cross and uptrend),
    }


def extract_one_ticker(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Yfinance Download mit group_by='ticker' oder Einzel-Ticker robust auslesen."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            values = raw.columns.get_level_values(level)
            if ticker in values:
                return raw.xs(ticker, axis=1, level=level)
        return pd.DataFrame()
    return raw if "Close" in raw.columns else pd.DataFrame()


def toy_market() -> dict[str, pd.DataFrame]:
    """Synthetische Testdaten; niemals als reale Marktdaten ausgeben."""
    import numpy as np
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=130)
    series = {}
    for i, ticker in enumerate(["DEMO-A", "DEMO-B", "DEMO-C", "DEMO-D", "DEMO-E"]):
        random = np.random.default_rng(seed=i + 21)
        trend = [0.15, 0.00, -0.18, 0.28, -0.08][i]
        x = 75 + trend * np.arange(len(idx)) + np.cumsum(random.normal(0, 0.5, len(idx)))
        series[ticker] = pd.DataFrame({"Close": x}, index=idx)
    return series
