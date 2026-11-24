"""📱 SignalBot Marktscanner: Lern-/Analyse-App, keine automatisierten Echtgeldorders.

Lokal: streamlit run app.py
Cloud: Dateien app.py, scanner_core.py und requirements.txt in ein GitHub-Repo.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO
import time

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from scanner_core import (
    FALLBACK_TICKERS,
    analyze_ticker,
    extract_one_ticker,
    tidy_tickers,
    toy_market,
)

st.set_page_config(page_title="SignalBot Marktscanner", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container { max-width: 1150px; padding-top: .7rem; padding-bottom: 2.5rem; }
h1 { font-size: clamp(1.7rem, 5vw, 2.6rem); }
div[data-testid="stMetric"] {background: rgba(126,139,163,.11); border-radius: 13px; padding: 11px; }
[data-testid="stDataFrame"] { border-radius: 9px; }
@media (max-width: 600px) { .block-container { padding-left: .8rem; padding-right: .8rem; } }
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_tickers() -> tuple[list[str], str]:
    """Online-Indexliste oder begrenzte Notfall-Liste; transparenter Status."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; EducationMarketScanner/1.0)"}, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text), attrs={"id": "constituents"})
        symbols = tidy_tickers(" ".join(tables[0]["Symbol"].astype(str)), max_count=600)
        if len(symbols) >= 400:
            return symbols, "S&P-500-Liste online abgerufen (Wikipedia; kein Echtzeit-Indexbestand)."
        raise ValueError("Unvollständige Online-Liste")
    except Exception:
        return FALLBACK_TICKERS, (
            "⚠️ Online-Indexliste nicht erreichbar. Es wird nur die eingebaute Beispielauswahl gescannt; "
            "sie ist weder vollständig noch eine aktuelle S&P-500-Zusammensetzung."
        )


@st.cache_data(ttl=1800, show_spinner=False)
def batch_history(tickers: tuple[str, ...]) -> pd.DataFrame:
    """30-minütiger Cache reduziert unnötige Yahoo-Abfragen."""
    return yf.download(
        tickers=list(tickers), period="9mo", interval="1d",
        group_by="ticker", multi_level_index=True, auto_adjust=True,
        progress=False, threads=True, timeout=18,
    )


def do_scan(symbols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict] = []
    failed: list[str] = []
    batch_size = 25
    progress = st.progress(0.0, text="Marktdaten werden geladen …")
    status = st.empty()
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start : start + batch_size]
        status.caption(f"Untersuche Werte {start + 1}–{min(start + len(batch), len(symbols))} von {len(symbols)} …")
        try:
            data = batch_history(tuple(batch))
            for symbol in batch:
                frame = extract_one_ticker(data, symbol)
                record = analyze_ticker(symbol, frame)
                if record is None:
                    failed.append(symbol)
                else:
                    rows.append(record)
        except Exception:
            failed.extend(batch)
        progress.progress(min(1.0, (start + len(batch)) / len(symbols)), text="Aktien werden analysiert …")
        if start + batch_size < len(symbols):
            time.sleep(0.35)  # mäßige Anfragefrequenz, keine Garantie gegen Rate Limits
    progress.empty()
    status.empty()
    if not rows:
        return pd.DataFrame(), failed
    result = pd.DataFrame(rows)
    result = result.sort_values(["Score (0–100)", "Ticker"], ascending=[False, True]).reset_index(drop=True)
    return result, failed


st.title("📈 SignalBot · Marktscanner")
st.caption("Fürs Smartphone optimiert · Keine Broker-Anbindung · Kein automatischer Echtgeldhandel")
st.info("🔎 Der Scanner wertet mehrere Aktien **auf Knopfdruck** aus. Er zeigt technische Muster, **keine verlässlichen Kaufempfehlungen**. Keine künstliche Intelligenz in dieser Version.")

with st.container(border=True):
    market = st.radio("Welche Werte sollen untersucht werden?", ["S&P 500 (US-Aktien)", "Eigene Liste", "Demo ohne Kursabruf"], horizontal=False)
    if market == "S&P 500 (US-Aktien)":
        number = st.select_slider("Anzahl untersuchter Aktien", options=[25, 50, 100, 200, 300, 400, 500], value=200)
        st.caption("Standard: 200. Es werden die ersten verfügbaren Symbole der Indexliste gewählt, nicht vorher ausgewählte Favoriten. Größere Scans können dauern oder durch den Datenanbieter begrenzt werden.")
    elif market == "Eigene Liste":
        custom = st.text_area(
            "Yahoo-Finance-Ticker (Komma oder Zeilenumbruch)",
            value="AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, SAP.DE",
            height=90,
            help="Beispiel: AAPL, MSFT, NVDA. Maximal 500 Symbole. Für Punkt-Symbole des S&P 500 wird '.' in '-' umgewandelt.",
        )
    button_text = "🧪 Demo anzeigen" if market == "Demo ohne Kursabruf" else "🔍 Aktien jetzt durchsuchen"
    scan_clicked = st.button(button_text, type="primary", use_container_width=True)

if scan_clicked:
    if market == "Demo ohne Kursabruf":
        tests = [analyze_ticker(t, data) for t, data in toy_market().items()]
        st.session_state["scan_df"] = pd.DataFrame([t for t in tests if t is not None]).sort_values("Score (0–100)", ascending=False)
        st.session_state["failed_tickers"] = []
        st.session_state["scan_note"] = "🧪 SYNTHETISCHE DEMO – alle Werte sind erfunden, keine tatsächlichen Kurse oder Signale."
    else:
        if market == "S&P 500 (US-Aktien)":
            all_tickers, note = get_sp500_tickers()
            symbols = all_tickers[:number]
        else:
            symbols = tidy_tickers(custom, max_count=500)
            note = "Eigene Tickerliste; Namen und Börsenzugehörigkeit werden nicht geprüft."
        if not symbols:
            st.error("Bitte mindestens einen gültigen Ticker eingeben.")
            st.stop()
        with st.spinner("Historische Tageskurse werden heruntergeladen und analysiert …"):
            scan, failed = do_scan(symbols)
        st.session_state["scan_df"] = scan
        st.session_state["failed_tickers"] = failed
        st.session_state["scan_note"] = note
    st.session_state["scan_time_utc"] = datetime.now(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")

if "scan_df" not in st.session_state:
    st.markdown("### Start")
    st.write("Wähle oben eine Aktienliste und tippe auf **Aktien jetzt durchsuchen**. Die App arbeitet nur, solange du sie geöffnet hast und den Scan startest.")
    st.stop()

out = st.session_state["scan_df"]
failed = st.session_state.get("failed_tickers", [])
st.caption(st.session_state.get("scan_note", ""))
st.caption(f"Letzte Scan-Ausführung: {st.session_state.get('scan_time_utc', '?')} · Keine fortlaufende Überwachung")
if out.empty:
    st.error("Es konnten keine ausreichenden Daten abgerufen werden. Bitte kleinere Auswahl versuchen oder später erneut testen.")
    if failed:
        st.caption("Ticker ohne ausreichende Daten: " + ", ".join(failed[:40]))
    st.stop()

found = int(out["Kreuzung letzte 5 Tage"].sum())
positive = int(out["Positiver Trend"].sum())
a, b, c = st.columns(3)
a.metric("Ausgewertet", len(out))
b.metric("Kreuzung ≤ 5 Tage", found)
c.metric("Positiver SMA-Trend", positive)
if failed:
    st.warning(f"Für {len(failed)} von {len(out) + len(failed)} angefragten Symbolen fehlen Daten. Ergebnisse sind möglicherweise unvollständig.")
    with st.expander("Fehlende Symbole anzeigen"):
        st.write(", ".join(failed))

st.subheader("🔎 Gefundene technische Muster")
view = st.selectbox("Ergebnisse filtern", ["Nur neue Kreuzungen (5 Handelstage)", "Alle positiven Trends", "Alle Ergebnisse"], index=0)
if view == "Nur neue Kreuzungen (5 Handelstage)":
    filtered = out[out["Kreuzung letzte 5 Tage"]]
elif view == "Alle positiven Trends":
    filtered = out[out["Positiver Trend"]]
else:
    filtered = out

if filtered.empty:
    st.info("Mit diesem Filter wurden keine Werte gefunden. Wähle ‚Alle positiven Trends‘ oder ‚Alle Ergebnisse‘.")
else:
    visible_columns = ["Ticker", "Technischer Zustand", "Score (0–100)", "Letzter Schlusskurs", "RSI14", "10-Tage-Veränderung %", "Schlusskurstag"]
    st.dataframe(filtered[visible_columns], use_container_width=True, hide_index=True, height=min(620, 110 + 37 * len(filtered)))
    st.download_button(
        "⬇️ Gefilterte Ergebnisse als CSV",
        filtered.to_csv(index=False).encode("utf-8-sig"),
        file_name="signalbot_marktscanner.csv", mime="text/csv", use_container_width=True,
    )

st.markdown("### 📊 Einzelwert ansehen")
ticker = st.selectbox("Ticker", out["Ticker"].tolist())
selected = out[out["Ticker"] == ticker].iloc[0]
st.write(f"**{ticker}** · {selected['Technischer Zustand']} · Score **{selected['Score (0–100)']}/100**")
st.caption("Schlusskurs in der jeweiligen Börsenwährung, nicht automatisch Euro. Kein aktueller Ausführungskurs bei Trade Republic.")
st.write(f"SMA20: **{selected['SMA20']}** · SMA50: **{selected['SMA50']}** · RSI14: **{selected['RSI14']}**")

if ticker.startswith("DEMO-"):
    st.caption("Die Demo-Kurswerte sind künstlich erzeugt; echte Kurscharts gibt es nur für reale Ticker.")
elif st.button("📉 Chart für diesen Wert laden", use_container_width=True):
    try:
        history = extract_one_ticker(batch_history((ticker,)), ticker)
        close = pd.to_numeric(history["Close"], errors="coerce").dropna()
        if len(close) < 50:
            st.warning("Nicht genügend Kursdaten für den Chart vorhanden.")
        else:
            lines = pd.DataFrame({"Schlusskurs": close, "SMA20": close.rolling(20).mean(), "SMA50": close.rolling(50).mean()})
            st.line_chart(lines.tail(120), height=280)
    except Exception:
        st.warning("Chart konnte nicht geladen werden. Bitte Datenanbieter/Ticker prüfen.")

with st.expander("🧮 Wie wird der Score berechnet?"):
    st.markdown("""
    Der Score ist eine **selbst gewählte Sortierregel** (0–100), **keine Gewinnwahrscheinlichkeit**:

    - +40 Punkte, wenn SMA20 über SMA50 liegt.
    - +20 Punkte, wenn in den letzten 5 Handelstagen ein Aufwärtskreuz aufgetreten ist.
    - +15 Punkte, wenn der Schlusskurs über SMA20 liegt.
    - +10 Punkte bei RSI14 zwischen 45 und 65.
    - +10 Punkte bei positivem 10-Tage-Kursverlauf.
    - +5 Punkte bei steigendem SMA20.

    Eine Kreuzung wird nur beim Wechsel von SMA20 unter SMA50 zu über SMA50 gezählt. Ein positiver Trend **ohne Kreuzung** ist **kein neues Kaufsignal**. Das System verwendet historische Kurse und keine KI. Es enthält keine Fundamentalanalyse und keinen seriös validierten Backtest.
    """)
with st.expander("⚠️ Grenzen, Kosten und Sicherheit"):
    st.markdown("""
    - Die Yahoo-Finance-Daten können verzögert, unvollständig oder nicht verfügbar sein. Bitte stets den Ausführungskurs **in Trade Republic** prüfen.
    - Eine große Liste kann teilweise scheitern (Rate Limits); fehlende Werte werden separat gezählt.
    - Die berechneten Signale sind Lernindikatoren, **keine Anlageberatung**. Trading kann zu Verlusten führen.
    - Gebühren, Steuern, Währungsumrechnung, Spreads, Liquidität und persönliche Risikotoleranz sind nicht berücksichtigt.
    - Keine automatische Orderausführung, kein Zugriff auf Broker-Passwörter, keine Push-Mitteilungen und **kein Hintergrundbetrieb**.
    - Die öffentliche Streamlit-Bereitstellung ersetzt keinen verlässlichen kommerziellen Marktdatenanbieter. Prüfe die Nutzungsbedingungen der Datenquelle.
    """)
