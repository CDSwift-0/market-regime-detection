#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Intelligence — détection de régimes de marché.

Dépendances :
    python3 -m pip install numpy pandas yfinance matplotlib scikit-learn

Principes :
- logique du modèle conservée : Gaussian Mixture, données hebdomadaires, 7 features ;
- le changement d'horizon ne retélécharge pas les prix et ne réentraîne pas le modèle ;
- l'analyse d'un actif est exécutée hors du thread Tkinter ;
- défilement direct, sans animation artificielle supplémentaire ;
- design aligné sur le dashboard Macro Intelligence / taux.py.
"""

from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import tkinter as tk
from tkinter import ttk

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Style — même ligne visuelle que taux.py
# =============================================================================

FONT = "Avenir Next"

COLORS = {
    "bg": "#f3f5f7",
    "panel": "#ffffff",
    "panel_alt": "#f8fafc",
    "panel_soft": "#f1f5f9",
    "ink": "#0f172a",
    "ink_soft": "#1e293b",
    "muted": "#64748b",
    "muted_light": "#94a3b8",
    "line": "#cbd5e1",
    "line_soft": "#e2e8f0",
    "grid": "#e7edf3",
    "blue": "#2563eb",
    "blue_bg": "#e3ecff",
    "green": "#0f9f6e",
    "green_bg": "#ddf3ea",
    "red": "#d64545",
    "red_bg": "#f8e3e3",
    "gold": "#b7791f",
    "gold_bg": "#f7ecd5",
    "purple": "#7c5ba7",
    "purple_bg": "#eee8f6",
    "teal": "#0f8b78",
    "teal_bg": "#dff3ef",
    "neutral": "#64748b",
    "neutral_bg": "#edf1f5",
}

REGIME_COLORS = {
    "Bear Market": COLORS["red"],
    "Transition": COLORS["neutral"],
    "Bull Market": COLORS["green"],
}

REGIME_BG = {
    "Bear Market": COLORS["red_bg"],
    "Transition": COLORS["neutral_bg"],
    "Bull Market": COLORS["green_bg"],
}

REGIME_ZONE_COLORS = {
    "Bear Market": "#f3dddd",
    "Transition": "#e7ebef",
    "Bull Market": "#dcefe8",
}

PERIODS = {
    "2 ans": 730,
    "3 ans": 1095,
    "5 ans": 1825,
    "10 ans": 3650,
    "15 ans": 5475,
    "20 ans": 7300,
    "Max": None,
}

MODEL_START = datetime(1990, 1, 1)
FREQUENCY_LABEL = "Semaine"
ASSETS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Or": "GLD",
    "STOXX 50": "^STOXX50E",
    "MSCI World": "URTH",
    "SMI": "^SSMI",
    "CAC 40": "^FCHI",
}
TICKERS = tuple(ASSETS)
FEATURES = ["Return", "Volatility", "Volume_Ratio", "RSI", "Momentum", "Range", "Drawdown"]
PRICE_CACHE: dict[str, pd.DataFrame] = {}


# =============================================================================
# Modèles de données
# =============================================================================

@dataclass(frozen=True)
class RegimeStat:
    name: str
    probability: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    persistence: float
    duration: float
    avg_rsi: float
    avg_momentum: float


@dataclass(frozen=True)
class AnalysisResult:
    ticker: str
    period: str
    frequency: str
    raw: pd.DataFrame
    model_data: pd.DataFrame
    probabilities: pd.DataFrame
    dominant: pd.Series
    transition: pd.DataFrame
    forecast: pd.DataFrame
    stats: list[RegimeStat]
    current_regime: str
    current_probability: float
    confidence: str
    summary: str


# =============================================================================
# Données et modèle — logique conservée
# =============================================================================

def fmt_pct(value: float | None, digits: int = 1, signed: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "n.d."
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:.{digits}f}%"


def calculate_rsi(price: pd.Series, window: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def download_prices(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=8,
    )
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    data["Price"] = data[price_col]
    for column in ("Open", "High", "Low", "Close"):
        if column not in data.columns:
            data[column] = data["Price"]
    if "Volume" not in data.columns:
        data["Volume"] = 0.0
    data["Volume"] = data["Volume"].fillna(0.0)
    return data[["Open", "High", "Low", "Close", "Price", "Volume"]].dropna()


def build_daily_features(raw: pd.DataFrame) -> pd.DataFrame:
    data = raw.copy()
    data["Return"] = data["Price"].pct_change(fill_method=None)
    data["Volatility"] = data["Return"].rolling(20).std() * np.sqrt(252)
    volume = data["Volume"].replace(0, np.nan)
    data["Volume_Ratio"] = (volume / volume.rolling(20).mean()).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    data["RSI"] = calculate_rsi(data["Price"])
    data["Momentum"] = data["Price"] / data["Price"].shift(20) - 1
    data["Range"] = (data["High"] - data["Low"]) / data["Close"]
    data["Drawdown"] = data["Price"] / data["Price"].cummax() - 1
    return data.dropna()


def aggregate_features(daily: pd.DataFrame) -> pd.DataFrame:
    freq = "W-FRI"
    actual_dates = daily["Price"].resample(freq).apply(lambda series: series.index[-1] if len(series) else pd.NaT)
    price = daily["Price"].resample(freq).last()
    out = pd.DataFrame(index=price.index)
    out["Price"] = price
    out["Return"] = price.pct_change(fill_method=None)
    out["Volatility"] = daily["Return"].resample(freq).std() * np.sqrt(52)
    out["Volume_Ratio"] = daily["Volume_Ratio"].resample(freq).mean()
    out["RSI"] = daily["RSI"].resample(freq).mean()
    out["Momentum"] = daily["Momentum"].resample(freq).mean()
    out["Range"] = daily["Range"].resample(freq).mean()
    out["Drawdown"] = daily["Drawdown"].resample(freq).last()
    out = out.dropna()
    out.index = pd.DatetimeIndex(actual_dates.loc[out.index])
    return out


def regime_names() -> list[str]:
    return ["Bear Market", "Transition", "Bull Market"]


def model_regime_names() -> list[str]:
    return ["Bear Market", "Bull Market"]


def classify_confidence(probabilities: pd.Series) -> str:
    ordered = probabilities.sort_values(ascending=False)
    top = float(ordered.iloc[0])
    spread = float(ordered.iloc[0] - ordered.iloc[1]) if len(ordered) > 1 else top
    if top >= 0.72 and spread >= 0.25:
        return "élevée"
    if top >= 0.55 and spread >= 0.12:
        return "moyenne"
    return "faible"


def transition_matrix(dominant: pd.Series, names: list[str]) -> pd.DataFrame:
    counts = pd.DataFrame(1.0, index=names, columns=names)
    previous = dominant.shift(1).dropna()
    current = dominant.loc[previous.index]
    for old, new in zip(previous, current):
        counts.loc[old, new] += 1
    return counts.div(counts.sum(axis=1), axis=0)


def forecast_probabilities(current: pd.Series, transition: pd.DataFrame) -> pd.DataFrame:
    vector = current.reindex(transition.index).fillna(0).to_numpy()
    matrix = transition.to_numpy()
    powers = {"Mois prochain": 4}
    rows = {label: vector @ np.linalg.matrix_power(matrix, power) for label, power in powers.items()}
    return pd.DataFrame(rows, index=transition.index).T


def transition_mask(base_probabilities: pd.DataFrame, base_dominant: pd.Series) -> pd.Series:
    margin = (base_probabilities["Bull Market"] - base_probabilities["Bear Market"]).abs()
    threshold = min(0.16, float(margin.quantile(0.08)))
    uncertain = margin <= threshold
    switches = base_dominant.ne(base_dominant.shift())
    switches.iloc[0] = False
    candidates = uncertain | switches
    limit = max(1, int(round(len(base_dominant) * 0.08)))
    if int(candidates.sum()) <= limit:
        return candidates

    rank = margin.copy()
    rank.loc[switches] = rank.loc[switches] - 0.05
    selected = rank.loc[candidates].nsmallest(limit).index
    mask = pd.Series(False, index=base_dominant.index)
    mask.loc[selected] = True
    return mask


def fit_regimes(model_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame, list[RegimeStat]]:
    names = regime_names()
    model_names = model_regime_names()
    n_regimes = len(model_names)
    annualizer = 52
    clean = model_data.dropna(subset=FEATURES).copy()
    minimum = max(24, n_regimes * 8)
    if len(clean) < minimum:
        raise RuntimeError(f"Pas assez de données propres pour entraîner le modèle ({len(clean)} observations).")

    x = StandardScaler().fit_transform(clean[FEATURES])
    model = GaussianMixture(
        n_components=n_regimes,
        covariance_type="full",
        n_init=10,
        max_iter=500,
        random_state=42,
    )
    model.fit(x)
    raw_probabilities = model.predict_proba(x)
    raw_labels = raw_probabilities.argmax(axis=1)

    raw_scores: list[tuple[int, float]] = []
    for index in range(n_regimes):
        subset = clean.iloc[raw_labels == index]
        if subset.empty:
            raw_scores.append((index, -999.0))
            continue
        annual_return = subset["Return"].mean() * annualizer * 100
        annual_vol = subset["Return"].std() * np.sqrt(annualizer) * 100
        momentum = subset["Momentum"].mean() * 100
        drawdown = subset["Drawdown"].mean() * 100
        score = annual_return - 0.45 * annual_vol + 0.25 * momentum + 0.10 * drawdown
        raw_scores.append((index, score))

    ordered_raw = [item[0] for item in sorted(raw_scores, key=lambda item: item[1])]
    mapping = {raw: model_names[rank] for rank, raw in enumerate(ordered_raw)}

    base_probabilities = pd.DataFrame(index=clean.index)
    for raw, name in mapping.items():
        base_probabilities[name] = raw_probabilities[:, raw]
    base_probabilities = base_probabilities[model_names]

    base_dominant = base_probabilities.idxmax(axis=1)
    is_transition = transition_mask(base_probabilities, base_dominant)
    margin = (base_probabilities["Bull Market"] - base_probabilities["Bear Market"]).abs()
    transition_weight = pd.Series(0.0, index=clean.index)
    transition_weight.loc[is_transition] = (1.0 - margin.loc[is_transition]).clip(0.38, 0.86)

    probabilities = pd.DataFrame(index=clean.index)
    probabilities["Bear Market"] = base_probabilities["Bear Market"] * (1.0 - transition_weight)
    probabilities["Transition"] = transition_weight
    probabilities["Bull Market"] = base_probabilities["Bull Market"] * (1.0 - transition_weight)
    probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)[names]
    dominant = probabilities.idxmax(axis=1)
    dominant.loc[is_transition] = "Transition"
    trans = transition_matrix(dominant, names)
    forecast = forecast_probabilities(probabilities.iloc[-1], trans)

    stats: list[RegimeStat] = []
    for name in names:
        subset = clean.loc[dominant == name]
        if subset.empty:
            stats.append(RegimeStat(name, 0, 0, 0, 0, 0, 0, 0, 0))
            continue
        annual_return = subset["Return"].mean() * annualizer * 100
        annual_volatility = subset["Return"].std() * np.sqrt(annualizer) * 100
        sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0
        stats.append(
            RegimeStat(
                name=name,
                probability=float(probabilities[name].iloc[-1]),
                annual_return=float(annual_return),
                annual_volatility=float(annual_volatility),
                sharpe=float(sharpe),
                persistence=float(trans.loc[name, name]),
                duration=float((dominant == name).mean()),
                avg_rsi=float(subset["RSI"].mean()),
                avg_momentum=float(subset["Momentum"].mean() * 100),
            )
        )
    return probabilities, dominant, trans, forecast, stats


def build_summary(current_regime: str, probability: float, confidence: str, stats: list[RegimeStat]) -> str:
    stat = next(item for item in stats if item.name == current_regime)
    if current_regime == "Bull Market":
        reading = "le marché reste orienté risk-on, avec une lecture majoritairement haussière"
    elif current_regime == "Bear Market":
        reading = "le modèle détecte une phase risk-off, donc la prudence domine"
    else:
        reading = "le signal principal est une zone de transition, sans direction nette"
    return (
        f"{reading}. Probabilité dominante : {probability * 100:.1f}%, confiance {confidence}. "
        f"Sur les observations classées dans ce régime, le rendement moyen ressort à "
        f"{stat.annual_return:.1f}% annualisé pour {stat.annual_volatility:.1f}% de volatilité."
    )


def run_analysis(ticker: str, period: str, force_download: bool = False) -> AnalysisResult:
    end = datetime.now()
    symbol = ASSETS.get(ticker, ticker)
    if force_download:
        PRICE_CACHE.pop(symbol, None)
    if symbol in PRICE_CACHE:
        raw = PRICE_CACHE[symbol].copy()
    else:
        raw = download_prices(symbol, MODEL_START, end)
        PRICE_CACHE[symbol] = raw.copy()
    if raw.empty:
        raise RuntimeError("Impossible de télécharger les prix.")

    daily = build_daily_features(raw)
    model_data = aggregate_features(daily)
    probabilities, dominant, trans, forecast, stats = fit_regimes(model_data)
    current_regime = str(dominant.iloc[-1])
    current_probability = float(probabilities.iloc[-1].max())
    confidence = classify_confidence(probabilities.iloc[-1])
    summary = build_summary(current_regime, current_probability, confidence, stats)
    return AnalysisResult(
        ticker=ticker,
        period=period,
        frequency=FREQUENCY_LABEL,
        raw=raw,
        model_data=model_data.loc[probabilities.index],
        probabilities=probabilities,
        dominant=dominant,
        transition=trans,
        forecast=forecast,
        stats=stats,
        current_regime=current_regime,
        current_probability=current_probability,
        confidence=confidence,
        summary=summary,
    )


# =============================================================================
# Interface
# =============================================================================

class Dashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Machine Learning Intelligence")

        screen_w = max(self.winfo_screenwidth(), 1280)
        screen_h = max(self.winfo_screenheight(), 820)
        width = min(1920, max(1280, int(screen_w * 0.96)))
        height = min(1120, max(820, int(screen_h * 0.93)))
        self.geometry(f"{width}x{height}+{max(0, (screen_w-width)//2)}+{max(0, (screen_h-height)//2)}")
        self.minsize(1240, 800)
        self.configure(bg=COLORS["bg"])

        self.queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.ticker = tk.StringVar(value="S&P 500")
        self.period = tk.StringVar(value="5 ans")
        self.status = tk.StringVar(value="Initialisation")
        self.range_text = tk.StringVar(value="Plage calculée après analyse")
        self.model_text = tk.StringVar(value="Modèle hebdomadaire")
        self.result: AnalysisResult | None = None
        self.loading = False
        self.spinner_job: str | None = None
        self.spinner_angle = 0

        self.kpi_values: dict[str, tk.Label] = {}
        self.period_buttons: dict[str, tk.Button] = {}
        self._crosshair_refs: list[Any] = []

        self._styles()
        self._shell()
        self.after(100, self._poll)
        self.after(180, self.start_analysis)

    # ------------------------------------------------------------------ shell

    def _styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(
            "ML.Vertical.TScrollbar",
            troughcolor=COLORS["bg"],
            background=COLORS["line"],
            bordercolor=COLORS["bg"],
            arrowcolor=COLORS["muted"],
            lightcolor=COLORS["line"],
            darkcolor=COLORS["line"],
        )

    def _shell(self) -> None:
        self.page = tk.Frame(self, bg=COLORS["bg"])
        self.page.pack(fill="both", expand=True)

        self._header()
        self._control_strip()
        self._kpi_strip()

        self.footer = tk.Label(
            self.page,
            text="",
            bg=COLORS["bg"],
            fg=COLORS["muted_light"],
            font=(FONT, 8),
            anchor="w",
        )
        self.footer.pack(side="bottom", fill="x", padx=20, pady=(4, 7))

        wrap = tk.Frame(self.page, bg=COLORS["bg"])
        wrap.pack(fill="both", expand=True, padx=(18, 10), pady=(0, 2))

        self.body_canvas = tk.Canvas(
            wrap,
            bg=COLORS["bg"],
            highlightthickness=0,
            borderwidth=0,
            yscrollincrement=1,
        )
        self.scrollbar = ttk.Scrollbar(
            wrap,
            orient="vertical",
            command=self.body_canvas.yview,
            style="ML.Vertical.TScrollbar",
        )
        self.body_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.body_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y", padx=(5, 0))

        self.body = tk.Frame(self.body_canvas, bg=COLORS["bg"])
        self.body_window = self.body_canvas.create_window((0, 0), window=self.body, anchor="nw")
        for col in range(2):
            self.body.grid_columnconfigure(col, weight=1, uniform="main")

        self.body.bind("<Configure>", self._sync_scroll_region)
        self.body_canvas.bind("<Configure>", self._fit_body_width)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Button-4>", self._on_mousewheel)
        self.bind_all("<Button-5>", self._on_mousewheel)

        self._render_loading_body("Préparation du modèle…")

    def _header(self) -> None:
        head = tk.Frame(self.page, bg=COLORS["bg"])
        head.pack(fill="x", padx=20, pady=(16, 10))

        left = tk.Frame(head, bg=COLORS["bg"])
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="MACHINE LEARNING", bg=COLORS["bg"], fg=COLORS["blue"], font=(FONT, 10, "bold")).pack(anchor="w")
        tk.Label(left, text="Régimes de marché", bg=COLORS["bg"], fg=COLORS["ink"], font=(FONT, 27, "bold")).pack(anchor="w")
        tk.Label(
            left,
            text="Gaussian Mixture · 7 variables · fréquence hebdomadaire · probabilités de régime · projection à 1 mois",
            bg=COLORS["bg"],
            fg=COLORS["muted"],
            font=(FONT, 11),
        ).pack(anchor="w")

        right = tk.Frame(head, bg=COLORS["bg"])
        right.pack(side="right", anchor="ne", pady=(8, 0))
        self.spinner = tk.Canvas(right, width=24, height=24, bg=COLORS["bg"], highlightthickness=0)
        self.spinner.pack(side="left", padx=(0, 8), pady=(3, 0))
        tk.Label(right, textvariable=self.status, bg=COLORS["bg"], fg=COLORS["muted"], font=(FONT, 10, "bold")).pack(side="left", padx=(0, 12), pady=(5, 0))
        self.refresh_button = tk.Button(
            right,
            text="Actualiser les données",
            command=lambda: self.start_analysis(force_download=True),
            bg=COLORS["ink"],
            fg="white",
            activebackground=COLORS["ink_soft"],
            activeforeground="white",
            borderwidth=0,
            padx=18,
            pady=10,
            cursor="hand2",
            font=(FONT, 10, "bold"),
        )
        self.refresh_button.pack(side="left")

    def _control_strip(self) -> None:
        outer = tk.Frame(
            self.page,
            bg=COLORS["panel"],
            highlightbackground=COLORS["line_soft"],
            highlightthickness=1,
            height=96,
        )
        outer.pack(fill="x", padx=20, pady=(0, 12))
        outer.pack_propagate(False)

        asset = tk.Frame(outer, bg=COLORS["panel"], width=255)
        asset.pack(side="left", fill="y", padx=(18, 10), pady=13)
        asset.pack_propagate(False)
        tk.Label(asset, text="ACTIF ANALYSÉ", bg=COLORS["panel"], fg=COLORS["ink"], font=(FONT, 10, "bold")).pack(anchor="w")

        select_row = tk.Frame(asset, bg=COLORS["panel"])
        select_row.pack(fill="x", pady=(6, 0))
        selector = tk.Menubutton(
            select_row,
            textvariable=self.ticker,
            bg=COLORS["panel_soft"],
            fg=COLORS["ink"],
            activebackground=COLORS["line_soft"],
            activeforeground=COLORS["ink"],
            borderwidth=0,
            relief="flat",
            anchor="w",
            padx=11,
            pady=7,
            cursor="hand2",
            font=(FONT, 10, "bold"),
        )
        menu = tk.Menu(selector, tearoff=False, bg=COLORS["panel"], fg=COLORS["ink"], activebackground=COLORS["panel_soft"], activeforeground=COLORS["ink"], bd=0)
        for value in TICKERS:
            menu.add_command(label=value, command=lambda selected=value: self._select_asset(selected), font=(FONT, 10))
        selector.configure(menu=menu)
        selector.pack(side="left", fill="x", expand=True)

        tk.Button(
            select_row,
            text="Analyser",
            command=self.start_analysis,
            bg=COLORS["blue"],
            fg="white",
            activebackground=COLORS["ink_soft"],
            activeforeground="white",
            borderwidth=0,
            padx=12,
            pady=7,
            cursor="hand2",
            font=(FONT, 9, "bold"),
        ).pack(side="left", padx=(6, 0))

        segment = tk.Frame(outer, bg=COLORS["panel_soft"], highlightbackground=COLORS["line_soft"], highlightthickness=1)
        segment.pack(side="left", fill="both", expand=True, padx=8, pady=15)
        for label in PERIODS:
            button = tk.Button(
                segment,
                text=label,
                command=lambda value=label: self._set_period(value),
                borderwidth=0,
                padx=12,
                pady=9,
                cursor="hand2",
                font=(FONT, 9, "bold"),
            )
            button.pack(side="left", fill="both", expand=True, padx=2, pady=2)
            self.period_buttons[label] = button
        self._style_period_buttons()

        info = tk.Frame(outer, bg=COLORS["panel"], width=265)
        info.pack(side="right", fill="y", padx=(10, 18), pady=13)
        info.pack_propagate(False)
        tk.Label(info, text="PLAGE AFFICHÉE", bg=COLORS["panel"], fg=COLORS["muted_light"], font=(FONT, 9, "bold")).pack(anchor="e")
        tk.Label(info, textvariable=self.range_text, bg=COLORS["panel"], fg=COLORS["ink"], font=(FONT, 10, "bold")).pack(anchor="e", pady=(3, 0))
        tk.Label(info, textvariable=self.model_text, bg=COLORS["panel"], fg=COLORS["muted"], font=(FONT, 8)).pack(anchor="e", pady=(2, 0))

    def _kpi_strip(self) -> None:
        frame = tk.Frame(self.page, bg=COLORS["bg"])
        frame.pack(fill="x", padx=20, pady=(0, 11))
        for col in range(6):
            frame.grid_columnconfigure(col, weight=1, uniform="kpi")

        specs = [
            ("regime", "RÉGIME ACTUEL", COLORS["blue"]),
            ("prob", "PROBABILITÉ", COLORS["purple"]),
            ("forecast", "PROJECTION 1 MOIS", COLORS["blue"]),
            ("vol", "VOLATILITÉ", COLORS["gold"]),
            ("momentum", "MOMENTUM", COLORS["teal"]),
            ("drawdown", "DRAWDOWN", COLORS["red"]),
        ]
        for idx, (key, label, accent) in enumerate(specs):
            card = tk.Frame(frame, bg=COLORS["panel"], highlightbackground=COLORS["line_soft"], highlightthickness=1)
            card.grid(row=0, column=idx, sticky="nsew", padx=3)
            tk.Frame(card, bg=accent, width=3).pack(side="left", fill="y")
            inner = tk.Frame(card, bg=COLORS["panel"])
            inner.pack(fill="both", expand=True, padx=9, pady=7)
            tk.Label(inner, text=label, bg=COLORS["panel"], fg=COLORS["muted"], font=(FONT, 8, "bold")).pack(anchor="w")
            value = tk.Label(inner, text="—", bg=COLORS["panel"], fg=COLORS["ink"], font=(FONT, 14, "bold"), wraplength=190, justify="left")
            value.pack(anchor="w", pady=(2, 0))
            self.kpi_values[key] = value

    # --------------------------------------------------------------- analysis

    def _select_asset(self, value: str) -> None:
        self.ticker.set(value)

    def start_analysis(self, force_download: bool = False) -> None:
        if self.loading:
            return

        ticker = self.ticker.get()
        period = self.period.get()

        if self.result is not None and self.result.ticker == ticker and not force_download:
            if self.result.period != period:
                self.result = replace(self.result, period=period)
                self._render_dashboard()
            return

        self.loading = True
        self.status.set("Analyse en cours")
        self.refresh_button.configure(state="disabled")
        self._start_spinner()
        if self.result is None:
            self._render_loading_body("Téléchargement des prix et entraînement du modèle…")

        threading.Thread(
            target=self._load_data,
            args=(ticker, period, force_download),
            daemon=True,
        ).start()

    def _load_data(self, ticker: str, period: str, force_download: bool) -> None:
        try:
            result = run_analysis(ticker, period, force_download=force_download)
            self.queue.put(("ok", result))
        except Exception as exc:
            self.queue.put(("error", exc))

    def _poll(self) -> None:
        handled = False
        while True:
            try:
                kind, payload = self.queue.get_nowait()
            except queue.Empty:
                break
            handled = True
            self.loading = False
            self._stop_spinner()
            self.refresh_button.configure(state="normal")
            if kind == "ok":
                self.result = payload
                self.status.set("À jour")
                self._render_dashboard()
            else:
                self.status.set("Erreur")
                self._render_error(payload)
        self.after(80 if handled else 120, self._poll)

    # --------------------------------------------------------------- spinner

    def _start_spinner(self) -> None:
        if self.spinner_job is not None:
            return

        def animate() -> None:
            self.spinner.delete("all")
            for i in range(8):
                self.spinner.create_arc(
                    4, 4, 20, 20,
                    start=self.spinner_angle + i * 40,
                    extent=18,
                    style="arc",
                    width=2,
                    outline=COLORS["blue"],
                )
            self.spinner_angle = (self.spinner_angle + 20) % 360
            self.spinner_job = self.after(130, animate)

        animate()

    def _stop_spinner(self) -> None:
        if self.spinner_job is not None:
            try:
                self.after_cancel(self.spinner_job)
            except Exception:
                pass
        self.spinner_job = None
        self.spinner.delete("all")

    # --------------------------------------------------------------- scroll

    def _sync_scroll_region(self, _event: tk.Event | None = None) -> None:
        self.body_canvas.configure(scrollregion=self.body_canvas.bbox("all"))

    def _fit_body_width(self, event: tk.Event) -> None:
        self.body_canvas.itemconfigure(self.body_window, width=max(1, int(event.width)))

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        bbox = self.body_canvas.bbox("all")
        if not bbox:
            return None
        if bbox[3] - bbox[1] <= self.body_canvas.winfo_height():
            return None

        if getattr(event, "num", None) == 4:
            pixels = -72
        elif getattr(event, "num", None) == 5:
            pixels = 72
        else:
            delta = float(getattr(event, "delta", 0) or 0)
            if delta == 0:
                return None
            pixels = int(round(-delta * 2.6)) if abs(delta) < 120 else int(round(-(delta / 120.0) * 78))

        if pixels:
            self.body_canvas.yview_scroll(pixels, "units")
        return "break"

    # --------------------------------------------------------------- helpers

    def _clear_body(self) -> None:
        self._crosshair_refs.clear()
        for child in self.body.winfo_children():
            child.destroy()

    def _panel(self, title: str, subtitle: str, row: int, col: int, colspan: int, height: int) -> tk.Frame:
        panel = tk.Frame(
            self.body,
            bg=COLORS["panel"],
            highlightbackground=COLORS["line_soft"],
            highlightthickness=1,
            height=height,
        )
        panel.grid(row=row, column=col, columnspan=colspan, sticky="nsew", padx=6, pady=6)
        panel.grid_propagate(False)
        panel.pack_propagate(False)

        head = tk.Frame(panel, bg=COLORS["panel"], height=48)
        head.pack(fill="x", padx=14, pady=(9, 3))
        head.pack_propagate(False)
        tk.Label(head, text=title, bg=COLORS["panel"], fg=COLORS["ink"], font=(FONT, 13, "bold")).pack(side="left")
        tk.Label(head, text=subtitle, bg=COLORS["panel"], fg=COLORS["muted"], font=(FONT, 9)).pack(side="right")
        return panel

    def _plot_holder(self, panel: tk.Frame) -> tk.Frame:
        holder = tk.Frame(panel, bg=COLORS["panel_alt"])
        holder.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        return holder

    def _render_loading_body(self, message: str) -> None:
        self._clear_body()
        panel = self._panel("Analyse du marché", "Machine learning hebdomadaire", 0, 0, 2, 260)
        inner = tk.Frame(panel, bg=COLORS["blue_bg"], highlightbackground=COLORS["line_soft"], highlightthickness=1)
        inner.pack(fill="both", expand=True, padx=14, pady=(2, 14))
        tk.Frame(inner, bg=COLORS["blue"], width=5).pack(side="left", fill="y")
        text = tk.Frame(inner, bg=COLORS["blue_bg"])
        text.pack(fill="both", expand=True, padx=18, pady=22)
        tk.Label(text, text="Préparation de l’analyse", bg=COLORS["blue_bg"], fg=COLORS["ink"], font=(FONT, 22, "bold")).pack(anchor="w")
        tk.Label(text, text=message, bg=COLORS["blue_bg"], fg=COLORS["muted"], font=(FONT, 11)).pack(anchor="w", pady=(6, 0))

    def _render_error(self, exc: Exception) -> None:
        self._clear_body()
        panel = self._panel("Analyse impossible", "La dernière requête n’a pas abouti", 0, 0, 2, 260)
        inner = tk.Frame(panel, bg=COLORS["red_bg"], highlightbackground=COLORS["line_soft"], highlightthickness=1)
        inner.pack(fill="both", expand=True, padx=14, pady=(2, 14))
        tk.Frame(inner, bg=COLORS["red"], width=5).pack(side="left", fill="y")
        text = tk.Frame(inner, bg=COLORS["red_bg"])
        text.pack(fill="both", expand=True, padx=18, pady=22)
        tk.Label(text, text="Impossible d’obtenir l’analyse", bg=COLORS["red_bg"], fg=COLORS["ink"], font=(FONT, 20, "bold")).pack(anchor="w")
        tk.Label(text, text=str(exc), bg=COLORS["red_bg"], fg=COLORS["muted"], font=(FONT, 10), wraplength=1050, justify="left").pack(anchor="w", pady=(6, 0))

    # --------------------------------------------------------------- dashboard

    def _render_dashboard(self) -> None:
        result = self.result
        if result is None:
            return

        self._clear_body()
        for row, height in enumerate((330, 590, 335, 390, 465)):
            self.body.grid_rowconfigure(row, minsize=height)

        self._update_top_info(result)
        self._render_signal_panel(result)
        self._render_probabilities_panel(result)
        self._render_main_chart(result)
        self._render_current_zone(result)
        self._render_regime_stats(result)
        self._render_transition_matrix(result)
        self._render_probability_history(result)
        self._render_inputs_table(result)

        self.body_canvas.yview_moveto(0.0)
        self.after_idle(self._sync_scroll_region)

    def _update_top_info(self, result: AnalysisResult) -> None:
        data = self.display_model_data(result)
        latest = result.model_data.iloc[-1]
        forecast_regime, forecast_prob, expected_return = self._forecast_summary(result)

        self.range_text.set(f"{data.index[0].strftime('%d.%m.%Y')}  →  {data.index[-1].strftime('%d.%m.%Y')}")
        self.model_text.set(f"{len(result.model_data):,} observations hebdomadaires · données jusqu’au {result.model_data.index[-1].strftime('%d.%m.%Y')}".replace(",", " "))

        self.kpi_values["regime"].configure(text=result.current_regime, fg=REGIME_COLORS[result.current_regime])
        self.kpi_values["prob"].configure(text=fmt_pct(result.current_probability * 100, 1))
        self.kpi_values["forecast"].configure(text=f"{forecast_regime} · {forecast_prob:.0f}%")
        self.kpi_values["vol"].configure(text=fmt_pct(float(latest["Volatility"]) * 100, 1))
        self.kpi_values["momentum"].configure(text=fmt_pct(float(latest["Momentum"]) * 100, 1, True))
        self.kpi_values["drawdown"].configure(text=fmt_pct(float(latest["Drawdown"]) * 100, 1, True))

        self.footer.configure(
            text=(
                f"{result.ticker} · modèle entraîné sur tout l’historique disponible à partir de 1990 · "
                f"zoom {result.period} · projection indicative à 1 mois ({expected_return:+.2f}%)"
            )
        )
        self._style_period_buttons()

    def _set_period(self, value: str) -> None:
        if value == self.period.get():
            return
        self.period.set(value)
        self._style_period_buttons()
        if self.result is not None:
            self.result = replace(self.result, period=value)
            self._render_dashboard()

    def _style_period_buttons(self) -> None:
        current = self.period.get()
        for label, button in self.period_buttons.items():
            selected = label == current
            button.configure(
                bg=COLORS["ink"] if selected else COLORS["panel_soft"],
                fg="white" if selected else COLORS["ink_soft"],
                activebackground=COLORS["ink_soft"] if selected else COLORS["line_soft"],
                activeforeground="white" if selected else COLORS["ink"],
            )

    # --------------------------------------------------------------- signal

    def _render_signal_panel(self, result: AnalysisResult) -> None:
        panel = self._panel("Lecture du modèle", "Signal actuel et anticipation à un mois", 0, 0, 1, 318)
        regime = result.current_regime
        tone = REGIME_COLORS[regime]
        tone_bg = REGIME_BG[regime]

        hero = tk.Frame(panel, bg=tone_bg, highlightbackground=COLORS["line_soft"], highlightthickness=1)
        hero.pack(fill="x", padx=12, pady=(2, 9))
        tk.Frame(hero, bg=tone, width=5).pack(side="left", fill="y")
        inner = tk.Frame(hero, bg=tone_bg)
        inner.pack(fill="both", expand=True, padx=13, pady=11)

        badge = tk.Label(inner, text="RÉGIME ACTUEL", bg=tone, fg="white", font=(FONT, 8, "bold"), padx=9, pady=4)
        badge.pack(anchor="w")
        tk.Label(inner, text=regime, bg=tone_bg, fg=COLORS["ink"], font=(FONT, 21, "bold")).pack(anchor="w", pady=(5, 1))
        tk.Label(
            inner,
            text=f"Probabilité {result.current_probability*100:.1f}% · confiance {result.confidence}",
            bg=tone_bg,
            fg=COLORS["muted"],
            font=(FONT, 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            inner,
            text=result.summary,
            bg=tone_bg,
            fg=COLORS["ink_soft"],
            font=(FONT, 10),
            wraplength=560,
            justify="left",
        ).pack(anchor="w", pady=(7, 0))

        forecast_regime, forecast_prob, expected_return = self._forecast_summary(result)
        strip = tk.Frame(panel, bg=COLORS["panel_alt"], highlightbackground=COLORS["line_soft"], highlightthickness=1)
        strip.pack(fill="x", padx=12, pady=(0, 10))
        tk.Frame(strip, bg=REGIME_COLORS[forecast_regime], width=4).pack(side="left", fill="y")
        left = tk.Frame(strip, bg=COLORS["panel_alt"])
        left.pack(side="left", fill="both", expand=True, padx=11, pady=8)
        tk.Label(left, text="PROJECTION À 1 MOIS", bg=COLORS["panel_alt"], fg=COLORS["muted_light"], font=(FONT, 8, "bold")).pack(anchor="w")
        tk.Label(left, text=f"{forecast_regime} · {forecast_prob:.1f}%", bg=COLORS["panel_alt"], fg=COLORS["ink"], font=(FONT, 12, "bold")).pack(anchor="w", pady=(2, 0))
        tk.Label(strip, text=f"Rendement attendu\n{expected_return:+.2f}%", bg=COLORS["panel_alt"], fg=REGIME_COLORS[forecast_regime], font=(FONT, 10, "bold"), justify="right").pack(side="right", padx=12)

    def _render_probabilities_panel(self, result: AnalysisResult) -> None:
        panel = self._panel("Probabilités du régime", "Distribution actuelle du signal", 0, 1, 1, 318)
        values = result.probabilities.iloc[-1]
        ordered = values.sort_values(ascending=False)
        margin = float(ordered.iloc[0] - ordered.iloc[1]) * 100
        entropy = -sum(float(p) * math.log(float(p)) for p in values if p > 0) / math.log(len(values))
        concentration = (1 - entropy) * 100

        holder = tk.Frame(panel, bg=COLORS["panel"])
        holder.pack(fill="both", expand=True, padx=13, pady=(2, 10))

        for name in regime_names():
            row = tk.Frame(holder, bg=COLORS["panel"])
            row.pack(fill="x", pady=5)
            top = tk.Frame(row, bg=COLORS["panel"])
            top.pack(fill="x")
            tk.Label(top, text=name, bg=COLORS["panel"], fg=COLORS["ink"], font=(FONT, 10, "bold")).pack(side="left")
            tk.Label(top, text=f"{values[name]*100:.1f}%", bg=COLORS["panel"], fg=REGIME_COLORS[name], font=(FONT, 10, "bold")).pack(side="right")
            bar = tk.Frame(row, bg=COLORS["panel_soft"], height=11, highlightbackground=COLORS["line_soft"], highlightthickness=1)
            bar.pack(fill="x", pady=(4, 0))
            bar.pack_propagate(False)
            fill = tk.Frame(bar, bg=REGIME_COLORS[name])
            fill.place(x=0, y=0, relheight=1, relwidth=max(0.002, float(values[name])))

        metrics = tk.Frame(holder, bg=COLORS["panel"])
        metrics.pack(fill="x", pady=(10, 0))
        self._small_metric(metrics, "ÉCART #1 / #2", f"{margin:.1f} pts", 0)
        self._small_metric(metrics, "CONCENTRATION", f"{concentration:.1f}%", 1)
        self._small_metric(metrics, "CONFIANCE", result.confidence.capitalize(), 2)

    def _small_metric(self, parent: tk.Frame, label: str, value: str, column: int) -> None:
        parent.grid_columnconfigure(column, weight=1, uniform="small")
        card = tk.Frame(parent, bg=COLORS["panel_alt"], highlightbackground=COLORS["line_soft"], highlightthickness=1)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 4, 0))
        tk.Label(card, text=label, bg=COLORS["panel_alt"], fg=COLORS["muted_light"], font=(FONT, 7, "bold")).pack(anchor="w", padx=8, pady=(6, 0))
        tk.Label(card, text=value, bg=COLORS["panel_alt"], fg=COLORS["ink"], font=(FONT, 10, "bold")).pack(anchor="w", padx=8, pady=(2, 6))

    # --------------------------------------------------------------- main chart

    def _render_main_chart(self, result: AnalysisResult) -> None:
        panel = self._panel("Prix et régimes de marché", "Zones historiques et projection à 1 mois", 1, 0, 2, 578)
        holder = self._plot_holder(panel)

        fig = Figure(figsize=(13.5, 5.05), dpi=90, facecolor=COLORS["panel_alt"])
        ax = fig.add_subplot(111, facecolor=COLORS["panel_alt"])
        fig.subplots_adjust(left=0.055, right=0.985, top=0.94, bottom=0.12)

        data = self.display_model_data(result)
        dominant = result.dominant.loc[data.index]
        prices = data["Price"]
        ax.plot(data.index, prices, color=COLORS["ink"], linewidth=2.1, zorder=4)

        self._shade_regime_zones(ax, dominant)
        self._highlight_current_zone(ax, dominant)
        self._draw_projection(ax, result, data)
        self._mark_current_point(ax, data)
        self._style_axis(ax, "Prix")

        span_days = max(1, (data.index[-1] - data.index[0]).days)
        if span_days > 4200:
            ax.xaxis.set_major_locator(mdates.YearLocator(base=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        elif span_days > 2200:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        elif span_days > 900:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        canvas = FigureCanvasTkAgg(fig, master=holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._add_smooth_crosshair(canvas, ax, data)

    def _draw_projection(self, ax: Any, result: AnalysisResult, data: pd.DataFrame) -> None:
        forecast_regime, _probability, expected_return = self._forecast_summary(result)
        latest_date = data.index[-1]
        latest_price = float(data["Price"].iloc[-1])
        next_date = latest_date + timedelta(days=30)
        projected_price = latest_price * (1 + expected_return / 100)
        color = REGIME_COLORS[forecast_regime]
        span_days = max(1, (data.index[-1] - data.index[0]).days)
        right_pad = max(45, min(120, int(span_days * 0.045)))

        ax.axvspan(latest_date, next_date, color=COLORS["panel"], alpha=0.75, linewidth=0, zorder=1)
        ax.axvline(latest_date, color=COLORS["ink"], linewidth=0.8, linestyle="--", alpha=0.35, zorder=5)
        ax.plot([latest_date, next_date], [latest_price, projected_price], color=color, linewidth=2.1, linestyle=(0, (4, 3)), zorder=9)
        ax.scatter([next_date], [projected_price], s=40, color=color, edgecolor=COLORS["panel_alt"], linewidth=1.0, zorder=10)
        ax.text(next_date, projected_price, "  1 mois", color=color, fontsize=8.5, fontweight="bold", va="center", ha="left", zorder=11)
        ax.set_xlim(data.index[0], latest_date + timedelta(days=right_pad))

    def _shade_regime_zones(self, ax: Any, dominant: pd.Series) -> None:
        date_nums = mdates.date2num(dominant.index.to_pydatetime())
        if len(date_nums) == 1:
            edges = np.array([date_nums[0] - 15, date_nums[0] + 15])
        else:
            edges = np.empty(len(date_nums) + 1)
            edges[1:-1] = (date_nums[:-1] + date_nums[1:]) / 2
            edges[0] = date_nums[0] - (edges[1] - date_nums[0])
            edges[-1] = date_nums[-1] + (date_nums[-1] - edges[-2])

        groups = dominant.ne(dominant.shift()).cumsum()
        positions = pd.Series(np.arange(len(dominant)), index=dominant.index)
        for _, group in dominant.groupby(groups):
            name = str(group.iloc[0])
            first = int(positions.loc[group.index[0]])
            last = int(positions.loc[group.index[-1]])
            ax.axvspan(edges[first], edges[last + 1], color=REGIME_ZONE_COLORS[name], alpha=1.0, linewidth=0, zorder=0)

    def _current_zone_edges(self, dominant: pd.Series) -> tuple[float, float]:
        date_nums = mdates.date2num(dominant.index.to_pydatetime())
        if len(date_nums) == 1:
            return date_nums[0] - 15, date_nums[0] + 15
        edges = np.empty(len(date_nums) + 1)
        edges[1:-1] = (date_nums[:-1] + date_nums[1:]) / 2
        edges[0] = date_nums[0] - (edges[1] - date_nums[0])
        edges[-1] = date_nums[-1] + (date_nums[-1] - edges[-2])
        groups = dominant.ne(dominant.shift()).cumsum()
        group_id = groups.iloc[-1]
        positions = pd.Series(np.arange(len(dominant)), index=dominant.index)
        current_group = dominant.loc[groups == group_id]
        first = int(positions.loc[current_group.index[0]])
        last = int(positions.loc[current_group.index[-1]])
        return float(edges[first]), float(edges[last + 1])

    def _highlight_current_zone(self, ax: Any, dominant: pd.Series) -> None:
        start, end = self._current_zone_edges(dominant)
        regime = str(dominant.iloc[-1])
        ax.axvspan(start, end, facecolor="none", edgecolor=REGIME_COLORS[regime], linewidth=1.8, zorder=2)
        ax.axvline(start, color=REGIME_COLORS[regime], linewidth=1.1, alpha=0.9, zorder=2)

    def _mark_current_point(self, ax: Any, data: pd.DataFrame) -> None:
        prices = data["Price"].to_numpy(dtype=float)
        ax.axhline(prices[-1], color=COLORS["ink"], linewidth=0.8, linestyle="--", alpha=0.3, zorder=5)
        ax.scatter([data.index[-1]], [prices[-1]], s=55, marker="x", color=COLORS["ink"], linewidth=2.0, zorder=8)

    def _add_smooth_crosshair(self, canvas: FigureCanvasTkAgg, ax: Any, data: pd.DataFrame) -> None:
        date_nums = mdates.date2num(data.index.to_pydatetime())
        prices = data["Price"].to_numpy(dtype=float)
        vertical = ax.axvline(data.index[-1], color=COLORS["blue"], linewidth=0.8, alpha=0.0, animated=True, zorder=20)
        horizontal = ax.axhline(prices[-1], color=COLORS["blue"], linewidth=0.8, alpha=0.0, animated=True, zorder=20)
        dot = ax.scatter([data.index[-1]], [prices[-1]], s=22, color=COLORS["blue"], alpha=0.0, animated=True, zorder=21)
        state: dict[str, Any] = {"background": None, "pending": None, "job": None, "visible": False}

        def nearest_index(x_value: float) -> int:
            pos = int(np.searchsorted(date_nums, x_value))
            if pos <= 0:
                return 0
            if pos >= len(date_nums):
                return len(date_nums) - 1
            left = pos - 1
            return left if abs(date_nums[left] - x_value) <= abs(date_nums[pos] - x_value) else pos

        def capture_background(_event: Any | None = None) -> None:
            try:
                state["background"] = canvas.copy_from_bbox(ax.bbox)
            except Exception:
                state["background"] = None

        def draw_crosshair(index: int) -> None:
            if state["background"] is None:
                capture_background()
            if state["background"] is None:
                return
            day = data.index[index]
            price = prices[index]
            vertical.set_xdata([day, day])
            horizontal.set_ydata([price, price])
            dot.set_offsets([[mdates.date2num(day), price]])
            vertical.set_alpha(0.58)
            horizontal.set_alpha(0.45)
            dot.set_alpha(0.9)
            state["visible"] = True
            canvas.restore_region(state["background"])
            ax.draw_artist(vertical)
            ax.draw_artist(horizontal)
            ax.draw_artist(dot)
            canvas.blit(ax.bbox)

        def flush() -> None:
            state["job"] = None
            if state["pending"] is not None:
                draw_crosshair(int(state["pending"]))

        def on_move(event: Any) -> None:
            if event.inaxes != ax or event.xdata is None:
                return
            state["pending"] = nearest_index(event.xdata)
            if state["job"] is None:
                state["job"] = canvas.get_tk_widget().after(14, flush)

        def on_leave(_event: Any) -> None:
            if not state["visible"] or state["background"] is None:
                return
            vertical.set_alpha(0.0)
            horizontal.set_alpha(0.0)
            dot.set_alpha(0.0)
            state["visible"] = False
            canvas.restore_region(state["background"])
            canvas.blit(ax.bbox)

        refs = [
            canvas.mpl_connect("draw_event", capture_background),
            canvas.mpl_connect("motion_notify_event", on_move),
            canvas.mpl_connect("axes_leave_event", on_leave),
        ]
        self._crosshair_refs.append((canvas, refs, state))
        capture_background()

    # --------------------------------------------------------------- zone

    def _render_current_zone(self, result: AnalysisResult) -> None:
        panel = self._panel("Zone actuelle", "Durée et état technique du régime en cours", 2, 0, 1, 323)
        data = result.model_data
        prices = data["Price"]
        latest = data.iloc[-1]
        latest_price = float(prices.iloc[-1])
        latest_date = data.index[-1]
        high_price = float(prices.max())
        one_year = min(52, max(1, len(data) - 1))
        trailing_return = (latest_price / float(prices.iloc[-one_year - 1]) - 1) * 100 if len(data) > one_year else float("nan")
        current_start = self._current_zone_start(result.dominant)
        zone_length = len(result.dominant.loc[current_start:])
        regime = result.current_regime

        hero = tk.Frame(panel, bg=REGIME_BG[regime], highlightbackground=COLORS["line_soft"], highlightthickness=1)
        hero.pack(fill="x", padx=12, pady=(2, 10))
        tk.Frame(hero, bg=REGIME_COLORS[regime], width=5).pack(side="left", fill="y")
        inner = tk.Frame(hero, bg=REGIME_BG[regime])
        inner.pack(fill="x", expand=True, padx=13, pady=10)
        tk.Label(inner, text=regime, bg=REGIME_BG[regime], fg=COLORS["ink"], font=(FONT, 18, "bold")).pack(anchor="w")
        tk.Label(
            inner,
            text=f"Depuis le {current_start.strftime('%d/%m/%Y')} · {zone_length} observation{'s' if zone_length > 1 else ''} · dernière donnée {latest_date.strftime('%d/%m/%Y')}",
            bg=REGIME_BG[regime], fg=COLORS["muted"], font=(FONT, 9, "bold")
        ).pack(anchor="w", pady=(3, 0))

        metrics = tk.Frame(panel, bg=COLORS["panel"])
        metrics.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        items = [
            ("Prix", f"{latest_price:,.2f}"),
            ("Plus haut", f"{high_price:,.2f}"),
            ("Perf. 1 an", fmt_pct(trailing_return, 1, True)),
            ("Volatilité", fmt_pct(float(latest["Volatility"]) * 100, 1)),
            ("Drawdown", fmt_pct(float(latest["Drawdown"]) * 100, 1, True)),
            ("RSI", f"{float(latest['RSI']):.1f}"),
        ]
        for idx, (label, value) in enumerate(items):
            row, col = divmod(idx, 3)
            metrics.grid_columnconfigure(col, weight=1, uniform="zone")
            box = tk.Frame(metrics, bg=COLORS["panel_alt"], highlightbackground=COLORS["line_soft"], highlightthickness=1)
            box.grid(row=row, column=col, sticky="nsew", padx=3, pady=3)
            tk.Label(box, text=label.upper(), bg=COLORS["panel_alt"], fg=COLORS["muted_light"], font=(FONT, 8, "bold")).pack(anchor="w", padx=8, pady=(6, 0))
            tk.Label(box, text=value, bg=COLORS["panel_alt"], fg=COLORS["ink"], font=(FONT, 11, "bold")).pack(anchor="w", padx=8, pady=(2, 6))

    # --------------------------------------------------------------- new: stats

    def _render_regime_stats(self, result: AnalysisResult) -> None:
        panel = self._panel("Profil des régimes", "Performance historique conditionnelle au régime", 2, 1, 1, 323)
        table = tk.Frame(panel, bg=COLORS["line_soft"])
        table.pack(fill="both", expand=True, padx=12, pady=(2, 11))

        headers = ("Régime", "Prob.", "Rend. ann.", "Vol. ann.", "Sharpe", "Persistance", "Poids hist.")
        for col, header in enumerate(headers):
            table.grid_columnconfigure(col, weight=1)
            tk.Label(table, text=header.upper(), bg=COLORS["panel_soft"], fg=COLORS["muted"], font=(FONT, 7, "bold"), padx=6, pady=7).grid(row=0, column=col, sticky="nsew", padx=(0, 1), pady=(0, 1))

        for row_idx, stat in enumerate(result.stats, start=1):
            bg = COLORS["panel"] if row_idx % 2 else COLORS["panel_alt"]
            values = (
                stat.name,
                fmt_pct(stat.probability * 100, 1),
                fmt_pct(stat.annual_return, 1, True),
                fmt_pct(stat.annual_volatility, 1),
                f"{stat.sharpe:.2f}",
                fmt_pct(stat.persistence * 100, 1),
                fmt_pct(stat.duration * 100, 1),
            )
            for col, value in enumerate(values):
                fg = REGIME_COLORS[stat.name] if col == 0 else COLORS["ink"]
                tk.Label(table, text=value, bg=bg, fg=fg, font=(FONT, 9, "bold" if col in (0, 1) else "normal"), padx=6, pady=10, anchor="w").grid(row=row_idx, column=col, sticky="nsew", padx=(0, 1), pady=(0, 1))

    # --------------------------------------------------------------- new: transition

    def _render_transition_matrix(self, result: AnalysisResult) -> None:
        panel = self._panel("Matrice de transition", "Probabilité hebdomadaire de rester ou changer de régime", 3, 0, 1, 378)
        names = regime_names()
        matrix = result.transition

        wrap = tk.Frame(panel, bg=COLORS["panel"])
        wrap.pack(fill="both", expand=True, padx=12, pady=(2, 11))
        tk.Label(wrap, text="DE  ↓   /   VERS  →", bg=COLORS["panel"], fg=COLORS["muted_light"], font=(FONT, 8, "bold")).pack(anchor="w", pady=(0, 8))

        grid = tk.Frame(wrap, bg=COLORS["line_soft"])
        grid.pack(fill="both", expand=True)
        grid.grid_columnconfigure(0, weight=1)
        for col in range(1, 4):
            grid.grid_columnconfigure(col, weight=1)

        tk.Label(grid, text="", bg=COLORS["panel_soft"]).grid(row=0, column=0, sticky="nsew", padx=(0, 1), pady=(0, 1))
        for col, name in enumerate(names, start=1):
            tk.Label(grid, text=name, bg=REGIME_BG[name], fg=REGIME_COLORS[name], font=(FONT, 8, "bold"), padx=5, pady=8).grid(row=0, column=col, sticky="nsew", padx=(0, 1), pady=(0, 1))

        for row, old in enumerate(names, start=1):
            tk.Label(grid, text=old, bg=REGIME_BG[old], fg=REGIME_COLORS[old], font=(FONT, 8, "bold"), padx=7, pady=12, anchor="w").grid(row=row, column=0, sticky="nsew", padx=(0, 1), pady=(0, 1))
            for col, new in enumerate(names, start=1):
                value = float(matrix.loc[old, new]) * 100
                bg = REGIME_BG[new] if old == new else COLORS["panel_alt"]
                fg = REGIME_COLORS[new] if old == new else COLORS["ink"]
                tk.Label(grid, text=f"{value:.1f}%", bg=bg, fg=fg, font=(FONT, 11, "bold" if old == new else "normal"), padx=5, pady=12).grid(row=row, column=col, sticky="nsew", padx=(0, 1), pady=(0, 1))

    # --------------------------------------------------------------- new: probability history

    def _render_probability_history(self, result: AnalysisResult) -> None:
        panel = self._panel("Probabilités dans le temps", "Évolution de la conviction du modèle", 3, 1, 1, 378)
        holder = self._plot_holder(panel)

        probs = self.display_probabilities(result)
        fig = Figure(figsize=(6.4, 2.95), dpi=88, facecolor=COLORS["panel_alt"])
        ax = fig.add_subplot(111, facecolor=COLORS["panel_alt"])
        fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.17)

        for name in regime_names():
            ax.plot(probs.index, probs[name] * 100, color=REGIME_COLORS[name], linewidth=1.55, label=name)
        ax.set_ylim(0, 100)
        self._style_axis(ax, "Probabilité (%)")
        ax.legend(loc="upper left", ncol=3, frameon=False, fontsize=8, labelcolor=COLORS["muted"])

        span_days = max(1, (probs.index[-1] - probs.index[0]).days)
        if span_days > 2200:
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        elif span_days > 900:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        else:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        canvas = FigureCanvasTkAgg(fig, master=holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # --------------------------------------------------------------- inputs

    def _render_inputs_table(self, result: AnalysisResult) -> None:
        panel = self._panel("Variables utilisées par le modèle", "Valeur actuelle, moyenne historique et écart standardisé", 4, 0, 2, 453)

        note = tk.Label(
            panel,
            text=(
                f"Dernière observation : {result.model_data.index[-1].strftime('%d/%m/%Y')}. "
                "Les variables sont calculées chaque semaine puis standardisées avant l'estimation du Gaussian Mixture."
            ),
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=(FONT, 9),
            anchor="w",
        )
        note.pack(fill="x", padx=14, pady=(0, 9))

        table = tk.Frame(panel, bg=COLORS["line_soft"])
        table.pack(fill="both", expand=True, padx=12, pady=(0, 11))
        headers = ("Variable", "Valeur actuelle", "Moyenne historique", "Écart au normal", "Ce que ça mesure")
        weights = (1, 1, 1, 1, 3)
        for col, (header, weight) in enumerate(zip(headers, weights)):
            table.grid_columnconfigure(col, weight=weight)
            tk.Label(table, text=header.upper(), bg=COLORS["panel_soft"], fg=COLORS["muted"], anchor="w", padx=9, pady=8, font=(FONT, 8, "bold")).grid(row=0, column=col, sticky="nsew", padx=(0, 1), pady=(0, 1))

        for row_idx, row in enumerate(self._feature_rows(result), start=1):
            bg = COLORS["panel"] if row_idx % 2 else COLORS["panel_alt"]
            for col, value in enumerate(row):
                tk.Label(
                    table,
                    text=value,
                    bg=bg,
                    fg=COLORS["ink"] if col != 4 else COLORS["muted"],
                    anchor="w",
                    padx=9,
                    pady=7,
                    font=(FONT, 9, "bold" if col == 0 else "normal"),
                ).grid(row=row_idx, column=col, sticky="nsew", padx=(0, 1), pady=(0, 1))

    # --------------------------------------------------------------- calculations for UI

    def display_model_data(self, result: AnalysisResult) -> pd.DataFrame:
        days = PERIODS[result.period]
        if days is None:
            return result.model_data
        cutoff = result.model_data.index[-1] - timedelta(days=days)
        visible = result.model_data.loc[result.model_data.index >= cutoff]
        return visible if len(visible) >= 4 else result.model_data.tail(4)

    def display_probabilities(self, result: AnalysisResult) -> pd.DataFrame:
        days = PERIODS[result.period]
        if days is None:
            return result.probabilities
        cutoff = result.probabilities.index[-1] - timedelta(days=days)
        visible = result.probabilities.loc[result.probabilities.index >= cutoff]
        return visible if len(visible) >= 4 else result.probabilities.tail(4)

    def _forecast_summary(self, result: AnalysisResult) -> tuple[str, float, float]:
        probabilities = result.forecast.iloc[0]
        regime = str(probabilities.idxmax())
        probability = float(probabilities.max()) * 100
        annual_returns = {item.name: item.annual_return for item in result.stats}
        expected_return = sum(float(probabilities[name]) * annual_returns[name] for name in result.forecast.columns) * 4 / 52
        return regime, probability, expected_return

    def _current_zone_start(self, dominant: pd.Series) -> pd.Timestamp:
        groups = dominant.ne(dominant.shift()).cumsum()
        current_group = groups.iloc[-1]
        return pd.Timestamp(dominant.loc[groups == current_group].index[0])

    def _feature_rows(self, result: AnalysisResult) -> list[tuple[str, str, str, str, str]]:
        specs = [
            ("Rendement", "Return", "pct", "Variation du prix sur la semaine : direction immédiate du marché."),
            ("Volatilité", "Volatility", "pct", "Risque réalisé annualisé : intensité des fluctuations récentes."),
            ("Volume relatif", "Volume_Ratio", "ratio", "Participation du marché par rapport à sa moyenne récente."),
            ("RSI", "RSI", "number", "Force technique entre 0 et 100 : excès haussier ou faiblesse récente."),
            ("Momentum", "Momentum", "pct", "Tendance récente sur environ un mois de cotation."),
            ("Amplitude", "Range", "pct", "Écart hauts-bas rapporté au cours : nervosité intrapériode."),
            ("Drawdown", "Drawdown", "pct", "Distance au plus haut historique : profondeur de la baisse courante."),
        ]
        latest = result.model_data.iloc[-1]
        means = result.model_data[FEATURES].mean()
        stds = result.model_data[FEATURES].std().replace(0, np.nan)

        rows: list[tuple[str, str, str, str, str]] = []
        for label, column, kind, role in specs:
            current = float(latest[column])
            mean = float(means[column])
            std = float(stds[column]) if math.isfinite(float(stds[column])) else float("nan")
            z_score = (current - mean) / std if math.isfinite(std) else float("nan")
            rows.append((label, self._format_feature(current, kind), self._format_feature(mean, kind), self._format_zscore(z_score), role))
        return rows

    def _format_feature(self, value: float, kind: str) -> str:
        if not math.isfinite(value):
            return "n.d."
        if kind == "pct":
            return fmt_pct(value * 100, 2, True)
        if kind == "ratio":
            return f"{value:.2f}x"
        return f"{value:.1f}"

    def _format_zscore(self, value: float) -> str:
        if not math.isfinite(value):
            return "n.d."
        if abs(value) < 0.35:
            label = "normal"
        elif value > 0:
            label = "au-dessus"
        else:
            label = "en-dessous"
        return f"{value:+.2f} σ · {label}"

    def _style_axis(self, ax: Any, ylabel: str = "") -> None:
        ax.tick_params(colors=COLORS["muted"], labelsize=8.5)
        ax.grid(True, color=COLORS["grid"], linewidth=0.75, alpha=0.85)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(COLORS["line"])
        ax.spines["bottom"].set_color(COLORS["line"])
        if ylabel:
            ax.set_ylabel(ylabel, color=COLORS["muted"], fontsize=9, fontweight="bold")


# =============================================================================
# Entrée
# =============================================================================

def main() -> None:
    Dashboard().mainloop()


if __name__ == "__main__":
    main()
