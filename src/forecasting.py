"""Modellazione SARIMAX con regressori esogeni PESTEL, selezione ordine e backtest.

Nota metodologica: con serie trimestrali di ~19 osservazioni, il numero di regressori
esogeni stimabili in modo affidabile e' molto limitato (si usa la regola empirica
n_regressori <= n_osservazioni / 8, quindi tipicamente 2 regressori al massimo).
La dashboard applica questo limite e avvisa l'utente se prova a superarlo.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")

MAX_REGRESSORS_RULE = 8  # n_osservazioni // MAX_REGRESSORS_RULE = numero massimo regressori consigliato


def max_recommended_regressors(n_obs: int) -> int:
    return max(1, n_obs // MAX_REGRESSORS_RULE)


def linear_trend_forecast(y: pd.Series, future_index) -> pd.Series:
    """Estrapolazione lineare (OLS su indice temporale intero) della serie storica, senza
    stagionalità ne' effetto di regressori: la 'retta di tendenza' pura usata come ancora
    per il controllo di reattivita'.
    """
    t_hist = np.arange(len(y))
    coeffs = np.polyfit(t_hist, y.values.astype(float), deg=1)
    t_future = np.arange(len(y), len(y) + len(future_index))
    values = np.polyval(coeffs, t_future)
    return pd.Series(values, index=future_index)


def apply_reactivity(mean: pd.Series, ci: pd.DataFrame, y_hist: pd.Series, reactivity_pct: float):
    """Fonde la previsione (gia' comprensiva di regressori e leve di scenario) con una retta
    di tendenza lineare pura, in proporzione a reactivity_pct.

    A reactivity_pct=100 la previsione resta quella stimata dal modello (invariata). A 0
    diventa una retta di tendenza pura: nessuna stagionalita', nessun effetto di regressori
    o scenari PESTEL. Valori >100 amplificano sia le oscillazioni stagionali sia gli effetti
    di regressori/scenario rispetto alla retta di tendenza. L'intervallo di confidenza viene
    traslato della stessa quantita' della media, mantenendo invariata la sua ampiezza
    (che riflette l'incertezza del modello, indipendente dal grado di 'linearizzazione').
    """
    trend = linear_trend_forecast(y_hist, mean.index)
    alpha = reactivity_pct / 100
    mean_final = trend + alpha * (mean - trend)
    delta = mean_final - mean
    ci_final = ci.add(delta, axis=0)
    return mean_final, ci_final


def _period_index_to_datetime(idx: pd.PeriodIndex) -> pd.DatetimeIndex:
    return idx.to_timestamp(how="start")


def grid_search_sarimax(y: pd.Series, exog: pd.DataFrame | None, seasonal_periods: int = 4):
    """Piccola grid search su ordini (p,d,q)(P,D,Q,s) selezionando il modello con AIC minimo.

    Griglia volutamente ridotta (dati trimestrali corti): p,q in {0,1,2}, d in {0,1},
    P,Q in {0,1}, D in {0,1}.
    """
    y_dt = y.copy()
    y_dt.index = _period_index_to_datetime(y.index)
    exog_dt = None
    if exog is not None and exog.shape[1] > 0:
        exog_dt = exog.copy()
        exog_dt.index = _period_index_to_datetime(exog.index)

    best = None
    n = len(y_dt)
    seasonal_ok = n >= 2 * seasonal_periods + 2

    p_vals = range(0, 3)
    d_vals = range(0, 2)
    q_vals = range(0, 3)
    P_vals = range(0, 2) if seasonal_ok else [0]
    D_vals = range(0, 2) if seasonal_ok else [0]
    Q_vals = range(0, 2) if seasonal_ok else [0]
    s = seasonal_periods if seasonal_ok else 0

    for p, d, q in itertools.product(p_vals, d_vals, q_vals):
        for P, D, Q in itertools.product(P_vals, D_vals, Q_vals):
            if p == d == q == 0 and P == D == Q == 0:
                continue
            n_params_est = p + q + P + Q + (1 if exog_dt is None else exog_dt.shape[1] + 1)
            if n_params_est >= n - 2:
                continue
            try:
                model = sm.tsa.statespace.SARIMAX(
                    y_dt,
                    exog=exog_dt,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    enforce_stationarity=True,
                    enforce_invertibility=True,
                )
                res = model.fit(disp=False)
                if not res.mle_retvals.get("converged", True):
                    continue
                ar_params = [v for k, v in res.params.items() if k.startswith("ar.")]
                if any(abs(v) > 0.97 for v in ar_params):
                    # Coefficiente AR troppo vicino alla radice unitaria: fit degenere/instabile
                    # (AIC apparentemente ottimo ma previsione non attendibile). Scartato.
                    continue
                if best is None or res.aic < best["aic"]:
                    best = {
                        "order": (p, d, q),
                        "seasonal_order": (P, D, Q, s),
                        "aic": res.aic,
                        "result": res,
                    }
            except Exception:
                continue

    if best is None:
        model = sm.tsa.statespace.SARIMAX(
            y_dt, exog=exog_dt, order=(0, 1, 0), seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=True, enforce_invertibility=True,
        )
        res = model.fit(disp=False)
        best = {"order": (0, 1, 0), "seasonal_order": (0, 0, 0, 0), "aic": res.aic, "result": res}

    return best


def forecast_future(y: pd.Series, exog: pd.DataFrame | None, order, seasonal_order,
                     steps: int, exog_future: pd.DataFrame | None):
    y_dt = y.copy()
    y_dt.index = _period_index_to_datetime(y.index)
    exog_dt = None
    exog_future_dt = None
    if exog is not None and exog.shape[1] > 0:
        exog_dt = exog.copy()
        exog_dt.index = _period_index_to_datetime(exog.index)
        exog_future_dt = exog_future.copy()
        exog_future_dt.index = _period_index_to_datetime(exog_future.index)

    model = sm.tsa.statespace.SARIMAX(
        y_dt, exog=exog_dt, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=True, enforce_invertibility=True,
    )
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps, exog=exog_future_dt)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.2)
    return res, mean, ci


def backtest(y: pd.Series, exog: pd.DataFrame | None, order, seasonal_order, n_test: int = 4):
    """Walk-forward backtest sugli ultimi n_test trimestri: rifit ad ogni step, previsione 1 passo avanti."""
    y_dt = y.copy()
    y_dt.index = _period_index_to_datetime(y.index)
    exog_dt = None
    if exog is not None and exog.shape[1] > 0:
        exog_dt = exog.copy()
        exog_dt.index = _period_index_to_datetime(exog.index)

    n = len(y_dt)
    n_test = min(n_test, max(1, n - 6))
    errors_abs_pct = []
    errors_sq = []
    actuals, preds = [], []

    for i in range(n - n_test, n):
        y_train = y_dt.iloc[:i]
        exog_train = exog_dt.iloc[:i] if exog_dt is not None else None
        exog_test = exog_dt.iloc[i:i + 1] if exog_dt is not None else None
        try:
            model = sm.tsa.statespace.SARIMAX(
                y_train, exog=exog_train, order=order, seasonal_order=seasonal_order,
                enforce_stationarity=True, enforce_invertibility=True,
            )
            res = model.fit(disp=False)
            pred = res.get_forecast(steps=1, exog=exog_test).predicted_mean.iloc[0]
        except Exception:
            pred = y_train.iloc[-1]

        actual = y_dt.iloc[i]
        actuals.append(actual)
        preds.append(pred)
        errors_abs_pct.append(abs((actual - pred) / actual) * 100 if actual != 0 else np.nan)
        errors_sq.append((actual - pred) ** 2)

    mape = float(np.nanmean(errors_abs_pct))
    rmse = float(np.sqrt(np.nanmean(errors_sq)))
    return {"mape": mape, "rmse": rmse, "actuals": actuals, "preds": preds,
            "index": y.index[n - n_test:n]}
