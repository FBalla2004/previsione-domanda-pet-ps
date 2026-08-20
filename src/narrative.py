"""Calcolo delle leve di scenario PESTEL (obbligo PPWR di contenuto riciclato, crescita del
mercato lattiero-caseario) e generazione di una spiegazione sintetica del risultato.
"""

from __future__ import annotations

import pandas as pd

from pestel_data import PESTEL_LABELS, PPWR_TARGET_PERIOD

TYPE_LABELS = {
    "vergine": "vergine",
    "riciclato": "riciclato (scaglia)",
    "macinato_interno": "macinato interno",
    "misto": "vergine + riciclato",
}


def recycled_share_ramp(s0: float, target_pct: float, start_period: pd.Period,
                         index: pd.PeriodIndex, target_period: pd.Period = PPWR_TARGET_PERIOD) -> pd.Series:
    """Transizione lineare della quota di riciclato da s0 (oggi) a target_pct% (al target_period).

    Prima di start_period resta a s0, dopo target_period resta al target. Rappresenta
    l'ipotesi semplificativa che l'azienda adegui gradualmente il mix vergine/riciclato in
    vista della scadenza normativa, invece di un salto discontinuo.
    """
    target = target_pct / 100
    start_ord, target_ord = start_period.ordinal, target_period.ordinal
    vals = []
    for p in index:
        p_period = p if isinstance(p, pd.Period) else pd.Timestamp(p).to_period("Q")
        if target_ord <= start_ord or p_period >= target_period:
            vals.append(target)
        else:
            frac = max(0.0, min(1.0, (p_period.ordinal - start_ord) / (target_ord - start_ord)))
            vals.append(s0 + frac * (target - s0))
    return pd.Series(vals, index=index)


def ppwr_content_factor(mtype: str, s0: float | None, target_pct: float, start_period: pd.Period,
                         index: pd.PeriodIndex) -> pd.Series:
    """Fattore moltiplicativo da applicare alla previsione statistica di base per ciascun trimestre.

    riciclato -> quota di riciclato richiesta / quota attuale (fattore > 1 se il target e' piu' alto);
    vergine -> quota di vergine implicita / quota di vergine attuale (fattore < 1 se il riciclato deve salire);
    altrimenti (macinato interno, famiglie aggregate, materiali senza target) -> nessun effetto (1.0).
    """
    if s0 is None or mtype not in ("vergine", "riciclato") or not (0 < s0 < 1):
        return pd.Series(1.0, index=index)
    share = recycled_share_ramp(s0, target_pct, start_period, index)
    if mtype == "riciclato":
        return share / s0
    return (1 - share) / (1 - s0)


def dairy_growth_factor(growth_pct_annual: float, start_period: pd.Period, index: pd.PeriodIndex) -> pd.Series:
    """Fattore di crescita composta dovuto all'espansione del mercato di destinazione
    (lattiero-caseario), applicato in egual misura a vergine e riciclato: riflette la
    crescita del volume totale imballato, non uno spostamento di composizione.
    """
    rate = growth_pct_annual / 100
    vals = []
    for p in index:
        p_period = p if isinstance(p, pd.Period) else pd.Timestamp(p).to_period("Q")
        years = (p_period.ordinal - start_period.ordinal) / 4
        vals.append((1 + rate) ** years)
    return pd.Series(vals, index=index)


def _trend_line(y: pd.Series, mean: pd.Series) -> str:
    hist_avg = y.iloc[-4:].mean() if len(y) >= 4 else y.mean()
    fc_avg = mean.iloc[: min(4, len(mean))].mean()
    if hist_avg == 0:
        return ""
    pct = (fc_avg / hist_avg - 1) * 100
    direzione = "in crescita" if pct > 2 else ("in calo" if pct < -2 else "stabile")
    return f"Previsione **{direzione}**: {pct:+.1f}% rispetto alla media degli ultimi 4 trimestri."


def _regressor_lines(res, exog_cols) -> list[str]:
    lines = []
    for col in exog_cols or []:
        if res is None or col not in res.params.index:
            continue
        coef = res.params[col]
        label = PESTEL_LABELS.get(col, col)
        segno = "positivo" if coef > 0 else "negativo"
        lines.append(f"{label}: effetto {segno} sulla domanda.")
    return lines


def _ppwr_line(mtype: str, family: str, s0: float | None,
               target_pct: float, factor_series: pd.Series) -> str:
    if mtype == "misto":
        return "Obbligo riciclato PPWR: effetto neutro sul totale (sposta vergine → riciclato, non il volume)."
    if mtype == "macinato_interno":
        return "Obbligo riciclato PPWR: non applicabile (macinato interno)."
    if s0 is None:
        return f"Obbligo riciclato PPWR: nessun target normativo individuato per {family}."

    target_year = PPWR_TARGET_PERIOD.year
    effetto = (factor_series.iloc[-1] - 1) * 100
    segno = "+" if effetto >= 0 else ""
    return (
        f"Obbligo riciclato PPWR: {family} dal {s0*100:.0f}% al {target_pct:.0f}% entro il {target_year} "
        f"→ {segno}{effetto:.0f}% sulla domanda di {TYPE_LABELS.get(mtype, mtype)}."
    )


def _plastic_tax_line(mtype: str, plastic_tax_impact: int) -> str | None:
    sign = {"vergine": 1, "riciclato": -1}.get(mtype, 0)
    if sign == 0:
        return None
    effective = plastic_tax_impact * sign
    return f"Plastic Tax IT (dal 2027): {effective:+d}% sulla domanda."


def build_explanation(
    mtype: str,
    family: str,
    y: pd.Series,
    mean: pd.Series,
    res,
    exog_cols: list[str],
    s0: float | None,
    target_pct: float,
    ppwr_factor: pd.Series,
    dairy_growth_pct: float,
    plastic_tax_scenario: bool,
    plastic_tax_impact: int,
    reactivity_pct: float = 100,
) -> str:
    lines = []
    trend = _trend_line(y, mean)
    if trend:
        lines.append(trend)
    if reactivity_pct < 100:
        lines.append(f"Reattività al {reactivity_pct:.0f}%: previsione più lineare (meno stagionalità e fattori esterni).")

    lines.extend(_regressor_lines(res, exog_cols))
    lines.append(_ppwr_line(mtype, family, s0, target_pct, ppwr_factor))
    if dairy_growth_pct:
        lines.append(f"Crescita mercato lattiero-caseario: +{dairy_growth_pct:.2f}%/anno su tutta la domanda.")
    if plastic_tax_scenario:
        tax_line = _plastic_tax_line(mtype, plastic_tax_impact)
        if tax_line:
            lines.append(tax_line)

    body = "  \n".join(f"- {line}" for line in lines[1:] if line)
    return f"{lines[0]}  \n{body}" if lines else ""
