"""Generazione di spiegazioni testuali della previsione, ancorate ai coefficienti stimati
dal SARIMAX e alle leve di scenario PESTEL applicate (obbligo PPWR di contenuto riciclato,
contesto di mercato lattiero-caseario).
"""

import pandas as pd

from data_prep import MATERIAL_FAMILY, MATERIAL_TYPE
from pestel_data import PESTEL_LABELS, PPWR_TARGET_PERIOD

TYPE_LABELS = {
    "vergine": "vergine",
    "riciclato": "riciclato (scaglia / rigenerato)",
    "macinato_interno": "macinato interno (scarto di produzione rilavorato)",
    "misto": "aggregato vergine + riciclato",
}


def material_type(material: str) -> str:
    """'vergine' | 'riciclato' | 'macinato_interno' | 'misto' (per le famiglie aggregate)."""
    return MATERIAL_TYPE.get(material, "misto")


def material_family(material: str) -> str:
    """Famiglia (PET/PS/PE) del materiale selezionato, sia esso famiglia o sottocategoria."""
    return MATERIAL_FAMILY.get(material, material)


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


def _trend_text(y: pd.Series, mean: pd.Series) -> str:
    hist_avg = y.iloc[-4:].mean() if len(y) >= 4 else y.mean()
    fc_avg = mean.iloc[: min(4, len(mean))].mean()
    if hist_avg == 0:
        return ""
    pct = (fc_avg / hist_avg - 1) * 100
    direzione = "in crescita" if pct > 2 else ("in calo" if pct < -2 else "sostanzialmente stabile")
    return (
        f"La previsione risulta **{direzione}**: la media dei primi trimestri previsti "
        f"({fc_avg:,.0f} kg/trimestre) rispetto alla media degli ultimi 4 trimestri storici "
        f"({hist_avg:,.0f} kg/trimestre) corrisponde a una variazione del **{pct:+.1f}%**."
    )


def _regressor_text(res, exog_cols, future_exog: pd.DataFrame | None) -> list[str]:
    lines = []
    if not exog_cols:
        return lines
    for col in exog_cols:
        if col not in res.params.index:
            continue
        coef = res.params[col]
        pval = res.pvalues.get(col, float("nan"))
        label = PESTEL_LABELS.get(col, col)
        segno = "positiva" if coef > 0 else "negativa"
        sig = "statisticamente significativa" if pval < 0.1 else (
            "non statisticamente significativa (atteso, vista la ridotta numerosità campionaria)"
        )
        assunzione = ""
        if future_exog is not None and col in future_exog.columns:
            v0 = future_exog[col].iloc[0]
            v1 = future_exog[col].iloc[-1]
            if abs(v1 - v0) < 1e-9:
                assunzione = f" Nel periodo di previsione il valore è mantenuto costante a {v1:,.2f}."
            else:
                assunzione = f" Nel periodo di previsione il valore passa da {v0:,.2f} a {v1:,.2f}."
        lines.append(
            f"**{label}**: relazione stimata **{segno}** con la domanda (coefficiente = {coef:,.1f}, "
            f"p-value = {pval:.2f}) — {sig}.{assunzione}"
        )
    return lines


_MARKET_CONTEXT = (
    "L'azienda produce prevalentemente imballaggi per il settore **lattiero-caseario** "
    "(vaschette e vasetti PET/PS per yogurt e formaggi freschi), un mercato di destinazione "
    "in forte evoluzione. Dati recenti (Ismea Mercati, Mordor Intelligence, luglio-agosto "
    "2025): la spesa delle famiglie italiane in lattiero-caseario è cresciuta del +6,6% nei "
    "primi 7 mesi del 2025, trainata da yogurt (+5,4%) e formaggi freschi (+4,1%) — categorie "
    "che usano tipicamente vaschette termoformate PET/PS — mentre latte fresco (-3,3%) e UHT "
    "(-1,2%), con packaging diverso, sono in calo. La plastica resta il materiale predominante "
    "negli imballaggi lattiero-caseari (quota 36,4%, +2,3% attesa fino al 2026), ma con "
    "pressione crescente di GDO e normativa verso formati senza vassoio e alternative in carta. "
    "**Lettura netta**: la crescita di yogurt/formaggi freschi sostiene la domanda di "
    "imballaggio PET/PS; la spinta verso formati senza vassoio è un freno strutturale "
    "aggiuntivo, distinto dall'obbligo di contenuto riciclato. Nessuna serie numerica "
    "affidabile per includerlo come regressore statistico: fattore trattato qualitativamente."
)


def _ppwr_content_text(material: str, mtype: str, family: str, s0: float | None,
                        target_pct: float, factor_series: pd.Series) -> str:
    tipo_label = TYPE_LABELS.get(mtype, mtype)
    if mtype == "misto":
        return (
            f"**Obbligo di contenuto riciclato PPWR — a livello di famiglia**: {material} aggrega "
            "materiale vergine e riciclato. L'obbligo normativo sposta la *composizione* della "
            "domanda tra le due categorie più che il *volume totale* di famiglia: l'effetto netto "
            "sul totale è quindi assunto neutro in questa previsione. Per vedere l'effetto "
            "direzionale reale (vergine in calo, riciclato in aumento) selezionare il livello "
            "'Sottocategoria'."
        )
    if mtype == "macinato_interno":
        return (
            f"**Obbligo di contenuto riciclato PPWR — materiale macinato interno**: {material} è "
            "scarto di produzione rilavorato internamente, non chiaramente equiparabile al "
            "riciclato post-consumo (PCR) rilevante per l'obbligo PPWR. Il suo utilizzo dipende "
            "più dal tasso di scarto interno che dalla normativa: l'effetto è assunto neutro in "
            "questa previsione."
        )
    if s0 is None:
        return (
            f"**Obbligo di contenuto riciclato PPWR**: per la famiglia {family} non è stato "
            "individuato un target normativo specifico in questa analisi (regola definita solo "
            "per PET e PS nel Regolamento UE 2025/40, Allegato II): nessun aggiustamento applicato."
        )

    target_year = PPWR_TARGET_PERIOD.year
    effetto_finale = (factor_series.iloc[-1] - 1) * 100
    direzione = "aumento" if effetto_finale > 0 else ("riduzione" if effetto_finale < 0 else "nessuna variazione")

    last_dt = factor_series.index[-1]
    last_period = last_dt if isinstance(last_dt, pd.Period) else pd.Timestamp(last_dt).to_period("Q")
    if last_period >= PPWR_TARGET_PERIOD:
        quando = f"a partire dal {target_year} (raggiunto entro l'orizzonte di previsione)"
    else:
        quando = f"entro la fine dell'orizzonte di previsione ({last_period}), che non arriva ancora al {target_year}"

    return (
        f"**Obbligo di contenuto riciclato PPWR (Reg. UE 2025/40, Allegato II)**: per la famiglia "
        f"{family} il regolamento richiede almeno il **{target_pct:.0f}% di riciclato (PCR)** entro "
        f"il 1° gennaio {target_year}, contro una quota attuale osservata negli ultimi 4 trimestri "
        f"del **{s0*100:.1f}%**. Essendo {material} un materiale **{tipo_label}**, questa "
        f"transizione (modellata come rampa lineare da oggi al {target_year}) si traduce in un "
        f"{direzione} della domanda rispetto alla previsione statistica di base del "
        f"**{abs(effetto_finale):.0f}%** {quando}."
    )


def _plastic_tax_text(material: str, mtype: str, plastic_tax_impact: int) -> str | None:
    sign = {"vergine": 1, "riciclato": -1}.get(mtype, 0)
    if sign == 0:
        return None
    effective_tax = plastic_tax_impact * sign
    direzione_tax = "un aumento" if effective_tax > 0 else ("una riduzione" if effective_tax < 0 else "nessun effetto")
    return (
        f"**Plastic Tax IT (scenario dal 01/2027)**: {direzione_tax} della domanda del "
        f"{abs(effective_tax)}%, in quanto la tassa colpisce specificamente il materiale vergine "
        "e rende relativamente più conveniente il riciclato."
    )


def _reactivity_text(reactivity_pct: float) -> str | None:
    if reactivity_pct == 100:
        return None
    if reactivity_pct == 0:
        return (
            "**Reattività ai fattori di mercato impostata allo 0%**: la previsione è stata ridotta a una "
            "semplice retta di tendenza lineare, calcolata sullo storico della serie. Stagionalità, "
            "regressori stimati e leve di scenario PESTEL sono completamente esclusi."
        )
    if reactivity_pct < 100:
        return (
            f"**Reattività ai fattori di mercato impostata al {reactivity_pct:.0f}%**: la previsione è "
            "stata resa più lineare, fondendo la stima del modello (stagionalità, regressori, leve di "
            "scenario PESTEL) con una retta di tendenza storica pura. Più il valore è basso, più la "
            "previsione si avvicina a una linea retta, sia nelle oscillazioni stagionali sia negli "
            "effetti dei fattori esterni."
        )
    return (
        f"**Reattività ai fattori di mercato impostata al {reactivity_pct:.0f}%**: la previsione è stata "
        "resa più reattiva, amplificando sia le oscillazioni stagionali sia l'effetto dei regressori "
        "stimati e delle leve di scenario PESTEL rispetto alla retta di tendenza storica."
    )


def build_explanation(
    material: str,
    y: pd.Series,
    mean: pd.Series,
    res,
    exog_cols: list[str],
    future_exog: pd.DataFrame | None,
    s0: float | None,
    target_pct: float,
    ppwr_factor: pd.Series,
    plastic_tax_scenario: bool,
    plastic_tax_impact: int,
    bt_mape: float,
    reactivity_pct: float = 100,
) -> str:
    mtype = material_type(material)
    family = material_family(material)
    parts = [_trend_text(y, mean), _reactivity_text(reactivity_pct), _MARKET_CONTEXT]

    reg_lines = _regressor_text(res, exog_cols, future_exog)
    if reg_lines:
        parts.append("**Fattori economici stimati nel modello:**")
        parts.extend(f"- {line}" for line in reg_lines)
    else:
        parts.append(
            "Nessun regressore esogeno è stato incluso nella stima: la previsione si basa solo "
            "sulla dinamica storica della serie (trend e stagionalità)."
        )

    scen_lines = [_ppwr_content_text(material, mtype, family, s0, target_pct, ppwr_factor)]
    if plastic_tax_scenario:
        tax_line = _plastic_tax_text(material, mtype, plastic_tax_impact)
        if tax_line:
            scen_lines.append(tax_line)
    parts.append("**Leve di scenario PESTEL applicate:**")
    parts.extend(f"- {line}" for line in scen_lines)

    parts.append(
        f"**Affidabilità**: nel backtest sugli ultimi trimestri il modello ha un errore medio (MAPE) del "
        f"{bt_mape:.1f}%. Con sole {len(y)} osservazioni trimestrali disponibili, sia i coefficienti "
        "stimati sia le leve di scenario vanno interpretati come indicazioni di tendenza, non come stime "
        "puntuali precise."
    )

    return "\n\n".join(p for p in parts if p)
