import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from data_prep import build_dataset, current_recycled_share  # noqa: E402
from forecasting import (  # noqa: E402
    apply_reactivity,
    backtest,
    forecast_future,
    grid_search_sarimax,
    max_recommended_regressors,
)
from narrative import (  # noqa: E402
    TYPE_LABELS,
    build_explanation,
    dairy_growth_factor,
    material_family,
    material_type,
    ppwr_content_factor,
)
from pestel_data import (  # noqa: E402
    DAIRY_GROWTH_RATE_ANNUAL,
    PESTEL_CATEGORY,
    PPWR_RECYCLED_TARGET_2030,
    PPWR_TARGET_PERIOD,
    build_pestel_quarterly,
)

st.set_page_config(page_title="Previsione Domanda PET/PS", layout="wide")


@st.cache_data
def load_data():
    return build_dataset()


famiglia_q, consumi_q, prezzo_q = load_data()

st.title("Sistema di previsione della domanda — mercato plastico PET/PS")
st.caption(
    "Tesi — pianificazione della produzione (Coexpan). Modello statistico SARIMAX con "
    "regressori esogeni PESTEL, applicato ai consumi di materia prima come proxy della domanda."
)

with st.expander("Metodologia e fonti dei dati PESTEL", expanded=False):
    st.markdown(
        """
**Dati storici (endogeni)**: consumi mensili di materia prima (convertiti in tonnellate) per
PET VG, SCAGLIA PET, DA MAC PET, PS, PS SCAGLIA, PE — aggregati a trimestri completi
(ottobre 2021 - giugno 2026). Il trimestre in corso, incompleto, viene escluso per non
introdurre un crollo artificiale.

**Regressore economico** (stimato nel modello): prezzo materiale (EUR/kg), dato interno
aziendale. Prezzo Brent e cambio EUR/USD sono stati volutamente esclusi: per un
trasformatore che acquista resina (non greggio), il prezzo materiale internalizza già
questi effetti macro, e includerli avrebbe saturato il budget di regressori stimabili
(max ~2 con 19 osservazioni) senza aggiungere potere esplicativo diretto.

**Mercato di destinazione (lattiero-caseario)**: l'azienda produce prevalentemente
imballaggi per yogurt e formaggi freschi. Dati Ismea Mercati/Mordor Intelligence 2025
mostrano consumi in crescita in queste categorie (yogurt +5,4%, formaggi freschi +4,1%)
ma anche una spinta strutturale della GDO verso formati senza vassoio/carta. La crescita
di volume (media yogurt/formaggi freschi ≈ 4,75%/anno) è inclusa come **leva di scenario
quantificata** (non stimata nel modello, varianza storica nulla), applicata in egual
misura a vergine e riciclato perché riflette la crescita del volume totale imballato, non
uno spostamento di composizione.

**Regressori politico/ambientali**: PPWR in vigore dall'11/02/2025 (Regolamento UE
2025/40) è stimato nel modello. L'**obbligo di contenuto riciclato PPWR** (30% PCR per il
PET, 10% per il PS entro il 1/1/2030, Allegato II) e la **Plastic Tax IT** (rinviata al
01/01/2027) hanno varianza storica nulla nel campione: sono trattati come **leve di
scenario** applicate alla previsione statistica, non come regressori stimati. A
differenza della crescita di mercato, l'obbligo PPWR sposta la *composizione* tra
vergine e riciclato: il riciclato cresce quindi per due effetti che si sommano — la
crescita del mercato di destinazione E lo spostamento verso il riciclato imposto dalla
normativa.

**Fattori sociali e tecnologici**: non è stata reperita una serie trimestrale pubblica
affidabile per consapevolezza ambientale dei consumatori o capacità di riciclo rPET oltre
al contesto lattiero-caseario sopra citato. Discussi qualitativamente in tesi (§5.2 Limiti
del lavoro), non entrano come regressori numerici.

**Limite campionario**: con ~19 osservazioni trimestrali, il numero di regressori stimabili
in modo affidabile è ridotto (regola empirica: n. regressori ≤ n. osservazioni / 8).
        """
    )

st.sidebar.header("Configurazione")
level = st.sidebar.radio("Livello di dettaglio", ["Famiglia (PET/PS/PE)", "Sottocategoria"])
if level == "Famiglia (PET/PS/PE)":
    series_options = list(famiglia_q.columns)
    data_source = famiglia_q
else:
    series_options = list(consumi_q.columns)
    data_source = consumi_q

default_idx = series_options.index("PET") if "PET" in series_options else 0
material = st.sidebar.selectbox("Materiale", series_options, index=default_idx)
y = data_source[material].astype(float)
y.name = material

n_obs = len(y)
max_reg = max_recommended_regressors(n_obs)
mtype = material_type(material)
family = material_family(material)
st.sidebar.markdown(f"**Osservazioni disponibili:** {n_obs} trimestri")
st.sidebar.markdown(f"**Regressori consigliati (max):** {max_reg}")
st.sidebar.markdown(f"**Tipo materiale:** {TYPE_LABELS.get(mtype, mtype)}")

pestel = build_pestel_quarterly(y.index)
pestel["prezzo_materiale"] = prezzo_q.reindex(y.index)

fit_regressor_options = {
    "prezzo_materiale": "Prezzo materiale (EUR/kg)",
    "dummy_ppwr_forza": "PPWR in vigore (dummy dal 02/2025)",
}

st.sidebar.subheader("Regressori esogeni stimati nel modello")
selected_regressors = []
for key, label in fit_regressor_options.items():
    cat = PESTEL_CATEGORY.get(key, "Economico")
    default = key == "prezzo_materiale"
    if st.sidebar.checkbox(f"{label}  · {cat}", value=default, key=f"reg_{key}"):
        selected_regressors.append(key)

if len(selected_regressors) > max_reg:
    st.sidebar.warning(
        f"{len(selected_regressors)} regressori selezionati, ma con {n_obs} osservazioni se ne "
        f"consigliano al massimo {max_reg}. Rischio di overfitting: i risultati potrebbero non "
        f"generalizzare bene."
    )

st.sidebar.subheader("Orizzonte di previsione")
horizon_years = st.sidebar.slider("Anni di previsione", 1, 5, 3)
horizon_quarters = horizon_years * 4

st.sidebar.subheader("Sensibilità della previsione")
reattivita = st.sidebar.slider(
    "Reattività ai fattori di mercato (%)", 0, 100, 100, step=10,
    help=(
        "0% = previsione ridotta a una retta di tendenza lineare (nessuna stagionalità, nessun "
        "effetto di regressori o scenari PESTEL). 100% = previsione così com'è stimata dal "
        "modello, con i regressori e le leve di scenario a piena intensità."
    ),
)

st.sidebar.subheader("Scenari PESTEL (aggiustamento post-stima)")
st.sidebar.caption(
    "Aggiustamenti applicati alla previsione statistica, non stimati nel modello. Il segno "
    "viene invertito automaticamente per i materiali riciclati."
)

dairy_growth_pct = st.sidebar.slider(
    "Crescita mercato lattiero-caseario (%/anno)", 0.0, 15.0, DAIRY_GROWTH_RATE_ANNUAL, step=0.25,
    help=(
        "Crescita composta annua applicata a tutta la domanda (vergine e riciclato in egual "
        "misura). Default = media crescita consumi yogurt (+5,4%) e formaggi freschi (+4,1%), "
        "fonte Ismea Mercati 2025 — le due categorie che usano tipicamente vaschette PET/PS."
    ),
)

s0 = current_recycled_share(consumi_q, family) if family in ("PET", "PS") else None
default_target = PPWR_RECYCLED_TARGET_2030.get(family, 20.0)
if s0 is not None:
    st.sidebar.markdown(f"**Quota riciclato attuale ({family}, ultimi 4 trim.):** {s0*100:.1f}%")
    target_pct = st.sidebar.slider(
        f"Obiettivo minimo riciclato {family} al {PPWR_TARGET_PERIOD.year} (PPWR), %",
        0, 60, int(default_target),
        help="Regolamento UE 2025/40, Allegato II: 30% PET / 10% PS entro il 1/1/2030.",
    )
else:
    target_pct = default_target
    st.sidebar.caption(f"Nessun obbligo PPWR di contenuto riciclato individuato per {family}.")

plastic_tax_scenario = st.sidebar.checkbox("Simula introduzione Plastic Tax IT dal 01/2027", value=False)
plastic_tax_impact = (
    st.sidebar.slider("Impatto Plastic Tax IT, % su domanda vergine", -30, 0, -10)
    if plastic_tax_scenario
    else 0
)

st.subheader(f"Storico trimestrale — {material}")
hist_fig = go.Figure()
hist_fig.add_trace(
    go.Scatter(
        x=y.index.to_timestamp().astype(str), y=y.values, mode="lines+markers", name=material
    )
)
hist_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="tonnellate")
st.plotly_chart(hist_fig, use_container_width=True)

annual_hist = y.copy()
annual_hist.index = annual_hist.index.year
annual_hist = annual_hist.groupby(level=0).sum()
with st.expander("Totali annuali storici (somma trimestri disponibili)"):
    st.dataframe(annual_hist.rename("tonnellate totali").to_frame().style.format("{:,.1f}"))

run = st.button("Genera previsione", type="primary")

if run:
    with st.spinner("Stima modello SARIMAX e backtest in corso..."):
        exog = pestel[selected_regressors] if selected_regressors else None

        best = grid_search_sarimax(y, exog)
        n_test = max(2, min(4, n_obs // 5))
        bt = backtest(y, exog, best["order"], best["seasonal_order"], n_test=n_test)

        future_idx = pd.period_range(y.index[-1] + 1, periods=horizon_quarters, freq="Q")
        future_pestel = build_pestel_quarterly(future_idx)
        future_pestel["prezzo_materiale"] = prezzo_q.iloc[-1]
        future_exog = future_pestel[selected_regressors] if selected_regressors else None

        res, mean, ci = forecast_future(
            y, exog, best["order"], best["seasonal_order"], horizon_quarters, future_exog
        )

        ppwr_factor = ppwr_content_factor(mtype, s0, target_pct, y.index[-1], mean.index)
        dairy_factor = dairy_growth_factor(dairy_growth_pct, y.index[-1], mean.index)

        tax_sign = {"vergine": 1, "riciclato": -1}.get(mtype, 0)
        effective_tax = plastic_tax_impact * tax_sign
        tax_start = pd.Timestamp("2027-01-01")
        tax_factor = pd.Series(1.0, index=mean.index)
        if plastic_tax_scenario:
            for dt in mean.index:
                if dt >= tax_start:
                    tax_factor.loc[dt] = 1 + effective_tax / 100

        adj = ppwr_factor * tax_factor * dairy_factor
        mean_adj = mean * adj
        ci_adj = ci.multiply(adj, axis=0)

        mean_final, ci_final = apply_reactivity(mean_adj, ci_adj, y, reattivita)

        explanation = build_explanation(
            material=material, y=y, mean=mean_final, res=res, exog_cols=selected_regressors,
            future_exog=future_exog, s0=s0, target_pct=target_pct, ppwr_factor=ppwr_factor,
            dairy_growth_pct=dairy_growth_pct, plastic_tax_scenario=plastic_tax_scenario,
            plastic_tax_impact=plastic_tax_impact, bt_mape=bt["mape"], reactivity_pct=reattivita,
        )

        st.session_state["result"] = dict(
            best=best, bt=bt, mean=mean_final, ci=ci_final, material=material, y=y, explanation=explanation
        )

if "result" in st.session_state and st.session_state["result"]["material"] == material:
    r = st.session_state["result"]
    best, bt, mean, ci = r["best"], r["bt"], r["mean"], r["ci"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Ordine SARIMA scelto", f"{best['order']} x {best['seasonal_order']}")
    c2.metric("MAPE backtest", f"{bt['mape']:.1f}%")
    c3.metric("RMSE backtest", f"{bt['rmse']:,.1f} t")

    st.subheader("Previsione")
    fc_fig = go.Figure()
    fc_fig.add_trace(
        go.Scatter(x=y.index.to_timestamp().astype(str), y=y.values, mode="lines+markers", name="Storico")
    )
    lower_col = [c for c in ci.columns if c.startswith("lower")][0]
    upper_col = [c for c in ci.columns if c.startswith("upper")][0]
    fc_x = mean.index.astype(str)
    fc_fig.add_trace(
        go.Scatter(
            x=list(fc_x) + list(fc_x[::-1]),
            y=list(ci[upper_col]) + list(ci[lower_col][::-1]),
            fill="toself",
            fillcolor="rgba(99,110,250,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Intervallo 80%",
            showlegend=True,
        )
    )
    fc_fig.add_trace(go.Scatter(x=fc_x, y=mean.values, mode="lines+markers", name="Previsione"))
    fc_fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10), yaxis_title="tonnellate")
    st.plotly_chart(fc_fig, use_container_width=True)

    st.subheader("Perché questa previsione?")
    st.markdown(r["explanation"])

    st.subheader("Totali annuali — storico + previsione")
    st.caption(
        "La colonna 'trimestri' indica quanti trimestri compongono ciascun totale annuale: "
        "gli anni con meno di 4 trimestri sono parziali (inizio/fine serie) e NON vanno "
        "confrontati direttamente con gli anni completi."
    )
    hist_y = r["y"]
    hist_annual = hist_y.groupby(hist_y.index.year).sum().rename("t (storico)")
    hist_n = hist_y.groupby(hist_y.index.year).size().rename("trimestri (storico)")

    fc_annual = mean.groupby(mean.index.year).sum().rename("t (previsione)")
    fc_n = mean.groupby(mean.index.year).size().rename("trimestri (previsione)")

    annual_table = pd.concat([hist_n, hist_annual, fc_n, fc_annual], axis=1)
    annual_table = annual_table[["trimestri (storico)", "t (storico)", "trimestri (previsione)", "t (previsione)"]]
    st.dataframe(
        annual_table.style.format(
            {"t (storico)": "{:,.1f}", "t (previsione)": "{:,.1f}",
             "trimestri (storico)": "{:.0f}", "trimestri (previsione)": "{:.0f}"},
            na_rep="—",
        )
    )

    csv = mean.rename("previsione_t").to_frame().join(ci).to_csv().encode("utf-8")
    st.download_button("Scarica previsione trimestrale (CSV)", csv, file_name=f"previsione_{material}.csv")

    with st.expander("Dettaglio backtest (ultimi trimestri, 1 passo avanti)"):
        bt_df = pd.DataFrame(
            {"actual": bt["actuals"], "pred": bt["preds"]},
            index=bt["index"].astype(str),
        )
        st.dataframe(bt_df.style.format("{:,.1f}"))

st.divider()
st.caption(
    "Fonti PESTEL: Regolamento UE 2025/40 (PPWR) per date normative e target di contenuto "
    "riciclato; L. Bilancio 2026 per la Plastic Tax italiana; Ismea Mercati e Mordor "
    "Intelligence (2025) per il contesto di mercato lattiero-caseario. Dati di consumo e "
    "prezzo materiale: registrazioni interne aziendali."
)
