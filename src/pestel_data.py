"""
Indicatori PESTEL esogeni per il modello di previsione della domanda PET/PS.

Fonti e metodologia
--------------------
ECONOMICO
- Prezzo materiale (EUR/kg): dato interno aziendale (dati_prezzi_kg.xlsx), unico regressore
  economico stimato nel SARIMAX. Prezzo Brent e cambio EUR/USD sono stati esclusi su
  richiesta: per un trasformatore che acquista direttamente resina (non greggio) il prezzo
  materiale gia' internalizza questi effetti macro, e includerli separatamente avrebbe
  saturato il budget di regressori stimabili (max ~2 con 19 osservazioni) senza aggiungere
  potere esplicativo diretto.

- Mercato di destinazione (lattiero-caseario): l'azienda produce prevalentemente imballaggi
  per il settore lattiero-caseario (vaschette/vasetti PET-PS per yogurt, formaggi freschi).
  Non esiste una serie trimestrale pubblica granulare integrabile nel SARIMAX, ma i dati
  di mercato disponibili (fonte principale: Ismea Mercati, report "Consumi alimentari delle
  famiglie - primo semestre 2025", pubblicato novembre 2025 — il piu' recente disponibile)
  sono un riferimento qualitativo e quantitativo importante:
  - Spesa delle famiglie italiane in lattiero-caseario: +6,3% nel primo semestre 2025.
  - Consumi in volume (H1 2025 vs H1 2024): yogurt +5,7%, formaggi freschi +5,3% (categorie
    che usano tipicamente vaschette termoformate PET/PS) — mentre latte fresco -3,9% e
    formaggi duri sostanzialmente stabili (-0,4%), packaging diverso, meno rilevante per
    PET/PS.
  - Mercato lattiero-caseario italiano: CAGR atteso 5,83% fino al 2031 (Mordor Intelligence),
    trainato da mozzarella e Parmigiano-Reggiano.
  - La plastica resta il materiale predominante negli imballaggi lattiero-caseari (quota
    36,4% nel 2025, +2,3% attesa fino al 2026), ma con pressione crescente della GDO e del
    PPWR verso formati "senza vassoio" e alternative in carta.
  Lettura netta: la crescita dei consumi di yogurt/formaggi freschi e' un fattore di
  sostegno alla domanda di imballaggio PET/PS; la spinta a formati senza vassoio/carta e'
  un fattore di freno strutturale, distinto e aggiuntivo rispetto all'obbligo di contenuto
  riciclato PPWR. La crescita di volume (media yogurt +5,7% / formaggi freschi +5,3% =
  ~5,5%/anno, dato H1 2025, il piu' aggiornato reperibile) e' quantificata come LEVA DI
  SCENARIO (DAIRY_GROWTH_RATE_ANNUAL, regolabile in dashboard anche in negativo per
  simulare una contrazione del mercato di destinazione), applicata in egual misura a
  vergine e riciclato: non e' un regressore stimato nel SARIMAX (nessuna serie storica
  trimestrale disponibile per farlo rientrare nella stima), ma una crescita/contrazione
  composta imposta esplicitamente sulla previsione.

POLITICO / AMBIENTALE
- dummy_ppwr_forza: 0 prima di febbraio 2025, 1 da febbraio 2025 in poi (variabile stimabile
  nel SARIMAX, ha varianza nel campione). Fonte: Regolamento UE 2025/40 (PPWR), adottato
  19/12/2024, in vigore dall'11/02/2025, applicazione generale dal 12/08/2026.

- Obbligo di contenuto riciclato (PCR) PPWR — leva di scenario, non regressore stimato
  (varianza storica nulla, l'obbligo scatta nel periodo di previsione):
  * PET a contatto sensibile: minimo 30% PCR entro il 1/1/2030, 50% entro il 1/1/2040.
  * Altri polimeri a contatto sensibile (PS incluso): minimo 10% PCR entro il 1/1/2030,
    25% entro il 1/1/2040.
  * Imballaggi in plastica non a contatto sensibile: 35% entro il 1/1/2030, 65% entro il
    1/1/2040.
  Fonte: Regolamento UE 2025/40 (PPWR), Allegato II.
  Nella dashboard questo si traduce in una transizione lineare, per la famiglia di
  materiale selezionata, dalla quota di riciclato osservata negli ultimi 4 trimestri fino
  alla quota-obiettivo 2030, applicata come fattore moltiplicativo relativo alla previsione
  statistica di ciascuna sottocategoria (in aumento per il riciclato, in calo per il
  vergine, a parita' di volume totale di famiglia).

- plastic_tax_IT: costante a 0 nell'intero periodo storico. La plastic tax italiana
  (L.160/2019) e' stata rinviata ripetutamente (ultimo rinvio: 1/1/2027, L. Bilancio 2026).
  Non ha varianza storica quindi non e' utilizzabile come regressore stimato, ma resta
  disponibile come leva di scenario opzionale nella dashboard.

SOCIALE / TECNOLOGICO
- Non sono state reperite serie trimestrali pubbliche affidabili e granulari per
  consapevolezza ambientale dei consumatori o capacita' di riciclo rPET nella finestra
  temporale richiesta, oltre al contesto di mercato lattiero-caseario sopra riportato.
  Questi fattori sono trattati QUALITATIVAMENTE nel testo di tesi (vedi 5.2 Limiti del
  lavoro) e non entrano come regressori numerici nel SARIMAX, per evitare di introdurre
  variabili proxy arbitrarie e non verificabili in un lavoro accademico. Questa e' una
  scelta metodologica esplicita, non una svista.
"""

import pandas as pd

_PPWR_FORZA_DATE = pd.Timestamp("2025-02-11")
_PLASTIC_TAX_IT_DATE = pd.Timestamp("2027-01-01")

# Obiettivi minimi di contenuto riciclato (PCR) PPWR al 2030, per famiglia di materiale
# (Regolamento UE 2025/40, Allegato II). Espressi in percentuale (0-100).
PPWR_RECYCLED_TARGET_2030 = {
    "PET": 30.0,
    "PS": 10.0,
}
PPWR_TARGET_PERIOD = pd.Period("2030Q1", freq="Q")

# Crescita annua composta del mercato di destinazione (lattiero-caseario), applicata in
# egual misura a vergine e riciclato. Media tra crescita consumi yogurt (+5,7%) e formaggi
# freschi (+5,3%), fonte Ismea Mercati, report primo semestre 2025 (pubblicato nov. 2025,
# il piu' aggiornato disponibile). Regolabile in dashboard anche in negativo.
DAIRY_GROWTH_RATE_ANNUAL = 5.5


def build_pestel_quarterly(q_index: pd.PeriodIndex) -> pd.DataFrame:
    """Costruisce il dataframe di regressori esogeni PESTEL (dummy) per i trimestri richiesti."""
    df = pd.DataFrame(index=q_index)
    q_start = q_index.to_timestamp(how="start")
    df["dummy_ppwr_forza"] = (q_start >= _PPWR_FORZA_DATE).astype(int)
    df["dummy_plastic_tax_it"] = (q_start >= _PLASTIC_TAX_IT_DATE).astype(int)
    return df


PESTEL_LABELS = {
    "dummy_ppwr_forza": "PPWR in vigore (dummy, dal 02/2025)",
    "dummy_plastic_tax_it": "Plastic tax IT (dummy scenario, dal 01/2027)",
    "prezzo_materiale": "Prezzo materiale (EUR/kg, dato interno)",
}

PESTEL_CATEGORY = {
    "prezzo_materiale": "Economico",
    "dummy_ppwr_forza": "Politico/Ambientale",
    "dummy_plastic_tax_it": "Politico/Ambientale",
}
