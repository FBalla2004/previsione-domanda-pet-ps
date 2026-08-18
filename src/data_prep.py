"""Caricamento e aggregazione trimestrale dei dati storici di consumo e prezzo materiali."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MATERIAL_COLUMNS = {
    "PET VG": 1,
    "SCAGLIA PET": 4,
    "DA MAC PET": 5,
    "PS": 8,
    "PS SCAGLIA": 9,
    "PE": 12,
}

MATERIAL_FAMILY = {
    "PET VG": "PET",
    "SCAGLIA PET": "PET",
    "DA MAC PET": "PET",
    "PS": "PS",
    "PS SCAGLIA": "PS",
    "PE": "PE",
}

# Tipo di materiale, usato per orientare correttamente le leve di scenario normative
# (PPWR/Plastic Tax spingono la domanda VERSO il riciclato e VIA dal vergine: l'effetto
# ha quindi segno opposto tra le due categorie). "DA MAC PET" e' macinato di scarto di
# produzione rilavorato internamente: non e' chiaramente equiparabile al riciclato
# post-consumo rilevante per gli obblighi normativi PPWR, quindi resta neutro (0).
MATERIAL_TYPE = {
    "PET VG": "vergine",
    "SCAGLIA PET": "riciclato",
    "DA MAC PET": "macinato_interno",
    "PS": "vergine",
    "PS SCAGLIA": "riciclato",
    "PE": "vergine",
}

_MESI_IT = {
    "gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5, "giugno": 6,
    "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10, "novembre": 11, "dicembre": 12,
}


def load_consumi_mensili() -> pd.DataFrame:
    """Legge storico mp.xlsx e restituisce consumi mensili (kg) per sottocategoria materiale."""
    raw = pd.read_excel(DATA_DIR / "storico mp.xlsx", sheet_name="Foglio1", header=None)
    dates = pd.to_datetime(raw.iloc[1:, 0], errors="coerce")

    out = {}
    for name, col in MATERIAL_COLUMNS.items():
        out[name] = pd.to_numeric(raw.iloc[1:, col], errors="coerce").values

    df = pd.DataFrame(out, index=dates)
    df = df[df.index.notna()]
    df.index.name = "data"
    df = df.fillna(0.0)
    return df.sort_index()


def load_prezzo_mensile() -> pd.Series:
    """Legge dati_prezzi_kg.xlsx e restituisce il prezzo medio mensile (EUR/kg)."""
    raw = pd.read_excel(DATA_DIR / "dati_prezzi_kg.xlsx", sheet_name="Dati")
    raw.columns = ["mese", "anno", "prezzo"]

    def parse_mese(m):
        m = str(m).strip().lower()
        m = re.sub(r"\s+(inizio|fine)$", "", m)
        return _MESI_IT.get(m)

    raw["mese_num"] = raw["mese"].apply(parse_mese)
    raw = raw.dropna(subset=["mese_num", "anno", "prezzo"])
    raw["data"] = pd.to_datetime(dict(year=raw["anno"].astype(int), month=raw["mese_num"].astype(int), day=1))

    prezzo = raw.groupby("data")["prezzo"].mean().sort_index()
    prezzo.name = "prezzo_materiale"
    return prezzo


def to_quarterly(monthly_df: pd.DataFrame, agg: str = "sum") -> pd.DataFrame:
    """Aggrega un dataframe mensile (index=datetime) a trimestri (index=PeriodIndex freq=Q)."""
    q = monthly_df.copy()
    q.index = q.index.to_period("Q")
    return q.groupby(level=0).agg(agg)


def _last_complete_quarter(monthly_index: pd.DatetimeIndex) -> pd.Period:
    """Ultimo trimestre per cui il mese piu' recente disponibile ne copre la fine.

    Non filtra buchi interni nella serie (es. un mese mancante per un errore di
    inserimento dati): serve solo a scartare il trimestre finale ancora in corso.
    """
    last_month = monthly_index.max()
    last_q = last_month.to_period("Q")
    q_end_month = last_q.end_time.month
    if last_month.month < q_end_month:
        return last_q - 1
    return last_q


KG_PER_TONNE = 1000.0


def build_dataset():
    """Ritorna (consumi_trimestrali_per_famiglia, consumi_trimestrali_per_sottocategoria, prezzo_trimestrale).

    I consumi sono in TONNELLATE (il dato grezzo in Excel e' in kg, convertito qui una volta
    per tutte cosi' che modello, metriche e dashboard lavorino tutti nella stessa unita').
    Il prezzo materiale resta in EUR/kg, unita' standard per i prezzi di resina.

    I trimestri incompleti (mesi mancanti, tipicamente l'ultimo trimestre in corso) vengono
    esclusi per evitare di introdurre un crollo artificiale nella serie storica.
    """
    consumi_m = load_consumi_mensili()
    prezzo_m = load_prezzo_mensile()

    consumi_q = to_quarterly(consumi_m, agg="sum")
    prezzo_q = to_quarterly(prezzo_m.to_frame(), agg="mean")["prezzo_materiale"]

    last_ok_q = min(_last_complete_quarter(consumi_m.index), _last_complete_quarter(prezzo_m.index))

    common_idx = consumi_q.index.intersection(prezzo_q.index)
    common_idx = common_idx[common_idx <= last_ok_q]
    consumi_q = consumi_q.loc[common_idx] / KG_PER_TONNE
    prezzo_q = prezzo_q.loc[common_idx]

    famiglia_q = pd.DataFrame(index=consumi_q.index)
    for fam in ("PET", "PS", "PE"):
        cols = [c for c, f in MATERIAL_FAMILY.items() if f == fam]
        famiglia_q[fam] = consumi_q[cols].sum(axis=1)

    return famiglia_q, consumi_q, prezzo_q


def current_recycled_share(consumi_q: pd.DataFrame, family: str, n_quarters: int = 4) -> float | None:
    """Quota di riciclato (SCAGLIA) sul totale della famiglia, media sugli ultimi n_quarters.

    Il macinato interno (DA MAC PET) e' escluso dal numeratore (non e' chiaramente
    equiparabile al riciclato post-consumo rilevante per l'obbligo PPWR) ma resta nel
    denominatore in quanto input di produzione della famiglia. Ritorna None se la famiglia
    non ha una sottocategoria "riciclato" definita (es. PE).
    """
    cols_tot = [c for c, f in MATERIAL_FAMILY.items() if f == family]
    cols_ric = [c for c in cols_tot if MATERIAL_TYPE.get(c) == "riciclato"]
    if not cols_ric:
        return None
    tot = consumi_q[cols_tot].iloc[-n_quarters:].sum().sum()
    ric = consumi_q[cols_ric].iloc[-n_quarters:].sum().sum()
    if tot == 0:
        return None
    return float(ric / tot)
