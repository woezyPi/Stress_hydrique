"""
Vague 2 — etape 1 : extraire le niveau Propluvia par departement
au 15 juillet 2022, a partir du CSV VigiEau.

Strategie : pour chaque departement, on prend le niveau le plus severe
parmi tous les arretes actifs ce jour-la. C'est conservative et
correspond a la lecture operationnelle "le pire qui s'applique".

Inputs  : data/propluvia/arretes.csv
Outputs : data/propluvia/dept_niveau_2022_07_15.csv
          (colonnes : code_dept, niveau_text, niveau_int, n_arretes)
"""

from __future__ import annotations
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_IN = ROOT / "data" / "propluvia" / "arretes.csv"
CSV_OUT = ROOT / "data" / "propluvia" / "dept_niveau_2022_07_15.csv"

TARGET_DATE = pd.Timestamp("2022-07-15")

NIVEAU_MAP = {
    "vigilance": 1,
    "alerte": 2,
    "alerte_renforcee": 3,
    "crise": 4,
}


def parse_niveau(raw) -> list[str]:
    """Le champ peut etre un str simple ou un JSON list-string."""
    if pd.isna(raw):
        return []
    s = str(raw).strip()
    if s.startswith("["):
        try:
            return [x for x in json.loads(s) if x in NIVEAU_MAP]
        except json.JSONDecodeError:
            return []
    return [s] if s in NIVEAU_MAP else []


def severity_max(niveaux: list[str]) -> int:
    if not niveaux:
        return 0
    return max(NIVEAU_MAP[n] for n in niveaux)


def main():
    df = pd.read_csv(CSV_IN, low_memory=False)
    df["date_debut"] = pd.to_datetime(df["date_debut"], errors="coerce")
    df["date_fin"] = pd.to_datetime(df["date_fin"], errors="coerce")

    active = df[(df["date_debut"] <= TARGET_DATE) & (df["date_fin"] >= TARGET_DATE)].copy()
    print(f"Arretes actifs au {TARGET_DATE.date()} : {len(active)}")

    active["niveaux_list"] = active["zones_alerte.niveau_gravite"].apply(parse_niveau)
    active["niveau_int_arrete"] = active["niveaux_list"].apply(severity_max)

    # filtre arretes sans niveau valide
    active = active[active["niveau_int_arrete"] > 0]

    # Agregation par departement : on garde le max
    grouped = (
        active.groupby("departement")
              .agg(niveau_int=("niveau_int_arrete", "max"),
                   n_arretes=("id", "count"))
              .reset_index()
              .rename(columns={"departement": "code_dept"})
    )
    inv_map = {v: k for k, v in NIVEAU_MAP.items()}
    grouped["niveau_text"] = grouped["niveau_int"].map(inv_map)
    grouped = grouped[["code_dept", "niveau_text", "niveau_int", "n_arretes"]]
    grouped["code_dept"] = grouped["code_dept"].astype(str).str.zfill(2)

    grouped.to_csv(CSV_OUT, index=False)
    print(f"Departements couverts : {len(grouped)} / 96 (metropole)")
    print(f"\nDistribution des niveaux :")
    print(grouped["niveau_text"].value_counts().to_string())
    print(f"\nEcrit : {CSV_OUT}")
    print("\nTop 10 :")
    print(grouped.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
