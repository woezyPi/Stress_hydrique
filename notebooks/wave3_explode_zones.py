"""
Vague 3 — etape 1 : eclater arretes.csv en 1 ligne par zone d'alerte
active au 15 juillet 2022.

Chaque arrete dans arretes.csv peut couvrir plusieurs zones. Les colonnes
zones_alerte.id / niveau_gravite / communes sont alors des listes JSON
de meme longueur. On les zippe pour produire une ligne par (arrete x zone).

Output : data/propluvia/zones_2022_07_15.csv
   colonnes : zone_id, zone_nom, code_dept, niveau_int, niveau_text,
              communes (str CSV des codes INSEE), n_communes
"""

from __future__ import annotations
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_IN = ROOT / "data" / "propluvia" / "arretes.csv"
CSV_OUT = ROOT / "data" / "propluvia" / "zones_2022_07_15.csv"
TARGET = pd.Timestamp("2022-07-15")

NIVEAU_MAP = {
    "vigilance": 1,
    "alerte": 2,
    "alerte_renforcee": 3,
    "crise": 4,
}


def parse_list(raw):
    """Le champ peut etre str simple ou JSON list-string."""
    if pd.isna(raw):
        return []
    s = str(raw).strip()
    if s.startswith("["):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return []
    return [s]


def main():
    df = pd.read_csv(CSV_IN, low_memory=False)
    df["date_debut"] = pd.to_datetime(df["date_debut"], errors="coerce")
    df["date_fin"] = pd.to_datetime(df["date_fin"], errors="coerce")
    active = df[(df["date_debut"] <= TARGET) & (df["date_fin"] >= TARGET)].copy()
    print(f"Arretes actifs au {TARGET.date()} : {len(active)}")

    rows = []
    for _, r in active.iterrows():
        ids = parse_list(r["zones_alerte.id"])
        noms = parse_list(r["zones_alerte.nom"])
        niveaux = parse_list(r["zones_alerte.niveau_gravite"])
        communes = parse_list(r["zones_alerte.communes"])
        # all lists should be aligned in length
        n = max(len(ids), len(noms), len(niveaux), len(communes))
        for i in range(n):
            zid = ids[i] if i < len(ids) else None
            nom = noms[i] if i < len(noms) else None
            niv = niveaux[i] if i < len(niveaux) else None
            com = communes[i] if i < len(communes) else ""
            if niv not in NIVEAU_MAP:
                continue
            rows.append({
                "zone_id": zid,
                "zone_nom": nom,
                "code_dept": str(r["departement"]).zfill(2),
                "niveau_text": niv,
                "niveau_int": NIVEAU_MAP[niv],
                "communes": com,
            })

    z = pd.DataFrame(rows)
    print(f"Lignes (arrete x zone) avant dedup : {len(z)}")

    # dedup : pour une meme zone_id, on garde le niveau max (regle conservative)
    z = (z.sort_values("niveau_int", ascending=False)
           .drop_duplicates("zone_id", keep="first"))
    print(f"Zones uniques apres dedup : {len(z)}")

    # n_communes
    z["n_communes"] = z["communes"].apply(
        lambda s: len([c for c in str(s).split(",") if c.strip()])
    )

    z = z.sort_values(["code_dept", "zone_id"]).reset_index(drop=True)
    z.to_csv(CSV_OUT, index=False)
    print(f"\nEcrit : {CSV_OUT}")
    print(f"\nDistribution niveaux :")
    print(z["niveau_text"].value_counts().to_string())
    print(f"\nDistribution n_communes :")
    print(z["n_communes"].describe().round(1).to_string())
    print(f"\nApercu :")
    print(z[["code_dept", "zone_id", "zone_nom", "niveau_text", "n_communes"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
