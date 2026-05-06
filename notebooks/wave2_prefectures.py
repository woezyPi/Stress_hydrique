"""
Vague 2 — etape 2 : recuperer les coordonnees des prefectures
de chaque departement via l'API geo.api.gouv.fr (publique, gratuite),
et joindre avec les niveaux Propluvia 2022-07-15.

Output : data/propluvia/sites_dept_2022.csv
   colonnes : code_dept, nom_dept, prefecture, lat, lon, niveau_int, niveau_text
"""

from __future__ import annotations
import requests
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROPLUVIA_CSV = ROOT / "data" / "propluvia" / "dept_niveau_2022_07_15.csv"
OUT_CSV = ROOT / "data" / "propluvia" / "sites_dept_2022.csv"

API_DEPTS = "https://geo.api.gouv.fr/departements?fields=nom,code,chefLieu"
API_COMMUNE = "https://geo.api.gouv.fr/communes/{code}?fields=nom,centre"


def fetch_departements() -> pd.DataFrame:
    print("GET", API_DEPTS)
    r = requests.get(API_DEPTS, timeout=30)
    r.raise_for_status()
    rows = []
    for d in r.json():
        rows.append({
            "code_dept": d["code"],
            "nom_dept": d["nom"],
            "code_chef_lieu": d["chefLieu"],
        })
    return pd.DataFrame(rows)


def fetch_commune_centre(code: str) -> tuple[str, float, float] | None:
    r = requests.get(API_COMMUNE.format(code=code), timeout=15)
    if r.status_code != 200:
        return None
    j = r.json()
    centre = j.get("centre")
    if not centre or "coordinates" not in centre:
        return None
    lon, lat = centre["coordinates"]
    return j["nom"], float(lat), float(lon)


def main():
    depts = fetch_departements()
    print(f"Departements API : {len(depts)}")

    propluvia = pd.read_csv(PROPLUVIA_CSV, dtype={"code_dept": str})
    print(f"Departements Propluvia : {len(propluvia)}")

    rows = []
    for _, d in depts.iterrows():
        info = fetch_commune_centre(d["code_chef_lieu"])
        if info is None:
            print(f"  [skip] {d['code_dept']} {d['nom_dept']} (pas de centroide)")
            continue
        nom, lat, lon = info
        rows.append({
            "code_dept": d["code_dept"],
            "nom_dept": d["nom_dept"],
            "prefecture": nom,
            "lat": lat,
            "lon": lon,
        })
    sites = pd.DataFrame(rows)
    print(f"Prefectures geocodees : {len(sites)}")

    merged = sites.merge(propluvia, on="code_dept", how="inner")
    print(f"Apres jointure Propluvia : {len(merged)} sites")

    merged = merged[
        ["code_dept", "nom_dept", "prefecture", "lat", "lon",
         "niveau_int", "niveau_text", "n_arretes"]
    ].sort_values("code_dept")

    merged.to_csv(OUT_CSV, index=False)
    print(f"\nEcrit : {OUT_CSV}")
    print("\nApercu :")
    print(merged.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
