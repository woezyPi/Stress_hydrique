"""
Téléchargement des données piézométriques Hub'Eau (BRGM).

API Hub'Eau — Niveaux des nappes d'eau souterraine :
    https://hubeau.eaufrance.fr/page/api-piezometrie

Principe :
    1. Lister les stations piézométriques actives en France métro
    2. Télécharger les chroniques de niveaux récentes (12 derniers mois)
    3. Télécharger les chroniques historiques (baseline) pour chaque station
    4. Calculer l'anomalie piézométrique (niveau actuel vs moyenne historique)

L'anomalie est ensuite interpolée spatialement et fusionnée avec le GWS
Aqueduct + BRGM dans groundwater_model.py.

Usage :
    downloader = HubEauPiezoDownloader(dest_dir=Path("data/raw/hubeau"))
    stations_gdf = downloader.fetch_stations()
    anomaly_gdf = downloader.compute_anomalies(year=2024)
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from loguru import logger
from shapely.geometry import Point


# Hub'Eau Piézométrie API
_BASE_URL = "https://hubeau.eaufrance.fr/api/v1/niveaux_nappes"
_STATIONS_ENDPOINT = f"{_BASE_URL}/stations"
_CHRONIQUES_ENDPOINT = f"{_BASE_URL}/chroniques"

# France métropolitaine bbox
_BBOX_FRANCE = [-5.2, 41.2, 9.8, 51.2]

# Nombre minimum de mesures pour considérer une station fiable
_MIN_MESURES_BASELINE = 120   # ~10 ans de mensuelles
_MIN_MESURES_RECENTES = 6     # au moins 6 mois sur les 12 derniers


class HubEauPiezoDownloader:
    """
    Télécharge et traite les données piézométriques Hub'Eau.

    Args:
        dest_dir: Répertoire de cache local
        baseline_years: Nombre d'années pour la baseline historique (défaut 20)
    """

    def __init__(
        self,
        dest_dir: Path,
        baseline_years: int = 20,
    ):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_years = baseline_years
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ──────────────────────────────────────────────────────────────────────
    # 1. Récupérer les stations actives
    # ──────────────────────────────────────────────────────────────────────

    def fetch_stations(self, year: int = 2022, force: bool = False) -> gpd.GeoDataFrame:
        """
        Récupère toutes les stations piézométriques actives en France métro.

        Filtre :
            - bbox France
            - nb_mesures_piezo >= MIN_MESURES_BASELINE
            - date_fin_mesure active pendant l'année demandée (year - 2)

        Args:
            year: Année de référence pour le filtre d'activité des stations

        Returns:
            GeoDataFrame avec colonnes : code_bss, nom_commune, geometry, ...
        """
        cache_path = self.dest_dir / f"stations_piezo_{year}.parquet"
        if cache_path.exists() and not force:
            gdf = gpd.read_parquet(str(cache_path))
            logger.info(f"Stations piézo en cache : {len(gdf)} stations")
            return gdf

        logger.info("Hub'Eau : récupération des stations piézométriques...")

        all_stations = []
        page = 1
        page_size = 1000

        while True:
            params = {
                "format": "json",
                "size": page_size,
                "page": page,
                "bbox": ",".join(str(x) for x in _BBOX_FRANCE),
            }
            resp = self._get(_STATIONS_ENDPOINT, params)
            if resp is None:
                break
            data = resp.get("data", [])
            if not data:
                break

            all_stations.extend(data)
            logger.debug(f"  Page {page} : {len(data)} stations")

            if len(data) < page_size:
                break
            page += 1
            time.sleep(0.2)  # politesse API

        if not all_stations:
            logger.warning("Aucune station piézométrique trouvée")
            return gpd.GeoDataFrame()

        df = pd.DataFrame(all_stations)
        logger.info(f"Hub'Eau : {len(df)} stations brutes récupérées")

        # Filtrer : assez de mesures + données récentes
        cutoff_date = f"{year - 2}-01-01"
        df = df[
            (df["nb_mesures_piezo"].fillna(0) >= _MIN_MESURES_BASELINE)
            & (df["date_fin_mesure"] >= cutoff_date)
        ].copy()

        # Construire le GeoDataFrame
        geometry = [
            Point(row["x"], row["y"])
            if pd.notna(row.get("x")) and pd.notna(row.get("y"))
            else None
            for _, row in df.iterrows()
        ]
        gdf = gpd.GeoDataFrame(
            df,
            geometry=geometry,
            crs="EPSG:4326",
        ).dropna(subset=["geometry"])

        # Colonnes utiles
        keep_cols = [
            "code_bss", "bss_id", "nom_commune", "code_departement",
            "nom_departement", "altitude_station", "nb_mesures_piezo",
            "date_debut_mesure", "date_fin_mesure",
            "profondeur_investigation", "libelle_pe",
            "geometry",
        ]
        gdf = gdf[[c for c in keep_cols if c in gdf.columns]]

        # Cache
        gdf.to_parquet(str(cache_path))
        logger.success(
            f"Hub'Eau : {len(gdf)} stations piézo actives "
            f"(≥{_MIN_MESURES_BASELINE} mesures, données < 2 ans)"
        )
        return gdf

    # ──────────────────────────────────────────────────────────────────────
    # 2. Télécharger les chroniques et calculer les anomalies
    # ──────────────────────────────────────────────────────────────────────

    def compute_anomalies(
        self,
        stations_gdf: Optional[gpd.GeoDataFrame] = None,
        year: Optional[int] = None,
        force: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Calcule l'anomalie piézométrique par station.

        Anomalie = (niveau_moyen_récent - moyenne_historique) / std_historique

        Normalisée ensuite vers [0, 1] :
            0 = nappe très haute (aucun stress)
            1 = nappe très basse (stress maximal)

        Args:
            stations_gdf: GeoDataFrame des stations (None → fetch_stations())
            year: Année de référence (None → année courante)
            force: Recalculer même si cache existe

        Returns:
            GeoDataFrame avec colonnes :
                code_bss, geometry, niveau_moyen, niveau_baseline,
                std_baseline, anomaly_zscore, piezo_stress_norm
        """
        if year is None:
            raise ValueError(
                "year est requis pour compute_anomalies(). "
                "Spécifier l'année explicitement (ex: year=2022)."
            )

        cache_path = self.dest_dir / f"piezo_anomalies_{year}.parquet"
        if cache_path.exists() and not force:
            gdf = gpd.read_parquet(str(cache_path))
            logger.info(f"Anomalies piézo {year} en cache : {len(gdf)} stations")
            return gdf

        if stations_gdf is None:
            stations_gdf = self.fetch_stations()

        if stations_gdf.empty:
            return gpd.GeoDataFrame()

        logger.info(
            f"Calcul des anomalies piézométriques {year} "
            f"({len(stations_gdf)} stations)..."
        )

        # Dates
        date_recent_start = f"{year}-01-01"
        date_recent_end = f"{year}-12-31"
        baseline_start = f"{year - self.baseline_years}-01-01"
        baseline_end = f"{year - 1}-12-31"

        results = []
        n_total = len(stations_gdf)

        for i, (_, station) in enumerate(stations_gdf.iterrows()):
            code_bss = station["code_bss"]

            if (i + 1) % 100 == 0:
                logger.info(f"  Progression : {i+1}/{n_total} stations")

            try:
                anomaly = self._compute_station_anomaly(
                    code_bss=code_bss,
                    date_recent_start=date_recent_start,
                    date_recent_end=date_recent_end,
                    baseline_start=baseline_start,
                    baseline_end=baseline_end,
                )
                if anomaly is not None:
                    anomaly["geometry"] = station.geometry
                    results.append(anomaly)
            except Exception as e:
                logger.debug(f"  Station {code_bss} échouée : {e}")

            # Politesse API : 5 req/s max
            if (i + 1) % 5 == 0:
                time.sleep(1.0)

        if not results:
            logger.warning("Aucune anomalie piézo calculée")
            return gpd.GeoDataFrame()

        gdf = gpd.GeoDataFrame(results, crs="EPSG:4326")

        # Normaliser le z-score en stress [0, 1]
        # z-score négatif = nappe basse = stress élevé
        # On inverse : stress = sigmoid(-z_score)
        # z=-2 → stress≈0.88, z=0 → stress=0.50, z=+2 → stress≈0.12
        gdf["piezo_stress_norm"] = (
            1.0 / (1.0 + np.exp(0.8 * gdf["anomaly_zscore"]))
        ).clip(0.0, 1.0)

        # Cache
        gdf.to_parquet(str(cache_path))

        n_stress = (gdf["piezo_stress_norm"] >= 0.6).sum()
        logger.success(
            f"Anomalies piézométriques {year} :\n"
            f"  Stations valides  : {len(gdf)}\n"
            f"  Z-score moyen     : {gdf['anomaly_zscore'].mean():.2f}\n"
            f"  Stress moyen      : {gdf['piezo_stress_norm'].mean():.2f}\n"
            f"  Stations en stress (≥0.6) : {n_stress} ({n_stress/len(gdf)*100:.0f}%)"
        )
        return gdf

    def _compute_station_anomaly(
        self,
        code_bss: str,
        date_recent_start: str,
        date_recent_end: str,
        baseline_start: str,
        baseline_end: str,
    ) -> Optional[dict]:
        """Calcule l'anomalie pour une station."""

        # Chronique récente (année en cours)
        recent = self._fetch_chronique(
            code_bss, date_recent_start, date_recent_end
        )
        if recent is None or len(recent) < _MIN_MESURES_RECENTES:
            return None

        # Chronique baseline (historique)
        baseline = self._fetch_chronique(
            code_bss, baseline_start, baseline_end
        )
        if baseline is None or len(baseline) < _MIN_MESURES_BASELINE:
            return None

        niveau_recent = recent["niveau_nappe_eau"].mean()
        niveau_baseline = baseline["niveau_nappe_eau"].mean()
        std_baseline = baseline["niveau_nappe_eau"].std()

        if std_baseline < 0.01:  # nappe quasi constante
            return None

        z_score = (niveau_recent - niveau_baseline) / std_baseline

        return {
            "code_bss": code_bss,
            "niveau_moyen_recent": round(float(niveau_recent), 2),
            "niveau_baseline": round(float(niveau_baseline), 2),
            "std_baseline": round(float(std_baseline), 2),
            "anomaly_zscore": round(float(z_score), 3),
            "n_mesures_recent": len(recent),
            "n_mesures_baseline": len(baseline),
        }

    def _fetch_chronique(
        self,
        code_bss: str,
        date_debut: str,
        date_fin: str,
    ) -> Optional[pd.DataFrame]:
        """Récupère les mesures piézo pour une station sur une période."""

        all_data = []
        page = 1
        page_size = 2000

        while True:
            params = {
                "code_bss": code_bss,
                "date_debut_mesure": date_debut,
                "date_fin_mesure": date_fin,
                "size": page_size,
                "page": page,
                "format": "json",
            }
            resp = self._get(_CHRONIQUES_ENDPOINT, params)
            if resp is None:
                return None

            data = resp.get("data", [])
            if not data:
                break

            all_data.extend(data)
            if len(data) < page_size:
                break
            page += 1

        if not all_data:
            return None

        df = pd.DataFrame(all_data)
        df["niveau_nappe_eau"] = pd.to_numeric(
            df["niveau_nappe_eau"], errors="coerce"
        )
        df = df.dropna(subset=["niveau_nappe_eau"])
        return df

    def _get(self, url: str, params: dict) -> Optional[dict]:
        """GET avec retry."""
        for attempt in range(3):
            try:
                resp = self._session.get(url, params=params, timeout=30)
                if resp.status_code in (200, 206):
                    return resp.json()
                if resp.status_code == 429:
                    time.sleep(5)
                    continue
                logger.debug(f"HTTP {resp.status_code} pour {url}")
                return None
            except requests.RequestException as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    logger.debug(f"Requête échouée après 3 tentatives : {e}")
                    return None
        return None
