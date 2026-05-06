"""
Téléchargeur BRGM BDLISA v3 — Base de Données des Limites des Systèmes Aquifères.

Source   : https://infoterre.brgm.fr / https://geoservices.brgm.fr
Licence  : Etalab Open License 2.0 (données publiques BRGM)

Données téléchargées :
    Entités hydrogéologiques BDLISA (masse d'eau souterraine)
    → polygones avec attributs productivité, type aquifère, état quantitatif

Structure BDLISA :
    - CODE_BSH    : identifiant unique de la masse d'eau
    - LB_NATUR    : nature de l'entité (Domaine/Système/Sous-système)
    - LB_TYPEHG   : type hydrogéologique (poreux libre, captif, fissuré, karstique...)
    - LB_ETATQ    : état quantitatif (Bon, Médiocre...)
    - LB_PRODUC   : classe de productivité
    - RECHARGE    : taux de recharge estimé (mm/an)

Méthode d'accès :
    1. WFS BRGM (requête OGC WFS 2.0) — recommandé, pas de compte requis
    2. ATOM feed (téléchargement direct GPKG/shapefile) — backup
    3. Infoterre API (JSON) — données tabulaires

Utilisation dans le modèle :
    La productivité des aquifères sert de proxy pour le stress en eau souterraine
    là où Aqueduct 4.0 a des données insuffisantes (résolution ~10 km).
    Combinée avec les données Aqueduct GWS, elle améliore la précision en France.

Usage :
    dl = BRGMDownloader(dest_dir=Path("data/raw/brgm"))
    gpkg_path = dl.download()
    gdf = dl.load()                # GeoDataFrame avec toutes les entités France
    stress = dl.compute_gw_stress(gdf)  # Series de stress normalisé [0-1]
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from loguru import logger
from shapely.geometry import box

# ──────────────────────────────────────────────────────────────────────────────
# URLs des services BRGM
# ──────────────────────────────────────────────────────────────────────────────

# WFS BRGM Géologie / BDLISA
_BRGM_WFS_URL = "https://geoservices.brgm.fr/geologie"
_BRGM_WFS_LAYER = "ms:BDLISA_ENTITES_HYDROGEOLOGIQUES_v3"

# ATOM feed (téléchargement direct GPKG) — service INSPIRE BRGM
_BRGM_ATOM_URL = (
    "https://inspire.brgm.fr/atom/download/fr-143582031-gdl-v3-BDLISA_ENTITES_HYDROGEOLOGIQUES_v3"
)

# Lien direct GeoPackage (officiel BRGM Infoterre, peut changer)
_BRGM_DIRECT_GPKG = (
    "https://infoterre.brgm.fr/telechargements/BDtopo/bdlisa_entites_hydrogeologiques_v3.gpkg"
)

# Couche BDLISA à lire (Niveau 3 = le plus fin, entités hydrogéologiques)
_BDLISA_LAYER = "ENTITES_NIVEAU3_ORDRES"

# Mapping milieueh (milieu hydrogéologique) → score de stress souterrain [0-1]
# Basé sur la classification BDLISA v3 et la productivité aquifère
# Référence : BRGM (2022) "Guide de lecture BDLISA v3"
# milieueh: 1=Poreux, 2=Fissuré, 3=Mixte, 4=Karstique, 5=Non aquifère, X=Indéterminé
_MILIEU_STRESS_MAP = {
    "1": 0.15,   # Poreux — très productif, faible stress
    "2": 0.50,   # Fissuré — productivité variable
    "3": 0.35,   # Mixte (poreux + fissuré)
    "4": 0.55,   # Karstique — vulnérable à la pollution et à l'épuisement
    "5": 0.85,   # Non aquifère — pas de ressource souterraine
    "X": 0.50,   # Indéterminé — valeur neutre
}

# Mapping themeeh (contexte géologique) → facteur correctif
# themeeh: 1=Alluvial, 2=Sédimentaire, 3=Socle/métamorphique, 4=Volcanique, 5=Karst
_THEME_FACTOR = {
    "1": 0.85,   # Alluvial → recharge facile, stress réduit
    "2": 0.95,   # Sédimentaire → productif
    "3": 1.15,   # Socle/métamorphique → peu productif, stress accru
    "4": 1.00,   # Volcanique → variable
    "5": 1.10,   # Karst → vulnérable
}

# Legacy mapping (pour compatibilité avec anciennes données WFS)
_TYPE_STRESS_MAP = {
    "Domaine à aquifère poreux continu libre":          0.15,
    "Domaine à aquifère poreux continu captif":         0.20,
    "Domaine à aquifère poreux discontinu":             0.35,
    "Domaine à aquifère fissuré continu":               0.45,
    "Domaine à aquifère fissuré discontinu":            0.60,
    "Domaine à aquifère karstique":                     0.55,
    "Domaine à socle peu perméable en petit fond":      0.80,
    "Domaine à socle peu perméable en grand fond":      0.85,
    "Non renseigné":                                    0.50,
}

# Mapping état quantitatif → facteur multiplicatif de stress
_ETAT_QUANT_FACTOR = {
    "Bon état":      1.0,
    "Médiocre":      1.5,
    "Inconnu":       1.0,
    "Non évalué":    1.0,
}

# Bbox France élargie pour WFS
_BBOX_FRANCE = (-6.0, 41.0, 10.0, 52.0)


class BRGMDownloader:
    """
    Télécharge les données BDLISA v3 via les services BRGM.

    Args:
        dest_dir: Répertoire de destination
        force: Forcer le re-téléchargement
        bbox: Bounding box [minlon, minlat, maxlon, maxlat]
    """

    def __init__(
        self,
        dest_dir: Path,
        force: bool = False,
        bbox: tuple[float, float, float, float] = _BBOX_FRANCE,
    ):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.bbox = bbox

    # ──────────────────────────────────────────────────────────────────────────
    # Téléchargement
    # ──────────────────────────────────────────────────────────────────────────

    def download(self) -> Path:
        """
        Télécharge BDLISA via WFS, puis ATOM, puis lien direct.
        Supporte aussi les GPKG régionaux (BDLISA_V3_XXX-gpkg/).

        Returns:
            Path du GeoPackage local
        """
        gpkg_path = self.dest_dir / "bdlisa_v3.gpkg"

        if gpkg_path.exists() and not self.force:
            logger.info(f"BDLISA déjà présente : {gpkg_path}")
            return gpkg_path

        # Chercher des GPKG régionaux (téléchargement manuel InfoTerre)
        regional_dirs = sorted(self.dest_dir.glob("BDLISA_V3_*-gpkg"))
        if regional_dirs:
            regional_gpkgs = [
                d / "BDLISA_V3.gpkg" for d in regional_dirs
                if (d / "BDLISA_V3.gpkg").exists()
            ]
            if regional_gpkgs:
                logger.info(
                    f"Fusion de {len(regional_gpkgs)} GPKG régionaux BDLISA..."
                )
                chunks = []
                for rg in regional_gpkgs:
                    region = rg.parent.name.replace("BDLISA_V3_", "").replace("-gpkg", "")
                    logger.debug(f"  Lecture {region} : {rg}")
                    try:
                        gdf_r = gpd.read_file(str(rg), layer=_BDLISA_LAYER)
                    except Exception:
                        logger.debug(f"    Couche {_BDLISA_LAYER} absente, lecture par défaut")
                        gdf_r = gpd.read_file(str(rg))
                    # Harmoniser les noms de colonnes en lowercase
                    gdf_r.columns = [c.lower() if c != "geometry" else c for c in gdf_r.columns]
                    chunks.append(gdf_r)
                crs = chunks[0].crs
                gdf_merged = gpd.pd.concat(chunks, ignore_index=True)
                gdf_merged = gpd.GeoDataFrame(gdf_merged, crs=crs)
                gdf_merged.to_file(str(gpkg_path), driver="GPKG")
                logger.success(
                    f"BDLISA fusionnée : {len(gdf_merged)} entités "
                    f"({len(regional_gpkgs)} régions) → {gpkg_path}"
                )
                return gpkg_path

        logger.info("Téléchargement BRGM BDLISA v3...")

        # Méthode 1 : WFS BRGM (paginé, données complètes)
        try:
            gdf = self._download_via_wfs()
            gdf.to_file(str(gpkg_path), driver="GPKG")
            logger.success(f"BDLISA téléchargée via WFS : {gpkg_path}")
            return gpkg_path
        except Exception as e:
            logger.warning(f"WFS BRGM échoué ({e})")

        # Méthode 2 : ATOM feed (lien direct GPKG INSPIRE)
        try:
            self._download_direct(gpkg_path)
            logger.success(f"BDLISA téléchargée via ATOM : {gpkg_path}")
            return gpkg_path
        except Exception as e:
            logger.warning(f"ATOM BRGM échoué ({e})")

        # Méthode 3 : Lien direct Infoterre
        try:
            self._download_file(_BRGM_DIRECT_GPKG, gpkg_path)
            logger.success(f"BDLISA téléchargée (lien direct) : {gpkg_path}")
            return gpkg_path
        except Exception as e:
            logger.error(
                f"Tous les téléchargements BRGM ont échoué ({e}).\n"
                f"Solution manuelle :\n"
                f"  1. Aller sur https://infoterre.brgm.fr\n"
                f"  2. Rechercher 'BDLISA entités hydrogéologiques'\n"
                f"  3. Télécharger le GPKG et le placer dans : {gpkg_path}\n"
            )
            raise

    def _download_via_wfs(self) -> gpd.GeoDataFrame:
        """
        Télécharge BDLISA via WFS OGC 2.0 avec pagination.

        La requête GetFeature est paginée par chunks de 500 features
        pour éviter les timeouts serveur BRGM.
        """
        from owslib.wfs import WebFeatureService
        import xml.etree.ElementTree as ET

        minx, miny, maxx, maxy = self.bbox

        logger.info(f"WFS BRGM : {_BRGM_WFS_URL}")
        wfs = WebFeatureService(url=_BRGM_WFS_URL, version="2.0.0", timeout=60)

        chunks = []
        start_index = 0
        count = 500

        while True:
            logger.debug(f"WFS page {start_index // count + 1}...")
            try:
                resp = wfs.getfeature(
                    typename=_BRGM_WFS_LAYER,
                    bbox=(minx, miny, maxx, maxy, "EPSG:4326"),
                    outputFormat="application/json",
                    startindex=start_index,
                    maxfeatures=count,
                )
                data = json.loads(resp.read())
                features = data.get("features", [])
                if not features:
                    break

                chunk = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
                chunks.append(chunk)
                start_index += len(features)

                if len(features) < count:
                    break  # Dernière page
            except Exception as e:
                if start_index == 0:
                    raise  # Première page → erreur fatale
                logger.warning(f"Pagination WFS interrompue à {start_index} features : {e}")
                break

        if not chunks:
            raise ValueError("WFS BRGM : aucune feature retournée")

        gdf = gpd.pd.concat(chunks, ignore_index=True)
        gdf = gdf.to_crs("EPSG:4326")
        logger.success(f"WFS : {len(gdf)} entités BDLISA chargées")
        return gdf

    def _download_direct(self, dest: Path) -> None:
        """Télécharge directement depuis l'ATOM INSPIRE BRGM."""
        # L'ATOM renvoie un XML listant les fichiers téléchargeables
        resp = requests.get(_BRGM_ATOM_URL, timeout=60)
        resp.raise_for_status()

        # Parser l'ATOM XML pour trouver le lien GPKG
        import xml.etree.ElementTree as ET
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        gpkg_url = None
        for link in root.findall(".//atom:link", ns):
            href = link.get("href", "")
            if href.endswith(".gpkg") or "gpkg" in href.lower():
                gpkg_url = href
                break

        if not gpkg_url:
            raise ValueError("Aucun lien GPKG dans l'ATOM BRGM")

        self._download_file(gpkg_url, dest)

    def _download_file(self, url: str, dest: Path, chunk_size: int = 8192) -> None:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)

    # ──────────────────────────────────────────────────────────────────────────
    # Chargement
    # ──────────────────────────────────────────────────────────────────────────

    def load(self, gpkg_path: Optional[Path] = None) -> gpd.GeoDataFrame:
        """
        Charge BDLISA (couche ENTITES_NIVEAU3_ORDRES) et standardise les colonnes.

        Returns:
            GeoDataFrame avec colonnes :
            codeeh, libelleeh, natureeh, milieueh, themeeh, geometry
        """
        if gpkg_path is None:
            gpkg_path = self.dest_dir / "bdlisa_v3.gpkg"

        if not gpkg_path.exists():
            gpkg_path = self.download()

        # Lire la couche Niveau 3 (la plus fine)
        try:
            gdf = gpd.read_file(str(gpkg_path), layer=_BDLISA_LAYER)
        except Exception:
            logger.warning(f"Couche {_BDLISA_LAYER} absente, lecture par défaut")
            gdf = gpd.read_file(str(gpkg_path))

        # Reprojeter en WGS84 puis filtrer par bbox France
        if gdf.crs and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = self.bbox
        gdf = gdf.cx[minx:maxx, miny:maxy]

        # Harmoniser les noms de colonnes en lowercase
        gdf.columns = [c.lower() if c != "geometry" else c for c in gdf.columns]

        logger.success(
            f"BDLISA chargée : {len(gdf)} entités, "
            f"colonnes clés : milieueh={('milieueh' in gdf.columns)}, "
            f"themeeh={('themeeh' in gdf.columns)}"
        )
        return gdf

    # ──────────────────────────────────────────────────────────────────────────
    # Calcul du stress en eau souterraine
    # ──────────────────────────────────────────────────────────────────────────

    def compute_gw_stress(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calcule un score de stress souterrain normalisé [0-1] par entité BDLISA.

        Formule (Niveau 3, milieueh + themeeh) :
            GWS_score = milieu_stress × theme_factor

        Fallback (anciennes données WFS, LB_TYPEHG) :
            GWS_score = type_stress × etat_factor

        Clampé à [0, 1].

        Args:
            gdf: GeoDataFrame BDLISA (output de load())

        Returns:
            Même GeoDataFrame avec colonne 'gws_score' ajoutée [0-1]
        """
        gdf = gdf.copy()

        if "milieueh" in gdf.columns and gdf["milieueh"].notna().sum() > 0:
            # Nouvelle méthode : milieueh + themeeh (Niveau 3)
            gdf["gws_milieu_score"] = gdf["milieueh"].astype(str).map(
                _MILIEU_STRESS_MAP
            ).fillna(0.5)

            gdf["gws_theme_factor"] = gdf["themeeh"].astype(str).map(
                _THEME_FACTOR
            ).fillna(1.0) if "themeeh" in gdf.columns else 1.0

            gdf["gws_score"] = (
                gdf["gws_milieu_score"] * gdf["gws_theme_factor"]
            ).clip(0.0, 1.0)

            logger.info(
                f"GWS BRGM (milieueh+themeeh) : "
                f"moy={gdf['gws_score'].mean():.3f}, "
                f"max={gdf['gws_score'].max():.3f}, "
                f"{len(gdf)} entités"
            )
        else:
            # Fallback : anciennes données WFS (LB_TYPEHG)
            gdf["gws_type_score"] = gdf["LB_TYPEHG"].map(
                _TYPE_STRESS_MAP
            ).fillna(0.5)

            gdf["gws_etat_factor"] = gdf["LB_ETATQ"].map(
                _ETAT_QUANT_FACTOR
            ).fillna(1.0)

            gdf["gws_score"] = (
                gdf["gws_type_score"] * gdf["gws_etat_factor"]
            ).clip(0.0, 1.0)

            logger.info(
                f"GWS BRGM (LB_TYPEHG fallback) : "
                f"moy={gdf['gws_score'].mean():.3f}, "
                f"max={gdf['gws_score'].max():.3f}"
            )
        return gdf

    def rasterize_gws(
        self,
        gdf: gpd.GeoDataFrame,
        resolution_deg: float = 0.0833,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Rasterise le score GWS BRGM en GeoTIFF EPSG:4326.

        Args:
            gdf: GeoDataFrame avec colonne 'gws_score'
            resolution_deg: Résolution de sortie (degrés)
            output_path: Chemin de sortie (None → dest_dir/brgm_gws.tif)

        Returns:
            Path du GeoTIFF
        """
        import rasterio
        from rasterio.features import rasterize as rio_rasterize
        from rasterio.transform import from_bounds

        if output_path is None:
            output_path = self.dest_dir / "brgm_gws.tif"

        if "gws_score" not in gdf.columns:
            gdf = self.compute_gw_stress(gdf)

        minx, miny, maxx, maxy = self.bbox
        width = int((maxx - minx) / resolution_deg) + 1
        height = int((maxy - miny) / resolution_deg) + 1
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        shapes = [
            (geom, val)
            for geom, val in zip(gdf.geometry, gdf["gws_score"])
            if geom is not None and not geom.is_empty
        ]

        raster = rio_rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=-9999.0,
            dtype="float32",
        )

        with rasterio.open(
            str(output_path),
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=-9999.0,
            compress="lzw",
        ) as dst:
            dst.write(raster, 1)

        logger.success(f"GWS BRGM rasterisé : {output_path}")
        return output_path


def run_brgm_download(dest_dir: Path, force: bool = False) -> Path:
    """Point d'entrée pipeline."""
    dl = BRGMDownloader(dest_dir=dest_dir, force=force)
    gpkg_path = dl.download()
    gdf = dl.load(gpkg_path)
    gdf = dl.compute_gw_stress(gdf)
    tif_path = dl.rasterize_gws(gdf)
    return tif_path


if __name__ == "__main__":
    from water_model.config import get_path
    dest = get_path("raw_data") / "brgm"
    tif = run_brgm_download(dest)
    print(f"\nGWS BRGM : {tif}")