"""
Téléchargeur WRI Aqueduct 4.0 — Water Risk Atlas.

Données : WRI Aqueduct 4.0 (Hofste et al., 2019 — mise à jour 2023)
Source   : https://www.wri.org/data/aqueduct-water-risk-atlas
Licence  : CC BY 4.0

Variables extraites (colonnes GDB baseline_annual) :
    bws_score  → bws_raw.tif   Baseline Water Stress      (échelle 0-5)
    gtd_raw    → gws_raw.tif   Groundwater Table Decline  (m/yr → converti 0-5)
    drr_raw    → drr_raw.tif   Drought Risk               ([0,1])

Format source : FileGeoDatabase (.gdb) ou GeoPackage (.gpkg)
Le module lit le GDB local et convertit les polygones en rasters GeoTIFF EPSG:4326.

Résolution native : ≈ 5 arc-minutes (~10 km à l'équateur).

Normalisation GWS (gtd_raw → 0-5) :
    gtd_raw ≤ 0     → 0.0  (recharge active, pas de stress)
    gtd_raw = 0.1   → 1.0  (dépletion faible)
    gtd_raw = 0.3   → 3.0  (dépletion significative)
    gtd_raw ≥ 0.5   → 5.0  (dépletion critique)
    Formule : gws_score = clip(max(gtd_raw, 0) × 10, 0, 5)

Usage (GDB local) :
    dl = AqueductDownloader(dest_dir=Path("data/raw/aqueduct"))
    tifs = dl.extract_rasters_from_gdb(gdb_path)  # {'bws_raw': Path, ...}

Usage (téléchargement) :
    dl = AqueductDownloader(dest_dir=Path("data/raw/aqueduct"))
    dl.download()                      # télécharge le GeoPackage
    tifs = dl.extract_rasters()        # rasterise → dict[var: Path]
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import requests
from loguru import logger
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# ──────────────────────────────────────────────────────────────────────────────
# Emplacement du GDB téléchargé manuellement depuis WRI
# ──────────────────────────────────────────────────────────────────────────────
# Chemin relatif au répertoire water_model/ (racine du module)
_GDB_RELATIVE_PATHS = [
    "data/Aqueduct40_waterrisk_download_Y2023M07D05/GDB/Aq40_Y2023D07M05.gdb",
    "data/raw/aqueduct/Aq40_Y2023D07M05.gdb",
]
_GDB_LAYER = "baseline_annual"
_FRANCE_GID0 = "FRA"

# ──────────────────────────────────────────────────────────────────────────────
# Mapping colonnes GDB → fichiers TIF de sortie
#
# Tuple (colonne_gdb, nom_sortie, description)
# La transformation numérique est gérée dans load_from_local_gdb()
# ──────────────────────────────────────────────────────────────────────────────
_GDB_COLUMNS = [
    # bws_score (0-5) → bws_raw.tif, pipeline normalise par /5.0
    ("bws_score",        "bws_raw",  "Baseline Water Stress (0-5)"),
    # gtd_raw (m/yr) → colonne intermédiaire gws_raw_computed (0-5)
    ("gws_raw_computed", "gws_raw",  "Groundwater Table Decline (0-5)"),
    # drr_raw ([0,1]) → drr_raw.tif pour référence
    ("drr_raw",          "drr_raw",  "Drought Risk ([0,1])"),
]

# ──────────────────────────────────────────────────────────────────────────────
# URL de téléchargement automatique (secours si pas de GDB local)
# ──────────────────────────────────────────────────────────────────────────────
_AQUEDUCT_URL_ZIP = (
    "https://files.wri.org/d8/s3fs-public/2023-09/"
    "Aqueduct40_waterrisk_annual_y2023m07d05.zip"
)

# Variables GPKG (ancien format, pour compatibilité)
_VARIABLES_GPKG = {
    "bws_raw": "bws_raw",
    "gws_raw": "gws_raw",
    "drr_raw": "drr_raw",
    "sev_raw": "sev_raw",
}

# Grille de rasterisation (0.0833° ≈ 5 arc-minutes)
_RASTER_RES = 0.0833
_BBOX_FRANCE = (-6.0, 41.0, 10.0, 52.0)

# Répertoire racine du module water_model (pour trouver le GDB)
_MODULE_ROOT = Path(__file__).parent.parent


class AqueductDownloader:
    """
    Lit et rasterise les données WRI Aqueduct 4.0.

    Priorité de source :
        1. GDB local (téléchargé manuellement depuis WRI)
        2. GPKG existant dans dest_dir
        3. Téléchargement automatique (peut échouer selon l'accès WRI)

    Args:
        dest_dir: Répertoire de destination des TIFs produits
        bbox: Bounding box [minlon, minlat, maxlon, maxlat] pour clip France
        force: Forcer la re-rasterisation même si les TIFs existent
    """

    def __init__(
        self,
        dest_dir: Path,
        bbox: tuple[float, float, float, float] = _BBOX_FRANCE,
        force: bool = False,
    ):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.bbox = bbox
        self.force = force

    # ──────────────────────────────────────────────────────────────────────────
    # Localisation du GDB local
    # ──────────────────────────────────────────────────────────────────────────

    def find_local_gdb(self) -> Optional[Path]:
        """
        Cherche le fichier GDB Aqueduct 4.0 téléchargé localement.

        Cherche dans les emplacements standards relatifs au module water_model/.

        Returns:
            Path du GDB si trouvé, None sinon.
        """
        # 1. Emplacements standards
        for rel in _GDB_RELATIVE_PATHS:
            candidate = _MODULE_ROOT / rel
            if candidate.exists():
                logger.info(f"GDB Aqueduct trouvé : {candidate}")
                return candidate

        # 2. Recherche récursive dans data/ (au cas où le nom du dossier varie)
        data_dir = _MODULE_ROOT / "data"
        if data_dir.exists():
            found = list(data_dir.rglob("*.gdb"))
            if found:
                gdb_path = found[0]
                logger.info(f"GDB Aqueduct trouvé (recherche) : {gdb_path}")
                return gdb_path

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Lecture GDB → GeoDataFrame France
    # ──────────────────────────────────────────────────────────────────────────

    def load_from_local_gdb(self, gdb_path: Path) -> gpd.GeoDataFrame:
        """
        Charge les bassins versants France depuis le GDB Aqueduct 4.0.

        Colonnes créées / transformées :
            bws_score        → conservée telle quelle (0-5)
            gws_raw_computed → dérivée de gtd_raw (m/yr) → [0-5] :
                               clip(max(gtd_raw, 0) × 10, 0, 5)
            drr_raw          → conservée telle quelle ([0,1])

        Args:
            gdb_path: Chemin vers le .gdb WRI Aqueduct

        Returns:
            GeoDataFrame France en EPSG:4326
        """
        logger.info(
            f"Lecture GDB Aqueduct 4.0 : {gdb_path}\n"
            f"  layer = {_GDB_LAYER}, filtre = gid_0 == '{_FRANCE_GID0}'"
        )

        gdf = gpd.read_file(str(gdb_path), layer=_GDB_LAYER)
        logger.info(f"Lignes lues (monde) : {len(gdf)}")

        # ── Filtre France ──────────────────────────────────────────────────
        if "gid_0" in gdf.columns:
            gdf_fra = gdf[gdf["gid_0"] == _FRANCE_GID0].copy()
        else:
            # Fallback spatial si la colonne gid_0 est absente
            minx, miny, maxx, maxy = self.bbox
            gdf_fra = gdf.cx[minx:maxx, miny:maxy].copy()

        if len(gdf_fra) == 0:
            raise ValueError(
                f"Aucun bassin versant trouvé pour gid_0='{_FRANCE_GID0}' "
                f"dans {gdb_path}. Vérifier la colonne 'gid_0'."
            )

        logger.info(f"Bassins France : {len(gdf_fra)}")

        # ── Reprojection → EPSG:4326 ───────────────────────────────────────
        if gdf_fra.crs is None:
            logger.warning("CRS absent → forçage EPSG:4326")
            gdf_fra = gdf_fra.set_crs("EPSG:4326")
        elif gdf_fra.crs.to_epsg() != 4326:
            gdf_fra = gdf_fra.to_crs("EPSG:4326")

        # ── Transformation GWS : gtd_raw (m/yr) → 0-5 ────────────────────
        if "gtd_raw" in gdf_fra.columns:
            gtd = gdf_fra["gtd_raw"].astype(float).values
            # Masquer les valeurs nodata (-9999) → NaN
            # NaN sera exclu par le valid_mask lors de la rasterisation
            gtd_clean = np.where(gtd < -9000.0, np.nan, gtd)
            # clip(max(gtd, 0) × 10, 0, 5) :
            #   0.5 m/yr de dépletion → 5.0 (stress critique)
            #   valeurs négatives (recharge) → 0.0 (pas de stress)
            #   nodata → NaN (exclus de la rasterisation)
            gws_computed = np.where(
                np.isnan(gtd_clean),
                np.nan,
                np.clip(np.maximum(gtd_clean, 0.0) * 10.0, 0.0, 5.0),
            )
            gdf_fra["gws_raw_computed"] = gws_computed.astype(np.float32)

            finite = gtd_clean[np.isfinite(gtd_clean)]
            n_depleting = int(np.sum(gtd_clean > 0))
            n_nodata = int(np.sum(gtd < -9000))
            logger.info(
                f"  gtd_raw : [{finite.min():.3f}, {finite.max():.3f}] m/yr\n"
                f"  {n_depleting}/{len(gtd)} bassins en dépletion (>0), "
                f"{n_nodata} nodata\n"
                f"  gws_raw_computed : [{np.nanmin(gws_computed):.3f}, "
                f"{np.nanmax(gws_computed):.3f}]"
            )
        else:
            logger.warning("Colonne 'gtd_raw' absente → gws_raw non disponible")

        # ── Logs statistiques ──────────────────────────────────────────────
        for gdb_col, out_name, desc in _GDB_COLUMNS:
            if gdb_col in gdf_fra.columns:
                raw = gdf_fra[gdb_col].astype(float).values
                # Masquer les nodata (-9999) pour les stats
                valid = raw[(raw > -9000.0) & np.isfinite(raw)]
                logger.info(
                    f"  {out_name} ({gdb_col}) : "
                    f"[{valid.min():.3f}, {valid.max():.3f}], "
                    f"{len(valid)}/{len(gdf_fra)} valides — {desc}"
                )
            elif gdb_col != "gws_raw_computed":
                logger.warning(f"  {gdb_col} absent (pour {out_name})")

        return gdf_fra

    # ──────────────────────────────────────────────────────────────────────────
    # Rasterisation depuis GDB
    # ──────────────────────────────────────────────────────────────────────────

    def extract_rasters_from_gdb(
        self,
        gdb_path: Path,
        resolution_deg: float = _RASTER_RES,
    ) -> dict[str, Path]:
        """
        Rasterise les variables Aqueduct depuis le GDB local en GeoTIFFs.

        Args:
            gdb_path: Chemin vers le .gdb WRI Aqueduct
            resolution_deg: Résolution de sortie en degrés

        Returns:
            dict {'bws_raw': Path, 'gws_raw': Path, 'drr_raw': Path}
        """
        gdf = self.load_from_local_gdb(gdb_path)

        minx, miny, maxx, maxy = self.bbox
        width  = int((maxx - minx) / resolution_deg) + 1
        height = int((maxy - miny) / resolution_deg) + 1
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        output_paths: dict[str, Path] = {}

        for gdb_col, out_name, desc in _GDB_COLUMNS:
            if gdb_col not in gdf.columns:
                logger.warning(f"Colonne '{gdb_col}' absente → skip {out_name}")
                continue

            out_path = self.dest_dir / f"{out_name}.tif"

            if out_path.exists() and not self.force:
                logger.info(f"TIF en cache : {out_path}")
                output_paths[out_name] = out_path
                continue

            logger.info(f"Rasterisation {out_name} ({width}×{height} px)...")

            values = gdf[gdb_col].astype(float).values
            valid_mask = np.isfinite(values) & (values > -9000.0)

            gdf_valid = gdf[valid_mask].copy()
            gdf_valid["_val"] = values[valid_mask]

            # Trier par aire croissante : petits bassins spécialisés écrasent
            # les grands bassins génériques (priorité aux données locales)
            try:
                areas = gdf_valid.geometry.area
                gdf_valid = gdf_valid.iloc[areas.argsort().values]
            except Exception:
                pass  # ordre original si l'aire est indisponible

            shapes = [
                (geom, float(val))
                for geom, val in zip(gdf_valid.geometry, gdf_valid["_val"])
                if geom is not None and not geom.is_empty
            ]

            if not shapes:
                logger.warning(f"Aucun polygone valide pour {out_name} → skip")
                continue

            raster = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=-9999.0,
                dtype="float32",
            )

            import rasterio
            with rasterio.open(
                str(out_path),
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

            valid_vals = raster[raster != -9999.0]
            logger.success(
                f"TIF créé : {out_path}\n"
                f"  Pixels valides : {len(valid_vals)} / {width * height}\n"
                f"  Valeurs : [{valid_vals.min():.3f}, {valid_vals.max():.3f}], "
                f"moy={valid_vals.mean():.3f}  — {desc}"
            )
            output_paths[out_name] = out_path

        return output_paths

    # ──────────────────────────────────────────────────────────────────────────
    # Téléchargement automatique (GPKG)
    # ──────────────────────────────────────────────────────────────────────────

    def download(self) -> Path:
        """
        Télécharge le GeoPackage Aqueduct 4.0 depuis WRI.

        Returns:
            Path du GeoPackage local (.gpkg)

        Note:
            WRI requiert parfois une inscription. Si le téléchargement échoue,
            télécharger manuellement depuis https://www.wri.org/data/aqueduct-water-risk-atlas
            et placer le .gdb dans water_model/data/Aqueduct40_waterrisk_download_*/GDB/
        """
        gpkg_path = self.dest_dir / "aqueduct40_annual.gpkg"
        zip_path  = self.dest_dir / "aqueduct40_annual.zip"

        if gpkg_path.exists() and not self.force:
            logger.info(f"Aqueduct GPKG déjà présent : {gpkg_path}")
            return gpkg_path

        logger.info("Téléchargement WRI Aqueduct 4.0...")
        try:
            self._download_file(_AQUEDUCT_URL_ZIP, zip_path)
            self._extract_gpkg(zip_path, gpkg_path)
            return gpkg_path
        except Exception as e:
            logger.error(
                f"Téléchargement Aqueduct échoué ({e}).\n"
                f"Solution manuelle :\n"
                f"  1. Aller sur https://www.wri.org/data/aqueduct-water-risk-atlas\n"
                f"  2. Télécharger 'Aqueduct 4.0 Annual' (GDB ou GPKG)\n"
                f"  3. Extraire le .gdb dans :\n"
                f"     water_model/data/Aqueduct40_waterrisk_download_*/GDB/\n"
            )
            raise

    def _download_file(self, url: str, dest: Path, chunk_size: int = 8192) -> None:
        with requests.get(url, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        logger.debug(f"  {downloaded/total*100:.0f}% ({downloaded/1e6:.1f} MB)")
        logger.success(f"Téléchargé : {dest} ({dest.stat().st_size/1e6:.1f} MB)")

    def _extract_gpkg(self, zip_path: Path, gpkg_dest: Path) -> None:
        with zipfile.ZipFile(zip_path, "r") as zf:
            gpkg_names = [n for n in zf.namelist() if n.endswith(".gpkg")]
            if not gpkg_names:
                raise ValueError("Aucun .gpkg trouvé dans le ZIP Aqueduct")
            with zf.open(gpkg_names[0]) as src, open(gpkg_dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

    # ──────────────────────────────────────────────────────────────────────────
    # Rasterisation depuis GPKG (ancien format)
    # ──────────────────────────────────────────────────────────────────────────

    def load_france(self, gpkg_path: Optional[Path] = None) -> gpd.GeoDataFrame:
        """Charge le GPKG Aqueduct et filtre sur la France (bbox)."""
        if gpkg_path is None:
            gpkg_path = self.dest_dir / "aqueduct40_annual.gpkg"
        if not gpkg_path.exists():
            gpkg_path = self.download()

        minx, miny, maxx, maxy = self.bbox
        gdf = gpd.read_file(str(gpkg_path), bbox=(minx, miny, maxx, maxy))
        gdf = gdf.to_crs("EPSG:4326")
        logger.success(f"Aqueduct GPKG chargé : {len(gdf)} polygones")
        return gdf

    def extract_rasters(
        self,
        gpkg_path: Optional[Path] = None,
        resolution_deg: float = _RASTER_RES,
    ) -> dict[str, Path]:
        """Rasterise depuis GPKG (colonnes bws_raw, gws_raw, drr_raw, sev_raw)."""
        gdf = self.load_france(gpkg_path)

        minx, miny, maxx, maxy = self.bbox
        width  = int((maxx - minx) / resolution_deg) + 1
        height = int((maxy - miny) / resolution_deg) + 1
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        output_paths: dict[str, Path] = {}

        for col_name, var_name in _VARIABLES_GPKG.items():
            if col_name not in gdf.columns:
                logger.warning(f"Colonne {col_name} absente du GPKG → skip")
                continue

            out_path = self.dest_dir / f"{var_name}.tif"
            if out_path.exists() and not self.force:
                output_paths[var_name] = out_path
                continue

            values = gdf[col_name].astype(float).values
            valid_mask = np.isfinite(values) & (values > -9000)

            gdf_valid = gdf[valid_mask].copy()
            gdf_valid["_val"] = values[valid_mask]

            try:
                areas = gdf_valid.geometry.area
                gdf_valid = gdf_valid.iloc[areas.argsort().values]
            except Exception:
                pass

            shapes = [
                (geom, float(val))
                for geom, val in zip(gdf_valid.geometry, gdf_valid["_val"])
                if geom is not None and not geom.is_empty
            ]

            raster = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=-9999.0,
                dtype="float32",
            )

            import rasterio
            with rasterio.open(
                str(out_path), "w", driver="GTiff",
                height=height, width=width, count=1,
                dtype="float32", crs="EPSG:4326",
                transform=transform, nodata=-9999.0, compress="lzw",
            ) as dst:
                dst.write(raster, 1)

            output_paths[var_name] = out_path
            valid_vals = raster[raster != -9999.0]
            logger.success(f"TIF créé : {out_path} (moy={valid_vals.mean():.2f})")

        return output_paths


# ──────────────────────────────────────────────────────────────────────────────
# Fonction utilitaire haut niveau — point d'entrée pour pipeline.py
# ──────────────────────────────────────────────────────────────────────────────

def run_aqueduct_download(dest_dir: Path, force: bool = False) -> dict[str, Path]:
    """
    Point d'entrée principal pour le pipeline.

    Stratégie :
        1. GDB local téléchargé manuellement (priorité)
        2. GPKG existant dans dest_dir
        3. Téléchargement automatique depuis WRI (peut échouer)

    Returns:
        dict {'bws_raw': Path, 'gws_raw': Path, 'drr_raw': Path}
    """
    dl = AqueductDownloader(dest_dir=dest_dir, force=force)

    # 1. GDB local (téléchargement manuel WRI)
    gdb_path = dl.find_local_gdb()
    if gdb_path is not None:
        try:
            tifs = dl.extract_rasters_from_gdb(gdb_path)
            if tifs:
                logger.success(
                    f"Aqueduct chargé depuis GDB local : {list(tifs.keys())}"
                )
                return tifs
        except Exception as e:
            logger.warning(f"Lecture GDB échouée ({e}) → tentative GPKG")

    # 2. GPKG existant dans dest_dir
    gpkg_path = dest_dir / "aqueduct40_annual.gpkg"
    if gpkg_path.exists():
        try:
            tifs = dl.extract_rasters(gpkg_path=gpkg_path)
            if tifs:
                logger.success(f"Aqueduct chargé depuis GPKG : {list(tifs.keys())}")
                return tifs
        except Exception as e:
            logger.warning(f"Lecture GPKG échouée ({e}) → tentative téléchargement")

    # 3. Téléchargement automatique
    try:
        gpkg_path = dl.download()
        return dl.extract_rasters(gpkg_path=gpkg_path)
    except Exception as e:
        logger.error(
            f"Aqueduct : toutes les sources ont échoué ({e}).\n"
            f"  Télécharger depuis https://www.wri.org/data/aqueduct-water-risk-atlas\n"
            f"  Extraire le .gdb dans water_model/data/\n"
        )
        return {}


if __name__ == "__main__":
    from water_model.config import get_path

    dest = get_path("raw_data") / "aqueduct"
    tifs = run_aqueduct_download(dest)
    print("\nFichiers Aqueduct générés :")
    for name, path in tifs.items():
        print(f"  {name}: {path}")