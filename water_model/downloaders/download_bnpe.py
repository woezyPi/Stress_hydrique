"""
Téléchargeur BNPE — Banque Nationale des Prélèvements en Eau.

Source   : Hub'Eau API (https://hubeau.eaufrance.fr/api/v1/prelevements)
Licence  : Licence Ouverte / Open Licence (Etalab)

Télécharge les volumes annuels de prélèvements par point (lat/lon)
et rasterise sur la grille L93 5km pour produire une carte de pression
de prélèvement locale (m³/an/km²).

Usage :
    dl = BNPEDownloader(dest_dir=Path("data/raw/bnpe"), year=2022)
    dl.download()
    withdrawal_grid = dl.rasterize_to_grid(resolution_m=5000)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from loguru import logger
from pyproj import Transformer

from water_model.preprocessing.resampling import build_reference_grid


_API_URL = "https://hubeau.eaufrance.fr/api/v1/prelevements/chroniques"
_PAGE_SIZE = 10000


class BNPEDownloader:
    """Télécharge et rasterise les prélèvements BNPE."""

    def __init__(self, dest_dir: Path, year: int = 2022):
        self.dest_dir = Path(dest_dir)
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        self.year = year
        self._cache_path = self.dest_dir / f"bnpe_prelevements_{year}.parquet"

    def download(self, force: bool = False) -> pd.DataFrame:
        """
        Télécharge tous les prélèvements France pour l'année.

        Returns:
            DataFrame avec colonnes: latitude, longitude, volume, code_usage
        """
        if self._cache_path.exists() and not force:
            df = pd.read_parquet(self._cache_path)
            logger.info(f"BNPE {self.year} en cache : {len(df)} prélèvements")
            return df

        logger.info(f"Téléchargement BNPE {self.year} via Hub'Eau...")

        all_records = []
        page = 1

        while True:
            params = {
                "annee": self.year,
                "size": _PAGE_SIZE,
                "page": page,
                "format": "json",
            }

            try:
                r = requests.get(_API_URL, params=params, timeout=120)
                r.raise_for_status()
            except requests.RequestException as e:
                logger.warning(f"Erreur page {page} : {e}")
                break

            data = r.json()
            records = data.get("data", [])
            if not records:
                break

            all_records.extend(records)
            logger.debug(f"  Page {page} : {len(records)} enregistrements (total {len(all_records)})")

            # Plus de pages ?
            if data.get("next") is None:
                break

            page += 1
            time.sleep(0.3)

        df = pd.DataFrame(all_records)
        # Nettoyer
        df = df.dropna(subset=["latitude", "longitude", "volume"])
        df = df[df["volume"] > 0]
        df["latitude"] = df["latitude"].astype(float)
        df["longitude"] = df["longitude"].astype(float)
        df["volume"] = df["volume"].astype(float)

        df.to_parquet(self._cache_path, index=False)
        logger.success(f"BNPE {self.year} : {len(df)} prélèvements téléchargés")
        return df

    def rasterize_to_grid(
        self,
        resolution_m: float = 5000.0,
    ) -> xr.DataArray:
        """
        Rasterise les prélèvements sur la grille L93 5km.

        Chaque pixel = somme des volumes (m³/an) dans la cellule,
        normalisé par la surface du pixel → m³/an/km².

        Returns:
            DataArray (y, x) : densité de prélèvement [m³/an/km²]
        """
        df = self.download()

        # Projeter WGS84 → L93
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
        x_l93, y_l93 = transformer.transform(
            df["longitude"].values, df["latitude"].values
        )

        # Grille de référence
        xs, ys = build_reference_grid(resolution_m)
        half = resolution_m / 2.0

        # Trier ys en ordre croissant pour digitize
        ys_sorted = np.sort(ys)
        ys_descending = ys[0] > ys[-1] if len(ys) > 1 else False

        # Binning vectorisé avec np.digitize
        # Bordures des cellules : centre ± half
        x_edges = np.concatenate([xs - half, [xs[-1] + half]])
        y_edges = np.concatenate([ys_sorted - half, [ys_sorted[-1] + half]])

        ix_all = np.digitize(x_l93, x_edges) - 1
        iy_all = np.digitize(y_l93, y_edges) - 1

        # Filtrer les points hors grille
        valid = (
            (ix_all >= 0) & (ix_all < len(xs))
            & (iy_all >= 0) & (iy_all < len(ys_sorted))
        )

        grid = np.zeros((len(ys), len(xs)), dtype=np.float64)
        volumes = df["volume"].values
        for i in np.where(valid)[0]:
            iy_grid = (len(ys) - 1 - iy_all[i]) if ys_descending else iy_all[i]
            grid[iy_grid, ix_all[i]] += volumes[i]

        # Normaliser par surface du pixel (km²)
        pixel_area_km2 = (resolution_m / 1000.0) ** 2
        grid_density = grid / pixel_area_km2

        da = xr.DataArray(
            grid_density.astype(np.float32),
            dims=["y", "x"],
            coords={"y": ys, "x": xs},
            name="withdrawal_density",
            attrs={
                "long_name": "Water Withdrawal Density",
                "units": "m3/year/km2",
                "source": "BNPE via Hub'Eau",
                "year": self.year,
            },
        )

        valid = grid_density[grid_density > 0]
        logger.success(
            f"BNPE rasterisé : {len(valid)} pixels actifs\n"
            f"  Densité moyenne  : {np.mean(valid):,.0f} m³/an/km²\n"
            f"  Densité médiane  : {np.median(valid):,.0f} m³/an/km²\n"
            f"  Densité max      : {np.max(valid):,.0f} m³/an/km²\n"
            f"  Pixels avec prélèvements : {len(valid)}/{grid.size}"
        )
        return da

    def normalize_withdrawal_stress(
        self,
        withdrawal_da: xr.DataArray,
        p95_cap: bool = True,
    ) -> xr.DataArray:
        """
        Normalise la densité de prélèvement en score de stress [0, 1].

        Normalisation min-max après capping au P95 (robuste aux outliers) :
            stress = density / max(density_capped)

        Args:
            withdrawal_da: DataArray densité de prélèvement
            p95_cap: Si True, cap au 95e percentile (évite outliers extrêmes)

        Returns:
            DataArray [0, 1] stress de prélèvement local
        """
        arr = withdrawal_da.values.copy()
        valid_mask = arr > 0

        if valid_mask.sum() == 0:
            logger.warning("Aucun prélèvement BNPE → stress uniforme 0")
            return xr.zeros_like(withdrawal_da)

        if p95_cap:
            p95 = np.percentile(arr[valid_mask], 95)
            arr = np.clip(arr, 0, p95)

        # Normalisation min-max sur les pixels actifs
        vmax = arr[valid_mask].max()
        if vmax > 0:
            stress = arr / vmax
        else:
            stress = np.zeros_like(arr)

        stress = np.clip(stress, 0.0, 1.0).astype(np.float32)

        da = xr.DataArray(
            stress,
            dims=withdrawal_da.dims,
            coords=withdrawal_da.coords,
            name="bnpe_withdrawal_stress",
            attrs={
                "long_name": "BNPE Local Withdrawal Stress",
                "units": "dimensionless",
                "range": "[0, 1]",
                "source": "BNPE via Hub'Eau",
                "normalization": "percentile rank, p95 capped" if p95_cap else "min-max",
            },
        )
        return da
