#!/usr/bin/env python3
"""
Diagnostic script: compare CI model vs reference.
Prints both gen-weighted and load-weighted aggregations.

Usage:
    python diagnose_ci_gap.py [--year 2022] [--test]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    # Imports
    from carbon_model.config import get_path
    from carbon_model.tracing.carbon_tracing import compute_country_carbon_intensity
    from carbon_model.validation.validation import aggregate_national_ci
    from carbon_model.downloaders.download_rte import RTEDownloader
    import xarray as xr
    import pypsa

    year = args.year
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year}-01-07 23:00") if args.test else pd.Timestamp(f"{year}-12-31 23:00")

    # Load CI nodal
    ci_path = get_path("outputs") / "ci_nodal" / f"ci_nodal_{year}.nc"
    if not ci_path.exists():
        print(f"ERROR: {ci_path} not found. Run the pipeline first.")
        sys.exit(1)

    with xr.open_dataset(str(ci_path)) as _ds:
        ci_ds = _ds.load()

    # Load network
    processed_dir = get_path("processed_data")
    net_path = processed_dir / "network_france.nc"
    if not net_path.exists():
        print(f"ERROR: {net_path} not found.")
        sys.exit(1)
    n = pypsa.Network(str(net_path))

    # Load production data for reference
    from carbon_model.config import get_api_keys
    keys = get_api_keys()
    raw_dir = get_path("raw_data")
    rte_dl = RTEDownloader(
        client_id=keys.get("rte_client_id", ""),
        client_secret=keys.get("rte_client_secret", ""),
        dest_dir=raw_dir / "rte",
    )
    production_fr = rte_dl.get_generation(start, end).resample("h").mean().fillna(0)

    _GEN_COLS = {
        "Nuclear", "Wind Onshore", "Solar", "Hydro Run-of-river",
        "Biomass", "Fossil Gas", "Fossil Oil", "Fossil Hard Coal",
        "Hydro Pumped Storage",
    }
    gen_cols = [c for c in production_fr.columns if c in _GEN_COLS]

    # Production-side reference (same EFs as model)
    ref_prod = compute_country_carbon_intensity(production_fr[gen_cols])

    # Consumption-side reference (RTE co2_rate) if available
    ref_cons = None
    if "co2_rate_gco2_kwh" in production_fr.columns:
        ref_cons = production_fr["co2_rate_gco2_kwh"].dropna()

    # Model CI aggregations
    ci_gen = aggregate_national_ci(ci_ds, n, weighting="generation")
    ci_load = aggregate_national_ci(ci_ds, n, weighting="load")

    def compute_metrics(model, ref, label):
        common = model.dropna().index.intersection(ref.dropna().index)
        if len(common) < 24:
            print(f"  [{label}] Not enough common timestamps ({len(common)})")
            return
        m = model[common].values
        r = ref[common].values
        mape = float(np.mean(np.abs(m - r) / np.maximum(r, 1.0)) * 100)
        bias = float(np.mean(m - r))
        ss_res = np.sum((m - r) ** 2)
        ss_tot = np.sum((r - r.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        print(f"  [{label}]  ref_mean={r.mean():.1f}  model_mean={m.mean():.1f}  "
              f"MAPE={mape:.1f}%  bias={bias:+.1f}  R2={r2:.3f}  n={len(common)}")

    print(f"\n{'='*70}")
    print(f"  GreenDC CI Diagnostic — Year {year} {'[TEST]' if args.test else ''}")
    print(f"{'='*70}")

    print(f"\n  CI model gen-weighted mean : {ci_gen.dropna().mean():.1f} gCO2/kWh")
    print(f"  CI model load-weighted mean: {ci_load.dropna().mean():.1f} gCO2/kWh")
    print(f"  CI ref production-side mean: {ref_prod.dropna().mean():.1f} gCO2/kWh")
    if ref_cons is not None:
        print(f"  CI ref consumption-side mean (RTE co2_rate): {ref_cons.mean():.1f} gCO2/kWh")

    print(f"\n--- vs Production-side reference (hard criterion) ---")
    compute_metrics(ci_gen, ref_prod, "gen-weighted")
    compute_metrics(ci_load, ref_prod, "load-weighted")

    if ref_cons is not None:
        print(f"\n--- vs Consumption-side reference (RTE co2_rate, informative) ---")
        compute_metrics(ci_gen, ref_cons, "gen-weighted")
        compute_metrics(ci_load, ref_cons, "load-weighted")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
