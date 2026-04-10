# -*- coding: utf-8 -*-
"""
summarize_unique_precursors_from_csvs.py  (NO command-line arguments)

Goal
----
Count UNIQUE precursors (precursor_mz) across ALL input CSV files and save to CSV.

This script follows the same "extract CSV files" style as MoNA_match_modified_cos.py:
- No CLI: edit CONFIG
- Collect multiple CSVs via (directory / glob / explicit list) + optional filename filter
- Read CSV robustly with encoding fallback

Supported input CSV formats
---------------------------
A) Peak table CSV:
   columns include: precursor_mz, mz, intensity (or intensity_sample)

B) Other summary CSV:
   columns include: precursor_mz
   (If mz/intensity are absent, only counts based on rows/files are computed.)

Outputs
-------
OUT_DIR/
  - unique_precursors_summary.csv          (one row per unique precursor_mz)
  - per_file_unique_precursors.csv         (per file stats)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import pandas as pd

# ===================== CONFIG (edit these) =====================
CSV_DIR: Optional[str] = r"E:\model\singlecell\singlecell_data\GSE\SNR_0.001_adaptive"      # directory containing many CSVs (or set None)
CSV_GLOB: Optional[str] = None             # e.g., r"/mnt/data/*.csv" or r"E:\...\*.csv"
CSV_FILES: List[str] = []                  # explicit file list, if you prefer

FILENAME_MUST_CONTAIN: Optional[str] = None  # e.g., "filtered" (case-insensitive). None = no filter

PRECURSOR_ROUND_DIGITS: int = 6

OUT_DIR: str = r"E:\model\singlecell\singlecell_data\GSE\0308\precursor_summary_out"
OUT_UNIQUE_PRECURSORS: str = "unique_precursors_summary.csv"
OUT_PER_FILE: str = "per_file_unique_precursors.csv"
# ===============================================================


def _read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "gb18030", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def _filename_ok(p: Path, must_contain: Optional[str]) -> bool:
    if not must_contain:
        return True
    return must_contain.lower() in p.name.lower()


def iter_csvs(
    csv_dir: Optional[Union[str, Path]] = None,
    csv_glob: Optional[str] = None,
    csv_files: Optional[Iterable[Union[str, Path]]] = None,
    filename_must_contain: Optional[str] = FILENAME_MUST_CONTAIN,
) -> List[Path]:
    paths: List[Path] = []

    if csv_dir:
        d = Path(csv_dir)
        if d.exists() and d.is_dir():
            paths.extend(sorted(d.glob("*.csv")))

    if csv_glob:
        paths.extend(sorted(Path().glob(csv_glob)))

    if csv_files:
        paths.extend([Path(p) for p in csv_files])

    uniq: Dict[str, Path] = {}
    for p in paths:
        if not p.exists():
            continue
        if not _filename_ok(p, filename_must_contain):
            continue
        rp = str(p.resolve()) if p.exists() else str(p)
        uniq.setdefault(rp, p)

    return list(uniq.values())


def _norm_precursor(x, ndigits: int) -> float:
    try:
        return round(float(x), ndigits)
    except Exception:
        return float("nan")


def _pick_intensity_col(df: pd.DataFrame) -> Optional[str]:
    if "intensity" in df.columns:
        return "intensity"
    if "intensity_sample" in df.columns:
        return "intensity_sample"
    return None


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = iter_csvs(CSV_DIR, CSV_GLOB, CSV_FILES, FILENAME_MUST_CONTAIN)
    if not csv_paths:
        raise FileNotFoundError("No input CSV files found. Check CSV_DIR / CSV_GLOB / CSV_FILES.")

    agg: Dict[float, Dict[str, float]] = {}
    files_seen: Dict[float, set] = {}

    per_file_rows = []

    for p in csv_paths:
        df = _read_csv_safely(p)

        if "precursor_mz" not in df.columns:
            # skip non-matching CSVs silently
            continue

        df = df.copy()
        df["prec_norm"] = df["precursor_mz"].apply(lambda x: _norm_precursor(x, PRECURSOR_ROUND_DIGITS))
        df = df[df["prec_norm"].notna()]

        int_col = _pick_intensity_col(df)
        has_peak_cols = ("mz" in df.columns) and (int_col is not None)

        uniq_prec = sorted(set(df["prec_norm"].dropna().tolist()))
        per_file_rows.append({
            "file": p.name,
            "n_rows": int(len(df)),
            "n_unique_precursors": int(len(uniq_prec)),
            "has_peak_cols": int(bool(has_peak_cols)),
        })

        if has_peak_cols:
            df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
            df[int_col] = pd.to_numeric(df[int_col], errors="coerce").fillna(0.0)
            df = df[df["mz"].notna()].copy()

        for prec, sub in df.groupby("prec_norm", dropna=True):
            if prec != prec:  # NaN
                continue
            prec = float(prec)

            if prec not in agg:
                agg[prec] = {
                    "precursor_mz_norm": prec,
                    "total_rows": 0.0,
                    "n_files": 0.0,
                    "has_peak_info": 0.0,
                    "total_intensity": 0.0,
                    "max_intensity": 0.0,
                    "min_mz": float("nan"),
                    "max_mz": float("nan"),
                    "unique_mz_count_sum": 0.0,
                }
                files_seen[prec] = set()

            agg[prec]["total_rows"] += float(len(sub))
            files_seen[prec].add(p.name)

            if has_peak_cols:
                agg[prec]["has_peak_info"] = 1.0
                total_int = float(sub[int_col].sum())
                max_int = float(sub[int_col].max()) if len(sub) else 0.0
                agg[prec]["total_intensity"] += total_int
                agg[prec]["max_intensity"] = max(float(agg[prec]["max_intensity"]), max_int)

                mn = float(sub["mz"].min()) if len(sub) else float("nan")
                mx = float(sub["mz"].max()) if len(sub) else float("nan")
                if pd.isna(agg[prec]["min_mz"]) or (not pd.isna(mn) and mn < agg[prec]["min_mz"]):
                    agg[prec]["min_mz"] = mn
                if pd.isna(agg[prec]["max_mz"]) or (not pd.isna(mx) and mx > agg[prec]["max_mz"]):
                    agg[prec]["max_mz"] = mx

                agg[prec]["unique_mz_count_sum"] += float(sub["mz"].nunique(dropna=True))

    rows = []
    for prec, d in agg.items():
        dd = d.copy()
        dd["n_files"] = float(len(files_seen.get(prec, set())))
        rows.append(dd)

    if not rows:
        raise ValueError("No rows aggregated. Ensure your CSVs include 'precursor_mz' column.")

    out_unique = pd.DataFrame(rows).sort_values("precursor_mz_norm").reset_index(drop=True)
    out_per_file = pd.DataFrame(per_file_rows).sort_values(["n_unique_precursors", "file"], ascending=[False, True])

    out_unique_path = out_dir / OUT_UNIQUE_PRECURSORS
    out_per_file_path = out_dir / OUT_PER_FILE

    out_unique.to_csv(out_unique_path, index=False, encoding="utf-8-sig")
    out_per_file.to_csv(out_per_file_path, index=False, encoding="utf-8-sig")

    print("Input CSV files scanned:", len(csv_paths))
    print("Files with precursor_mz:", len(out_per_file))
    print("Unique precursors:", len(out_unique))
    print("Wrote:", out_unique_path)
    print("Wrote:", out_per_file_path)


if __name__ == "__main__":
    main()
