# -*- coding: utf-8 -*-
"""
find_inchikey_in_4libs.py

Goal:
  Read pred_match_inchikey_in_truth.csv (predicted results),
  then for each InChIKey, check whether ANY spectrum exists in four libraries:
    - MoNA-export-*.json / .jsonl
    - ALL_GNPS_cleaned.json / .jsonl
    - MSMS-Public_experimentspectra-*.msp
    - HMDB export json/jsonl (with keys similar to your hmdb_positive_spectra.json)

Outputs:
  1) inchikey_4libs_presence_summary.csv  (one row per input InChIKey)
  2) inchikey_4libs_matches.csv          (one row per matched spectrum record)
  3) (optional) matched spectra jsonl per library, for downstream use
"""

from __future__ import annotations

import ast
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd


# =======================
# CONFIG (edit these paths)
# =======================
BASE_DIR = Path(r"E:\model\singlecell\results\HGC\0330\GM_DecoID\data")  # set to your folder if needed

INPUT_CSV = BASE_DIR / "COS0p8andFR10.csv"

# 4 libraries (same types as in get_4lib_precursor_mz.py)
MONA_JSON = Path(r"E:\model\singlecell\spectra_library\MoNA-export-LC-MS-MS_Positive_Mode.json")
GNPS_JSON = Path(r"E:\model\singlecell\spectra_library\ALL_GNPS_cleaned.json")
MSP_FILE  = Path(r"E:\model\singlecell\spectra_library\MSMS-Public_experimentspectra-pos-VS19.msp")
HMDB_JSON = Path(r"E:\model\singlecell\spectra_library\hmdb_positive_spectra.json")

# HMDB column keys (adjust if your hmdb json uses different keys)
HMDB_INCHIKEY_COL = "INCHI_KEY"
HMDB_SPECTRUM_ID_COL = "SPECTRUM_ID"
HMDB_NAME_COL = "NAME"
HMDB_PEAKS_COL = "PEAKS"
HMDB_MONO_COL = "MONO_MASS"  # neutral mono mass; may not be needed for inchikey lookup

# Output
OUTPUT_DIR = BASE_DIR / "inchikey_4libs_out"
EXPORT_MATCHED_JSONL = False  # set False if you only want CSV

# Input InChIKey column candidates
INCHIKEY_COL_CANDIDATES = ["inchikey", "InChIKey", "INCHIKEY", "INCHI_KEY", "inchi_key"]


# =======================
# Helpers (mostly copied/compatible with get_4lib_precursor_mz.py)
# =======================

_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            m = _FLOAT_RE.search(s)
            if not m:
                return None
            return float(m.group(0))
        return float(x)
    except Exception:
        return None

def _norm_inchikey(s: str) -> str:
    return (s or "").strip().upper()

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise ValueError(f"Cannot find InChIKey column. candidates={candidates}, cols={cols}")

def _first_non_ws_char(path: Path) -> str:
    with path.open("rb") as f:
        while True:
            b = f.read(1)
            if not b:
                return ""
            if b not in b" \t\r\n":
                return chr(b[0])

def iter_json_entries_stream(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    """
    Stream JSON entries from:
      - .jsonl: one JSON object per line
      - .json: large array of objects (requires ijson), else small dict/list with json.load
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p.resolve()}")

    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
        return

    ch = _first_non_ws_char(p)
    if ch == "[":
        try:
            import ijson  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"{p.name} looks like a JSON array. Please install ijson for streaming:\n"
                f"  pip install ijson\n"
                f"Import error: {e}"
            )
        with p.open("rb") as f:
            for obj in ijson.items(f, "item"):
                if isinstance(obj, dict):
                    yield obj
        return

    with p.open("r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it

def _find_metadata_value(meta_list: Any, names_lower: set[str]) -> Optional[str]:
    if not isinstance(meta_list, list):
        return None
    for m in meta_list:
        if not isinstance(m, dict):
            continue
        n = str(m.get("name", "")).strip().lower()
        if n in names_lower:
            v = m.get("value")
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
    return None

def extract_inchikey_mona_gnps(entry: Dict[str, Any]) -> str:
    # common keys
    for k in ("InChIKey", "inchikey", "InChIKey_smiles"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # nested compound meta
    comp = entry.get("compound")
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        v = _find_metadata_value(comp[0].get("metaData"), {"inchikey"})
        if v:
            return v.strip()

    # entry metadata
    v = _find_metadata_value(entry.get("metaData") or entry.get("metadata"), {"inchikey"})
    return (v or "").strip()

def extract_precursor_mz(entry: Dict[str, Any]) -> Optional[float]:
    for k in ("Precursor_MZ", "precursor_mz", "precursorMz", "PEPMASS", "pepmass"):
        if k in entry:
            mz = _safe_float(entry.get(k))
            if mz is not None:
                return mz

    v = _find_metadata_value(entry.get("metaData") or entry.get("metadata"),
                             {"precursor m/z", "precursormz", "pepmass", "precursor"})
    mz = _safe_float(v) if v is not None else None
    if mz is not None:
        return mz

    spec = entry.get("spectrum")
    if isinstance(spec, dict):
        return extract_precursor_mz(spec)

    comp = entry.get("compound")
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        return extract_precursor_mz(comp[0])

    return None

def extract_name(entry: Dict[str, Any]) -> str:
    for k in ("Compund_Name", "compound_name", "Name", "name", "TITLE"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    comp = entry.get("compound")
    if isinstance(comp, list) and comp:
        c0 = comp[0]
        if isinstance(c0, dict):
            names = c0.get("names")
            if isinstance(names, list) and names and isinstance(names[0], dict):
                nm = names[0].get("name")
                if isinstance(nm, str) and nm.strip():
                    return nm.strip()
    return ""

def has_peaks_mona_gnps(entry: Dict[str, Any]) -> bool:
    if isinstance(entry.get("spectrum"), str) and entry["spectrum"].strip():
        return True
    if isinstance(entry.get("peaks_json"), str) and entry["peaks_json"].strip():
        return True
    peaks_obj = entry.get("peaks") or entry.get("Peaks") or entry.get("PEAKS")
    return isinstance(peaks_obj, list) and len(peaks_obj) > 0

def best_spectrum_id(entry: Dict[str, Any], fallback_prefix: str, counter: int) -> str:
    for k in ("spectrum_id", "SpectrumID", "id", "ID", "scan", "accession"):
        v = entry.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return f"{fallback_prefix}_{counter}"

# --- MSP parser ---
_MSP_KV_RE = re.compile(r"^\s*([^:]+)\s*:\s*(.*)\s*$")
_MSP_NUMPEAKS_KEYS = {"num peaks", "numpeaks", "number of peaks"}
_MSP_ID_KEYS = {"id", "spectrumid", "spectrum id", "recordid", "record id", "db#", "dbid", "accession"}
_MSP_PRECURSOR_KEYS = {"precursormz", "precursor mz", "precursor_mz", "precursor m/z", "parent mass", "pepmass"}

def _safe_open_text(path: Path):
    try:
        return path.open("r", encoding="utf-8", errors="strict")
    except UnicodeDecodeError:
        return path.open("r", encoding="utf-8", errors="replace")

def _parse_peak_line(line: str) -> Optional[Tuple[float, float]]:
    parts = re.split(r"[\s,;]+", line.strip())
    if len(parts) < 2:
        return None
    mz = _safe_float(parts[0])
    it = _safe_float(parts[1])
    if mz is None or it is None or it <= 0:
        return None
    return float(mz), float(it)

def iter_msp_entries(msp_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    msp_path = Path(msp_path)
    entry: Dict[str, Any] = {}
    peaks: List[Tuple[float, float]] = []
    in_peaks = False
    record_index = 0

    def flush():
        nonlocal entry, peaks, in_peaks, record_index
        if entry or peaks:
            record_index += 1
            out = dict(entry)
            out["peaks"] = list(peaks)
            if not out.get("spectrum_id"):
                out["spectrum_id"] = f"MSP_{record_index}"
            yield out
        entry = {}
        peaks = []
        in_peaks = False

    with _safe_open_text(msp_path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                yield from flush()
                continue

            m = _MSP_KV_RE.match(line)
            if m and not in_peaks:
                key = m.group(1).strip().lower()
                val = m.group(2).strip()

                if key == "name":
                    entry["name"] = val
                elif key in _MSP_ID_KEYS:
                    entry["spectrum_id"] = val
                elif key in _MSP_NUMPEAKS_KEYS:
                    in_peaks = True
                elif key in {"inchikey", "inchi key", "inchi_key"}:
                    entry["inchikey"] = val
                elif key in _MSP_PRECURSOR_KEYS:
                    entry["precursor_mz"] = _safe_float(val)
                elif key in {"adduct"}:
                    entry["adduct"] = val
                continue

            in_peaks = True
            pk = _parse_peak_line(line)
            if pk:
                peaks.append(pk)

    yield from flush()

# --- HMDB peaks count (optional) ---
def parse_hmdb_peaks(peaks_cell: Any) -> List[Tuple[float, float]]:
    if peaks_cell is None:
        return []
    if isinstance(peaks_cell, (list, tuple)):
        obj = peaks_cell
    else:
        s = str(peaks_cell).strip()
        if not s or s.lower() in {"nan", "none"}:
            return []
        obj = None
        try:
            obj = json.loads(s)
        except Exception:
            obj = None
        if obj is None:
            try:
                obj = ast.literal_eval(s)
            except Exception:
                obj = None

    out: List[Tuple[float, float]] = []
    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                mz = _safe_float(it[0])
                inten = _safe_float(it[1])
                if mz is None or inten is None or inten <= 0:
                    continue
                out.append((float(mz), float(inten)))
        out.sort(key=lambda x: x[0])
        return out

    # fallback regex
    s2 = str(peaks_cell)
    nums = [float(x) for x in _FLOAT_RE.findall(s2)]
    for i in range(0, len(nums) - 1, 2):
        mz, inten = nums[i], nums[i + 1]
        if inten > 0:
            out.append((float(mz), float(inten)))
    out.sort(key=lambda x: x[0])
    return out


# =======================
# Matching core
# =======================

@dataclass
class MatchRecord:
    input_inchikey: str
    source: str
    spectrum_id: str
    precursor_mz: Optional[float]
    name: str
    adduct: str
    n_peaks: Optional[int]

def _write_jsonl(path: Path, rec: Dict[str, Any]):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def scan_mona_or_gnps(json_path: Path, want: set[str], source_label: str, out_jsonl: Optional[Path]) -> List[MatchRecord]:
    if not json_path.exists():
        print(f"[SKIP] not found: {json_path}")
        return []
    matches: List[MatchRecord] = []
    counter = 0
    for entry in iter_json_entries_stream(json_path):
        if not isinstance(entry, dict):
            continue
        ik = _norm_inchikey(extract_inchikey_mona_gnps(entry))
        if not ik or ik not in want:
            continue
        if not has_peaks_mona_gnps(entry):
            # still count as presence? usually you want a spectrum with peaks
            continue
        counter += 1
        spec_id = best_spectrum_id(entry, source_label, counter)
        pmz = extract_precursor_mz(entry)
        nm = extract_name(entry)
        rec = MatchRecord(
            input_inchikey=ik,
            source=source_label,
            spectrum_id=str(spec_id),
            precursor_mz=float(pmz) if pmz is not None else None,
            name=nm,
            adduct=str(entry.get("adduct") or entry.get("Adduct") or ""),
            n_peaks=None
        )
        matches.append(rec)

        if out_jsonl:
            _write_jsonl(out_jsonl, {
                "source": source_label,
                "spectrum_id": spec_id,
                "inchikey": ik,
                "precursor_mz": pmz,
                "name": nm,
                "raw_entry": entry,  # keep original for later extraction if needed
            })
    return matches


def scan_msp(msp_path: Path, want: set[str], out_jsonl: Optional[Path]) -> List[MatchRecord]:
    if not msp_path.exists():
        print(f"[SKIP] not found: {msp_path}")
        return []
    matches: List[MatchRecord] = []
    for entry in iter_msp_entries(msp_path):
        ik = _norm_inchikey(str(entry.get("inchikey") or ""))
        if not ik or ik not in want:
            continue
        spec_id = str(entry.get("spectrum_id") or "")
        pmz = _safe_float(entry.get("precursor_mz"))
        nm = str(entry.get("name") or "")
        adduct = str(entry.get("adduct") or "")
        n_peaks = len(entry.get("peaks") or []) if isinstance(entry.get("peaks"), list) else None

        matches.append(MatchRecord(
            input_inchikey=ik,
            source="MSP",
            spectrum_id=spec_id,
            precursor_mz=float(pmz) if pmz is not None else None,
            name=nm,
            adduct=adduct,
            n_peaks=n_peaks
        ))

        if out_jsonl:
            _write_jsonl(out_jsonl, {
                "source": "MSP",
                "spectrum_id": spec_id,
                "inchikey": ik,
                "precursor_mz": pmz,
                "name": nm,
                "adduct": adduct,
                "peaks": entry.get("peaks", []),
            })
    return matches


def scan_hmdb(hmdb_json: Path, want: set[str], out_jsonl: Optional[Path]) -> List[MatchRecord]:
    if not hmdb_json.exists():
        print(f"[SKIP] not found: {hmdb_json}")
        return []
    matches: List[MatchRecord] = []
    counter = 0
    for entry in iter_json_entries_stream(hmdb_json):
        if not isinstance(entry, dict):
            continue
        # HMDB keys may differ in case
        lower_map = {str(k).lower(): k for k in entry.keys()}
        def _get(col: str) -> str:
            if col in entry:
                return str(entry.get(col) or "").strip()
            lk = col.lower()
            if lk in lower_map:
                return str(entry.get(lower_map[lk]) or "").strip()
            return ""

        ik = _norm_inchikey(_get(HMDB_INCHIKEY_COL))
        if not ik or ik not in want:
            continue

        counter += 1
        spec_id = _get(HMDB_SPECTRUM_ID_COL) or f"HMDB_{counter}"
        nm = _get(HMDB_NAME_COL)

        # HMDB may not have precursor_mz directly; keep MONO_MASS as precursor_mz=None
        pmz = _safe_float(_get(HMDB_MONO_COL))
        peaks_raw = entry.get(lower_map.get(HMDB_PEAKS_COL.lower(), HMDB_PEAKS_COL), None)
        n_peaks = len(parse_hmdb_peaks(peaks_raw)) if peaks_raw is not None else None

        matches.append(MatchRecord(
            input_inchikey=ik,
            source="HMDB",
            spectrum_id=spec_id,
            precursor_mz=float(pmz) if pmz is not None else None,
            name=nm,
            adduct=str(entry.get("adduct") or ""),
            n_peaks=n_peaks
        ))

        if out_jsonl:
            _write_jsonl(out_jsonl, {
                "source": "HMDB",
                "spectrum_id": spec_id,
                "inchikey": ik,
                "mono_mass": pmz,
                "name": nm,
                "peaks": peaks_raw,
                "raw_entry": entry,
            })
    return matches


def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV.resolve()}")

    df = pd.read_csv(INPUT_CSV)
    ik_col = _pick_col(df, INCHIKEY_COL_CANDIDATES)

    inchikeys = sorted({_norm_inchikey(x) for x in df[ik_col].dropna().astype(str).tolist() if _norm_inchikey(x)})
    want = set(inchikeys)
    print(f"[INFO] Input inchikey count: {len(inchikeys)}")

    # Scan each library
    matched_records: List[MatchRecord] = []

    mona_jsonl = OUTPUT_DIR / "matched_mona.jsonl" if EXPORT_MATCHED_JSONL else None
    gnps_jsonl = OUTPUT_DIR / "matched_gnps.jsonl" if EXPORT_MATCHED_JSONL else None
    msp_jsonl  = OUTPUT_DIR / "matched_msp.jsonl"  if EXPORT_MATCHED_JSONL else None
    hmdb_jsonl = OUTPUT_DIR / "matched_hmdb.jsonl" if EXPORT_MATCHED_JSONL else None

    print("[1/4] Scanning MoNA...")
    matched_records += scan_mona_or_gnps(MONA_JSON, want, "MoNA", mona_jsonl)
    print(f"  MoNA matches: {sum(1 for r in matched_records if r.source=='MoNA')}")

    print("[2/4] Scanning GNPS...")
    before = len(matched_records)
    matched_records += scan_mona_or_gnps(GNPS_JSON, want, "GNPS", gnps_jsonl)
    print(f"  GNPS matches: {len(matched_records)-before}")

    print("[3/4] Scanning MSP...")
    before = len(matched_records)
    matched_records += scan_msp(MSP_FILE, want, msp_jsonl)
    print(f"  MSP matches: {len(matched_records)-before}")

    print("[4/4] Scanning HMDB...")
    before = len(matched_records)
    matched_records += scan_hmdb(HMDB_JSON, want, hmdb_jsonl)
    print(f"  HMDB matches: {len(matched_records)-before}")

    # Build presence summary in-memory (per unique inchikey)


    pres = {ik: {"inchikey": ik, "in_mona": False, "in_gnps": False, "in_msp": False, "in_hmdb": False,


                 "mona_count": 0, "gnps_count": 0, "msp_count": 0, "hmdb_count": 0}


            for ik in inchikeys}



    for r in matched_records:


        row = pres.get(r.input_inchikey)


        if not row:


            continue


        if r.source == "MoNA":


            row["in_mona"] = True


            row["mona_count"] += 1


        elif r.source == "GNPS":


            row["in_gnps"] = True


            row["gnps_count"] += 1


        elif r.source == "MSP":


            row["in_msp"] = True


            row["msp_count"] += 1


        elif r.source == "HMDB":


            row["in_hmdb"] = True


            row["hmdb_count"] += 1



    # Still output the per-inchikey summary (optional / kept for compatibility)


    summary_df = pd.DataFrame(list(pres.values()))


    summary_out = OUTPUT_DIR / "inchikey_4libs_presence_summary.csv"


    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")



    # === NEW: fill results back into the input table and save as a new CSV ===


    # Normalize input inchikey column for mapping


    df["_inchikey_norm_"] = df[ik_col].astype(str).map(_norm_inchikey)



    def _map_col(colname: str):


        return df["_inchikey_norm_"].map(lambda k: pres.get(k, {}).get(colname))



    # add / overwrite columns in the original df


    df["in_mona"] = _map_col("in_mona").fillna(False).astype(bool)


    df["in_gnps"] = _map_col("in_gnps").fillna(False).astype(bool)


    df["in_msp"]  = _map_col("in_msp").fillna(False).astype(bool)


    df["in_hmdb"] = _map_col("in_hmdb").fillna(False).astype(bool)



    df["mona_count"] = _map_col("mona_count").fillna(0).astype(int)


    df["gnps_count"] = _map_col("gnps_count").fillna(0).astype(int)


    df["msp_count"]  = _map_col("msp_count").fillna(0).astype(int)


    df["hmdb_count"] = _map_col("hmdb_count").fillna(0).astype(int)



    # drop helper col and save a new file (do NOT overwrite the original input)


    df.drop(columns=["_inchikey_norm_"], inplace=True)


    filled_out = OUTPUT_DIR / "pred_match_inchikey_in_truth_filled.csv"


    df.to_csv(filled_out, index=False, encoding="utf-8-sig")



    print(f"[OK] Saved filled result CSV: {filled_out.resolve()}")

    # Matched spectrum records
    matches_out = OUTPUT_DIR / "inchikey_4libs_matches.csv"
    with matches_out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "input_inchikey", "source", "spectrum_id", "precursor_mz", "name", "adduct", "n_peaks"
        ])
        w.writeheader()
        for r in matched_records:
            w.writerow({
                "input_inchikey": r.input_inchikey,
                "source": r.source,
                "spectrum_id": r.spectrum_id,
                "precursor_mz": "" if r.precursor_mz is None else r.precursor_mz,
                "name": r.name,
                "adduct": r.adduct,
                "n_peaks": "" if r.n_peaks is None else r.n_peaks,
            })

    print("\n[OK] Done.")
    print(f"  Presence summary: {summary_out.resolve()}")
    print(f"  Match records:    {matches_out.resolve()}")
    if EXPORT_MATCHED_JSONL:
        print(f"  Matched jsonl:    {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run()
