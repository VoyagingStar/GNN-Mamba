# -*- coding: utf-8 -*-
"""
1) GNPS 谱库读取、QUERY CSV 扫描、实验谱聚合方式参考 GNPS_match_reverse_search.py
2) 候选筛选、矩阵构建、LASSO/NNLS 解卷积、结果输出逻辑参考 MS-DiAL_DecoID.py

用途：
- 读取 GNPS cleaned JSON / JSONL 谱库
- 读取实验 QUERY CSV（每个 precursor_mz 聚合成一个实验谱）
- 在 precursor m/z 容差范围内筛选 GNPS 候选谱
- 用 DecoID 风格的 LASSO / NNLS 对未知谱进行解卷积
- 输出：
    * 每个 query CSV 的 component ranking
    * 每个 query CSV 的 library weights
    * 所有 query CSV 的总表
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.linear_model as linModel


# ===================== CONFIG =====================

QUERY_CSV_DIR: Optional[str] = r"E:\model\singlecell\singlecell_data\HGC\SNR"
QUERY_GLOB: Optional[str] = None
QUERY_CSVS: List[str] = []
QUERY_FILENAME_MUST_CONTAIN: Optional[str] = "filtered"

GNPS_JSON_PATH: str = r"E:\model\singlecell\spectra_library\ALL_GNPS_cleaned.json"

OUTPUT_DIR: str = r"E:\model\singlecell\results\HGC\0330\GNPS_DecoID"

PRECURSOR_TOL: float = 1.0

COS_MZ_TOL: float = 0.05
MATRIX_MZ_TOL: float = 0.025
COS_USE_SQRT_INTENSITY: bool = True
NORMALIZE_TO: Optional[float] = 1000.0

DECOID_RESOLUTION: int = 4
DECOID_RES_PENALTY: float = -1.0
REDUNDANCY_CHECK_THRESH: float = 0.9

QUERY_MIN_REL_INTENSITY: float = 0.001
LIBRARY_MIN_REL_INTENSITY: float = 0.005
MIN_INITIAL_COSINE: float = 0.05
TOP_N_CANDIDATES: int = 80
MIN_MATCHED_PEAKS: int = 2
FALLBACK_TO_NNLS: bool = True
EXP_TOPK_PEAKS: Optional[int] = None
LIB_TOPK_PEAKS: Optional[int] = None


@dataclass
class GNPSSpectrumLite:
    spectrum_id: str
    compound_name: str
    precursor_mz: Optional[float]
    peaks: List[Tuple[float, float]]
    inchikey: str = ""
    smiles: str = ""
    formula: str = ""
    adduct: str = ""
    ion_mode: str = ""
    instrument: str = ""
    collision_energy: str = ""
    source_file: str = ""


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        import numpy as _np
        if isinstance(x, (_np.integer, _np.floating)):
            return float(x)
    except Exception:
        pass
    try:
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
            if not m:
                return None
            return float(m.group(0))
        return float(x)
    except Exception:
        return None


def flatten(l):
    if len(l) > 0 and type(l[0]) == type(l):
        return [item for sublist in l for item in sublist]
    return l


def _filename_ok(p: Path, must_contain: Optional[str]) -> bool:
    if not must_contain:
        return True
    return must_contain.lower() in p.name.lower()


def _read_csv_safely(path: Path) -> pd.DataFrame:
    for enc in ["utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace")


def iter_query_csvs(
    query_csv_dir: Optional[Union[str, Path]] = None,
    query_glob: Optional[str] = None,
    query_csvs: Optional[Iterable[Union[str, Path]]] = None,
    filename_must_contain: Optional[str] = QUERY_FILENAME_MUST_CONTAIN,
) -> List[Path]:
    paths: List[Path] = []

    if query_csv_dir:
        d = Path(query_csv_dir)
        if d.exists() and d.is_dir():
            paths.extend(sorted(d.glob("*.csv")))

    if query_glob:
        paths.extend(sorted(Path().glob(query_glob)))

    if query_csvs:
        paths.extend([Path(p) for p in query_csvs])

    uniq: Dict[str, Path] = {}
    for p in paths:
        if not p.exists():
            continue
        if not _filename_ok(p, filename_must_contain):
            continue
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        if rp not in uniq:
            uniq[rp] = p

    return list(uniq.values())


def keep_topk_peaks(peaks: List[Tuple[float, float]], k: Optional[int]) -> List[Tuple[float, float]]:
    if not k or k <= 0 or len(peaks) <= k:
        return peaks
    peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)[:k]
    peaks_sorted.sort(key=lambda x: x[0])
    return peaks_sorted


def normalize_vector(vec: List[float], method: str = "sum") -> List[float]:
    arr = np.asarray(vec, dtype=float)
    if arr.size == 0:
        return []
    scale = np.sum(arr) if method == "sum" else np.max(arr)
    if not np.isfinite(scale) or scale <= 0:
        return [0.0 for _ in vec]
    out = arr / scale
    if np.nanmax(out) > 1.1:
        return [0.0 for _ in vec]
    return out.tolist()


def clean_and_normalize_peaks(
    peaks: List[Tuple[float, float]],
    rel_cutoff: float = 0.0,
    normalize_to: Optional[float] = 1.0,
    merge_tol: float = 0.0,
    topk: Optional[int] = None,
) -> List[Tuple[float, float]]:
    clean = []
    for mz, inten in peaks:
        mz_f = _safe_float(mz)
        it_f = _safe_float(inten)
        if mz_f is None or it_f is None or mz_f <= 0 or it_f <= 0:
            continue
        clean.append((float(mz_f), float(it_f)))
    if not clean:
        return []

    clean.sort(key=lambda x: x[0])

    if merge_tol > 0:
        merged: List[List[float]] = []
        for mz, inten in clean:
            if merged and abs(mz - merged[-1][0]) <= merge_tol:
                prev_mz, prev_int = merged[-1]
                merged[-1][0] = (prev_mz * prev_int + mz * inten) / (prev_int + inten)
                merged[-1][1] = prev_int + inten
            else:
                merged.append([mz, inten])
        clean = [(float(mz), float(inten)) for mz, inten in merged]

    max_int = max(i for _, i in clean)
    if rel_cutoff > 0 and max_int > 0:
        clean = [(mz, inten) for mz, inten in clean if inten >= max_int * rel_cutoff]
    if not clean:
        return []

    if normalize_to is not None:
        max_int = max(i for _, i in clean)
        if max_int > 0:
            clean = [(mz, inten / max_int * float(normalize_to)) for mz, inten in clean]

    clean = keep_topk_peaks(clean, topk)
    return clean


def peaks_to_dict(peaks: List[Tuple[float, float]]) -> Dict[float, float]:
    return {float(mz): float(inten) for mz, inten in peaks}


def merge_mz_axis(mzs: List[float], mz_tol: float) -> List[float]:
    if not mzs:
        return []
    mzs = sorted(float(x) for x in mzs)
    axis = [[mzs[0]]]
    for mz in mzs[1:]:
        if abs(mz - np.mean(axis[-1])) <= mz_tol:
            axis[-1].append(mz)
        else:
            axis.append([mz])
    return [float(np.mean(group)) for group in axis]


def project_spectrum_to_axis(spec: Dict[float, float], axis: List[float], mz_tol: float) -> List[float]:
    if not axis:
        return []
    axis_arr = np.asarray(axis, dtype=float)
    out = np.zeros(len(axis_arr), dtype=float)
    for mz, inten in spec.items():
        diffs = np.abs(axis_arr - float(mz))
        idx = int(np.argmin(diffs))
        if diffs[idx] <= mz_tol:
            out[idx] += float(inten)
    return out.tolist()


def build_query_spectra_from_avg_csv(
    csv_path: Union[str, Path],
    normalize_to: Optional[float] = NORMALIZE_TO,
    min_peaks: int = MIN_MATCHED_PEAKS,
) -> List[Dict[str, Any]]:
    csv_path = Path(csv_path)
    df = _read_csv_safely(csv_path)
    print(f"Loaded query CSV: {csv_path} | rows={len(df)}")

    if "intensity" in df.columns:
        intensity_col_in = "intensity"
    elif "intensity_sample" in df.columns:
        intensity_col_in = "intensity_sample"
    else:
        raise ValueError(f"Query CSV missing intensity column: {csv_path}")

    required = {"precursor_mz", "mz", intensity_col_in}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Query CSV missing columns {missing}: {csv_path}")

    df[intensity_col_in] = pd.to_numeric(df[intensity_col_in], errors="coerce").fillna(0.0)
    df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
    df["precursor_mz"] = pd.to_numeric(df["precursor_mz"], errors="coerce")

    df = df[df[intensity_col_in] > 0].copy()
    df = df[df["mz"].notna() & df["precursor_mz"].notna()].copy()
    if df.empty:
        return []

    spectra: List[Dict[str, Any]] = []
    for prec, sub in df.groupby("precursor_mz", dropna=False):
        peaks = list(zip(sub["mz"].astype(float).tolist(), sub[intensity_col_in].astype(float).tolist()))
        peaks = clean_and_normalize_peaks(
            peaks,
            rel_cutoff=QUERY_MIN_REL_INTENSITY,
            normalize_to=normalize_to,
            merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
            topk=EXP_TOPK_PEAKS,
        )
        if len(peaks) < min_peaks:
            continue
        spectra.append({"precursor_mz": float(prec), "peaks": peaks})

    print(f"Built query spectra: {len(spectra)} (one per precursor)")
    return spectra


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


def extract_precursor_mz(entry: Dict[str, Any]) -> Optional[float]:
    for k in ("Precursor_MZ", "precursor_mz", "precursorMz", "PEPMASS", "pepmass"):
        if k in entry:
            mz = _safe_float(entry.get(k))
            if mz is not None:
                return mz
    v = _find_metadata_value(entry.get("metaData") or entry.get("metadata"), {"precursor m/z", "precursormz", "pepmass", "precursor"})
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
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        names = comp[0].get("names")
        if isinstance(names, list) and names and isinstance(names[0], dict):
            nm = names[0].get("name")
            if isinstance(nm, str) and nm.strip():
                return nm.strip()
    return ""


def extract_inchikey(entry: Dict[str, Any]) -> str:
    for k in ("InChIKey", "inchikey", "InChIKey_smiles"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    comp = entry.get("compound")
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        v = _find_metadata_value(comp[0].get("metaData"), {"inchikey"})
        if v:
            return v.strip()
    v = _find_metadata_value(entry.get("metaData") or entry.get("metadata"), {"inchikey"})
    return (v or "").strip()


def extract_smiles(entry: Dict[str, Any]) -> str:
    for k in ("Smiles", "SMILES", "smiles"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    comp = entry.get("compound")
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        v = _find_metadata_value(comp[0].get("metaData"), {"smiles"})
        if v:
            return v.strip()
    v = _find_metadata_value(entry.get("metaData") or entry.get("metadata"), {"smiles"})
    return (v or "").strip()


def extract_formula(entry: Dict[str, Any]) -> str:
    for k in ("formula", "Formula", "molecularFormula"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    comp = entry.get("compound")
    if isinstance(comp, list) and comp and isinstance(comp[0], dict):
        v = _find_metadata_value(comp[0].get("metaData"), {"molecular formula", "formula"})
        if v:
            return v.strip()
    v = _find_metadata_value(entry.get("metaData") or entry.get("metadata"), {"molecular formula", "formula"})
    return (v or "").strip()


def extract_adduct(entry: Dict[str, Any]) -> str:
    for k in ("Adduct", "adduct"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = _find_metadata_value(entry.get("metaData") or entry.get("metadata"), {"precursor type", "adduct"})
    return (v or "").strip()


def extract_ion_mode(entry: Dict[str, Any]) -> str:
    for k in ("Ion_Mode", "ionMode", "ion_mode"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    v = _find_metadata_value(entry.get("metaData") or entry.get("metadata"), {"ionization mode", "ion mode", "ionmode"})
    return (v or "").strip()


def extract_instrument(entry: Dict[str, Any]) -> str:
    for k in ("instrument", "Instrument", "msMassAnalyzer", "msManufacturer"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
            return v.strip()
    v = _find_metadata_value(entry.get("metaData") or entry.get("metadata"), {"instrument", "instrument type"})
    return (v or "").strip()


def extract_collision_energy(entry: Dict[str, Any]) -> str:
    for k in ("collision_energy", "collisionEnergy", "CE"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip() and v.strip().lower() != "nan":
            return v.strip()
        if isinstance(v, (int, float)):
            return str(v)
    v = _find_metadata_value(entry.get("metaData") or entry.get("metadata"), {"collision energy", "collisionenergy", "ce"})
    return (v or "").strip()


def parse_mona_like_spectrum_string(spectrum: str) -> List[Tuple[float, float]]:
    spectrum = (spectrum or "").strip()
    if not spectrum:
        return []
    peaks: List[Tuple[float, float]] = []
    for tok in re.split(r"\s+", spectrum):
        tok = tok.strip()
        if not tok:
            continue
        if ":" in tok:
            a, b = tok.split(":", 1)
        elif "," in tok:
            a, b = tok.split(",", 1)
        else:
            continue
        mz = _safe_float(a)
        it = _safe_float(b)
        if mz is None or it is None or it <= 0:
            continue
        peaks.append((float(mz), float(it)))
    peaks.sort(key=lambda x: x[0])
    return peaks


def parse_peaks(entry: Dict[str, Any]) -> List[Tuple[float, float]]:
    spec = entry.get("spectrum")
    if isinstance(spec, str) and spec.strip():
        return parse_mona_like_spectrum_string(spec)

    pj = entry.get("peaks_json")
    if isinstance(pj, str) and pj.strip():
        obj = None
        try:
            obj = json.loads(pj)
        except Exception:
            try:
                obj = ast.literal_eval(pj)
            except Exception:
                obj = None
        if isinstance(obj, list):
            out: List[Tuple[float, float]] = []
            for it in obj:
                if isinstance(it, (list, tuple)) and len(it) >= 2:
                    mz = _safe_float(it[0])
                    inten = _safe_float(it[1])
                    if mz is None or inten is None or inten <= 0:
                        continue
                    out.append((float(mz), float(inten)))
            out.sort(key=lambda x: x[0])
            return out
    return []


def has_peaks(entry: Dict[str, Any]) -> bool:
    if isinstance(entry.get("spectrum"), str) and entry["spectrum"].strip():
        return True
    if isinstance(entry.get("peaks_json"), str) and entry["peaks_json"].strip():
        return True
    return False


def best_spectrum_id(entry: Dict[str, Any], fallback_prefix: str, counter: int) -> str:
    for k in ("spectrum_id", "SpectrumID", "id", "ID", "scan", "accession"):
        v = entry.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return f"{fallback_prefix}_{counter}"


def _first_non_ws_char(path: Path) -> str:
    with path.open("rb") as f:
        while True:
            b = f.read(1)
            if not b:
                return ""
            if b not in b" \t\r\n":
                return chr(b[0])


def iter_json_entries_stream(path: Union[str, Path]) -> Iterator[Tuple[Dict[str, Any], str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GNPS JSON file not found: {p}")

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
                    yield obj, str(p)
        return

    ch = _first_non_ws_char(p)
    if ch == "[":
        try:
            import ijson  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"{p.name} looks like a large JSON array. Please install ijson for streaming:\n"
                f"  pip install ijson\n"
                f"Import error: {e}"
            )
        with p.open("rb") as f:
            for obj in ijson.items(f, "item"):
                if isinstance(obj, dict):
                    yield obj, str(p)
        return

    with p.open("r", encoding="utf-8", errors="replace") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        yield obj, str(p)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                yield it, str(p)


def iter_gnps_library(path: Union[str, Path]) -> Iterator[GNPSSpectrumLite]:
    counter = 0
    for entry, src in iter_json_entries_stream(path):
        if not isinstance(entry, dict) or not has_peaks(entry):
            continue
        peaks = parse_peaks(entry)
        if not peaks:
            continue

        precursor = extract_precursor_mz(entry)
        if precursor is None or precursor <= 0:
            continue

        peaks = clean_and_normalize_peaks(
            peaks,
            rel_cutoff=LIBRARY_MIN_REL_INTENSITY,
            normalize_to=NORMALIZE_TO,
            merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
            topk=LIB_TOPK_PEAKS,
        )
        if not peaks:
            continue

        counter += 1
        yield GNPSSpectrumLite(
            spectrum_id=best_spectrum_id(entry, "GNPS", counter),
            compound_name=extract_name(entry) or "Unknown",
            precursor_mz=float(precursor),
            peaks=peaks,
            inchikey=extract_inchikey(entry),
            smiles=extract_smiles(entry),
            formula=extract_formula(entry),
            adduct=extract_adduct(entry),
            ion_mode=extract_ion_mode(entry),
            instrument=extract_instrument(entry),
            collision_energy=extract_collision_energy(entry),
            source_file=str(src),
        )


def load_gnps_spectra(json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"GNPS library not found: {json_path}")

    print(f"Loading GNPS library: {json_path}")
    spectra: List[Dict[str, Any]] = []
    counter = 0

    for rec in iter_gnps_library(json_path):
        counter += 1
        spectra.append(
            {
                "gnps_id": rec.spectrum_id or f"GNPS_{counter}",
                "spectrum_id": rec.spectrum_id or f"GNPS_{counter}",
                "precursor_mz": float(rec.precursor_mz),
                "peaks": rec.peaks,
                "compound_name": rec.compound_name,
                "inchikey": rec.inchikey,
                "smiles": rec.smiles,
                "formula": rec.formula,
                "best_adduct": rec.adduct,
                "ion_mode": rec.ion_mode,
                "instrument": rec.instrument,
                "collision_energy": rec.collision_energy,
                "source_file": rec.source_file,
            }
        )
        if counter % 50000 == 0:
            print(f"  Parsed GNPS spectra: {counter:,}")

    print(f"Parsed GNPS spectra: {len(spectra)}")
    return spectra


def cosine_similarity(
    query_peaks: List[Tuple[float, float]],
    ref_peaks: List[Tuple[float, float]],
    mz_tol: float,
    use_sqrt: bool,
) -> Tuple[float, int, int, int]:
    if not query_peaks or not ref_peaks:
        return 0.0, 0, len(query_peaks), len(ref_peaks)

    query_peaks = sorted(query_peaks, key=lambda x: x[0])
    ref_peaks = sorted(ref_peaks, key=lambda x: x[0])

    q_mz = np.asarray([float(mz) for mz, _ in query_peaks], dtype=float)
    r_mz = np.asarray([float(mz) for mz, _ in ref_peaks], dtype=float)

    q_int = np.asarray([float(np.sqrt(float(i))) if use_sqrt else float(i) for _, i in query_peaks], dtype=float)
    r_int = np.asarray([float(np.sqrt(float(i))) if use_sqrt else float(i) for _, i in ref_peaks], dtype=float)

    nq = float(np.linalg.norm(q_int))
    nr = float(np.linalg.norm(r_int))
    if nq == 0.0 or nr == 0.0:
        return 0.0, 0, len(query_peaks), len(ref_peaks)

    q_int = q_int / nq
    r_int = r_int / nr

    n_q = len(query_peaks)
    n_r = len(ref_peaks)
    cost_matrix = np.zeros((n_q, n_r), dtype=float)

    j_start = 0
    for i, mz_q in enumerate(q_mz):
        while j_start < n_r and (mz_q - float(mz_tol) > r_mz[j_start]):
            j_start += 1
        j = j_start
        while j < n_r and abs(mz_q - r_mz[j]) <= float(mz_tol):
            cost_matrix[i, j] = q_int[i] * r_int[j]
            j += 1

    if not np.any(cost_matrix > 0.0):
        return 0.0, 0, n_q, n_r

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)

    score = 0.0
    n_matched = 0
    for row, col in zip(row_ind, col_ind):
        pair_score = float(cost_matrix[row, col])
        if pair_score > 0.0:
            score += pair_score
            n_matched += 1

    score = max(min(float(score), 1.0), -1.0)
    return score, n_matched, n_q, n_r


def estimate_res_penalty(num_candidates: int, query_vec: List[float], user_penalty: float) -> float:
    if user_penalty is not None and float(user_penalty) > 0:
        return float(user_penalty)
    nonzero = max(1, int(np.sum(np.asarray(query_vec) > 0)))
    alpha = 0.02 * np.sqrt(max(1, num_candidates)) / np.sqrt(nonzero)
    return float(max(alpha, 1e-4))


def deconvolveLASSO(A, b, lb, ub, resPenalty=10):
    sparseQ = True
    if resPenalty == 0:
        sparseQ = False
        resPenalty = 1

    A_arr = np.asarray(A, dtype=float, order="F")
    b_arr = np.asarray(flatten(b), dtype=float)

    if A_arr.ndim != 2:
        raise ValueError(f"A must be 2D, got shape={A_arr.shape}")
    if b_arr.ndim != 1:
        b_arr = b_arr.reshape(-1)
    if A_arr.shape[0] != b_arr.shape[0]:
        raise ValueError(f"Dimension mismatch: A.shape={A_arr.shape}, len(b)={b_arr.shape[0]}")
    if np.isinf(resPenalty):
        return [[0 for _ in range(A_arr.shape[1])], 0.0]

    if sparseQ:
        model = linModel.Lasso(alpha=float(resPenalty), fit_intercept=False, positive=True, max_iter=20000)
        model.fit(A_arr, b_arr)
        params = np.asarray(model.coef_, dtype=float)
        foundSpectra = np.dot(A_arr, params)
        denom = float(np.sum(np.abs(foundSpectra - b_arr)))
        s2nR = float(np.sum(foundSpectra) / denom) if denom > 0 else 0.0
        return [params.tolist(), s2nR]

    params, _ = scipy.optimize.nnls(A_arr, b_arr, maxiter=int(1e4))
    foundSpectra = np.dot(A_arr, params)
    denom = float(np.sum(np.abs(foundSpectra - b_arr)))
    s2nR = float(np.sum(foundSpectra) / denom) if denom > 0 else 0.0
    return [params.tolist(), s2nR]


def solveSystem(S, o, resPenalty, maxQuant=1000):
    return deconvolveLASSO(np.transpose(S), [[x] for x in o], [0 for _ in S], [maxQuant for _ in S], resPenalty=resPenalty)


def pullMostSimilarSpectra(trees, spectra):
    returnDict = {}
    query_peaks = [(mz, i) for mz, i in spectra.items()]
    for tree, ms2Scans in trees.items():
        if len(ms2Scans) > 0:
            temp = []
            for spec_id, ms2 in ms2Scans.items():
                score, n_matched, _, _ = cosine_similarity(query_peaks, [(mz, i) for mz, i in ms2.items()], mz_tol=float(COS_MZ_TOL), use_sqrt=bool(COS_USE_SQRT_INTENSITY))
                temp.append([spec_id, ms2, score, n_matched])
            temp.sort(key=lambda x: (x[2], x[3]), reverse=True)
            returnDict[tree] = temp[0]
    return returnDict


def getMatricesForGroup(trees, spectra, mz_merge_tol, top_n_candidates, min_initial_cosine):
    spectrum = dict(spectra[0])
    for spec in spectra[1:]:
        for m, i in spec.items():
            spectrum[m] = spectrum.get(m, 0.0) + i

    compoundDict = pullMostSimilarSpectra(trees, spectrum)
    kept_items = []
    for key, val in compoundDict.items():
        spec_id, ms2, score, n_matched = val
        if score >= float(min_initial_cosine) and n_matched >= int(MIN_MATCHED_PEAKS):
            kept_items.append((key, spec_id, ms2, score, n_matched))

    if not kept_items:
        temp = [(key, val[0], val[1], val[2], val[3]) for key, val in compoundDict.items()]
        temp.sort(key=lambda x: (x[3], x[4]), reverse=True)
        kept_items = temp[: min(max(5, top_n_candidates // 4), len(temp))]
    else:
        kept_items.sort(key=lambda x: (x[3], x[4]), reverse=True)
        kept_items = kept_items[: min(top_n_candidates, len(kept_items))]

    keys = [x[0] for x in kept_items]
    gnps_ids = [x[0] for x in keys]
    masses = [x[1] for x in keys]
    compound_names = [x[2] for x in keys]
    best_adducts = [x[3] for x in keys]
    inchikeys = [x[4] for x in keys]
    smiles_list = [x[5] for x in keys]
    formulas = [x[6] for x in keys]
    ion_modes = [x[7] for x in keys]
    instruments = [x[8] for x in keys]
    collision_energies = [x[9] for x in keys]
    source_files = [x[10] for x in keys]

    spectraIDs = [x[1] for x in kept_items]
    matrix_dicts = [x[2] for x in kept_items]
    initial_scores = [float(x[3]) for x in kept_items]
    initial_matches = [int(x[4]) for x in kept_items]

    axis_mzs = list(spectrum.keys())
    for m in matrix_dicts:
        axis_mzs.extend(list(m.keys()))
    axis = merge_mz_axis(axis_mzs, mz_merge_tol)

    matrix = [normalize_vector(project_spectrum_to_axis(m, axis, mz_merge_tol), method="sum") for m in matrix_dicts]
    reduceSpec = [normalize_vector(project_spectrum_to_axis(spec, axis, mz_merge_tol), method="sum") for spec in spectra]

    return (
        gnps_ids, spectraIDs, matrix, masses, compound_names, best_adducts, inchikeys,
        smiles_list, formulas, ion_modes, instruments, collision_energies,
        source_files, axis, reduceSpec, initial_scores, initial_matches
    )


def vector_to_peak_list(vec, axis, resolution):
    peaks = []
    for mz, val in zip(axis, vec):
        if float(val) > 0:
            peaks.append((float(np.round(mz, resolution)), float(val)))
    peaks.sort(key=lambda x: x[0])
    return peaks


def score_components_against_library(
    component_vectors,
    gnps_ids,
    spectraIDs,
    masses,
    compound_names,
    best_adducts,
    inchikeys,
    smiles_list,
    formulas,
    ion_modes,
    instruments,
    collision_energies,
    source_files,
    axis,
    resolution,
    initial_scores,
    initial_matches,
):
    rows = []

    for comp_info in component_vectors:
        component_name = comp_info["component_name"]
        component_vec = comp_info["component_vector"]
        component_abundance = comp_info["component_abundance"]
        component_weight_raw = comp_info["component_weight_raw"]
        component_weight_rank = comp_info["component_weight_rank"]
        component_source_spectrum_id = comp_info["source_spectrum_id"]
        component_source_compound_name = comp_info["source_compound_name"]
        component_peaks = vector_to_peak_list(component_vec, axis, resolution)

        component_hits = []
        for gnps_id, spec_id, mass, name, best_adduct, inchikey, smiles, formula, ion_mode, instrument, ce, source_file, db_vec, init_score, init_n_match in zip(
            gnps_ids, spectraIDs, masses, compound_names, best_adducts, inchikeys, smiles_list, formulas,
            ion_modes, instruments, collision_energies, source_files,
            comp_info["db_matrix"], initial_scores, initial_matches
        ):
            db_peaks = vector_to_peak_list(db_vec, axis, resolution)
            score, n_matched, n_q, n_r = cosine_similarity(component_peaks, db_peaks, mz_tol=float(COS_MZ_TOL), use_sqrt=bool(COS_USE_SQRT_INTENSITY))
            component_hits.append(
                {
                    "gnps_id": gnps_id,
                    "spectrum_id": spec_id,
                    "compound_name": name,
                    "inchikey": "" if inchikey is None else str(inchikey),
                    "smiles": "" if smiles is None else str(smiles),
                    "formula": "" if formula is None else str(formula),
                    "best_adduct": str(best_adduct),
                    "ion_mode": "" if ion_mode is None else str(ion_mode),
                    "instrument": "" if instrument is None else str(instrument),
                    "collision_energy": "" if ce is None else str(ce),
                    "source_file": "" if source_file is None else str(source_file),
                    "library_precursor_mz": float(mass),
                    "component_score": float(score),
                    "component_score_0_to_100": float(score * 100.0),
                    "initial_candidate_score": float(init_score),
                    "initial_candidate_n_matched": int(init_n_match),
                    "n_matched_peaks": int(n_matched),
                    "n_component_peaks": int(n_q),
                    "n_ref_peaks": int(n_r),
                    "component_name": component_name,
                    "component_source_spectrum_id": component_source_spectrum_id,
                    "component_source_compound_name": component_source_compound_name,
                    "component_source_inchikey": comp_info.get("source_inchikey", ""),
                    "component_source_smiles": comp_info.get("source_smiles", ""),
                    "component_source_formula": comp_info.get("source_formula", ""),
                    "component_source_best_adduct": comp_info["best_adduct"],
                    "component_source_precursor_mz": float(comp_info["component_mass"]),
                    "component_abundance": float(component_abundance),
                    "component_weight_raw": float(component_weight_raw),
                    "component_weight_rank": int(component_weight_rank),
                }
            )

        component_hits.sort(key=lambda x: (x["component_score"], x["initial_candidate_score"]), reverse=True)
        for rank, row in enumerate(component_hits, start=1):
            row["component_hit_rank"] = rank
            rows.append(row)

    return rows


def build_library_weight_rows(
    query_precursor_mz,
    res_vector,
    gnps_ids,
    spectraIDs,
    masses,
    compound_names,
    best_adducts,
    inchikeys,
    smiles_list,
    formulas,
    ion_modes,
    instruments,
    collision_energies,
    source_files,
    initial_scores,
    initial_matches,
):
    rows = []
    res = np.asarray(res_vector, dtype=float)
    positive_sum = float(np.sum(res[res > 0])) if np.any(res > 0) else 0.0

    order = sorted(range(len(res)), key=lambda i: float(res[i]), reverse=True)
    rank_map = {idx: rank for rank, idx in enumerate(order, start=1)}

    for idx, (gnps_id, spec_id, mass, name, adduct, inchikey, smiles, formula, ion_mode, instrument, ce, source_file, init_score, init_n_match) in enumerate(
        zip(
            gnps_ids, spectraIDs, masses, compound_names, best_adducts, inchikeys, smiles_list, formulas,
            ion_modes, instruments, collision_energies, source_files,
            initial_scores, initial_matches
        )
    ):
        raw_weight = float(res[idx]) if idx < len(res) else 0.0
        norm_weight = float(raw_weight / positive_sum) if positive_sum > 0 and raw_weight > 0 else 0.0
        rows.append(
            {
                "precursor_mz": float(query_precursor_mz),
                "gnps_id": gnps_id,
                "spectrum_id": spec_id,
                "compound_name": name,
                "inchikey": "" if inchikey is None else str(inchikey),
                "smiles": "" if smiles is None else str(smiles),
                "formula": "" if formula is None else str(formula),
                "best_adduct": str(adduct),
                "ion_mode": "" if ion_mode is None else str(ion_mode),
                "instrument": "" if instrument is None else str(instrument),
                "collision_energy": "" if ce is None else str(ce),
                "source_file": "" if source_file is None else str(source_file),
                "library_precursor_mz": float(mass),
                "initial_candidate_score": float(init_score),
                "initial_candidate_n_matched": int(init_n_match),
                "lasso_weight_raw": raw_weight,
                "lasso_weight_normalized": norm_weight,
                "component_score": 0.0,
                "weight_rank": int(rank_map[idx]),
                "is_active_component": bool(raw_weight > 1e-12),
            }
        )
    return rows


def attach_component_scores_to_library_weights(weight_rows, ranking_rows):
    if not weight_rows:
        return weight_rows
    score_map = {}
    for row in ranking_rows or []:
        component_name = str(row.get("component_name", "")).strip()
        gnps_id = str(row.get("gnps_id", "")).strip()
        if component_name and gnps_id and component_name == gnps_id:
            score = float(row.get("component_score", 0.0))
            if component_name not in score_map or score > score_map[component_name]:
                score_map[component_name] = score

    for row in weight_rows:
        gnps_id = str(row.get("gnps_id", "")).strip()
        row["component_score"] = float(score_map.get(gnps_id, 0.0))
    return weight_rows


def build_component_vectors(
    originalSpectra,
    matrix,
    res,
    gnps_ids,
    masses,
    best_adducts,
    spectraIDs,
    compound_names,
    inchikeys,
    smiles_list,
    formulas,
):
    component_rows = []
    res = np.asarray(res, dtype=float)
    positive = np.where(res > 0)[0]
    if positive.size == 0:
        return component_rows

    original_arr = np.asarray(flatten(originalSpectra), dtype=float)
    if len(matrix) > 0:
        solved_spectra_total = np.dot(np.transpose(np.asarray(matrix, dtype=float)), res)
        num_components = max(1, int(len(positive)))
        differences = np.subtract(original_arr, solved_spectra_total) / float(num_components)
    else:
        differences = original_arr

    resSum = float(np.sum(res[positive])) if np.sum(res[positive]) > 0 else 1.0
    positive_order = list(positive[np.argsort(res[positive])[::-1]])
    positive_rank_map = {idx: rank for rank, idx in enumerate(positive_order, start=1)}

    for x in positive:
        component_vec = [max(float(res[x] * m + d), 0.0) for m, d in zip(np.asarray(matrix[x], dtype=float), differences)]
        component_rows.append(
            {
                "component_name": str(gnps_ids[x]),
                "component_abundance": float(res[x] / resSum),
                "component_weight_raw": float(res[x]),
                "component_weight_rank": int(positive_rank_map[x]),
                "component_vector": component_vec,
                "component_mass": float(masses[x]),
                "best_adduct": str(best_adducts[x]),
                "source_spectrum_id": str(spectraIDs[x]),
                "source_compound_name": str(compound_names[x]),
                "source_inchikey": "" if inchikeys[x] is None else str(inchikeys[x]),
                "source_smiles": "" if smiles_list[x] is None else str(smiles_list[x]),
                "source_formula": "" if formulas[x] is None else str(formulas[x]),
                "db_matrix": matrix,
            }
        )
    return component_rows


class GNPSLibraryAdapter:
    def __init__(self, gnps_spectra: List[Dict[str, Any]]):
        self.gnps_spectra = gnps_spectra

    def get_candidate_trees(self, lower_bound: float, upper_bound: float):
        trees: Dict[Tuple[str, float, str, str, str, str, str, str, str, str, str], Dict[str, Dict[float, float]]] = {}

        for rec in self.gnps_spectra:
            mz = float(rec["precursor_mz"])
            if lower_bound <= mz <= upper_bound:
                key = (
                    rec["gnps_id"],
                    mz,
                    rec.get("compound_name", "Unknown"),
                    rec.get("best_adduct", ""),
                    rec.get("inchikey", ""),
                    rec.get("smiles", ""),
                    rec.get("formula", ""),
                    rec.get("ion_mode", ""),
                    rec.get("instrument", ""),
                    rec.get("collision_energy", ""),
                    rec.get("source_file", ""),
                )
                spec_id = str(rec.get("spectrum_id", rec.get("gnps_id", "GNPS")))
                spec_dict = peaks_to_dict(rec["peaks"])
                trees.setdefault(key, {})[spec_id] = spec_dict

        return trees


def deconvolve_and_score_one_query(
    query_precursor_mz: float,
    query_peaks: List[Tuple[float, float]],
    library: GNPSLibraryAdapter,
    precursor_tol: float = PRECURSOR_TOL,
    res_penalty: float = DECOID_RES_PENALTY,
    resolution: int = DECOID_RESOLUTION,
) -> Dict[str, pd.DataFrame]:
    query_peaks = clean_and_normalize_peaks(
        query_peaks,
        rel_cutoff=QUERY_MIN_REL_INTENSITY,
        normalize_to=NORMALIZE_TO,
        merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
        topk=EXP_TOPK_PEAKS,
    )
    query_spec_dict = peaks_to_dict(query_peaks)

    empty_rank_row = {
        "precursor_mz": float(query_precursor_mz),
        "component_name": "",
        "component_source_spectrum_id": "",
        "component_source_compound_name": "",
        "component_source_inchikey": "",
        "component_source_smiles": "",
        "component_source_formula": "",
        "component_source_best_adduct": "",
        "component_source_precursor_mz": np.nan,
        "component_abundance": np.nan,
        "component_weight_raw": np.nan,
        "component_weight_rank": 0,
        "component_hit_rank": 0,
        "gnps_id": "",
        "spectrum_id": "",
        "compound_name": "",
        "inchikey": "",
        "smiles": "",
        "formula": "",
        "best_adduct": "",
        "ion_mode": "",
        "instrument": "",
        "collision_energy": "",
        "source_file": "",
        "library_precursor_mz": np.nan,
        "component_score": 0.0,
        "component_score_0_to_100": 0.0,
        "n_matched_peaks": 0,
        "n_component_peaks": len(query_peaks),
        "n_ref_peaks": 0,
        "n_candidates": 0,
        "s2n": np.nan,
        "lasso_alpha": np.nan,
    }

    empty_weight_row = {
        "precursor_mz": float(query_precursor_mz),
        "gnps_id": "",
        "spectrum_id": "",
        "compound_name": "",
        "inchikey": "",
        "smiles": "",
        "formula": "",
        "best_adduct": "",
        "ion_mode": "",
        "instrument": "",
        "collision_energy": "",
        "source_file": "",
        "library_precursor_mz": np.nan,
        "initial_candidate_score": 0.0,
        "initial_candidate_n_matched": 0,
        "lasso_weight_raw": np.nan,
        "lasso_weight_normalized": np.nan,
        "component_score": 0.0,
        "weight_rank": 0,
        "is_active_component": False,
        "n_candidates": 0,
        "s2n": np.nan,
        "lasso_alpha": np.nan,
    }

    lower_bound = float(query_precursor_mz) - float(precursor_tol)
    upper_bound = float(query_precursor_mz) + float(precursor_tol)
    trees = library.get_candidate_trees(lower_bound=lower_bound, upper_bound=upper_bound)

    if not trees:
        return {"ranking_df": pd.DataFrame([empty_rank_row]), "weights_df": pd.DataFrame([empty_weight_row])}

    (
        gnps_ids, spectraIDs, matrix, masses, compound_names, best_adducts, inchikeys,
        smiles_list, formulas, ion_modes, instruments, collision_energies,
        source_files, axis, reduceSpec, initial_scores, initial_matches
    ) = getMatricesForGroup(
        trees=trees,
        spectra=[query_spec_dict],
        mz_merge_tol=float(MATRIX_MZ_TOL),
        top_n_candidates=int(TOP_N_CANDIDATES),
        min_initial_cosine=float(MIN_INITIAL_COSINE),
    )

    if len(matrix) == 0:
        return {"ranking_df": pd.DataFrame([empty_rank_row]), "weights_df": pd.DataFrame([empty_weight_row])}

    query_vec = reduceSpec[0]
    use_penalty = estimate_res_penalty(len(matrix), query_vec, float(res_penalty))
    res_vector, s2n = solveSystem(matrix, query_vec, resPenalty=float(use_penalty))

    if FALLBACK_TO_NNLS and np.sum(np.asarray(res_vector) > 1e-12) == 0:
        res_vector, s2n = solveSystem(matrix, query_vec, resPenalty=0.0)
        use_penalty = 0.0

    component_vectors = build_component_vectors(
        originalSpectra=query_vec,
        matrix=matrix,
        res=res_vector,
        gnps_ids=gnps_ids,
        masses=masses,
        best_adducts=best_adducts,
        spectraIDs=spectraIDs,
        compound_names=compound_names,
        inchikeys=inchikeys,
        smiles_list=smiles_list,
        formulas=formulas,
    )

    rows = score_components_against_library(
        component_vectors=component_vectors,
        gnps_ids=gnps_ids,
        spectraIDs=spectraIDs,
        masses=masses,
        compound_names=compound_names,
        best_adducts=best_adducts,
        inchikeys=inchikeys,
        smiles_list=smiles_list,
        formulas=formulas,
        ion_modes=ion_modes,
        instruments=instruments,
        collision_energies=collision_energies,
        source_files=source_files,
        axis=axis,
        resolution=resolution,
        initial_scores=initial_scores,
        initial_matches=initial_matches,
    )

    weight_rows = build_library_weight_rows(
        query_precursor_mz=float(query_precursor_mz),
        res_vector=res_vector,
        gnps_ids=gnps_ids,
        spectraIDs=spectraIDs,
        masses=masses,
        compound_names=compound_names,
        best_adducts=best_adducts,
        inchikeys=inchikeys,
        smiles_list=smiles_list,
        formulas=formulas,
        ion_modes=ion_modes,
        instruments=instruments,
        collision_energies=collision_energies,
        source_files=source_files,
        initial_scores=initial_scores,
        initial_matches=initial_matches,
    )
    weight_rows = attach_component_scores_to_library_weights(weight_rows, rows)

    df_weights = pd.DataFrame(weight_rows) if weight_rows else pd.DataFrame([empty_weight_row])
    df_weights["n_candidates"] = int(len(gnps_ids))
    df_weights["s2n"] = float(s2n)
    df_weights["lasso_alpha"] = float(use_penalty)
    df_weights = df_weights.sort_values(["weight_rank", "lasso_weight_raw", "initial_candidate_score"], ascending=[True, False, False]).reset_index(drop=True)

    if not rows:
        rank_df = pd.DataFrame([{**empty_rank_row, "n_candidates": int(len(gnps_ids)), "s2n": float(s2n), "lasso_alpha": float(use_penalty)}])
        return {"ranking_df": rank_df, "weights_df": df_weights}

    df = pd.DataFrame(rows)
    df.insert(0, "precursor_mz", float(query_precursor_mz))
    df["n_candidates"] = int(len(gnps_ids))
    df["s2n"] = float(s2n)
    df["lasso_alpha"] = float(use_penalty)
    df = df.sort_values(["component_weight_rank", "component_name", "component_hit_rank", "component_score"], ascending=[True, True, True, False]).reset_index(drop=True)

    return {"ranking_df": df, "weights_df": df_weights}


def process_one_query_csv(query_csv: Path, gnps_spectra: List[Dict[str, Any]], out_root: Path) -> Dict[str, pd.DataFrame]:
    query_spectra = build_query_spectra_from_avg_csv(query_csv)
    if not query_spectra:
        print(f"[WARN] No query spectra after filtering: {query_csv.name}")
        return {"ranking_df": pd.DataFrame(), "weights_df": pd.DataFrame()}

    library = GNPSLibraryAdapter(gnps_spectra)
    all_parts = []
    all_weight_parts = []

    for q in query_spectra:
        result_one = deconvolve_and_score_one_query(
            query_precursor_mz=float(q["precursor_mz"]),
            query_peaks=q["peaks"],
            library=library,
            precursor_tol=PRECURSOR_TOL,
            res_penalty=DECOID_RES_PENALTY,
            resolution=DECOID_RESOLUTION,
        )
        df_one = result_one["ranking_df"].copy()
        df_one.insert(0, "query_csv", query_csv.name)
        all_parts.append(df_one)

        df_weight_one = result_one["weights_df"].copy()
        df_weight_one.insert(0, "query_csv", query_csv.name)
        all_weight_parts.append(df_weight_one)

    result = pd.concat(all_parts, axis=0, ignore_index=True) if all_parts else pd.DataFrame()
    if not result.empty:
        result = result.sort_values(["precursor_mz", "component_weight_rank", "component_name", "component_hit_rank"], ascending=[True, True, True, True]).reset_index(drop=True)

    result_weights = pd.concat(all_weight_parts, axis=0, ignore_index=True) if all_weight_parts else pd.DataFrame()
    if not result_weights.empty:
        result_weights = result_weights.sort_values(["precursor_mz", "weight_rank", "lasso_weight_raw", "initial_candidate_score"], ascending=[True, True, False, False]).reset_index(drop=True)

    out_file = out_root / f"{query_csv.stem}_decoid_component_ranking.csv"
    result.to_csv(out_file, index=False, encoding="utf-8-sig")

    out_weights = out_root / f"{query_csv.stem}_decoid_library_weights.csv"
    result_weights.to_csv(out_weights, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved per-query deconvolution summary: {out_file.resolve()}")
    print(f"[OK] Saved per-query library weights: {out_weights.resolve()}")
    return {"ranking_df": result, "weights_df": result_weights}


def main():
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    query_files = iter_query_csvs(
        query_csv_dir=QUERY_CSV_DIR,
        query_glob=QUERY_GLOB,
        query_csvs=QUERY_CSVS,
        filename_must_contain=QUERY_FILENAME_MUST_CONTAIN,
    )

    if not query_files:
        raise FileNotFoundError(
            f"No QUERY_CSV files found with filename containing {QUERY_FILENAME_MUST_CONTAIN!r}. "
            "Please check QUERY_CSV_DIR / QUERY_GLOB / QUERY_CSVS."
        )

    print(f"QUERY CSV files to process: {len(query_files)}")
    for p in query_files:
        print(f"  - {p}")

    gnps_spectra = load_gnps_spectra(GNPS_JSON_PATH)
    if not gnps_spectra:
        raise RuntimeError("No GNPS spectra parsed. Check GNPS_JSON_PATH.")

    all_dfs = []
    all_weight_dfs = []

    for qcsv in query_files:
        result_one = process_one_query_csv(qcsv, gnps_spectra, out_root)
        df_one = result_one["ranking_df"]
        df_weight_one = result_one["weights_df"]

        if not df_one.empty:
            all_dfs.append(df_one)
        if not df_weight_one.empty:
            all_weight_dfs.append(df_weight_one)

    df_all = pd.concat(all_dfs, axis=0, ignore_index=True) if all_dfs else pd.DataFrame()
    out_all = out_root / "decoid_component_ranking_all_queries.csv"
    df_all.to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved overall summary: {out_all.resolve()}")

    df_all_weights = pd.concat(all_weight_dfs, axis=0, ignore_index=True) if all_weight_dfs else pd.DataFrame()
    out_all_weights = out_root / "decoid_library_weights_all_queries.csv"
    df_all_weights.to_csv(out_all_weights, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved overall library weights: {out_all_weights.resolve()}")


if __name__ == "__main__":
    main()
