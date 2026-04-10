# -*- coding: utf-8 -*-
"""
1) 数据集读取、QUERY CSV 扫描、MSP 流式解析方式参考 MSP_match_reverse_search.py
2) 候选筛选、矩阵构建、LASSO/NNLS 解卷积、结果输出逻辑参考 MoNA_DecoID.py

用途：
- 读取 MS-DIAL 导出的 MSP 质谱库
- 读取实验 QUERY CSV（每个 precursor_mz 聚合成一个实验谱）
- 在 precursor m/z 容差范围内筛选候选库谱
- 用 DecoID 风格的 LASSO / NNLS 对未知谱进行解卷积
- 输出：
    * 每个 query CSV 的 component ranking
    * 每个 query CSV 的 library weights
    * 所有 query CSV 的总表
"""

from __future__ import annotations

import ast
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.linear_model as linModel


# ===================== CONFIG =====================

# Query CSV 来源
QUERY_CSV_DIR: Optional[str] = r"E:\model\singlecell\singlecell_data\HGC\SNR"
QUERY_GLOB: Optional[str] = None
QUERY_CSVS: List[str] = []
QUERY_FILENAME_MUST_CONTAIN: Optional[str] = "filtered"

# MS-DIAL MSP 谱库
MSDIAL_LIB_PATH: str = r"E:\model\singlecell\spectra_library\MSMS-Public_experimentspectra-pos-VS19.msp"

# 输出目录
OUTPUT_DIR: str = r"E:\model\singlecell\results\HGC\0330\MS-DiAL_DecoID"

# precursor 筛选
PRECURSOR_TOL: float = 1.0

# 峰匹配 / 矩阵投影
COS_MZ_TOL: float = 0.05
MATRIX_MZ_TOL: float = 0.025
COS_USE_SQRT_INTENSITY: bool = False
NORMALIZE_TO: Optional[float] = 1.0

# 解卷积参数
DECOID_RESOLUTION: int = 4
DECOID_RES_PENALTY: float = -1.0
REDUNDANCY_CHECK_THRESH: float = 0.9

# 谱清洗参数
QUERY_MIN_REL_INTENSITY: float = 0.001
LIBRARY_MIN_REL_INTENSITY: float = 0.005
MIN_INITIAL_COSINE: float = 0.05
TOP_N_CANDIDATES: int = 80
MIN_MATCHED_PEAKS: int = 2
FALLBACK_TO_NNLS: bool = True


# ===================== Data Classes =====================

@dataclass
class MSDialSpectrumLite:
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
    ontology: str = ""
    source_file: str = ""


# ===================== Generic Helpers =====================

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        import numpy as _np
        if isinstance(x, (_np.integer, _np.floating)):
            return float(x)
    except Exception:
        pass

    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return None

    s2 = s.replace(",", "")
    try:
        return float(s2)
    except Exception:
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s2)
        if not m:
            return None
        try:
            return float(m.group(0))
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


# ===================== QUERY CSV Loader =====================

def build_query_spectra_from_avg_csv(
    csv_path: Union[str, Path],
    normalize_to: Optional[float] = NORMALIZE_TO,
    min_peaks: int = MIN_MATCHED_PEAKS,
) -> List[Dict[str, Any]]:
    """
    每个 precursor_mz 聚合成一个实验谱
    必需列：
        precursor_mz, mz, intensity
    或
        precursor_mz, mz, intensity_sample
    """
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
        if sub.empty:
            continue

        peaks = list(
            zip(
                sub["mz"].astype(float).tolist(),
                sub[intensity_col_in].astype(float).tolist(),
            )
        )

        peaks = clean_and_normalize_peaks(
            peaks,
            rel_cutoff=QUERY_MIN_REL_INTENSITY,
            normalize_to=normalize_to,
            merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
        )

        if len(peaks) < min_peaks:
            continue

        spectra.append(
            {
                "precursor_mz": float(prec),
                "peaks": peaks,
            }
        )

    print(f"Built query spectra: {len(spectra)} (one per precursor)")
    return spectra


# ===================== MSP / MS-DIAL Library Parser =====================

_MSP_KV_RE = re.compile(r"^\s*([^:]+)\s*:\s*(.*)\s*$")
_MSP_NUMPEAKS_KEYS = {"num peaks", "numpeaks", "number of peaks"}
_MSP_ID_KEYS = {"id", "spectrumid", "spectrum id", "recordid", "record id", "db#", "dbid", "accession"}
_MSP_PRECURSOR_KEYS = {"precursormz", "precursor mz", "precursor_mz", "precursor m/z", "parent mass", "pepmass"}
_MSP_NAME_KEYS = {"name", "title"}
_MSP_ADDUCT_KEYS = {"adduct", "precursor type", "precursortype"}
_MSP_IONMODE_KEYS = {"ionmode", "ion mode", "ionization mode", "polarity"}
_MSP_INSTRUMENT_KEYS = {"instrument", "instrumenttype", "instrument type"}
_MSP_CE_KEYS = {"collisionenergy", "collision energy", "ce", "nce"}
_MSP_FORMULA_KEYS = {"formula", "molecular formula"}
_MSP_SMILES_KEYS = {"smiles"}
_MSP_INCHIKEY_KEYS = {"inchikey", "inchi key", "inchi_key"}
_MSP_ONTOLOGY_KEYS = {"ontology", "class", "superclass"}


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


def iter_msp_entries(msp_path: Union[str, Path]) -> Iterator[Tuple[Dict[str, Any], str]]:
    """
    流式读取 MSP，每条记录以空行分隔。
    """
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
                out["spectrum_id"] = f"MSDIAL_{record_index}"
            yield out, str(msp_path)
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

                if key in _MSP_NAME_KEYS:
                    entry["compound_name"] = val
                elif key in _MSP_ID_KEYS:
                    entry["spectrum_id"] = val
                elif key in _MSP_NUMPEAKS_KEYS:
                    in_peaks = True
                elif key in _MSP_PRECURSOR_KEYS:
                    entry["precursor_mz"] = _safe_float(val)
                elif key in _MSP_INCHIKEY_KEYS:
                    entry["inchikey"] = val
                elif key in _MSP_SMILES_KEYS:
                    entry["smiles"] = val
                elif key in _MSP_FORMULA_KEYS:
                    entry["formula"] = val
                elif key in _MSP_ADDUCT_KEYS:
                    entry["adduct"] = val
                elif key in _MSP_IONMODE_KEYS:
                    entry["ion_mode"] = val
                elif key in _MSP_INSTRUMENT_KEYS:
                    entry["instrument"] = val
                elif key in _MSP_CE_KEYS:
                    entry["collision_energy"] = val
                elif key in _MSP_ONTOLOGY_KEYS:
                    entry["ontology"] = val
                continue

            in_peaks = True
            pk = _parse_peak_line(line)
            if pk:
                peaks.append(pk)

    yield from flush()


def iter_msdial_library(
    path: Union[str, Path],
    ion_mode_keep: Optional[str] = None,
) -> Iterator[MSDialSpectrumLite]:
    for entry, src in iter_msp_entries(path):
        peaks = entry.get("peaks") or []
        if not isinstance(peaks, list) or not peaks:
            continue

        precursor = _safe_float(entry.get("precursor_mz"))
        if precursor is None or precursor <= 0:
            continue

        ion_mode = str(entry.get("ion_mode") or "").strip()
        if ion_mode_keep is not None and ion_mode:
            if ion_mode.lower() != str(ion_mode_keep).strip().lower():
                continue

        peaks = clean_and_normalize_peaks(
            peaks,
            rel_cutoff=LIBRARY_MIN_REL_INTENSITY,
            normalize_to=NORMALIZE_TO,
            merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
        )
        if not peaks:
            continue

        yield MSDialSpectrumLite(
            spectrum_id=str(entry.get("spectrum_id") or ""),
            compound_name=str(entry.get("compound_name") or "Unknown"),
            precursor_mz=float(precursor),
            peaks=peaks,
            inchikey=str(entry.get("inchikey") or ""),
            smiles=str(entry.get("smiles") or ""),
            formula=str(entry.get("formula") or ""),
            adduct=str(entry.get("adduct") or ""),
            ion_mode=ion_mode,
            instrument=str(entry.get("instrument") or ""),
            collision_energy=str(entry.get("collision_energy") or ""),
            ontology=str(entry.get("ontology") or ""),
            source_file=str(src),
        )


def load_msdial_spectra(
    msp_path: Union[str, Path],
    ion_mode_keep: Optional[str] = None,
) -> List[Dict[str, Any]]:
    msp_path = Path(msp_path)
    if not msp_path.exists():
        raise FileNotFoundError(f"MS-DIAL library not found: {msp_path}")

    print(f"Loading MS-DIAL library: {msp_path}")
    spectra: List[Dict[str, Any]] = []
    counter = 0

    for rec in iter_msdial_library(msp_path, ion_mode_keep=ion_mode_keep):
        counter += 1
        spectra.append(
            {
                "msdial_id": rec.spectrum_id or f"MSDIAL_{counter}",
                "spectrum_id": rec.spectrum_id or f"MSDIAL_{counter}",
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
                "ontology": rec.ontology,
                "source_file": rec.source_file,
            }
        )

        if counter % 50000 == 0:
            print(f"  Parsed library spectra: {counter:,}")

    print(f"Parsed MS-DIAL spectra: {len(spectra)}")
    return spectra


# ===================== Cosine Similarity =====================

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

    q_int = np.asarray(
        [float(np.sqrt(float(i))) if use_sqrt else float(i) for _, i in query_peaks],
        dtype=float,
    )
    r_int = np.asarray(
        [float(np.sqrt(float(i))) if use_sqrt else float(i) for _, i in ref_peaks],
        dtype=float,
    )

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


# ===================== DecoID-style Core =====================

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
        model = linModel.Lasso(
            alpha=float(resPenalty),
            fit_intercept=False,
            positive=True,
            max_iter=20000,
        )
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
    return deconvolveLASSO(
        np.transpose(S),
        [[x] for x in o],
        [0 for _ in S],
        [maxQuant for _ in S],
        resPenalty=resPenalty,
    )


def pullMostSimilarSpectra(trees, spectra):
    returnDict = {}
    query_peaks = [(mz, i) for mz, i in spectra.items()]
    for tree, ms2Scans in trees.items():
        if len(ms2Scans) > 0:
            temp = []
            for spec_id, ms2 in ms2Scans.items():
                score, n_matched, _, _ = cosine_similarity(
                    query_peaks,
                    [(mz, i) for mz, i in ms2.items()],
                    mz_tol=float(COS_MZ_TOL),
                    use_sqrt=bool(COS_USE_SQRT_INTENSITY),
                )
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
    msdial_ids = [x[0] for x in keys]
    masses = [x[1] for x in keys]
    compound_names = [x[2] for x in keys]
    best_adducts = [x[3] for x in keys]
    inchikeys = [x[4] for x in keys]
    smiles_list = [x[5] for x in keys]
    formulas = [x[6] for x in keys]
    ion_modes = [x[7] for x in keys]
    instruments = [x[8] for x in keys]
    collision_energies = [x[9] for x in keys]
    ontologies = [x[10] for x in keys]
    source_files = [x[11] for x in keys]

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
        msdial_ids, spectraIDs, matrix, masses, compound_names, best_adducts, inchikeys,
        smiles_list, formulas, ion_modes, instruments, collision_energies, ontologies,
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
    msdial_ids,
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
    ontologies,
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
        for msdial_id, spec_id, mass, name, best_adduct, inchikey, smiles, formula, ion_mode, instrument, ce, ontology, source_file, db_vec, init_score, init_n_match in zip(
            msdial_ids, spectraIDs, masses, compound_names, best_adducts, inchikeys, smiles_list, formulas,
            ion_modes, instruments, collision_energies, ontologies, source_files,
            comp_info["db_matrix"], initial_scores, initial_matches
        ):
            db_peaks = vector_to_peak_list(db_vec, axis, resolution)
            score, n_matched, n_q, n_r = cosine_similarity(
                component_peaks,
                db_peaks,
                mz_tol=float(COS_MZ_TOL),
                use_sqrt=bool(COS_USE_SQRT_INTENSITY),
            )

            component_hits.append(
                {
                    "msdial_id": msdial_id,
                    "spectrum_id": spec_id,
                    "compound_name": name,
                    "inchikey": "" if inchikey is None else str(inchikey),
                    "smiles": "" if smiles is None else str(smiles),
                    "formula": "" if formula is None else str(formula),
                    "best_adduct": str(best_adduct),
                    "ion_mode": "" if ion_mode is None else str(ion_mode),
                    "instrument": "" if instrument is None else str(instrument),
                    "collision_energy": "" if ce is None else str(ce),
                    "ontology": "" if ontology is None else str(ontology),
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
    msdial_ids,
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
    ontologies,
    source_files,
    initial_scores,
    initial_matches,
):
    rows = []
    res = np.asarray(res_vector, dtype=float)
    positive_sum = float(np.sum(res[res > 0])) if np.any(res > 0) else 0.0

    order = sorted(range(len(res)), key=lambda i: float(res[i]), reverse=True)
    rank_map = {idx: rank for rank, idx in enumerate(order, start=1)}

    for idx, (msdial_id, spec_id, mass, name, adduct, inchikey, smiles, formula, ion_mode, instrument, ce, ontology, source_file, init_score, init_n_match) in enumerate(
        zip(
            msdial_ids, spectraIDs, masses, compound_names, best_adducts, inchikeys, smiles_list, formulas,
            ion_modes, instruments, collision_energies, ontologies, source_files,
            initial_scores, initial_matches
        )
    ):
        raw_weight = float(res[idx]) if idx < len(res) else 0.0
        norm_weight = float(raw_weight / positive_sum) if positive_sum > 0 and raw_weight > 0 else 0.0
        rows.append(
            {
                "precursor_mz": float(query_precursor_mz),
                "msdial_id": msdial_id,
                "spectrum_id": spec_id,
                "compound_name": name,
                "inchikey": "" if inchikey is None else str(inchikey),
                "smiles": "" if smiles is None else str(smiles),
                "formula": "" if formula is None else str(formula),
                "best_adduct": str(adduct),
                "ion_mode": "" if ion_mode is None else str(ion_mode),
                "instrument": "" if instrument is None else str(instrument),
                "collision_energy": "" if ce is None else str(ce),
                "ontology": "" if ontology is None else str(ontology),
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
    """
    - 用稳定且此时已经存在的谱标识来做映射：component_name / msdial_id
    - 对同一 id 取最大 component_score
    """
    if not weight_rows:
        return weight_rows

    score_map = {}
    for row in ranking_rows or []:
        component_name = str(row.get("component_name", "")).strip()
        msdial_id = str(row.get("msdial_id", "")).strip()
        if component_name and msdial_id and component_name == msdial_id:
            score = float(row.get("component_score", 0.0))
            if component_name not in score_map or score > score_map[component_name]:
                score_map[component_name] = score

    for row in weight_rows:
        msdial_id = str(row.get("msdial_id", "")).strip()
        row["component_score"] = float(score_map.get(msdial_id, 0.0))

    return weight_rows


def build_component_vectors(
    originalSpectra,
    matrix,
    res,
    msdial_ids,
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
        component_vec = [
            max(float(res[x] * m + d), 0.0)
            for m, d in zip(np.asarray(matrix[x], dtype=float), differences)
        ]
        component_rows.append(
            {
                "component_name": str(msdial_ids[x]),
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


# ===================== Library Adapter =====================

class MSDialLibraryAdapter:
    def __init__(self, msdial_spectra: List[Dict[str, Any]]):
        self.msdial_spectra = msdial_spectra

    def get_candidate_trees(self, lower_bound: float, upper_bound: float):
        trees: Dict[
            Tuple[str, float, str, str, str, str, str, str, str, str, str, str],
            Dict[str, Dict[float, float]]
        ] = {}

        for rec in self.msdial_spectra:
            mz = float(rec["precursor_mz"])
            if lower_bound <= mz <= upper_bound:
                key = (
                    rec["msdial_id"],
                    mz,
                    rec.get("compound_name", "Unknown"),
                    rec.get("best_adduct", ""),
                    rec.get("inchikey", ""),
                    rec.get("smiles", ""),
                    rec.get("formula", ""),
                    rec.get("ion_mode", ""),
                    rec.get("instrument", ""),
                    rec.get("collision_energy", ""),
                    rec.get("ontology", ""),
                    rec.get("source_file", ""),
                )
                spec_id = str(rec.get("spectrum_id", rec.get("msdial_id", "MSDIAL")))
                spec_dict = peaks_to_dict(rec["peaks"])
                trees.setdefault(key, {})[spec_id] = spec_dict

        return trees


# ===================== One Query Deconvolution =====================

def deconvolve_and_score_one_query(
    query_precursor_mz: float,
    query_peaks: List[Tuple[float, float]],
    library: MSDialLibraryAdapter,
    precursor_tol: float = PRECURSOR_TOL,
    res_penalty: float = DECOID_RES_PENALTY,
    resolution: int = DECOID_RESOLUTION,
) -> Dict[str, pd.DataFrame]:
    query_peaks = clean_and_normalize_peaks(
        query_peaks,
        rel_cutoff=QUERY_MIN_REL_INTENSITY,
        normalize_to=NORMALIZE_TO,
        merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
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
        "msdial_id": "",
        "spectrum_id": "",
        "compound_name": "",
        "inchikey": "",
        "smiles": "",
        "formula": "",
        "best_adduct": "",
        "ion_mode": "",
        "instrument": "",
        "collision_energy": "",
        "ontology": "",
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
        "msdial_id": "",
        "spectrum_id": "",
        "compound_name": "",
        "inchikey": "",
        "smiles": "",
        "formula": "",
        "best_adduct": "",
        "ion_mode": "",
        "instrument": "",
        "collision_energy": "",
        "ontology": "",
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
        return {
            "ranking_df": pd.DataFrame([empty_rank_row]),
            "weights_df": pd.DataFrame([empty_weight_row]),
        }

    (
        msdial_ids, spectraIDs, matrix, masses, compound_names, best_adducts, inchikeys,
        smiles_list, formulas, ion_modes, instruments, collision_energies, ontologies,
        source_files, axis, reduceSpec, initial_scores, initial_matches
    ) = getMatricesForGroup(
        trees=trees,
        spectra=[query_spec_dict],
        mz_merge_tol=float(MATRIX_MZ_TOL),
        top_n_candidates=int(TOP_N_CANDIDATES),
        min_initial_cosine=float(MIN_INITIAL_COSINE),
    )

    if len(matrix) == 0:
        return {
            "ranking_df": pd.DataFrame([empty_rank_row]),
            "weights_df": pd.DataFrame([empty_weight_row]),
        }

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
        msdial_ids=msdial_ids,
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
        msdial_ids=msdial_ids,
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
        ontologies=ontologies,
        source_files=source_files,
        axis=axis,
        resolution=resolution,
        initial_scores=initial_scores,
        initial_matches=initial_matches,
    )

    weight_rows = build_library_weight_rows(
        query_precursor_mz=float(query_precursor_mz),
        res_vector=res_vector,
        msdial_ids=msdial_ids,
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
        ontologies=ontologies,
        source_files=source_files,
        initial_scores=initial_scores,
        initial_matches=initial_matches,
    )
    weight_rows = attach_component_scores_to_library_weights(weight_rows, rows)

    df_weights = pd.DataFrame(weight_rows) if weight_rows else pd.DataFrame([empty_weight_row])
    df_weights["n_candidates"] = int(len(msdial_ids))
    df_weights["s2n"] = float(s2n)
    df_weights["lasso_alpha"] = float(use_penalty)
    df_weights = df_weights.sort_values(
        ["weight_rank", "lasso_weight_raw", "initial_candidate_score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    if not rows:
        rank_df = pd.DataFrame([{
            **empty_rank_row,
            "n_candidates": int(len(msdial_ids)),
            "s2n": float(s2n),
            "lasso_alpha": float(use_penalty),
        }])
        return {"ranking_df": rank_df, "weights_df": df_weights}

    df = pd.DataFrame(rows)
    df.insert(0, "precursor_mz", float(query_precursor_mz))
    df["n_candidates"] = int(len(msdial_ids))
    df["s2n"] = float(s2n)
    df["lasso_alpha"] = float(use_penalty)
    df = df.sort_values(
        ["component_weight_rank", "component_name", "component_hit_rank", "component_score"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    return {"ranking_df": df, "weights_df": df_weights}


# ===================== CSV-level Processing =====================

def process_one_query_csv(
    query_csv: Path,
    msdial_spectra: List[Dict[str, Any]],
    out_root: Path,
) -> Dict[str, pd.DataFrame]:
    query_spectra = build_query_spectra_from_avg_csv(query_csv)
    if not query_spectra:
        print(f"[WARN] No query spectra after filtering: {query_csv.name}")
        return {"ranking_df": pd.DataFrame(), "weights_df": pd.DataFrame()}

    library = MSDialLibraryAdapter(msdial_spectra)
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
        result = result.sort_values(
            ["precursor_mz", "component_weight_rank", "component_name", "component_hit_rank"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    result_weights = pd.concat(all_weight_parts, axis=0, ignore_index=True) if all_weight_parts else pd.DataFrame()
    if not result_weights.empty:
        result_weights = result_weights.sort_values(
            ["precursor_mz", "weight_rank", "lasso_weight_raw", "initial_candidate_score"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)

    out_file = out_root / f"{query_csv.stem}_decoid_component_ranking.csv"
    result.to_csv(out_file, index=False, encoding="utf-8-sig")

    out_weights = out_root / f"{query_csv.stem}_decoid_library_weights.csv"
    result_weights.to_csv(out_weights, index=False, encoding="utf-8-sig")

    print(f"[OK] Saved per-query deconvolution summary: {out_file.resolve()}")
    print(f"[OK] Saved per-query library weights: {out_weights.resolve()}")
    return {"ranking_df": result, "weights_df": result_weights}


# ===================== Main =====================

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
            f"No QUERY_CSV files found with filename containing "
            f"{QUERY_FILENAME_MUST_CONTAIN!r}. "
            "Please check QUERY_CSV_DIR / QUERY_GLOB / QUERY_CSVS."
        )

    print(f"QUERY CSV files to process: {len(query_files)}")
    for p in query_files:
        print(f"  - {p}")

    msdial_spectra = load_msdial_spectra(MSDIAL_LIB_PATH, ion_mode_keep=None)
    if not msdial_spectra:
        raise RuntimeError("No MS-DIAL spectra parsed. Check MSDIAL_LIB_PATH.")

    all_dfs = []
    all_weight_dfs = []

    for qcsv in query_files:
        result_one = process_one_query_csv(qcsv, msdial_spectra, out_root)
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
