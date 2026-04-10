# -*- coding: utf-8 -*-
"""
GM_DecoID_hmdb_style_fix.py

整合来源：
1) GM 预测谱库读取方式参考 predicted_match_reverse.py
   - 按 precursor 文件夹扫描
   - 读取 predicted_candidates.csv
   - 解析 predicted_spectra/*.csv
   - 支持 pred_spectrum_path / HMDB_ID 两种定位方式
2) 拆谱、候选筛选、矩阵构建、LASSO/NNLS 解卷积、结果输出逻辑参考 GNPS_DecoID.py

用途：
- 读取 GM 预测生成的谱库（目录结构见 predicted_match_reverse.py）
- 读取实验 QUERY CSV（每个 precursor_mz 聚合成一个实验谱）
- 在 precursor m/z 容差范围内筛选候选预测谱
- 用 DecoID 风格的 LASSO / NNLS 对未知谱进行解卷积
- 输出：
    * 每个 query CSV 的 component ranking
    * 每个 query CSV 的 library weights
    * 所有 query CSV 的总表
"""

from __future__ import annotations

import csv
import re
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.linear_model as linModel


# ===================== CONFIG =====================

PREDICT_ROOT: str = r"E:\model\singlecell\results\HGC\results_HGC_predict_0325"

QUERY_CSV_DIR: Optional[str] = r"E:\model\singlecell\singlecell_data\HGC\SNR"
QUERY_GLOB: Optional[str] = None
QUERY_CSVS: List[str] = []
QUERY_FILENAME_MUST_CONTAIN: Optional[str] = "filtered"

OUTPUT_DIR: str = r"E:\model\singlecell\results\HGC\0330\GM_DecoID"

PRECURSOR_TOL: float = 1.0

COS_MZ_TOL: float = 0.05
MATRIX_MZ_TOL: float = 0.025
COS_USE_SQRT_INTENSITY: bool = True
NORMALIZE_TO: Optional[float] = 1000.0

DECOID_RESOLUTION: int = 4
DECOID_RES_PENALTY: float = -1.0

QUERY_MIN_REL_INTENSITY: float = 0.001
LIBRARY_MIN_REL_INTENSITY: float = 0.005
MIN_INITIAL_COSINE: float = 0.05
TOP_N_CANDIDATES: int = 80
MIN_MATCHED_PEAKS: int = 2
FALLBACK_TO_NNLS: bool = True

EXP_TOPK_PEAKS: Optional[int] = None
PRED_TOPK_PEAKS: Optional[int] = None

PEAKS_CACHE_MAX: int = 200_000


@dataclass
class GMSpectrumLite:
    library_id: str
    spectrum_id: str
    compound_name: str
    precursor_mz: Optional[float]
    peaks: List[Tuple[float, float]]
    hmdb_id: str = ""
    inchikey: str = ""
    smiles: str = ""
    matched_mass_column: str = ""
    matched_mass_value: Optional[float] = None
    mass_diff_da: Optional[float] = None
    adduct_used_for_model: str = ""
    adduct_fallback_used: bool = False
    source_file: str = ""
    source_folder: str = ""


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

    try:
        return float(s.replace(",", ""))
    except Exception:
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None


def _safe_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def flatten(l):
    if len(l) > 0 and type(l[0]) == type(l):
        return [item for sublist in l for item in sublist]
    return l


def _filename_ok(p: Path, must_contain: Optional[str]) -> bool:
    if not must_contain:
        return True
    return must_contain.lower() in p.name.lower()


def _read_csv_safely(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1", "iso-8859-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8", errors="replace", **kwargs)


def keep_topk_peaks(peaks: List[Tuple[float, float]], k: Optional[int]) -> List[Tuple[float, float]]:
    if not k or k <= 0 or len(peaks) <= k:
        return peaks
    peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)[:k]
    peaks_sorted.sort(key=lambda x: x[0])
    return peaks_sorted


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


def scan_predict_folders(predict_root: Path) -> Tuple[np.ndarray, List[Path]]:
    mz_list: List[float] = []
    folder_list: List[Path] = []
    for d in predict_root.iterdir():
        if not d.is_dir():
            continue
        cand = d / "predicted_candidates.csv"
        if not cand.exists():
            continue
        mz = _safe_float(d.name)
        if mz is None:
            continue
        mz_list.append(float(mz))
        folder_list.append(d)

    if not mz_list:
        return np.asarray([], dtype=float), []

    order = np.argsort(np.asarray(mz_list, dtype=float))
    mzs = np.asarray(mz_list, dtype=float)[order]
    folders = [folder_list[i] for i in order.tolist()]
    return mzs, folders


def resolve_pred_spectrum_path(pred_folder: Path, row: Mapping[str, Any]) -> Optional[Path]:
    raw = str(row.get("pred_spectrum_path", "") or "").strip()
    hmdb = str(row.get("HMDB_ID", "") or "").strip()

    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (pred_folder / p).resolve()
        if p.exists():
            return p

        base = Path(raw).name
        cand = pred_folder / "predicted_spectra" / base
        if cand.exists():
            return cand

    if hmdb:
        cand = pred_folder / "predicted_spectra" / f"{hmdb}.csv"
        if cand.exists():
            return cand

    return None


def _fast_read_peaks_numpy(path: Path) -> Optional[np.ndarray]:
    try:
        if not path.exists():
            return None
        if path.stat().st_size == 0:
            return np.empty((0, 2), dtype=float)

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            first = ""
            while True:
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    first = line.strip()
                    break

        if not first:
            return np.empty((0, 2), dtype=float)

        has_alpha = any(ch.isalpha() for ch in first)
        skip = 1 if has_alpha else 0

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"genfromtxt: Empty input file:.*", category=UserWarning)
            arr = np.genfromtxt(str(path), delimiter=",", skip_header=skip, dtype=float, invalid_raise=False)

        if arr is None:
            return None
        if arr.ndim == 1:
            if arr.size == 0:
                return np.empty((0, 2), dtype=float)
            if arr.size >= 2:
                arr = arr.reshape(1, -1)
        if arr.size == 0:
            return np.empty((0, 2), dtype=float)

        if arr.ndim == 2 and arr.shape[1] < 2:
            arr2 = np.genfromtxt(str(path), delimiter=None, skip_header=skip, dtype=float, invalid_raise=False)
            if arr2 is None:
                return None
            if arr2.ndim == 1:
                if arr2.size >= 2:
                    arr2 = arr2.reshape(1, -1)
                else:
                    return np.empty((0, 2), dtype=float)
            arr = arr2

        if arr.ndim != 2 or arr.shape[1] < 2:
            return None

        return arr[:, :2]
    except Exception:
        return None


def _normalize_colname(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" ", "").replace("\u00a0", "")
    return s


def read_peaks_csv_robust(path: Union[str, Path], normalize_to: Optional[float], topk: Optional[int]) -> Optional[List[Tuple[float, float]]]:
    p = Path(path)
    if not p.exists():
        return None

    try:
        if p.stat().st_size == 0:
            return []
    except Exception:
        pass

    try:
        df = _read_csv_safely(p, sep=None, engine="python")
    except Exception:
        df = pd.DataFrame()

    if df.empty or df.shape[1] == 1:
        try:
            df2 = _read_csv_safely(p, sep=None, engine="python", header=None)
            if not df2.empty:
                df = df2
        except Exception:
            pass

    if df.empty:
        return []

    mz_col = None
    inten_col = None

    if df.columns.dtype == object and any(isinstance(c, str) for c in df.columns):
        cols_norm = {_normalize_colname(c): c for c in df.columns}
        for k in ["mz", "m/z", "moverz", "masscharge", "mass-to-charge", "m_z", "m.z"]:
            kk = k.replace("/", "")
            if kk in cols_norm:
                mz_col = cols_norm[kk]
                break
            if k in cols_norm:
                mz_col = cols_norm[k]
                break
        for k in ["intensity", "inten", "int", "relativeintensity", "abundance", "height", "y"]:
            if k in cols_norm:
                inten_col = cols_norm[k]
                break

    if mz_col is None or inten_col is None:
        if df.shape[1] >= 2:
            mz_col = df.columns[0]
            inten_col = df.columns[1]
        else:
            return None

    try:
        mz = pd.to_numeric(df[mz_col], errors="coerce")
        it = pd.to_numeric(df[inten_col], errors="coerce").fillna(0.0)
    except Exception:
        return None

    mask = mz.notna() & (it > 0)
    mz = mz[mask]
    it = it[mask]
    if mz.empty:
        return []

    if normalize_to is not None:
        mx = float(it.max())
        if mx > 0:
            it = it / mx * float(normalize_to)

    peaks = list(zip(mz.astype(float).tolist(), it.astype(float).tolist()))
    peaks.sort(key=lambda x: x[0])
    peaks = keep_topk_peaks(peaks, topk)
    return peaks


def read_peaks_csv_fast(path: Union[str, Path], normalize_to: Optional[float], topk: Optional[int]) -> Optional[List[Tuple[float, float]]]:
    p = Path(path)
    arr = _fast_read_peaks_numpy(p)
    if arr is None:
        return None

    if arr.size == 0:
        return []

    mz = arr[:, 0]
    it = arr[:, 1]

    mask = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    if not np.any(mask):
        return []
    mz = mz[mask]
    it = it[mask]

    if mz.size == 0:
        return []

    if normalize_to is not None:
        mx = float(np.max(it))
        if mx > 0:
            it = it / mx * float(normalize_to)

    peaks = list(zip(mz.astype(float).tolist(), it.astype(float).tolist()))
    peaks.sort(key=lambda x: x[0])
    peaks = keep_topk_peaks(peaks, topk)
    return peaks


def _load_candidates_rows(pred_candidates_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with pred_candidates_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r:
                    rows.append(r)
    except Exception:
        try:
            df = _read_csv_safely(pred_candidates_path)
            if not df.empty:
                rows = df.to_dict(orient="records")
        except Exception:
            return []
    return rows


class PeaksCache:
    def __init__(self, max_size: int = PEAKS_CACHE_MAX):
        self.max_size = int(max_size)
        self.cache: OrderedDict[str, Optional[List[Tuple[float, float]]]] = OrderedDict()

    def get(self, path: Path) -> Optional[List[Tuple[float, float]]]:
        key = str(path)
        if key in self.cache:
            val = self.cache.pop(key)
            self.cache[key] = val
            return val
        return None

    def put(self, path: Path, peaks: Optional[List[Tuple[float, float]]]) -> None:
        key = str(path)
        if key in self.cache:
            self.cache.pop(key)
        self.cache[key] = peaks
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


def load_predicted_spectra(predict_root: Union[str, Path]) -> List[Dict[str, Any]]:
    predict_root = Path(predict_root)
    if not predict_root.exists():
        raise FileNotFoundError(f"PREDICT_ROOT not found: {predict_root}")

    print(f"Loading GM predicted library from: {predict_root}")
    pred_mzs, pred_folders = scan_predict_folders(predict_root)
    if pred_mzs.size == 0:
        raise FileNotFoundError(f"No predicted precursor folders found under {predict_root}")

    spectra: List[Dict[str, Any]] = []
    peaks_cache = PeaksCache(max_size=PEAKS_CACHE_MAX)

    n_folders = 0
    n_candidate_rows = 0
    n_loaded = 0

    for pred_folder in pred_folders:
        n_folders += 1
        pred_folder_mz = _safe_float(pred_folder.name)
        if pred_folder_mz is None:
            continue

        pred_candidates_path = pred_folder / "predicted_candidates.csv"
        if not pred_candidates_path.exists():
            continue

        cand_rows = _load_candidates_rows(pred_candidates_path)
        if not cand_rows:
            continue

        for row in cand_rows:
            n_candidate_rows += 1
            pred_file = resolve_pred_spectrum_path(pred_folder, row)
            if pred_file is None:
                continue

            peaks = peaks_cache.get(pred_file)
            if peaks is None:
                peaks = read_peaks_csv_fast(pred_file, normalize_to=NORMALIZE_TO, topk=PRED_TOPK_PEAKS)
                if peaks is None:
                    peaks = read_peaks_csv_robust(pred_file, normalize_to=NORMALIZE_TO, topk=PRED_TOPK_PEAKS)
                peaks_cache.put(pred_file, peaks)

            if not peaks:
                continue

            peaks = clean_and_normalize_peaks(
                peaks,
                rel_cutoff=LIBRARY_MIN_REL_INTENSITY,
                normalize_to=NORMALIZE_TO,
                merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
                topk=PRED_TOPK_PEAKS,
            )
            if not peaks:
                continue

            hmdb_id = str(row.get("HMDB_ID", "") or "").strip()
            inchikey = str(row.get("INCHI_KEY", "") or "").strip()
            name = str(row.get("NAME", "") or "").strip() or hmdb_id or pred_file.stem
            smiles = str(row.get("SMILES", "") or "").strip()
            matched_mass_column = str(row.get("matched_mass_column", "") or "").strip()
            matched_mass_value = _safe_float(row.get("matched_mass_value"))
            mass_diff_da = _safe_float(row.get("mass_diff_da"))
            adduct_used_for_model = str(row.get("adduct_used_for_model", "") or "").strip()
            adduct_fallback_used = _safe_bool(row.get("adduct_fallback_used"))

            spectrum_id = pred_file.stem or f"GM_{n_loaded + 1}"
            library_id = hmdb_id or spectrum_id

            spectra.append(
                {
                    "library_id": library_id,
                    "spectrum_id": spectrum_id,
                    "precursor_mz": float(pred_folder_mz),
                    "peaks": peaks,
                    "compound_name": name,
                    "hmdb_id": hmdb_id,
                    "inchikey": inchikey,
                    "smiles": smiles,
                    "matched_mass_column": matched_mass_column,
                    "matched_mass_value": matched_mass_value,
                    "mass_diff_da": mass_diff_da,
                    "adduct_used_for_model": adduct_used_for_model,
                    "adduct_fallback_used": adduct_fallback_used,
                    "source_file": str(pred_file),
                    "source_folder": str(pred_folder),
                }
            )
            n_loaded += 1

        if n_folders % 100 == 0:
            print(f"  Scanned folders: {n_folders:,} | loaded spectra: {n_loaded:,}")

    print(f"Parsed GM predicted spectra: {len(spectra)}")
    print(f"  Folders scanned: {n_folders:,}")
    print(f"  Candidate rows seen: {n_candidate_rows:,}")
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
    library_ids = [x[0] for x in keys]
    masses = [x[1] for x in keys]
    compound_names = [x[2] for x in keys]
    hmdb_ids = [x[3] for x in keys]
    inchikeys = [x[4] for x in keys]
    smiles_list = [x[5] for x in keys]
    matched_mass_columns = [x[6] for x in keys]
    matched_mass_values = [x[7] for x in keys]
    mass_diff_list = [x[8] for x in keys]
    adduct_used_list = [x[9] for x in keys]
    adduct_fallback_list = [x[10] for x in keys]
    source_files = [x[11] for x in keys]
    source_folders = [x[12] for x in keys]

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
        library_ids, spectraIDs, matrix, masses, compound_names, hmdb_ids, inchikeys, smiles_list,
        matched_mass_columns, matched_mass_values, mass_diff_list, adduct_used_list, adduct_fallback_list,
        source_files, source_folders, axis, reduceSpec, initial_scores, initial_matches
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
    library_ids,
    spectraIDs,
    masses,
    compound_names,
    hmdb_ids,
    inchikeys,
    smiles_list,
    matched_mass_columns,
    matched_mass_values,
    mass_diff_list,
    adduct_used_list,
    adduct_fallback_list,
    source_files,
    source_folders,
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
        for library_id, spec_id, mass, name, hmdb_id, inchikey, smiles, matched_mass_column, matched_mass_value, mass_diff_da, adduct_used_for_model, adduct_fallback_used, source_file, source_folder, db_vec, init_score, init_n_match in zip(
            library_ids, spectraIDs, masses, compound_names, hmdb_ids, inchikeys, smiles_list,
            matched_mass_columns, matched_mass_values, mass_diff_list, adduct_used_list, adduct_fallback_list,
            source_files, source_folders, comp_info["db_matrix"], initial_scores, initial_matches
        ):
            db_peaks = vector_to_peak_list(db_vec, axis, resolution)
            score, n_matched, n_q, n_r = cosine_similarity(component_peaks, db_peaks, mz_tol=float(COS_MZ_TOL), use_sqrt=bool(COS_USE_SQRT_INTENSITY))

            component_hits.append(
                {
                    "library_id": library_id,
                    "spectrum_id": spec_id,
                    "compound_name": name,
                    "hmdb_id": "" if hmdb_id is None else str(hmdb_id),
                    "inchikey": "" if inchikey is None else str(inchikey),
                    "smiles": "" if smiles is None else str(smiles),
                    "matched_mass_column": "" if matched_mass_column is None else str(matched_mass_column),
                    "matched_mass_value": np.nan if matched_mass_value is None else float(matched_mass_value),
                    "mass_diff_da": np.nan if mass_diff_da is None else float(mass_diff_da),
                    "adduct_used_for_model": "" if adduct_used_for_model is None else str(adduct_used_for_model),
                    "adduct_fallback_used": bool(adduct_fallback_used),
                    "source_file": "" if source_file is None else str(source_file),
                    "source_folder": "" if source_folder is None else str(source_folder),
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
                    "component_source_hmdb_id": comp_info.get("source_hmdb_id", ""),
                    "component_source_inchikey": comp_info.get("source_inchikey", ""),
                    "component_source_smiles": comp_info.get("source_smiles", ""),
                    "component_source_adduct_used_for_model": comp_info.get("adduct_used_for_model", ""),
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
    library_ids,
    spectraIDs,
    masses,
    compound_names,
    hmdb_ids,
    inchikeys,
    smiles_list,
    matched_mass_columns,
    matched_mass_values,
    mass_diff_list,
    adduct_used_list,
    adduct_fallback_list,
    source_files,
    source_folders,
    initial_scores,
    initial_matches,
):
    rows = []
    res = np.asarray(res_vector, dtype=float)
    positive_sum = float(np.sum(res[res > 0])) if np.any(res > 0) else 0.0

    order = sorted(range(len(res)), key=lambda i: float(res[i]), reverse=True)
    rank_map = {idx: rank for rank, idx in enumerate(order, start=1)}

    for idx, (library_id, spec_id, mass, name, hmdb_id, inchikey, smiles, matched_mass_column, matched_mass_value, mass_diff_da, adduct_used_for_model, adduct_fallback_used, source_file, source_folder, init_score, init_n_match) in enumerate(
        zip(
            library_ids, spectraIDs, masses, compound_names, hmdb_ids, inchikeys, smiles_list,
            matched_mass_columns, matched_mass_values, mass_diff_list, adduct_used_list, adduct_fallback_list,
            source_files, source_folders, initial_scores, initial_matches
        )
    ):
        raw_weight = float(res[idx]) if idx < len(res) else 0.0
        norm_weight = float(raw_weight / positive_sum) if positive_sum > 0 and raw_weight > 0 else 0.0
        rows.append(
            {
                "precursor_mz": float(query_precursor_mz),
                "library_id": library_id,
                "spectrum_id": spec_id,
                "compound_name": name,
                "hmdb_id": "" if hmdb_id is None else str(hmdb_id),
                "inchikey": "" if inchikey is None else str(inchikey),
                "smiles": "" if smiles is None else str(smiles),
                "matched_mass_column": "" if matched_mass_column is None else str(matched_mass_column),
                "matched_mass_value": np.nan if matched_mass_value is None else float(matched_mass_value),
                "mass_diff_da": np.nan if mass_diff_da is None else float(mass_diff_da),
                "adduct_used_for_model": "" if adduct_used_for_model is None else str(adduct_used_for_model),
                "adduct_fallback_used": bool(adduct_fallback_used),
                "source_file": "" if source_file is None else str(source_file),
                "source_folder": "" if source_folder is None else str(source_folder),
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
    参考 HMDB_DecoID.py 的回填逻辑。

    关键点：
    - 本函数调用时 ranking_rows 还没有插入 precursor_mz / query_csv
    - 因此不能依赖 precursor_mz 或 query_csv 去做键匹配
    - 只按稳定主键回填：
        1) 优先 (library_id, spectrum_id)
        2) 再回退到 library_id
    """
    if not weight_rows:
        return weight_rows

    exact_score_map = {}
    fallback_score_map = {}

    for row in ranking_rows or []:
        component_name = str(row.get("component_name", "")).strip()
        library_id = str(row.get("library_id", "")).strip()

        # 只取 component 自身对应 library 的那条记录
        if not (component_name and library_id and component_name == library_id):
            continue

        spectrum_id = str(row.get("spectrum_id", "")).strip()
        score = _safe_float(row.get("component_score"))
        if score is None:
            score = 0.0

        exact_key = (library_id, spectrum_id)
        fallback_key = library_id

        if exact_key not in exact_score_map or float(score) > float(exact_score_map[exact_key]):
            exact_score_map[exact_key] = float(score)
        if fallback_key not in fallback_score_map or float(score) > float(fallback_score_map[fallback_key]):
            fallback_score_map[fallback_key] = float(score)

    updated_rows = []
    for row in weight_rows:
        new_row = dict(row)
        library_id = str(new_row.get("library_id", "")).strip()
        spectrum_id = str(new_row.get("spectrum_id", "")).strip()

        exact_key = (library_id, spectrum_id)
        if exact_key in exact_score_map:
            new_row["component_score"] = float(exact_score_map[exact_key])
        else:
            new_row["component_score"] = float(fallback_score_map.get(library_id, 0.0))

        updated_rows.append(new_row)

    return updated_rows


def build_component_vectors(
    originalSpectra,
    matrix,
    res,
    library_ids,
    masses,
    spectraIDs,
    compound_names,
    hmdb_ids,
    inchikeys,
    smiles_list,
    adduct_used_list,
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
                "component_name": str(library_ids[x]),
                "component_abundance": float(res[x] / resSum),
                "component_weight_raw": float(res[x]),
                "component_weight_rank": int(positive_rank_map[x]),
                "component_vector": component_vec,
                "component_mass": float(masses[x]),
                "source_spectrum_id": str(spectraIDs[x]),
                "source_compound_name": str(compound_names[x]),
                "source_hmdb_id": "" if hmdb_ids[x] is None else str(hmdb_ids[x]),
                "source_inchikey": "" if inchikeys[x] is None else str(inchikeys[x]),
                "source_smiles": "" if smiles_list[x] is None else str(smiles_list[x]),
                "adduct_used_for_model": "" if adduct_used_list[x] is None else str(adduct_used_list[x]),
                "db_matrix": matrix,
            }
        )
    return component_rows


class GMLibraryAdapter:
    def __init__(self, gm_spectra: List[Dict[str, Any]]):
        self.gm_spectra = gm_spectra

    def get_candidate_trees(self, lower_bound: float, upper_bound: float):
        trees: Dict[Tuple[str, float, str, str, str, str, str, Any, Any, str, bool, str, str], Dict[str, Dict[float, float]]] = {}

        for rec in self.gm_spectra:
            mz = float(rec["precursor_mz"])
            if lower_bound <= mz <= upper_bound:
                key = (
                    rec["library_id"],
                    mz,
                    rec.get("compound_name", "Unknown"),
                    rec.get("hmdb_id", ""),
                    rec.get("inchikey", ""),
                    rec.get("smiles", ""),
                    rec.get("matched_mass_column", ""),
                    rec.get("matched_mass_value", np.nan),
                    rec.get("mass_diff_da", np.nan),
                    rec.get("adduct_used_for_model", ""),
                    bool(rec.get("adduct_fallback_used", False)),
                    rec.get("source_file", ""),
                    rec.get("source_folder", ""),
                )
                spec_id = str(rec.get("spectrum_id", rec.get("library_id", "GM")))
                spec_dict = peaks_to_dict(rec["peaks"])
                trees.setdefault(key, {})[spec_id] = spec_dict

        return trees


def deconvolve_and_score_one_query(
    query_precursor_mz: float,
    query_peaks: List[Tuple[float, float]],
    library: GMLibraryAdapter,
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
        "component_source_hmdb_id": "",
        "component_source_inchikey": "",
        "component_source_smiles": "",
        "component_source_adduct_used_for_model": "",
        "component_source_precursor_mz": np.nan,
        "component_abundance": np.nan,
        "component_weight_raw": np.nan,
        "component_weight_rank": 0,
        "component_hit_rank": 0,
        "library_id": "",
        "spectrum_id": "",
        "compound_name": "",
        "hmdb_id": "",
        "inchikey": "",
        "smiles": "",
        "matched_mass_column": "",
        "matched_mass_value": np.nan,
        "mass_diff_da": np.nan,
        "adduct_used_for_model": "",
        "adduct_fallback_used": False,
        "source_file": "",
        "source_folder": "",
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
        "library_id": "",
        "spectrum_id": "",
        "compound_name": "",
        "hmdb_id": "",
        "inchikey": "",
        "smiles": "",
        "matched_mass_column": "",
        "matched_mass_value": np.nan,
        "mass_diff_da": np.nan,
        "adduct_used_for_model": "",
        "adduct_fallback_used": False,
        "source_file": "",
        "source_folder": "",
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
        library_ids, spectraIDs, matrix, masses, compound_names, hmdb_ids, inchikeys, smiles_list,
        matched_mass_columns, matched_mass_values, mass_diff_list, adduct_used_list, adduct_fallback_list,
        source_files, source_folders, axis, reduceSpec, initial_scores, initial_matches
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
        library_ids=library_ids,
        masses=masses,
        spectraIDs=spectraIDs,
        compound_names=compound_names,
        hmdb_ids=hmdb_ids,
        inchikeys=inchikeys,
        smiles_list=smiles_list,
        adduct_used_list=adduct_used_list,
    )

    rows = score_components_against_library(
        component_vectors=component_vectors,
        library_ids=library_ids,
        spectraIDs=spectraIDs,
        masses=masses,
        compound_names=compound_names,
        hmdb_ids=hmdb_ids,
        inchikeys=inchikeys,
        smiles_list=smiles_list,
        matched_mass_columns=matched_mass_columns,
        matched_mass_values=matched_mass_values,
        mass_diff_list=mass_diff_list,
        adduct_used_list=adduct_used_list,
        adduct_fallback_list=adduct_fallback_list,
        source_files=source_files,
        source_folders=source_folders,
        axis=axis,
        resolution=resolution,
        initial_scores=initial_scores,
        initial_matches=initial_matches,
    )

    weight_rows = build_library_weight_rows(
        query_precursor_mz=float(query_precursor_mz),
        res_vector=res_vector,
        library_ids=library_ids,
        spectraIDs=spectraIDs,
        masses=masses,
        compound_names=compound_names,
        hmdb_ids=hmdb_ids,
        inchikeys=inchikeys,
        smiles_list=smiles_list,
        matched_mass_columns=matched_mass_columns,
        matched_mass_values=matched_mass_values,
        mass_diff_list=mass_diff_list,
        adduct_used_list=adduct_used_list,
        adduct_fallback_list=adduct_fallback_list,
        source_files=source_files,
        source_folders=source_folders,
        initial_scores=initial_scores,
        initial_matches=initial_matches,
    )
    weight_rows = attach_component_scores_to_library_weights(weight_rows, rows)

    df_weights = pd.DataFrame(weight_rows) if weight_rows else pd.DataFrame([empty_weight_row])
    df_weights["n_candidates"] = int(len(library_ids))
    df_weights["s2n"] = float(s2n)
    df_weights["lasso_alpha"] = float(use_penalty)
    df_weights = df_weights.sort_values(["weight_rank", "lasso_weight_raw", "initial_candidate_score"], ascending=[True, False, False]).reset_index(drop=True)

    if not rows:
        rank_df = pd.DataFrame([{**empty_rank_row, "n_candidates": int(len(library_ids)), "s2n": float(s2n), "lasso_alpha": float(use_penalty)}])
        return {"ranking_df": rank_df, "weights_df": df_weights}

    df = pd.DataFrame(rows)
    df.insert(0, "precursor_mz", float(query_precursor_mz))
    df["n_candidates"] = int(len(library_ids))
    df["s2n"] = float(s2n)
    df["lasso_alpha"] = float(use_penalty)
    df = df.sort_values(["component_weight_rank", "component_name", "component_hit_rank", "component_score"], ascending=[True, True, True, False]).reset_index(drop=True)

    return {"ranking_df": df, "weights_df": df_weights}


def process_one_query_csv(query_csv: Path, gm_spectra: List[Dict[str, Any]], out_root: Path) -> Dict[str, pd.DataFrame]:
    query_spectra = build_query_spectra_from_avg_csv(query_csv)
    if not query_spectra:
        print(f"[WARN] No query spectra after filtering: {query_csv.name}")
        return {"ranking_df": pd.DataFrame(), "weights_df": pd.DataFrame()}

    library = GMLibraryAdapter(gm_spectra)
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

    gm_spectra = load_predicted_spectra(PREDICT_ROOT)
    if not gm_spectra:
        raise RuntimeError("No GM predicted spectra parsed. Check PREDICT_ROOT.")

    all_dfs = []
    all_weight_dfs = []

    for qcsv in query_files:
        result_one = process_one_query_csv(qcsv, gm_spectra, out_root)
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
