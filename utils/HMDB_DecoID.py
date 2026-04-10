# -*- coding: utf-8 -*-
"""
HMDB_DecoID.py

Use HMDB library spectra to deconvolve unknown spectra with DecoID-style LASSO/NNLS,
while adapting the workflow for cases where query spectra do not have RT or MS1 fragment
screening information.

Key changes relative to the older version:
- candidate selection relies on precursor m/z only (no RT requirement)
- library/query spectra are normalized consistently before deconvolution
- m/z alignment for matrix construction uses tolerance-based merging instead of decimal rounding
- initial cosine prefilter reduces collinear/noisy candidates before LASSO
- component vectors are pure weighted library components (no averaged residual added back)
- optional fallback to NNLS if LASSO produces too sparse / empty solutions
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union, Iterator
import ast
import json

import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.linear_model as linModel


# ===================== CONFIG =====================

QUERY_CSV_DIR: Optional[str] = r"E:\model\singlecell\singlecell_data\HGC\SNR"
QUERY_GLOB: Optional[str] = None
QUERY_CSVS: List[str] = []
QUERY_FILENAME_MUST_CONTAIN = "filtered"

HMDB_LIB_PATH = r"E:\model\singlecell\spectra_library\hmdb_positive_spectra.json"
HMDB_IONIZATION_MODE_KEEP: Optional[str] = "Positive"

OUTPUT_DIR = r"E:\model\singlecell\results\HGC\0330\HMDB_DecoID"

# precursor filtering still needed because there is no RT; this is now the primary candidate selector
PRECURSOR_TOL = 1.0
ADDUCT_MASSES = {
    "M+H": 1.007276466812,
    "M+Na": 22.989218,
}
# peak matching tolerance for cosine scoring and for matrix-axis construction
COS_MZ_TOL = 0.05
# matrix construction no longer rounds mz to N decimals; instead it merges nearby mz using this tolerance
MATRIX_MZ_TOL = 0.025
COS_USE_SQRT_INTENSITY = False
NORMALIZE_TO = 1.0

# resolution kept only for pretty output of component peaks
DECOID_RESOLUTION = 4
# if <= 0, an adaptive alpha will be estimated from candidate count / query size
DECOID_RES_PENALTY = -1.0
REDUNDANCY_CHECK_THRESH = 0.9

# practical denoising / candidate-pruning knobs
QUERY_MIN_REL_INTENSITY = 0.001
LIBRARY_MIN_REL_INTENSITY = 0.005
MIN_INITIAL_COSINE = 0.05
TOP_N_CANDIDATES = 80
MIN_MATCHED_PEAKS = 2
FALLBACK_TO_NNLS = True


# ==================================================


def _safe_float(x):
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
        return None


def _candidate_precursor_entries_from_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert HMDB MONO_MASS to one or more precursor entries.
    Each entry keeps both precursor_mz and its adduct label, so the best adduct
    can be traced through candidate selection and written to the result table.

    Priority:
    1) If the record already has explicit precursor/adduct m/z fields, use them.
    2) Otherwise derive precursor m/z from MONO_MASS according to ionization mode.
       - Positive: MONO_MASS + each adduct in ADDUCT_MASSES
       - Negative: MONO_MASS - proton mass for [M-H]-
    """
    explicit_keys = [
        "PRECURSOR_MZ", "precursor_mz", "PRECURSOR", "precursor",
        "PRECURSOR_MASS", "precursor_mass", "EXACT_MASS_PRECURSOR",
    ]
    out: List[Dict[str, Any]] = []
    for k in explicit_keys:
        v = _safe_float(rec.get(k))
        if v is not None and v > 0:
            out.append({"precursor_mz": float(v), "best_adduct": "explicit"})
    if out:
        uniq = {}
        for e in out:
            uniq[round(float(e["precursor_mz"]), 10)] = e
        return [uniq[k] for k in sorted(uniq)]

    mono_mass_f = _safe_float(rec.get("MONO_MASS"))
    if mono_mass_f is None or mono_mass_f <= 0:
        return []

    ion_mode = str(rec.get("IONIZATION_MODE", "")).strip().lower()
    derived: List[Dict[str, Any]] = []

    if ion_mode == "negative":
        proton_mass = ADDUCT_MASSES.get("M+H", 1.007276466812)
        pmz = mono_mass_f - proton_mass
        if pmz > 0:
            derived.append({"precursor_mz": float(pmz), "best_adduct": "M-H"})
    else:
        for adduct, mass in ADDUCT_MASSES.items():
            pmz = mono_mass_f + float(mass)
            if pmz > 0:
                derived.append({"precursor_mz": float(pmz), "best_adduct": str(adduct)})

    uniq = {}
    for e in derived:
        uniq[(round(float(e["precursor_mz"]), 10), str(e["best_adduct"]))] = e
    return [uniq[k] for k in sorted(uniq)]




def flatten(l):
    if len(l) > 0 and type(l[0]) == type(l):
        return [item for sublist in l for item in sublist]
    return l


# ---------------- generic helpers ----------------

def _filename_ok(p: Path, must_contain: Optional[str]) -> bool:
    if not must_contain:
        return True
    return must_contain.lower() in p.name.lower()


def iter_query_csvs(
    query_csv_dir: Optional[Union[str, Path]] = None,
    query_glob: Optional[str] = None,
    query_csvs: Optional[List[Union[str, Path]]] = None,
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


def _parse_peaks_literal(peaks_raw: Any) -> List[Tuple[float, float]]:
    if isinstance(peaks_raw, list):
        parsed = peaks_raw
    else:
        try:
            parsed = ast.literal_eval(str(peaks_raw))
        except Exception:
            return []
    out = []
    for p in parsed:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            mz = _safe_float(p[0])
            inten = _safe_float(p[1])
            if mz is not None and inten is not None and inten > 0:
                out.append((float(mz), float(inten)))
    return out


# ---------------- HMDB loaders ----------------

def _iter_json_records(path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        pos = f.tell()
        samples: List[str] = []
        while len(samples) < 5:
            line = f.readline()
            if not line:
                break
            s = line.strip()
            if s:
                samples.append(s)
        f.seek(pos)

        jsonl_ok = False
        if samples and all(s.startswith("{") for s in samples):
            try:
                for s in samples:
                    obj = json.loads(s)
                    if not isinstance(obj, dict):
                        raise ValueError("jsonl sample not dict")
                jsonl_ok = True
            except Exception:
                jsonl_ok = False

        if jsonl_ok:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    yield obj
            return

        obj = json.load(f)

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(obj, dict):
        if "HMDB_ID" in obj and "PEAKS" in obj:
            yield obj
            return

        for v in obj.values():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v[: min(5, len(v))]):
                for item in v:
                    yield item
                return


def load_hmdb_spectra_from_json(hmdb_json: Union[str, Path], ion_mode_keep: Optional[str] = None) -> List[Dict[str, Any]]:
    records = list(_iter_json_records(hmdb_json))
    print(f"Loaded HMDB JSON: {hmdb_json} | records={len(records)}")

    spectra: List[Dict[str, Any]] = []
    for rec in records:
        if ion_mode_keep is not None:
            ion = rec.get("IONIZATION_MODE")
            if ion is not None and str(ion).strip().lower() != str(ion_mode_keep).strip().lower():
                continue

        hmdb_id = rec.get("HMDB_ID")
        
        inchikey = rec.get("INCHI_KEY") or rec.get("INCHIKEY") or rec.get("InChIKey")
        mono_mass = rec.get("MONO_MASS")
        peaks_raw = rec.get("PEAKS")

        mono_mass_f = _safe_float(mono_mass)
        if mono_mass_f is None or hmdb_id is None or peaks_raw is None:
            continue

        precursor_entries = _candidate_precursor_entries_from_record(rec)
        if not precursor_entries:
            continue

        peaks = clean_and_normalize_peaks(
            _parse_peaks_literal(peaks_raw),
            rel_cutoff=LIBRARY_MIN_REL_INTENSITY,
            normalize_to=NORMALIZE_TO,
            merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
        )
        if not peaks:
            continue

        base_entry = {
            "hmdb_id": str(hmdb_id),
            "spectrum_id": str(
                rec.get("spectrum_id")
                or rec.get("SPECTRUM_ID")
                or rec.get("ID")
                or rec.get("id")
                or hmdb_id),
            "peaks": peaks,
            "compound_name": rec.get("NAME", "Unknown"),
            "inchikey": "" if inchikey is None else str(inchikey),
        }
        for precursor_entry in precursor_entries:
            entry = dict(base_entry)
            entry["precursor_mz"] = float(precursor_entry["precursor_mz"])
            entry["best_adduct"] = str(precursor_entry.get("best_adduct", ""))
            spectra.append(entry)

    print(f"Parsed HMDB spectra: {len(spectra)}")
    return spectra


def load_hmdb_spectra_from_csv(hmdb_csv: Union[str, Path]) -> List[Dict[str, Any]]:
    df = pd.read_csv(hmdb_csv)
    print(f"Loaded HMDB CSV: {hmdb_csv} | rows={len(df)}")

    required = {"HMDB_ID", "MONO_MASS", "PEAKS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"HMDB CSV missing columns: {missing}")

    spectra: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        peaks = clean_and_normalize_peaks(
            _parse_peaks_literal(row["PEAKS"]),
            rel_cutoff=LIBRARY_MIN_REL_INTENSITY,
            normalize_to=NORMALIZE_TO,
            merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
        )
        if not peaks:
            continue

        precursor_entries = _candidate_precursor_entries_from_record(row.to_dict())
        if not precursor_entries:
            continue

        base_entry = {
            "hmdb_id": str(row["HMDB_ID"]),
            "spectrum_id": str(
                row.get("spectrum_id",
                row.get("SPECTRUM_ID",
                row.get("ID",
                row.get("id", row["HMDB_ID"]))))),
            "peaks": peaks,
            "compound_name": row.get("NAME", "Unknown"),
            "inchikey": str(row.get("INCHI_KEY", row.get("INCHIKEY", row.get("InChIKey", "")))),
        }
        for precursor_entry in precursor_entries:
            entry = dict(base_entry)
            entry["precursor_mz"] = float(precursor_entry["precursor_mz"])
            entry["best_adduct"] = str(precursor_entry.get("best_adduct", ""))
            spectra.append(entry)

    print(f"Parsed HMDB spectra: {len(spectra)}")
    return spectra


def load_hmdb_spectra(hmdb_path: Union[str, Path]) -> List[Dict[str, Any]]:
    hmdb_path = Path(hmdb_path)
    suf = hmdb_path.suffix.lower()
    if suf in {".json", ".jsonl"}:
        return load_hmdb_spectra_from_json(hmdb_path, ion_mode_keep=HMDB_IONIZATION_MODE_KEEP)
    if suf == ".csv":
        return load_hmdb_spectra_from_csv(hmdb_path)
    raise ValueError(f"Unsupported HMDB library file type: {hmdb_path.suffix}")


# ---------------- Query spectra ----------------

def build_query_spectra_from_avg_csv(
    csv_path: Union[str, Path],
    normalize_to: Optional[float] = NORMALIZE_TO,
    min_peaks: int = MIN_MATCHED_PEAKS,
) -> List[Dict[str, Any]]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    print(f"Loaded query CSV: {csv_path} | rows={len(df)}")

    if "intensity" in df.columns:
        intensity_col_in = "intensity"
    elif "intensity_sample" in df.columns:
        intensity_col_in = "intensity_sample"
    else:
        raise ValueError("Query CSV missing intensity column: needs 'intensity' or 'intensity_sample'")

    required = {"precursor_mz", "mz", intensity_col_in}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Query CSV missing columns: {missing}")

    df = df[pd.to_numeric(df[intensity_col_in], errors="coerce").fillna(0) > 0].copy()
    if df.empty:
        return []

    spectra: List[Dict[str, Any]] = []
    for prec, sub in df.groupby("precursor_mz", dropna=False):
        if sub.empty:
            continue
        peaks = list(zip(sub["mz"].astype(float).tolist(), pd.to_numeric(sub[intensity_col_in], errors="coerce").fillna(0).astype(float).tolist()))
        peaks = clean_and_normalize_peaks(
            peaks,
            rel_cutoff=QUERY_MIN_REL_INTENSITY,
            normalize_to=normalize_to,
            merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
        )
        if len(peaks) < min_peaks:
            continue
        spectra.append({"precursor_mz": float(prec), "peaks": peaks})

    print(f"Built query spectra: {len(spectra)} (one per precursor)")
    return spectra


# ---------------- Cosine similarity ----------------

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


# ---------------- DecoID-style core ----------------

def estimate_res_penalty(num_candidates: int, query_vec: List[float], user_penalty: float) -> float:
    if user_penalty is not None and float(user_penalty) > 0:
        return float(user_penalty)
    nonzero = max(1, int(np.sum(np.asarray(query_vec) > 0)))
    # moderate adaptive penalty: stronger when candidates are many, weaker when spectrum is richer
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
        raise ValueError(f"Dimension mismatch in deconvolveLASSO: A.shape={A_arr.shape}, len(b)={b_arr.shape[0]}")

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



def getVal(d, key):
    return d[key] if key in d else 0.0



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
        # fallback: keep the best few even if they are weak, to avoid empty candidate set
        temp = [(key, val[0], val[1], val[2], val[3]) for key, val in compoundDict.items()]
        temp.sort(key=lambda x: (x[3], x[4]), reverse=True)
        kept_items = temp[: min(max(5, top_n_candidates // 4), len(temp))]
    else:
        kept_items.sort(key=lambda x: (x[3], x[4]), reverse=True)
        kept_items = kept_items[: min(top_n_candidates, len(kept_items))]

    keys = [x[0] for x in kept_items]
    metIDs = [x[0] for x in keys]
    masses = [x[1] for x in keys]
    metNames = [x[2] for x in keys]
    best_adducts = [x[3] for x in keys]
    inchikeys = [x[4] for x in keys]
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

    return metIDs, spectraIDs, matrix, masses, metNames, best_adducts, inchikeys, axis, reduceSpec, initial_scores, initial_matches



def vector_to_peak_list(vec, axis, resolution):
    peaks = []
    for mz, val in zip(axis, vec):
        if float(val) > 0:
            peaks.append((float(np.round(mz, resolution)), float(val)))
    peaks.sort(key=lambda x: x[0])
    return peaks



def score_components_against_hmdb(
    component_vectors,
    metIDs,
    spectraIDs,
    masses,
    metNames,
    best_adducts,
    axis,
    resolution,
    initial_scores,
    initial_matches,
):
    rows = []

    for comp_rank_seed, comp_info in enumerate(component_vectors, start=1):
        component_name = comp_info["component_name"]
        component_vec = comp_info["component_vector"]
        component_abundance = comp_info["component_abundance"]
        component_peaks = vector_to_peak_list(component_vec, axis, resolution)

        component_hits = []
        for met_id, spec_id, mass, name, best_adduct, db_vec, init_score, init_n_match in zip(
            metIDs, spectraIDs, masses, metNames, best_adducts, comp_info["db_matrix"], initial_scores, initial_matches
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
                    "hmdb_id": met_id,
                    "spectrum_id": spec_id,
                    "compound_name": name,
                    "best_adduct": str(best_adduct),
                    "hmdb_precursor_mz": float(mass),
                    "component_score": float(score),
                    "component_score_0_to_100": float(score * 100.0),
                    "initial_candidate_score": float(init_score),
                    "initial_candidate_n_matched": int(init_n_match),
                    "n_matched_peaks": int(n_matched),
                    "n_component_peaks": int(n_q),
                    "n_ref_peaks": int(n_r),
                    "component_name": component_name,
                    "component_abundance": float(component_abundance),
                }
            )

        component_hits.sort(key=lambda x: (x["component_score"], x["initial_candidate_score"]), reverse=True)
        for rank, row in enumerate(component_hits, start=1):
            row["component_hit_rank"] = rank
            rows.append(row)

    return rows



def build_component_vectors(originalSpectra, matrix, res, metIDs, masses, best_adducts, centerMz):
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
    for x in positive:
        component_vec = [max(float(res[x] * m + d), 0.0) for m, d in zip(np.asarray(matrix[x], dtype=float), differences)]
        component_rows.append(
            {
                "component_name": str(metIDs[x]),
                "component_abundance": float(res[x] / resSum),
                "component_vector": component_vec,
                "component_mass": float(masses[x]),
                "best_adduct": str(best_adducts[x]),
                "db_matrix": matrix,
            }
        )
    return component_rows


class HMDBLibraryAdapter:
    def __init__(self, hmdb_spectra: List[Dict[str, Any]]):
        self.hmdb_spectra = hmdb_spectra

    def get_candidate_trees(self, lower_bound: float, upper_bound: float):
        trees: Dict[Tuple[str, float, str, str, str], Dict[str, Dict[float, float]]] = {}
        for rec in self.hmdb_spectra:
            mz = float(rec["precursor_mz"])
            if lower_bound <= mz <= upper_bound:
                key = (
                    rec["hmdb_id"],
                    mz,
                    rec.get("compound_name", "Unknown"),
                    rec.get("best_adduct", ""),
                    rec.get("inchikey", ""),
                )
                spec_id = str(rec.get("spectrum_id", rec.get("hmdb_id", "HMDB")))
                spec_dict = peaks_to_dict(rec["peaks"])
                trees.setdefault(key, {})[spec_id] = spec_dict
        return trees





def build_library_weights_rows(
    query_precursor_mz,
    metIDs,
    spectraIDs,
    metNames,
    inchikeys,
    best_adducts,
    masses,
    initial_scores,
    initial_matches,
    res_vector,
    s2n,
    use_penalty,
):
    n_candidates = int(len(metIDs))
    res_arr = np.asarray(res_vector, dtype=float)
    if res_arr.size == 0:
        return [{
            "precursor_mz": float(query_precursor_mz),
            "hmdb_id": "",
            "spectrum_id": "",
            "compound_name": "",
            "inchikey": "",
            "best_adduct": "",
            "hmdb_precursor_mz": np.nan,
            "initial_candidate_score": 0.0,
            "initial_candidate_n_matched": 0,
            "component_score": 0.0,
            "lasso_weight": 0.0,
            "lasso_weight_norm": 0.0,
            "weight_rank": 0,
            "is_active_component": False,
            "n_candidates": n_candidates,
            "s2n": float(s2n),
            "lasso_alpha": float(use_penalty),
        }]

    res_arr = np.where(res_arr > 0, res_arr, 0.0)
    weight_sum = float(np.sum(res_arr))
    ranked_indices = np.argsort(-res_arr, kind="mergesort") if res_arr.size > 0 else np.asarray([], dtype=int)
    ranks = {}
    for pos, idx in enumerate(ranked_indices, start=1):
        ranks[int(idx)] = int(pos)

    rows = []
    for idx, (met_id, spec_id, name, inchikey, best_adduct, mass, init_score, init_n_match) in enumerate(zip(
        metIDs, spectraIDs, metNames, inchikeys, best_adducts, masses, initial_scores, initial_matches
    )):
        weight = float(res_arr[idx]) if idx < len(res_arr) else 0.0
        rows.append({
            "precursor_mz": float(query_precursor_mz),
            "hmdb_id": str(met_id),
            "spectrum_id": str(spec_id),
            "compound_name": str(name),
            "inchikey": "" if inchikey is None else str(inchikey),
            "best_adduct": str(best_adduct),
            "hmdb_precursor_mz": float(mass),
            "initial_candidate_score": float(init_score),
            "initial_candidate_n_matched": int(init_n_match),
            "component_score": 0.0,
            "lasso_weight": weight,
            "lasso_weight_norm": float(weight / weight_sum) if weight_sum > 0 else 0.0,
            "weight_rank": int(ranks.get(idx, 0)),
            "is_active_component": bool(weight > 1e-12),
            "n_candidates": n_candidates,
            "s2n": float(s2n),
            "lasso_alpha": float(use_penalty),
        })

    rows.sort(key=lambda x: (x["lasso_weight"], x["initial_candidate_score"]), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["weight_rank"] = int(rank)
    return rows


def attach_component_scores_to_library_weights(library_weights_rows, component_rows):
    score_map = {}
    for row in component_rows:
        component_name = str(row.get("component_name", ""))
        hmdb_id = str(row.get("hmdb_id", ""))
        if component_name and hmdb_id and component_name == hmdb_id:
            key = hmdb_id
            score = _safe_float(row.get("component_score"))
            if score is None:
                score = 0.0
            if key not in score_map or float(score) > float(score_map[key]):
                score_map[key] = float(score)

    updated_rows = []
    for row in library_weights_rows:
        new_row = dict(row)
        hmdb_id = str(new_row.get("hmdb_id", ""))
        new_row["component_score"] = float(score_map.get(hmdb_id, 0.0))
        updated_rows.append(new_row)
    return updated_rows


def deconvolve_and_score_one_query(
    query_precursor_mz: float,
    query_peaks: List[Tuple[float, float]],
    library: HMDBLibraryAdapter,
    precursor_tol: float = PRECURSOR_TOL,
    res_penalty: float = DECOID_RES_PENALTY,
    resolution: int = DECOID_RESOLUTION,
) -> pd.DataFrame:
    query_peaks = clean_and_normalize_peaks(
        query_peaks,
        rel_cutoff=QUERY_MIN_REL_INTENSITY,
        normalize_to=NORMALIZE_TO,
        merge_tol=min(MATRIX_MZ_TOL, COS_MZ_TOL / 2.0),
    )
    query_spec_dict = peaks_to_dict(query_peaks)

    lower_bound = float(query_precursor_mz) - float(precursor_tol)
    upper_bound = float(query_precursor_mz) + float(precursor_tol)
    trees = library.get_candidate_trees(lower_bound=lower_bound, upper_bound=upper_bound)

    if not trees:
        empty_component_df = pd.DataFrame([
            {
                "precursor_mz": float(query_precursor_mz),
                "component_name": "",
                "component_abundance": np.nan,
                "component_hit_rank": 0,
                "hmdb_id": "",
                "compound_name": "",
                "best_adduct": "",
                "hmdb_precursor_mz": np.nan,
                "component_score": 0.0,
                "component_score_0_to_100": 0.0,
                "n_matched_peaks": 0,
                "n_component_peaks": len(query_peaks),
                "n_ref_peaks": 0,
                "n_candidates": 0,
            }
        ])
        empty_weights_df = pd.DataFrame(build_library_weights_rows(
            query_precursor_mz=float(query_precursor_mz),
            metIDs=[],
            spectraIDs=[],
            metNames=[],
            inchikeys=[],
            best_adducts=[],
            masses=[],
            initial_scores=[],
            initial_matches=[],
            res_vector=[],
            s2n=0.0,
            use_penalty=float(res_penalty if res_penalty > 0 else 0.0),
        ))
        return empty_component_df, empty_weights_df

    metIDs, spectraIDs, matrix, masses, metNames, best_adducts, inchikeys, axis, reduceSpec, initial_scores, initial_matches = getMatricesForGroup(
        trees=trees,
        spectra=[query_spec_dict],
        mz_merge_tol=float(MATRIX_MZ_TOL),
        top_n_candidates=int(TOP_N_CANDIDATES),
        min_initial_cosine=float(MIN_INITIAL_COSINE),
    )

    if len(matrix) == 0:
        empty_component_df = pd.DataFrame([
            {
                "precursor_mz": float(query_precursor_mz),
                "component_name": "",
                "component_abundance": np.nan,
                "component_hit_rank": 0,
                "hmdb_id": "",
                "compound_name": "",
                "best_adduct": "",
                "hmdb_precursor_mz": np.nan,
                "component_score": 0.0,
                "component_score_0_to_100": 0.0,
                "n_matched_peaks": 0,
                "n_component_peaks": len(query_peaks),
                "n_ref_peaks": 0,
                "n_candidates": 0,
            }
        ])
        empty_weights_df = pd.DataFrame(build_library_weights_rows(
            query_precursor_mz=float(query_precursor_mz),
            metIDs=[],
            spectraIDs=[],
            metNames=[],
            inchikeys=[],
            best_adducts=[],
            masses=[],
            initial_scores=[],
            initial_matches=[],
            res_vector=[],
            s2n=0.0,
            use_penalty=float(res_penalty if res_penalty > 0 else 0.0),
        ))
        return empty_component_df, empty_weights_df

    query_vec = reduceSpec[0]
    use_penalty = estimate_res_penalty(len(matrix), query_vec, float(res_penalty))
    res_vector, s2n = solveSystem(matrix, query_vec, resPenalty=float(use_penalty))

    # fallback: if adaptive/selected LASSO is too harsh, optionally relax with NNLS
    if FALLBACK_TO_NNLS and np.sum(np.asarray(res_vector) > 1e-12) == 0:
        res_vector, s2n = solveSystem(matrix, query_vec, resPenalty=0.0)
        use_penalty = 0.0

    library_weights_rows = build_library_weights_rows(
        query_precursor_mz=float(query_precursor_mz),
        metIDs=metIDs,
        spectraIDs=spectraIDs,
        metNames=metNames,
        inchikeys=inchikeys,
        best_adducts=best_adducts,
        masses=masses,
        initial_scores=initial_scores,
        initial_matches=initial_matches,
        res_vector=res_vector,
        s2n=s2n,
        use_penalty=use_penalty,
    )

    component_vectors = build_component_vectors(
        originalSpectra=query_vec,
        matrix=matrix,
        res=res_vector,
        metIDs=metIDs,
        masses=masses,
        best_adducts=best_adducts,
        centerMz=float(query_precursor_mz),
    )

    rows = score_components_against_hmdb(
        component_vectors=component_vectors,
        metIDs=metIDs,
        spectraIDs=spectraIDs,
        masses=masses,
        metNames=metNames,
        best_adducts=best_adducts,
        axis=axis,
        resolution=resolution,
        initial_scores=initial_scores,
        initial_matches=initial_matches,
    )

    library_weights_rows = attach_component_scores_to_library_weights(
        library_weights_rows=library_weights_rows,
        component_rows=rows,
    )

    if not rows:
        component_df = pd.DataFrame([
            {
                "precursor_mz": float(query_precursor_mz),
                "component_name": "",
                "component_abundance": np.nan,
                "component_hit_rank": 0,
                "hmdb_id": "",
                "compound_name": "",
                "best_adduct": "",
                "hmdb_precursor_mz": np.nan,
                "component_score": 0.0,
                "component_score_0_to_100": 0.0,
                "n_matched_peaks": 0,
                "n_component_peaks": len(query_peaks),
                "n_ref_peaks": 0,
                "n_candidates": len(metIDs),
                "s2n": float(s2n),
                "lasso_alpha": float(use_penalty),
            }
        ])
        weights_df = pd.DataFrame(library_weights_rows)
        return component_df, weights_df

    df = pd.DataFrame(rows)
    df.insert(0, "precursor_mz", float(query_precursor_mz))
    df["n_candidates"] = int(len(metIDs))
    df["s2n"] = float(s2n)
    df["lasso_alpha"] = float(use_penalty)
    component_df = df.sort_values(
        ["component_name", "component_hit_rank", "component_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    weights_df = pd.DataFrame(library_weights_rows)
    return component_df, weights_df



def process_one_query_csv(query_csv: Path, hmdb_spectra: List[Dict[str, Any]], out_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    query_spectra = build_query_spectra_from_avg_csv(query_csv)
    if not query_spectra:
        print(f"[WARN] No query spectra after filtering: {query_csv.name}")
        return pd.DataFrame(), pd.DataFrame()

    library = HMDBLibraryAdapter(hmdb_spectra)
    all_component_parts = []
    all_weight_parts = []

    for q in query_spectra:
        component_df_one, weight_df_one = deconvolve_and_score_one_query(
            query_precursor_mz=float(q["precursor_mz"]),
            query_peaks=q["peaks"],
            library=library,
            precursor_tol=PRECURSOR_TOL,
            res_penalty=DECOID_RES_PENALTY,
            resolution=DECOID_RESOLUTION,
        )
        component_df_one.insert(0, "query_csv", query_csv.name)
        weight_df_one.insert(0, "query_csv", query_csv.name)
        all_component_parts.append(component_df_one)
        all_weight_parts.append(weight_df_one)

    component_result = pd.concat(all_component_parts, axis=0, ignore_index=True) if all_component_parts else pd.DataFrame()
    if not component_result.empty:
        component_result = component_result.sort_values(["precursor_mz", "component_name", "component_hit_rank"], ascending=[True, True, True]).reset_index(drop=True)

    weight_result = pd.concat(all_weight_parts, axis=0, ignore_index=True) if all_weight_parts else pd.DataFrame()
    if not weight_result.empty:
        weight_result = weight_result.sort_values(["precursor_mz", "weight_rank", "initial_candidate_score"], ascending=[True, True, False]).reset_index(drop=True)

    out_component_file = out_root / f"{query_csv.stem}_decoid_component_ranking.csv"
    component_result.to_csv(out_component_file, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved per-query deconvolution summary: {out_component_file.resolve()}")

    out_weight_file = out_root / f"{query_csv.stem}_library_weights.csv"
    weight_result.to_csv(out_weight_file, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved per-query library weights: {out_weight_file.resolve()}")
    return component_result, weight_result



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
            f"No QUERY_CSV files found with filename containing '{QUERY_FILENAME_MUST_CONTAIN}'. "
            "Set QUERY_CSV_DIR, QUERY_GLOB, or QUERY_CSVS in CONFIG."
        )

    print(f"QUERY CSV files to process: {len(query_files)}")
    for p in query_files:
        print(f"  - {p}")

    hmdb_spectra = load_hmdb_spectra(HMDB_LIB_PATH)
    if not hmdb_spectra:
        raise RuntimeError("No HMDB spectra parsed. Check HMDB_LIB_PATH.")

    all_component_dfs = []
    all_weight_dfs = []
    for qcsv in query_files:
        component_df_one, weight_df_one = process_one_query_csv(qcsv, hmdb_spectra, out_root)
        if not component_df_one.empty:
            all_component_dfs.append(component_df_one)
        if not weight_df_one.empty:
            all_weight_dfs.append(weight_df_one)

    df_all = pd.concat(all_component_dfs, axis=0, ignore_index=True) if all_component_dfs else pd.DataFrame()
    out_all = out_root / "decoid_component_ranking_all_queries.csv"
    df_all.to_csv(out_all, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved overall deconvolution summary: {out_all.resolve()}")

    weights_all = pd.concat(all_weight_dfs, axis=0, ignore_index=True) if all_weight_dfs else pd.DataFrame()
    out_weights_all = out_root / "library_weights_all_queries.csv"
    weights_all.to_csv(out_weights_all, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved overall library weights summary: {out_weights_all.resolve()}")


if __name__ == "__main__":
    main()
