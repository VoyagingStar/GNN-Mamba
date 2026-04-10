# -*- coding: utf-8 -*-
"""
Predict-only pipeline for precursors listed in GSE_unique_precursor_mz.csv.

- Read precursor list from PRECURSOR_LIST_CSV (must contain column precursor_mz)
- Search candidate mother ions in METABOLITES_WITH_ADDUCTS within TOLERANCE_DA
- Generate predicted spectra for ALL candidates (no cosine / no ranking)
- Save candidates + predicted spectra per precursor_mz

Requirements: same project deps as GMsearch_onlyPredSpec.py
"""

import os
import re
import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import hashlib

from rdkit import Chem
from torch_geometric.data import Batch

from dataset.mol_dataset import MoleculeDataset
from dataset.data_utils import mol_to_graph_data_obj_pos, np_one_hot
from GNN_mamba import GNNMambaFusion


# ======================
# CONFIG (edit here; no CLI)
# ======================

# 1) Model config + checkpoint folder (best.pt must exist in run_d["save_dir"])
CONFIG_PATH = "GNN-Mamba.yaml"

# 2) precursor list CSV (must include precursor_mz)
PRECURSOR_LIST_CSV = "singlecell/HGC_unique_precursor_mz.csv"

# 3) metabolites library: must include SMILES + adduct mass columns
METABOLITES_WITH_ADDUCTS = "metabolites_with_adducts.csv"

# 4) optional experimental conditions (instrument_type / collision_energy)
EXP_CONDITIONS_FILE = "ExperimentalConditions.csv"

# 5) tolerance for precursor matching (Da)
TOLERANCE_DA = 1.0

# 6) adduct mass columns to match in the library
MASS_COLUMNS = [
    "M+H_MASS",
    "M+Na_MASS",
    # "M+K_MASS",
    # "M+NH4_MASS",
    # "M+H-H2O_MASS",
]

# 7) library mass column -> model adduct string
COL_TO_ADDUCT = {
    "M+H_MASS": "[M+H]+",
    "M+Na_MASS": "[M+Na]+",
    # "M+K_MASS": "[M+K]+",
    # "M+NH4_MASS": "[M+NH4]+",
    # "M+H-H2O_MASS": "[M+H-H2O]+",
}

# 8) output folder
OUTPUT_ROOT = "singlecell/results_HGC_predict_0325"

# 9) save predicted peaks above this normalized intensity
SAVE_PRED_MIN_INTENSITY = 0.001

# 10) if True, write summary.txt per precursor folder
SAVE_SUMMARY_TXT = True


# ======================
# Utils
# ======================

def sanitize_filename(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    return name or "unnamed"

def precursor_to_folder_name(precursor_mz: float) -> str:
    if precursor_mz is None or (isinstance(precursor_mz, float) and np.isnan(precursor_mz)):
        return "precursor_nan"
    s = f"{float(precursor_mz):.6f}".rstrip("0").rstrip(".")
    return sanitize_filename(s)

def make_hmdb_based_filename(hmdb_id: str, out_dir: str, ext: str = ".csv") -> str:
    hmdb_id = sanitize_filename(str(hmdb_id).strip())
    if not hmdb_id:
        hmdb_id = "HMDB_UNKNOWN"

    base = hmdb_id
    candidate = base + ext
    full = os.path.join(out_dir, candidate)
    k = 2
    while os.path.exists(full):
        candidate = f"{base}_{k}{ext}"
        full = os.path.join(out_dir, candidate)
        k += 1
    return candidate

def expected_bin_count(mz_max: float, mz_bin_res: float) -> int:
    return int(np.floor(mz_max / mz_bin_res))

def align_pred_vector(pred_vec: np.ndarray, mz_max: float, mz_bin_res: float) -> np.ndarray:
    exp_n = expected_bin_count(mz_max, mz_bin_res)
    v = pred_vec.astype(np.float32).reshape(-1)
    if v.shape[0] == exp_n:
        return v
    if v.shape[0] > exp_n:
        return v[:exp_n]
    out = np.zeros((exp_n,), dtype=np.float32)
    out[: v.shape[0]] = v
    return out

def vec_to_peaks(pred_vec: np.ndarray, mz_max: float, mz_bin_res: float, min_intensity: float = 0.0):
    bins = np.arange(mz_bin_res, mz_max + mz_bin_res, step=mz_bin_res).astype(np.float32)
    v = pred_vec.astype(np.float32)

    if v.shape[0] != bins.shape[0]:
        v = align_pred_vector(v, mz_max=mz_max, mz_bin_res=mz_bin_res)
        bins = np.arange(mz_bin_res, mz_max + mz_bin_res, step=mz_bin_res).astype(np.float32)

    mask = v > float(min_intensity)
    return bins[mask], v[mask]

def save_pred_spectrum_csv(out_path: str, pred_vec: np.ndarray, mz_max: float, mz_bin_res: float, min_intensity: float):
    mzs, ints = vec_to_peaks(pred_vec, mz_max=mz_max, mz_bin_res=mz_bin_res, min_intensity=min_intensity)
    pd.DataFrame({"mz": mzs, "intensity": ints}).to_csv(out_path, index=False, encoding="utf-8-sig")


def read_csv_with_fallback(path: str) -> pd.DataFrame:
    # 兼容你之前遇到的编码报错：utf-8 / gbk / gb18030 等
    encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "latin-1", "cp1252"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot read CSV: {path}. Last error: {last_err}")


# ======================
# Model / Dataset
# ======================

def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_d = cfg.get("data_d", {})
    run_d = cfg.get("run_d", {})
    return data_d, run_d

def build_dataset_for_meta(data_d):
    spec_df_name = data_d.get("spec_df", "spec_df_massspecgym.pkl")
    mol_df_name = data_d.get("mol_df", "mol_df_massspecgym.pkl")
    return MoleculeDataset(spec_df=spec_df_name, mol_df=mol_df_name, **data_d)

def build_model(ds: MoleculeDataset, run_d: dict, device: torch.device):
    spec_dim, meta_dim = ds.get_data_dims()
    model = GNNMambaFusion(
        hidden=256,
        n_bins=10000,
        meta_dim=meta_dim,
        use_adj_from_bond=True,
        n_mamba_layers=run_d["n_mamba_layers"],
        seg_head=run_d["use_seg_head"],
        gnn_type=run_d["gnn_type"],
    ).to(device)

    ckpt_root = run_d["save_dir"]
    ckpt_path = os.path.join(ckpt_root, "best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state"], strict=False)
    model.eval()
    return model, spec_dim

def build_spec_meta(inst_type: str, adduct: str, collision_energy: float, ds: MoleculeDataset) -> torch.Tensor:
    ce_vec = ds.ce_func(collision_energy)

    if inst_type not in ds.inst_type2id:
        inst_type = next(iter(ds.inst_type2id.keys()))
    inst_idx = ds.inst_type2id[inst_type]
    inst_onehot = torch.as_tensor(np_one_hot(inst_idx, num_classes=ds.num_inst_type), dtype=torch.float32)

    if adduct not in ds.adduct2id:
        raise ValueError(f"Unknown adduct for this model: {adduct}")
    adduct_idx = ds.adduct2id[adduct]
    adduct_onehot = torch.as_tensor(np_one_hot(adduct_idx, num_classes=ds.num_adduct), dtype=torch.float32)

    return torch.cat([ce_vec, inst_onehot, adduct_onehot], dim=0).unsqueeze(0)

def build_graph_from_smiles(smiles: str, idx: int):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    return mol_to_graph_data_obj_pos(mol, idx)

def load_conditions(conditions_path: str):
    inst_type = "UNKNOWN"
    collision_energy = 20.0
    if os.path.isfile(conditions_path):
        try:
            df = read_csv_with_fallback(conditions_path)
            if len(df) > 0:
                if "instrument_type" in df.columns:
                    inst_type = str(df.iloc[0]["instrument_type"])
                if "collision_energy" in df.columns:
                    collision_energy = float(df.iloc[0]["collision_energy"])
        except Exception:
            pass
    return inst_type, collision_energy


# ======================
# Candidate search
# ======================

def search_candidates_by_precursor(meta_df: pd.DataFrame, precursor_mz: float, tol_da: float) -> pd.DataFrame:
    lo, hi = precursor_mz - tol_da, precursor_mz + tol_da
    tasks = []

    for col in MASS_COLUMNS:
        if col not in meta_df.columns:
            continue
        s = pd.to_numeric(meta_df[col], errors="coerce")
        hits = meta_df[s.between(lo, hi)].copy()
        if hits.empty:
            continue
        hits["matched_mass_column"] = col
        hits["matched_mass_value"] = pd.to_numeric(hits[col], errors="coerce")
        hits["target_precursor_mz"] = precursor_mz
        hits["mass_diff_da"] = hits["matched_mass_value"] - precursor_mz
        hits["adduct"] = COL_TO_ADDUCT.get(col, "[M+H]+")
        tasks.append(hits)

    if not tasks:
        return pd.DataFrame()

    out = pd.concat(tasks, ignore_index=True)

    if "SMILES" not in out.columns:
        raise KeyError("metabolites_with_adducts.csv must contain 'SMILES' column for prediction.")

    # HMDB id column standardization for naming predicted spectra files
    hmdb_cols = ["DATABASE_ID", "HMDB_ID", "HMDB", "HMDBID", "hmdb_id", "hmdb", "hmdbid"]
    hmdb_col = next((c for c in hmdb_cols if c in out.columns), None)
    out["HMDB_ID_STD"] = out[hmdb_col].astype(str).str.strip() if hmdb_col else ""

     # InChIKey column standardization (optional but recommended)
    inchikey_cols = ["INCHI_KEY", "INCHIKEY", "InChIKey", "inchi_key", "inchikey", "inchiKey"]
    inchikey_col = next((c for c in inchikey_cols if c in out.columns), None)
    if inchikey_col is not None:
        out["INCHI_KEY_STD"] = out[inchikey_col].astype(str).str.strip()
    else:
        out["INCHI_KEY_STD"] = ""

    return out


def predict_for_precursor(
    candidates_df: pd.DataFrame,
    out_dir: str,
    ds: MoleculeDataset,
    model: torch.nn.Module,
    mz_max: float,
    mz_bin_res: float,
    inst_type: str,
    collision_energy: float,
    device: torch.device,
) -> pd.DataFrame:
    pred_dir = os.path.join(out_dir, "predicted_spectra")
    os.makedirs(pred_dir, exist_ok=True)

    results = []
    with torch.no_grad():
        for i, row in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc=f"Predict ({os.path.basename(out_dir)})"):
            smiles = str(row["SMILES"])
            adduct = str(row.get("adduct", "[M+H]+"))

            try:
                graph = build_graph_from_smiles(smiles, idx=i)
            except Exception:
                continue

            fallback_used = False
            try:
                spec_meta = build_spec_meta(inst_type, adduct, collision_energy, ds)
            except Exception:
                # fallback to [M+H]+
                try:
                    spec_meta = build_spec_meta(inst_type, "[M+H]+", collision_energy, ds)
                    fallback_used = True
                    adduct = "[M+H]+"
                except Exception:
                    continue

            graph.spec_meta = spec_meta
            batch = Batch.from_data_list([graph]).to(device)

            pred = model(batch)
            pred = torch.relu(pred)
            max_val = pred.amax(dim=-1, keepdim=True).clamp_min(1e-12)
            pred = pred / max_val
            pred_vec = pred.squeeze(0).cpu().numpy().astype(np.float32)
            pred_vec = align_pred_vector(pred_vec, mz_max=mz_max, mz_bin_res=mz_bin_res)

            hmdb_id = row.get("HMDB_ID_STD", "")
            fname = make_hmdb_based_filename(hmdb_id, pred_dir, ext=".csv")
            pred_spectrum_path = os.path.join(pred_dir, fname)

            save_pred_spectrum_csv(
                pred_spectrum_path,
                pred_vec,
                mz_max=mz_max,
                mz_bin_res=mz_bin_res,
                min_intensity=SAVE_PRED_MIN_INTENSITY,
            )

            results.append({
                "target_precursor_mz": float(row.get("target_precursor_mz", np.nan)),
                "matched_mass_column": row.get("matched_mass_column", ""),
                "matched_mass_value": float(row.get("matched_mass_value", np.nan)) if pd.notna(row.get("matched_mass_value", np.nan)) else np.nan,
                "mass_diff_da": float(row.get("mass_diff_da", np.nan)) if pd.notna(row.get("mass_diff_da", np.nan)) else np.nan,
                "adduct_used_for_model": adduct,
                "adduct_fallback_used": fallback_used,
                "SMILES": row.get("SMILES", ""),
                "INCHI_KEY": row.get("INCHI_KEY_STD", row.get("INCHI_KEY", "")),
                "HMDB_ID": row.get("HMDB_ID_STD", ""),
                "NAME": row.get("NAME", "") if "NAME" in candidates_df.columns else "",
                "pred_spectrum_path": pred_spectrum_path,
            })

    return pd.DataFrame(results)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load config + dataset + model
    data_d, run_d = load_config(CONFIG_PATH)
    mz_max = float(data_d.get("mz_max", 1000.0))
    mz_bin_res = float(data_d.get("mz_bin_res", 0.1))

    ds = build_dataset_for_meta(data_d)
    model, _ = build_model(ds, run_d, device)

    # Experimental conditions
    inst_type, collision_energy = load_conditions(EXP_CONDITIONS_FILE)
    print(f"[INFO] instrument_type={inst_type}, collision_energy={collision_energy}")

    # Read metabolite library
    if not os.path.isfile(METABOLITES_WITH_ADDUCTS):
        raise FileNotFoundError(f"Cannot find: {METABOLITES_WITH_ADDUCTS}")
    meta_df = read_csv_with_fallback(METABOLITES_WITH_ADDUCTS)

    # Read precursor list
    if not os.path.isfile(PRECURSOR_LIST_CSV):
        raise FileNotFoundError(f"Cannot find: {PRECURSOR_LIST_CSV}")
    pre_df = read_csv_with_fallback(PRECURSOR_LIST_CSV)

    # find precursor_mz column robustly
    lower_map = {c.lower(): c for c in pre_df.columns}
    if "precursor_mz" not in lower_map:
        raise KeyError(f"{PRECURSOR_LIST_CSV} must contain column 'precursor_mz'. Columns: {list(pre_df.columns)}")
    prec_col = lower_map["precursor_mz"]

    precursors = sorted(pd.to_numeric(pre_df[prec_col], errors="coerce").dropna().unique().tolist())
    print(f"[INFO] Found {len(precursors)} unique precursor_mz in {PRECURSOR_LIST_CSV}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary_records = []

    for precursor_mz in precursors:
        folder = precursor_to_folder_name(precursor_mz)
        out_dir = os.path.join(OUTPUT_ROOT, folder)
        os.makedirs(out_dir, exist_ok=True)

        # Search candidates
        candidates_df = search_candidates_by_precursor(meta_df, float(precursor_mz), TOLERANCE_DA)
        candidates_path = os.path.join(out_dir, "candidates.csv")

        if candidates_df.empty:
            pd.DataFrame().to_csv(candidates_path, index=False, encoding="utf-8-sig")
            if SAVE_SUMMARY_TXT:
                with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
                    f.write(f"precursor_mz={precursor_mz}\nNo candidates found within tolerance.\n")
            print(f"[WARN] precursor {precursor_mz}: no candidates")
            summary_records.append({"precursor_mz": float(precursor_mz), "candidates_total": 0, "predicted_total": 0})
            continue

        # attach target precursor mz to each row (for downstream)
        candidates_df = candidates_df.copy()
        candidates_df["target_precursor_mz"] = float(precursor_mz)
        candidates_df.to_csv(candidates_path, index=False, encoding="utf-8-sig")
        print(f"[INFO] precursor {precursor_mz}: {len(candidates_df)} candidates")

        # Predict
        pred_df = predict_for_precursor(
            candidates_df=candidates_df,
            out_dir=out_dir,
            ds=ds,
            model=model,
            mz_max=mz_max,
            mz_bin_res=mz_bin_res,
            inst_type=inst_type,
            collision_energy=collision_energy,
            device=device,
        )

        pred_list_path = os.path.join(out_dir, "predicted_candidates.csv")
        pred_df.to_csv(pred_list_path, index=False, encoding="utf-8-sig")

        summary_records.append({
            "precursor_mz": float(precursor_mz),
            "candidates_total": int(len(candidates_df)),
            "predicted_total": int(len(pred_df)) if pred_df is not None else 0,
        })

        if SAVE_SUMMARY_TXT:
            with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
                f.write(f"precursor_mz={precursor_mz}\n")
                f.write(f"candidates={len(candidates_df)}\n")
                f.write(f"predicted={len(pred_df)}\n")

        print(f"[OK] precursor {precursor_mz}: saved to {out_dir}")

    summary_df = pd.DataFrame(summary_records)
    summary_path = os.path.join(OUTPUT_ROOT, "summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[DONE] Summary saved to: {summary_path}")
    print(f"[DONE] All results saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
