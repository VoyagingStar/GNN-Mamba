# -*- coding: utf-8 -*-
"""
GM_DecoID_Frequency.py

参考：
1) GM_DecoID.py 的 library_weights_all_queries.csv 数据格式
2) HMDB_DecoID_Frequency.py 的统计逻辑

功能：
- 读取 GM_DecoID 生成的 decoid_library_weights_all_queries.csv
- 按 initial_candidate_n_matched 阈值筛选已注释成分
- 统计每个 query_csv 中去重后的 InChIKey 数量
- 输出不同阈值下的 summary
- 输出 occurrence 表，包含 occurrence_query_csv_count
"""

from pathlib import Path
import pandas as pd


# =========================
# User-editable parameters
# =========================
INPUT_CSV = Path(r"E:\model\singlecell\results\HGC\0330\GM_DecoID\decoid_library_weights_all_queries.csv")
OUTPUT_DIR = Path(r"E:\model\singlecell\results\HGC\0330\GM_DecoID\data")

# Thresholds requested by the user
MATCH_THRESHOLDS = [1, 4, 6]

# Output filenames
SUMMARY_FILE = "cell_annotation_count_summary_by_threshold.csv"
OCCURRENCE_FILE_T1 = "annotated_inchikey_occurrence_nmatched_ge1.csv"
OCCURRENCE_FILE_T4 = "annotated_inchikey_occurrence_nmatched_ge4.csv"
OCCURRENCE_FILE_T6 = "annotated_inchikey_occurrence_nmatched_ge6.csv"


def standardize_bool(series: pd.Series) -> pd.Series:
    """Convert common TRUE/FALSE-like values to pandas boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "t": True,
        "f": False,
    }
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(False)
        .astype(bool)
    )


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {
        "query_csv",
        "library_id",
        "spectrum_id",
        "compound_name",
        "hmdb_id",
        "inchikey",
        "smiles",
        "matched_mass_column",
        "matched_mass_value",
        "mass_diff_da",
        "adduct_used_for_model",
        "adduct_fallback_used",
        "source_file",
        "source_folder",
        "precursor_mz",
        "library_precursor_mz",
        "initial_candidate_score",
        "initial_candidate_n_matched",
        "lasso_weight_raw",
        "lasso_weight_normalized",
        "component_score",
        "weight_rank",
        "is_active_component",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    # basic normalization
    df["query_csv"] = df["query_csv"].astype(str).str.strip()
    df["library_id"] = df["library_id"].astype(str).str.strip()
    df["spectrum_id"] = df["spectrum_id"].astype(str).str.strip()
    df["compound_name"] = df["compound_name"].astype(str).str.strip()
    df["hmdb_id"] = df["hmdb_id"].astype(str).str.strip()
    df["inchikey"] = df["inchikey"].astype(str).str.strip()
    df["smiles"] = df["smiles"].astype(str).str.strip()
    df["matched_mass_column"] = df["matched_mass_column"].astype(str).str.strip()
    df["adduct_used_for_model"] = df["adduct_used_for_model"].astype(str).str.strip()
    df["source_file"] = df["source_file"].astype(str).str.strip()
    df["source_folder"] = df["source_folder"].astype(str).str.strip()

    # normalize missing IDs
    for col in ["hmdb_id", "inchikey", "smiles", "matched_mass_column", "adduct_used_for_model"]:
        df.loc[df[col].isin(["", "nan", "None", "none", "NaN"]), col] = pd.NA

    # booleans
    df["is_active_component"] = standardize_bool(df["is_active_component"])
    df["adduct_fallback_used"] = standardize_bool(df["adduct_fallback_used"])

    # numerics
    numeric_cols = [
        "precursor_mz",
        "library_precursor_mz",
        "matched_mass_value",
        "mass_diff_da",
        "initial_candidate_score",
        "initial_candidate_n_matched",
        "lasso_weight_raw",
        "lasso_weight_normalized",
        "component_score",
        "weight_rank",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filter_annotated(df: pd.DataFrame, min_matched: int) -> pd.DataFrame:
    """
    Definition of annotated:
    - is_active_component == True
    - initial_candidate_n_matched >= min_matched
    - inchikey is not empty
    """
    return df[
        (df["is_active_component"])
        & (df["initial_candidate_n_matched"] >= min_matched)
        & (df["inchikey"].notna())
    ].copy()


def build_count_summary(df: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    """
    For each threshold, count deduplicated InChIKeys per query_csv, then compute
    mean / variance / standard deviation across query_csv replicates.
    """
    records = []

    for threshold in thresholds:
        filtered = filter_annotated(df, threshold)

        per_query = (
            filtered.groupby("query_csv")["inchikey"]
            .nunique()
            .rename("annotated_unique_inchikey_count")
            .reset_index()
        )

        all_queries = pd.DataFrame({"query_csv": sorted(df["query_csv"].dropna().unique())})
        per_query = all_queries.merge(per_query, on="query_csv", how="left")
        per_query["annotated_unique_inchikey_count"] = (
            per_query["annotated_unique_inchikey_count"].fillna(0).astype(int)
        )

        count_series = per_query["annotated_unique_inchikey_count"]

        records.append(
            {
                "min_initial_candidate_n_matched": threshold,
                "n_query_csv_replicates": int(len(per_query)),
                "mean_annotated_unique_inchikeys_per_query_csv": float(count_series.mean()),
                "variance_annotated_unique_inchikeys_per_query_csv": float(count_series.var(ddof=1)),
                "std_annotated_unique_inchikeys_per_query_csv": float(count_series.std(ddof=1)),
                "min_count": int(count_series.min()),
                "max_count": int(count_series.max()),
            }
        )

        per_query_detail_path = OUTPUT_DIR / f"per_query_unique_inchikey_counts_nmatched_ge{threshold}.csv"
        per_query.insert(0, "min_initial_candidate_n_matched", threshold)
        per_query.to_csv(per_query_detail_path, index=False, encoding="utf-8-sig")

    return pd.DataFrame(records)


def choose_representative_values(group: pd.DataFrame) -> pd.Series:
    """
    Pick representative metadata for one InChIKey.
    Preference:
    1) highest lasso_weight_normalized
    2) highest initial_candidate_score
    3) highest initial_candidate_n_matched

    Notes:
    - group is already filtered by filter_annotated()
    - best_component_score / mean_component_score are both derived from
      decoid_library_weights_all_queries.csv
    """
    g = group.copy()

    numeric_cols = [
        "precursor_mz",
        "library_precursor_mz",
        "matched_mass_value",
        "mass_diff_da",
        "initial_candidate_score",
        "initial_candidate_n_matched",
        "lasso_weight_raw",
        "lasso_weight_normalized",
        "component_score",
        "weight_rank",
    ]
    for col in numeric_cols:
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce")

    g = g.sort_values(
        by=["lasso_weight_normalized", "initial_candidate_score", "initial_candidate_n_matched"],
        ascending=[False, False, False],
        na_position="last",
    )
    row = g.iloc[0]

    return pd.Series(
        {
            "library_id": row.get("library_id"),
            "spectrum_id": row.get("spectrum_id"),
            "compound_name": row.get("compound_name"),
            "hmdb_id": row.get("hmdb_id"),
            "smiles": row.get("smiles"),
            "matched_mass_column": row.get("matched_mass_column"),
            "matched_mass_value": row.get("matched_mass_value"),
            "mass_diff_da": row.get("mass_diff_da"),
            "adduct_used_for_model": row.get("adduct_used_for_model"),
            "adduct_fallback_used": row.get("adduct_fallback_used"),
            "source_file": row.get("source_file"),
            "source_folder": row.get("source_folder"),
            "precursor_mz": row.get("precursor_mz"),
            "library_precursor_mz": row.get("library_precursor_mz"),
            "best_initial_candidate_score": g["initial_candidate_score"].max(),
            "mean_initial_candidate_score": g["initial_candidate_score"].mean(),
            "best_initial_candidate_n_matched": g["initial_candidate_n_matched"].max(),
            "best_lasso_weight_raw": g["lasso_weight_raw"].max(),
            "best_lasso_weight_normalized": g["lasso_weight_normalized"].max(),
            "best_weight_rank": g["weight_rank"].min(),
            "best_component_score": g["component_score"].max(),
            "mean_component_score": g["component_score"].mean(),
        }
    )


def build_occurrence_table(df: pd.DataFrame, min_matched: int) -> pd.DataFrame:
    """
    Count in how many different query_csv each annotated InChIKey appears.
    Output contains occurrence_query_csv_count.
    """
    filtered = filter_annotated(df, min_matched)

    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "min_initial_candidate_n_matched",
                "inchikey",
                "compound_name",
                "hmdb_id",
                "library_id",
                "spectrum_id",
                "smiles",
                "matched_mass_column",
                "matched_mass_value",
                "mass_diff_da",
                "adduct_used_for_model",
                "adduct_fallback_used",
                "source_file",
                "source_folder",
                "precursor_mz",
                "library_precursor_mz",
                "occurrence_query_csv_count",
                "query_csv_list",
                "best_initial_candidate_score",
                "mean_initial_candidate_score",
                "best_initial_candidate_n_matched",
                "best_lasso_weight_raw",
                "best_lasso_weight_normalized",
                "best_weight_rank",
                "best_component_score",
                "mean_component_score",
            ]
        )

    occurrence = (
        filtered.groupby("inchikey")["query_csv"]
        .agg(
            occurrence_query_csv_count=lambda s: s.nunique(),
            query_csv_list=lambda s: "; ".join(sorted(pd.unique(s))),
        )
        .reset_index()
    )

    metadata = (
        filtered.groupby("inchikey", dropna=False)
        .apply(choose_representative_values, include_groups=False)
        .reset_index()
    )

    result = occurrence.merge(metadata, on="inchikey", how="left")
    result = result.sort_values(
        by=["occurrence_query_csv_count", "best_lasso_weight_normalized", "best_initial_candidate_score"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    result.insert(0, "min_initial_candidate_n_matched", min_matched)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(INPUT_CSV)

    summary_df = build_count_summary(df, MATCH_THRESHOLDS)
    summary_path = OUTPUT_DIR / SUMMARY_FILE
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    occurrence_t1 = build_occurrence_table(df, 1)
    occurrence_t1.to_csv(OUTPUT_DIR / OCCURRENCE_FILE_T1, index=False, encoding="utf-8-sig")

    occurrence_t4 = build_occurrence_table(df, 4)
    occurrence_t4.to_csv(OUTPUT_DIR / OCCURRENCE_FILE_T4, index=False, encoding="utf-8-sig")

    occurrence_t6 = build_occurrence_table(df, 6)
    occurrence_t6.to_csv(OUTPUT_DIR / OCCURRENCE_FILE_T6, index=False, encoding="utf-8-sig")

    print("Done.")
    print(f"Summary: {summary_path}")
    print(f"Occurrence >=1: {OUTPUT_DIR / OCCURRENCE_FILE_T1}")
    print(f"Occurrence >=4: {OUTPUT_DIR / OCCURRENCE_FILE_T4}")
    print(f"Occurrence >=6: {OUTPUT_DIR / OCCURRENCE_FILE_T6}")


if __name__ == "__main__":
    main()
