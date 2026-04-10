# -*- coding: utf-8 -*-
"""
功能：
1) 从 GNPS_DecoID.py 输出的 decoid_library_weights_all_queries.csv 读取结果
2) 依据 initial_candidate_n_matched 阈值筛选 annotated 结果
3) 统计每个 query_csv 中去重后的 InChIKey 数量
4) 统计每个 InChIKey 在多少个 query_csv 中出现（occurrence_query_csv_count）
5) 输出 summary 和 occurrence 表

说明：
- annotated 的定义：
    * is_active_component == True
    * initial_candidate_n_matched >= threshold
    * inchikey 非空
- 代表性元数据按以下优先级选取：
    1) lasso_weight_normalized 最大
    2) initial_candidate_score 最大
    3) initial_candidate_n_matched 最大
"""

from pathlib import Path
import pandas as pd


# =========================
# User-editable parameters
# =========================
INPUT_CSV = Path(r"E:\model\singlecell\results\HGC\0330\GNPS_DecoID\decoid_library_weights_all_queries.csv")
OUTPUT_DIR = Path(r"E:\model\singlecell\results\HGC\0330\GNPS_DecoID\data")

MATCH_THRESHOLDS = [1, 4, 6]

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
        "precursor_mz",
        "gnps_id",
        "spectrum_id",
        "compound_name",
        "inchikey",
        "smiles",
        "formula",
        "best_adduct",
        "ion_mode",
        "instrument",
        "collision_energy",
        "source_file",
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
    df["query_csv"] = df["query_csv"].astype(str).str.strip()
    df["is_active_component"] = standardize_bool(df["is_active_component"])

    numeric_cols = [
        "precursor_mz",
        "library_precursor_mz",
        "initial_candidate_score",
        "initial_candidate_n_matched",
        "lasso_weight_raw",
        "lasso_weight_normalized",
        "component_score",
        "weight_rank",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "inchikey",
        "gnps_id",
        "spectrum_id",
        "compound_name",
        "best_adduct",
        "smiles",
        "formula",
        "ion_mode",
        "instrument",
        "collision_energy",
        "source_file",
    ]:
        df[col] = df[col].astype(str).str.strip()

    df.loc[df["inchikey"].isin(["", "nan", "None", "none", "NaN", "<NA>"]), "inchikey"] = pd.NA
    df.loc[df["gnps_id"].isin(["", "nan", "None", "none", "NaN", "<NA>"]), "gnps_id"] = pd.NA

    return df


def filter_annotated(df: pd.DataFrame, min_matched: int) -> pd.DataFrame:
    """Definition of annotated:
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
    """For each threshold, count deduplicated InChIKeys per query_csv, then compute
    mean / variance / standard deviation across query_csv replicates.
    """
    records = []

    all_queries = pd.DataFrame({"query_csv": sorted(df["query_csv"].dropna().unique())})

    for threshold in thresholds:
        filtered = filter_annotated(df, threshold)

        per_query = (
            filtered.groupby("query_csv")["inchikey"]
            .nunique()
            .rename("annotated_unique_inchikey_count")
            .reset_index()
        )

        per_query = all_queries.merge(per_query, on="query_csv", how="left")
        per_query["annotated_unique_inchikey_count"] = (
            per_query["annotated_unique_inchikey_count"].fillna(0).astype(int)
        )

        count_series = per_query["annotated_unique_inchikey_count"]

        var_val = float(count_series.var(ddof=1)) if len(count_series) > 1 else 0.0
        std_val = float(count_series.std(ddof=1)) if len(count_series) > 1 else 0.0

        records.append(
            {
                "min_initial_candidate_n_matched": threshold,
                "n_query_csv_replicates": int(len(per_query)),
                "mean_annotated_unique_inchikeys_per_query_csv": float(count_series.mean()),
                "variance_annotated_unique_inchikeys_per_query_csv": var_val,
                "std_annotated_unique_inchikeys_per_query_csv": std_val,
                "min_count": int(count_series.min()) if len(count_series) > 0 else 0,
                "max_count": int(count_series.max()) if len(count_series) > 0 else 0,
            }
        )

        per_query_detail_path = OUTPUT_DIR / f"per_query_unique_inchikey_counts_nmatched_ge{threshold}.csv"
        per_query.insert(0, "min_initial_candidate_n_matched", threshold)
        per_query.to_csv(per_query_detail_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {per_query_detail_path}")

    return pd.DataFrame(records)


def choose_representative_values(group: pd.DataFrame) -> pd.Series:
    """Pick representative metadata for one InChIKey.

    Preference:
    1) highest lasso_weight_normalized
    2) highest initial_candidate_score
    3) highest initial_candidate_n_matched
    """
    g = group.copy()

    numeric_cols = [
        "precursor_mz",
        "library_precursor_mz",
        "lasso_weight_raw",
        "lasso_weight_normalized",
        "weight_rank",
        "initial_candidate_score",
        "initial_candidate_n_matched",
        "component_score",
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
            "gnps_id": row.get("gnps_id"),
            "spectrum_id": row.get("spectrum_id"),
            "compound_name": row.get("compound_name"),
            "best_adduct": row.get("best_adduct"),
            "smiles": row.get("smiles"),
            "formula": row.get("formula"),
            "ion_mode": row.get("ion_mode"),
            "instrument": row.get("instrument"),
            "collision_energy": row.get("collision_energy"),
            "source_file": row.get("source_file"),
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
    """Count in how many different query_csv each annotated InChIKey appears."""
    filtered = filter_annotated(df, min_matched)

    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "min_initial_candidate_n_matched",
                "inchikey",
                "compound_name",
                "gnps_id",
                "spectrum_id",
                "best_adduct",
                "smiles",
                "formula",
                "ion_mode",
                "instrument",
                "collision_energy",
                "source_file",
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

    value_columns = [c for c in filtered.columns if c != "inchikey"]
    metadata = (
        filtered.groupby("inchikey")[value_columns]
        .apply(choose_representative_values)
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
    print(f"Saved: {summary_path}")

    occurrence_t1 = build_occurrence_table(df, 1)
    occurrence_t1.to_csv(OUTPUT_DIR / OCCURRENCE_FILE_T1, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_DIR / OCCURRENCE_FILE_T1}")

    occurrence_t4 = build_occurrence_table(df, 4)
    occurrence_t4.to_csv(OUTPUT_DIR / OCCURRENCE_FILE_T4, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_DIR / OCCURRENCE_FILE_T4}")

    occurrence_t6 = build_occurrence_table(df, 6)
    occurrence_t6.to_csv(OUTPUT_DIR / OCCURRENCE_FILE_T6, index=False, encoding="utf-8-sig")
    print(f"Saved: {OUTPUT_DIR / OCCURRENCE_FILE_T6}")

    print("Done.")
    print(f"Summary: {summary_path}")
    print(f"Occurrence >=1: {OUTPUT_DIR / OCCURRENCE_FILE_T1}")
    print(f"Occurrence >=4: {OUTPUT_DIR / OCCURRENCE_FILE_T4}")
    print(f"Occurrence >=6: {OUTPUT_DIR / OCCURRENCE_FILE_T6}")


if __name__ == "__main__":
    main()
