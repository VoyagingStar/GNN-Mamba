import pandas as pd
import numpy as np


def merge_and_dedup_by_inchikey(
    hmdb_csv: str,
    gnps_csv: str,
    mona_csv: str,
    msp_csv: str,
    out_csv: str,
    source_names=("HMDB", "GNPS", "MoNA", "MSP"),
) -> pd.DataFrame:
    """
    合并四个数据库匹配结果CSV，并按 inchikey 去重。

    去重规则：
      1) inchikey 一致时，保留 appear_file_count 最大的行
      2) 若 appear_file_count 一致，保留 mean_cosine 最大的行
         (为了结果可复现，再加两个稳定的兜底规则：max_cosine 更大优先；source_db 字母序)
      3) 输出中注明：
         - selected_from: 最终保留行来自哪个库
         - all_sources: 同一 inchikey 出现在哪些库
         - other_sources: 除 selected_from 外还出现在哪些库
         - source_count: 出现库数量
         - is_in_multiple_sources: 是否出现在多个库

    兼容：
      - inchikey 列名可能是 'inchikey' 或 'INCHI_KEY'
      - inchikey 缺失/空的行：不参与去重，原样保留
    """
    paths = [hmdb_csv, gnps_csv, mona_csv, msp_csv]
    dfs = []

    for src, path in zip(source_names, paths):
        df = pd.read_csv(path).copy()
        df["source_db"] = src

        # 统一 inchikey 列名
        if "inchikey" not in df.columns:
            if "INCHI_KEY" in df.columns:
                df.rename(columns={"INCHI_KEY": "inchikey"}, inplace=True)

        if "inchikey" not in df.columns:
            raise ValueError(f"{src} 文件找不到 inchikey/INCHI_KEY 列：{path}")

        # 清洗 inchikey
        df["inchikey"] = df["inchikey"].astype(str).str.strip()
        df.loc[df["inchikey"].str.lower().isin(["", "nan", "none"]), "inchikey"] = np.nan

        # 数值列转数值（防止被读成字符串）
        for col in ["occurrence_query_csv_count", "mean_component_score", "best_component_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True, sort=False)

    # 统计每个 inchikey 出现在哪些库
    presence = (
        all_df.dropna(subset=["inchikey"])
        .groupby("inchikey")["source_db"]
        .apply(lambda s: ",".join(sorted(set(s))))
        .rename("all_sources")
    )

    # 仅对有 inchikey 的行去重
    work = all_df.dropna(subset=["inchikey"]).copy()
    work["all_sources"] = work["inchikey"].map(presence)

    # 排序：让“应保留的那行”排在每组最前
    work_sorted = work.sort_values(
        by=["inchikey", "occurrence_query_csv_count", "mean_component_score", "best_component_score", "source_db"],
        ascending=[True, False, False, False, True],
        kind="mergesort"  # 稳定排序，保证可复现
    )

    kept = work_sorted.groupby("inchikey", as_index=False).head(1).copy()

    kept["selected_from"] = kept["source_db"]
    kept["other_sources"] = kept.apply(
        lambda r: ",".join([s for s in r["all_sources"].split(",") if s != r["selected_from"]])
        if pd.notna(r["all_sources"]) else "",
        axis=1
    )
    kept["is_in_multiple_sources"] = kept["other_sources"].astype(str).str.len().gt(0)
    kept["source_count"] = kept["all_sources"].str.split(",").apply(len)

    # inchikey 缺失的行不参与去重，原样拼回
    missing = all_df[all_df["inchikey"].isna()].copy()
    if len(missing) > 0:
        missing["all_sources"] = missing["source_db"]
        missing["selected_from"] = missing["source_db"]
        missing["other_sources"] = ""
        missing["is_in_multiple_sources"] = False
        missing["source_count"] = 1

        # 对齐列
        for c in kept.columns:
            if c not in missing.columns:
                missing[c] = np.nan
        missing = missing[kept.columns]

        merged = pd.concat([kept, missing], ignore_index=True)
    else:
        merged = kept

    merged.to_csv(out_csv, index=False)
    return merged


if __name__ == "__main__":
    # 直接改成你的实际路径（Windows 路径建议用 r""）
    hmdb_path = r"E:\model\singlecell\results\HGC\0330\HMDB_DecoID\data\COS0p8andFR10.csv"
    gnps_path = r"E:\model\singlecell\results\HGC\0330\GNPS_DecoID\data\COS0p8andFR10.csv"
    mona_path = r"E:\model\singlecell\results\HGC\0330\MoNA_DecoID\data\COS0p8andFR10.csv"
    msp_path  = r"E:\model\singlecell\results\HGC\0330\MS-DiAL_DecoID\data\COS0p8andFR10.csv"

    out_path  = r"E:\model\singlecell\results\HGC\0330\COS0p8andFR10_merged_library_matches.csv"

    df_out = merge_and_dedup_by_inchikey(hmdb_path, gnps_path, mona_path, msp_path, out_path)
    print("Done. Output rows:", len(df_out))
    print("Saved to:", out_path)
