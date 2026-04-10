import pandas as pd
from pathlib import Path

# paths = {
#     "HMDB": r"E:\model\singlecell\results\GSE\0317\HMDB_DecoID\data\FR10.csv",
#     "MoNA": r"E:\model\singlecell\results\GSE\0317\MoNA_DecoID\data\FR10.csv",
#     "MS-DIAL": r"E:\model\singlecell\results\GSE\0317\MS-DiAL_DecoID\data\FR10.csv",
#     "GNPS": r"E:\model\singlecell\results\GSE\0317\GNPS_DecoID\data\FR10.csv",
# }
paths = {
    "HMDB": r"E:\model\singlecell\results\HGC\0330\HMDB_DecoID\data\COS0p8.csv",
    "MoNA": r"E:\model\singlecell\results\HGC\0330\MoNA_DecoID\data\COS0p8.csv",
    "MS-DIAL": r"E:\model\singlecell\results\HGC\0330\MS-DiAL_DecoID\data\COS0p8.csv",
    "GNPS": r"E:\model\singlecell\results\HGC\0330\GNPS_DecoID\data\COS0p8.csv",
}
# paths = {
#     "HMDB": r"E:\model\singlecell\results\HGC\0330\HMDB_DecoID\data\COS0p8andFR10.csv",
#     "MoNA": r"E:\model\singlecell\results\HGC\0330\MoNA_DecoID\data\COS0p8andFR10.csv",
#     "MS-DIAL": r"E:\model\singlecell\results\HGC\0330\MS-DiAL_DecoID\data\COS0p8andFR10.csv",
#     "GNPS": r"E:\model\singlecell\results\HGC\0330\GNPS_DecoID\data\COS0p8andFR10.csv",
# }

def _find_column(df: pd.DataFrame, aliases: list[str], fuzzy_contains: list[tuple[str, str]] | None = None) -> str:
    """Find a column by alias list and optional fuzzy contains rules."""
    norm = {c: c.lower().replace(" ", "").replace("_", "").replace("-", "") for c in df.columns}
    for a in aliases:
        a_norm = a.lower().replace(" ", "").replace("_", "").replace("-", "")
        for col, col_norm in norm.items():
            if col_norm == a_norm:
                return col

    if fuzzy_contains:
        for must1, must2 in fuzzy_contains:
            for col in df.columns:
                low = col.lower()
                if must1 in low and must2 in low:
                    return col

    raise ValueError(f"Cannot find required column. Columns: {list(df.columns)}")

def load_inchikey_set(csv_path: str) -> set[str]:
    df = pd.read_csv(csv_path)

    # 自动识别 InChIKey 列名（兼容 INCHI_KEY / inchikey 等）
    col = _find_column(
        df,
        aliases=["inchikey", "inchi_key", "inchi key", "inchikeys"],
        fuzzy_contains=[("inch", "key")],
    )

    # s = df[col].astype(str).str.strip().str.upper().str[:14]
    s = df[col].astype(str).str.strip().str.upper()
    s = s[~s.isin(["", "NAN", "NONE", "NULL"])]
    return set(s.tolist())

def load_precursor_mz_set(csv_path: str, decimals: int | None = 4) -> set[float]:
    """Load unique precursor_mz values.
    - decimals: if not None, round m/z to this number of decimals before de-duplication (recommended for float stability).
    """
    df = pd.read_csv(csv_path)

    col = _find_column(
        df,
        aliases=["precursor_mz", "precursormz", "precursorm/z", "mz", "precursor"],
        fuzzy_contains=[("precursor", "mz")],
    )

    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if decimals is not None:
        s = s.round(decimals)
    return set(s.astype(float).tolist())

# 读取每个谱库的 InChIKey / precursor_mz 集合
inchikey_sets = {lib: load_inchikey_set(p) for lib, p in paths.items()}
precursor_mz_sets = {lib: load_precursor_mz_set(p, decimals=4) for lib, p in paths.items()}

# 按要求的“累计加入”顺序
stages = [
    ("HMDB", ["HMDB"]),
    ("HMDB+MoNA", ["HMDB", "MoNA"]),
    ("HMDB+MoNA+MS-DIAL", ["HMDB", "MoNA", "MS-DIAL"]),
    ("HMDB+MoNA+MS-DIAL+GNPS", ["HMDB", "MoNA", "MS-DIAL", "GNPS"]),
]

summary_rows = []
detail_rows = []

prev_cum_ik = set()
prev_cum_mz = set()

for stage_name, libs in stages:
    cum_ik = set().union(*[inchikey_sets[l] for l in libs])
    cum_mz = set().union(*[precursor_mz_sets[l] for l in libs])

    newly_added_ik = cum_ik - prev_cum_ik
    newly_added_mz = cum_mz - prev_cum_mz

    summary_rows.append({
        "stage": stage_name,
        "libraries": ";".join(libs),
        "total_unique_inchikey": len(cum_ik),
        "new_unique_inchikey_vs_previous_stage": len(newly_added_ik),
        "total_unique_precursor_mz": len(cum_mz),
        "new_unique_precursor_mz_vs_previous_stage": len(newly_added_mz),
    })

    # 每一步新增的 InChIKey 明细 + 其来源谱库（在本阶段中出现在哪些库）
    for ik in sorted(newly_added_ik):
        origin_libs = [l for l in libs if ik in inchikey_sets[l]]
        detail_rows.append({
            "stage": stage_name,
            "type": "inchikey",
            "value": ik,
            "origin_libraries_in_stage": ";".join(origin_libs),
        })

    # （可选）每一步新增的 precursor_mz 明细 + 其来源谱库
    # for mz in sorted(newly_added_mz):
    #     origin_libs = [l for l in libs if mz in precursor_mz_sets[l]]
    #     detail_rows.append({
    #         "stage": stage_name,
    #         "type": "precursor_mz",
    #         "value": mz,
    #         "origin_libraries_in_stage": ";".join(origin_libs),
    #     })

    prev_cum_ik = cum_ik
    prev_cum_mz = cum_mz

summary_df = pd.DataFrame(summary_rows)
detail_df = pd.DataFrame(detail_rows)

# 输出
OUTPUT_SUMMARY = True
OUTPUT_DETAIL = False

summary_out = r"E:\model\singlecell\results\HGC\0330\COS0P8_growth_summary_14.csv"
detail_out  = r"E:\model\singlecell\results\HGC\0330\COS0P8_added_by_stage.csv"

if OUTPUT_SUMMARY:
    summary_df.to_csv(summary_out, index=False, encoding="utf-8-sig")
    print(f"Wrote: {Path(summary_out).name}")
else:
    print("Skip: over4_growth_summary.csv")

if OUTPUT_DETAIL:
    detail_df.to_csv(detail_out, index=False, encoding="utf-8-sig")
    print(f"Wrote: {Path(detail_out).name}")
else:
    print("Skip: over4_added_by_stage.csv")

print(summary_df)
