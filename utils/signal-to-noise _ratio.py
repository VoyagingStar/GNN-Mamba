import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterable, List, Union

def calculate_snr_and_filter_one_sample(
    blank_df: pd.DataFrame,
    sample_file: Union[str, Path],
    output_dir: Union[str, Path],
    snr_threshold: float = 3.0,
    intensity_threshold: float = 1000,
):
    """
    单个 sample 输出到同一个 output_dir：
      - 筛选结果汇总表：sampleStem_filtered.csv（包含所有通过筛选的行）
      - 汇总信息表：sampleStem_summary.csv（单行统计信息）
    不再额外保存每个母离子组的信息。
    """
    sample_file = Path(sample_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n====== 正在处理 sample: {sample_file.name} ======")
    print("正在读取样本数据...")
    sample_df = pd.read_parquet(sample_file)

    n_raw = len(sample_df)

    # 强度过滤
    before_sample = len(sample_df)
    sample_df = sample_df[sample_df["intensity"] > intensity_threshold].copy()
    n_after_intensity = len(sample_df)
    print(f"样本组：强度 > {intensity_threshold} 的峰数量 {n_after_intensity} / {before_sample}")

    # 对齐 key
    blank_df = blank_df.copy()
    blank_df["precursor_mz_key"] = blank_df["precursor_mz"].round(4)
    blank_df["mz_key"] = blank_df["mz"].round(4)

    sample_df["precursor_mz_key"] = sample_df["precursor_mz"].round(4)
    sample_df["mz_key"] = sample_df["mz"].round(4)

    print("正在计算信噪比...")
    merged_df = pd.merge(
        sample_df,
        blank_df[["precursor_mz_key", "mz_key", "intensity"]],
        left_on=["precursor_mz_key", "mz_key"],
        right_on=["precursor_mz_key", "mz_key"],
        how="left",
        suffixes=("_sample", "_blank"),
    )

    merged_df.rename(columns={"intensity": "intensity_blank"}, inplace=True)
    merged_df["intensity_blank"] = merged_df["intensity_blank"].fillna(0)

    merged_df["snr"] = np.where(
        merged_df["intensity_blank"] > 0,
        merged_df["intensity_sample"] / merged_df["intensity_blank"],
        np.inf,
    )

    filtered_df = merged_df[merged_df["snr"] > snr_threshold].copy()
    n_after_snr = len(filtered_df)
    print(f"信噪比大于 {snr_threshold} 的峰数量: {n_after_snr}")

    sample_stem = sample_file.stem

    # ✅ 不再按母离子分组另存；直接保存一个 sample 的总结果表
    result_cols = [
        "precursor_mz", "mz",
        "intensity_sample", "intensity_blank", "snr"
    ]
    # 如果原表里有 rt/scan 等字段，也可以追加到这里
    existing_cols = [c for c in result_cols if c in filtered_df.columns]
    filtered_out = filtered_df[existing_cols].copy()

    filtered_file = output_path / f"{sample_stem}_filtered.csv"
    filtered_out.to_csv(filtered_file, index=False)
    print(f"已保存该 sample 的筛选结果汇总表到：{filtered_file}")

    # 用于统计：母离子组数量（仅统计，不输出分组文件）
    precursor_count = filtered_df["precursor_mz"].nunique()

    # 每个 sample 一个汇总信息文件（同目录避免覆盖）
    summary = pd.DataFrame([{
        "sample_file": sample_file.name,
        "total_peaks_raw": n_raw,
        "peaks_after_intensity_filter": n_after_intensity,
        "peaks_after_snr_filter": n_after_snr,
        "precursor_groups_count": int(precursor_count),
        "snr_threshold": snr_threshold,
        "intensity_threshold": intensity_threshold,
        "filtered_result_file": filtered_file.name,
    }])
    summary_file = output_path / f"{sample_stem}_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"已保存 sample 汇总信息到：{summary_file}")

    return filtered_df


def iter_sample_files(sample_input: Union[str, Path, Iterable[Union[str, Path]]]) -> List[Path]:
    """
    支持：
      - 传入目录：自动收集该目录下所有 .parquet
      - 传入文件列表：逐个处理
      - 传入单个文件：当作单元素列表
    """
    if isinstance(sample_input, (str, Path)):
        p = Path(sample_input)
        if p.is_dir():
            return sorted(p.glob("*.parquet"))
        return [p]
    return [Path(x) for x in sample_input]


def process_multiple_samples(
    blank_file: Union[str, Path],
    sample_input: Union[str, Path, Iterable[Union[str, Path]]],
    output_root_dir: Union[str, Path],
    snr_threshold: float = 3.0,
    intensity_threshold: float = 1000,
):
    """
    循环处理多个 sample 文件（同目录输出）：
      - blank 只读一次
      - 所有 sample 的输出都写到 output_root_dir
    """
    blank_file = Path(blank_file)
    output_root_dir = Path(output_root_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)

    print("正在读取空白组数据（只读取一次）...")
    blank_df = pd.read_parquet(blank_file)

    sample_files = iter_sample_files(sample_input)
    if not sample_files:
        raise FileNotFoundError("未找到任何 sample 文件（.parquet）。")

    for sf in sample_files:
        calculate_snr_and_filter_one_sample(
            blank_df=blank_df,
            sample_file=sf,
            output_dir=output_root_dir,  
            snr_threshold=snr_threshold,
            intensity_threshold=intensity_threshold,
        )


if __name__ == "__main__":
    # ======== 按需修改这里的路径 ========
    blank_file = r"E:\model\singlecell\singlecell_data\GSE\bk4\avg_ms2_blank_mean.parquet"

    # 方式A：给一个目录（目录下所有 .parquet 都会被当作 sample 处理）
    sample_dir = r"E:\model\singlecell\singlecell_data\GSE"

    # 方式B：给一个列表（更精确）
    # sample_files = [
    #     r"e:\model\data\mzml_out\avg_ms2_rt23p43_24p06.parquet",
    #     r"e:\model\data\mzml_out\avg_ms2_rt9p23_9p90.parquet",
    # ]

    output_root_dir = r"E:\model\singlecell\singlecell_data\GSE\SNR"

    process_multiple_samples(
        blank_file=blank_file,
        sample_input=sample_dir,          # 或 sample_files
        output_root_dir=output_root_dir,
        snr_threshold=3.0,
        intensity_threshold=1000,
    )
