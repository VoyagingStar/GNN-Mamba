import subprocess
from pathlib import Path


def convert_raw_to_mzml(
    raw_path: Path,
    out_dir: Path,
    exe_path: Path,
    gzip: bool = False,
):
    """
    使用 ThermoRawFileParser.exe 将单个 RAW 转换为 mzML。
    通过 subprocess 调用外部 .exe，不需要用户手动在命令行敲命令。
    """
    if not exe_path.exists():
        raise FileNotFoundError(f"ThermoRawFileParser.exe 未找到: {exe_path}")

    if not raw_path.exists():
        raise FileNotFoundError(f"RAW 文件不存在: {raw_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 构造命令行参数（但由 Python 代为执行）
    cmd = [
        str(exe_path),
        "-i", str(raw_path),
        "-o", str(out_dir),
        "-f", "1",  # 1 = mzML; 2 = MGF; 3 = CSV; 4 = Parquet
    ]

    if gzip:
        cmd.append("-g")  # 输出 .mzML.gz

    print(f"[INFO] 开始转换 RAW -> mzML: {raw_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[ERROR] 转换失败: {raw_path}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)
        raise RuntimeError(f"ThermoRawFileParser 转换失败，返回码: {result.returncode}")
    else:
        print(f"[OK] 转换成功: {raw_path.name}")
        # 如需调试，可取消下面两行注释
        # print("STDOUT:")
        # print(result.stdout)
        # print("STDERR:")
        # print(result.stderr)


def batch_convert_folder(
    raw_dir: Path,
    out_dir: Path,
    exe_path: Path,
    gzip: bool = False,
):
    """
    批量转换文件夹中的所有 RAW 文件。
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"RAW 文件夹不存在: {raw_dir}")

    raw_files = sorted(list(raw_dir.glob("*.raw")))

    if not raw_files:
        print(f"[WARN] 未在 {raw_dir} 中找到任何 .RAW 文件")
        return

    print(f"[INFO] 在 {raw_dir} 中找到 {len(raw_files)} 个 RAW 文件")
    out_dir.mkdir(parents=True, exist_ok=True)

    for raw_file in raw_files:
        try:
            convert_raw_to_mzml(raw_file, out_dir, exe_path, gzip=gzip)
        except Exception as e:
            # 单个文件失败不会中断整个批处理
            print(f"[WARN] 文件 {raw_file.name} 转换时发生错误: {e}")


if __name__ == "__main__":
    # ========= 在这里按你的实际情况修改路径 =========
    # 1) ThermoRawFileParser.exe 路径
    exe_path = Path(r"E:\model\tools\ThermoRawFileParserGUI-master\resources\ThermoRawFileParser\ThermoRawFileParser.exe")

    # 2) 只转换一个 RAW 文件
    # raw_path = Path(r"E:\model\data\1_POS_DIA_2DA_AGC500_IT10ms_3bk_10QC.raw")
    # out_dir = Path(r"E:\model\data\mzml_out")
    # convert_raw_to_mzml(raw_path, out_dir, exe_path, gzip=False)

    # 3) 批量转换某个文件夹里的所有 RAW
    raw_dir = Path(r"E:\model\singlecell\singlecell_data")
    out_dir = Path(r"E:\model\singlecell\singlecell_data")
    batch_convert_folder(raw_dir, out_dir, exe_path, gzip=False)
