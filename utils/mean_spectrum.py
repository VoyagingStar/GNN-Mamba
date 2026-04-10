import xml.etree.ElementTree as ET
import numpy as np
import base64
import zlib
import json
from collections import defaultdict
from pathlib import Path
import pandas as pd 


def mz_to_bin_indices(mz_array, mz_bin_width):
    """
    使用四舍五入法将一组 m/z 映射到 bin 索引。
    bin 中心 = bin_index * mz_bin_width
    例如 bin_size=0.01，50.01 对应中心 50.01，范围 50.005–50.015。
    """
    return np.rint(mz_array / mz_bin_width).astype(np.int64)

def decode_binary_data(encoded_data, compression=None):
    """
    解码 mzML 中的二进制数据
    """
    # Base64解码
    decoded = base64.b64decode(encoded_data)
    
    # 如果是压缩数据，则解压缩
    if compression == 'zlib':
        decoded = zlib.decompress(decoded)
    
    # 将字节数据转换为64位浮点数数组
    data = np.frombuffer(decoded, dtype=np.float64)
    return data

def parse_mzml_file(file_path, max_spectra=5):
    """
    解析 mzML 文件并提取光谱信息
    """
    print(f"正在解析文件: {file_path}")
    
    # 使用 ElementTree 解析 XML
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # 获取命名空间
    namespace = {'mzml': 'http://psi.hupo.org/ms/mzml'}
    
    # 获取光谱列表
    spectrum_list = root.find('.//mzml:spectrumList', namespace)
    
    if spectrum_list is None:
        print("未找到光谱列表")
        return
    
    total_spectra = int(spectrum_list.get('count', 0))
    print(f"总光谱数: {total_spectra}")
    
    # 遍历前几个光谱
    spectra_parsed = 0
    for spectrum in spectrum_list.findall('mzml:spectrum', namespace):
        if spectra_parsed >= max_spectra:
            break
            
        spectrum_id = spectrum.get('id', 'Unknown')
        index = spectrum.get('index', 'Unknown')
        default_array_length = spectrum.get('defaultArrayLength', 'Unknown')
        
        print(f"\n=== 光谱 {spectra_parsed + 1} ===")
        print(f"ID: {spectrum_id}")
        print(f"索引: {index}")
        print(f"默认数组长度: {default_array_length}")
        
        # 查找二进制数据数组列表
        binary_data_list = spectrum.find('mzml:binaryDataArrayList', namespace)
        if binary_data_list is not None:
            arrays = []
            for binary_array in binary_data_list.findall('mzml:binaryDataArray', namespace):
                # 获取数组信息
                array_type = "Unknown"
                compression = None
                
                # 查找参数确定数组类型和压缩方式
                for cv_param in binary_array.findall('mzml:cvParam', namespace):
                    accession = cv_param.get('accession', '')
                    if accession == 'MS:1000514':
                        array_type = "m/z array"
                    elif accession == 'MS:1000515':
                        array_type = "intensity array"
                    elif accession == 'MS:1000574':
                        compression = "zlib"
                
                # 获取二进制数据
                binary = binary_array.find('mzml:binary', namespace)
                if binary is not None and binary.text:
                    try:
                        data = decode_binary_data(binary.text.strip(), compression)
                        arrays.append((array_type, data))
                        print(f"{array_type}: {len(data)} 个数据点")
                        
                        # 显示前几个数据点
                        if len(data) > 0:
                            print(f"  前5个{array_type}值: {data[:5]}")
                    except Exception as e:
                        print(f"解码 {array_type} 时出错: {e}")
            
            # 如果有m/z和强度数组，显示一些统计信息
            mz_array = None
            intensity_array = None
            for array_type, data in arrays:
                if "m/z" in array_type:
                    mz_array = data
                elif "intensity" in array_type:
                    intensity_array = data
            
            if mz_array is not None and intensity_array is not None:
                print(f"m/z 范围: {np.min(mz_array):.2f} - {np.max(mz_array):.2f}")
                print(f"强度范围: {np.min(intensity_array):.2f} - {np.max(intensity_array):.2f}")
                print(f"最高强度峰: m/z {mz_array[np.argmax(intensity_array)]:.4f}")
        
        spectra_parsed += 1

def extract_ms2_spectra_by_rt(mzml_file, rt_start, rt_end, precursor_mz=None, tolerance=0.01):
    """
    从mzML文件中提取指定保留时间范围内(可选指定母离子)的二级谱图
    
    返回:
        list[dict]，每个元素包含：
        {
            'scan_number': str,
            'index': str,
            'rt': float,
            'precursor_mz': float 或 None,
            'mz_array': np.ndarray,
            'intensity_array': np.ndarray
        }
    """
    print(f"正在解析文件: {mzml_file}")
    print(f"筛选条件: RT {rt_start}-{rt_end} min", end="")
    if precursor_mz is not None:
        print(f"，母离子 m/z ≈ {precursor_mz} (±{tolerance})")
    else:
        print()
        
    tree = ET.parse(mzml_file)
    root = tree.getroot()
    
    namespace = {
        'mzml': 'http://psi.hupo.org/ms/mzml'
    }
    
    spectrum_list = root.find('mzml:run/mzml:spectrumList', namespace)
    if spectrum_list is None:
        print("未找到光谱列表")
        return []
    
    ms2_spectra = []
    
    # 遍历所有光谱
    for spectrum in spectrum_list.findall('mzml:spectrum', namespace):
        ms_level = None
        scan_time = None
        precursor_ion_mz = None
        
        # 获取 ms level 和 RT（这里建议改用 .// 递归，保证能找到）
        for cv_param in spectrum.findall('.//mzml:cvParam', namespace):
            accession = cv_param.get('accession')
            if accession == 'MS:1000511':  # ms level
                ms_level = int(cv_param.get('value'))
            elif accession == 'MS:1000016':  # scan start time
                scan_time = float(cv_param.get('value'))
        
        # 只处理二级谱图
        if ms_level != 2:
            continue
        
        # 检查保留时间是否在指定范围内
        if scan_time is None or not (rt_start <= scan_time <= rt_end):
            continue
        
        # —— 总是尝试解析母离子 m/z —— #
        precursor_list = spectrum.find('mzml:precursorList', namespace)
        if precursor_list is not None:
            for precursor in precursor_list.findall('mzml:precursor', namespace):
                selected_ion_list = precursor.find('mzml:selectedIonList', namespace)
                if selected_ion_list is not None:
                    for selected_ion in selected_ion_list.findall('mzml:selectedIon', namespace):
                        for cv_param in selected_ion.findall('mzml:cvParam', namespace):
                            if cv_param.get('accession') == 'MS:1000744':  # selected ion m/z
                                precursor_ion_mz = float(cv_param.get('value'))
                                break
                        if precursor_ion_mz is not None:
                            break
                if precursor_ion_mz is not None:
                    break
        
        # 如果指定了目标母离子，则进一步筛选
        if precursor_mz is not None:
            if precursor_ion_mz is None or abs(precursor_ion_mz - precursor_mz) > tolerance:
                continue
        
        # 提取 m/z 和强度数组
        mz_array = None
        intensity_array = None
        
        binary_data_list = spectrum.find('mzml:binaryDataArrayList', namespace)
        if binary_data_list is not None:
            for binary_array in binary_data_list.findall('mzml:binaryDataArray', namespace):
                array_type = None
                compression = None
                
                # 确定数组类型和压缩方式
                for cv_param in binary_array.findall('mzml:cvParam', namespace):
                    accession = cv_param.get('accession')
                    if accession == 'MS:1000514':  # m/z array
                        array_type = 'mz'
                    elif accession == 'MS:1000515':  # intensity array
                        array_type = 'intensity'
                    elif accession == 'MS:1000574':  # zlib compression
                        compression = 'zlib'
                
                # 获取二进制数据
                binary = binary_array.find('mzml:binary', namespace)
                if binary is not None and binary.text:
                    try:
                        data = decode_binary_data(binary.text.strip(), compression)
                        if array_type == 'mz':
                            mz_array = data
                        elif array_type == 'intensity':
                            intensity_array = data
                    except Exception as e:
                        print(f"解码数据时出错: {e}")
                        continue
        
        # 创建谱图信息字典
        spectrum_info = {
            'scan_number': spectrum.get('id'),
            'index': spectrum.get('index'),
            'rt': scan_time,
            'precursor_mz': precursor_ion_mz,
            'mz_array': mz_array,
            'intensity_array': intensity_array
        }
        
        ms2_spectra.append(spectrum_info)
    
    print(f"找到 {len(ms2_spectra)} 个符合条件的二级谱图")
    return ms2_spectra

def build_avg_ms2_table(ms2_spectra, mz_bin_width=0.01):
    """
    将一组 MS2 谱图按母离子 m/z 分组，并对每个母离子的所有谱图做“分 bin + 平均”。

    规则：
      - 对每条谱图：用四舍五入法将 m/z 归到 bin，bin 内强度求和，得到这条谱图在每个 bin 的强度。
      - 对每个 bin：在所有谱图上取平均强度（谱图数为 N，就除以 N）。
      - 不做插值！

    返回 DataFrame 列：
      - precursor_mz  母离子 m/z
      - avg_rt        该母离子在该 RT 窗口内所有谱图 RT 的平均
      - n_spectra     参与平均的谱图个数
      - mz            bin 中心 m/z（k * mz_bin_width）
      - intensity     所有谱图在该 bin 的平均强度
    """
    from collections import defaultdict

    # 1. 按母离子 m/z 分组
    groups = defaultdict(list)
    for spec in ms2_spectra:
        pmz = spec.get("precursor_mz")
        mz_array = spec.get("mz_array")
        intensity_array = spec.get("intensity_array")
        rt = spec.get("rt")

        if pmz is None or mz_array is None or intensity_array is None:
            continue
        if len(mz_array) != len(intensity_array):
            continue

        groups[float(pmz)].append(
            {
                "mz": mz_array,
                "intensity": intensity_array,
                "rt": rt,
            }
        )

    if not groups:
        print("⚠️ 没有可用于平均的 MS2 谱图。")
        return pd.DataFrame(
            columns=["precursor_mz", "avg_rt", "n_spectra", "mz", "intensity"]
        )

    rows = []

    # 2. 对每个母离子做“所有谱图的平均”
    for pmz, specs in groups.items():
        n_spec = len(specs)
        if n_spec == 0:
            continue

        # 所有谱图的 RT 平均
        rts = [s["rt"] for s in specs if s["rt"] is not None]
        avg_rt = float(np.mean(rts)) if rts else None

        # bin 上的总强度（先按谱图内分 bin，再跨谱图相加）
        bin_intensity_sum = defaultdict(float)

        for s in specs:
            mz_array = s["mz"]
            inten_array = s["intensity"]

            # 本谱图内：bin -> 强度和
            spec_bin_sum = defaultdict(float)
            bin_indices = mz_to_bin_indices(mz_array, mz_bin_width)

            for bi, inten in zip(bin_indices, inten_array):
                spec_bin_sum[int(bi)] += float(inten)

            # 把本谱图的 bin 总强度累加到全局（谱图间）
            for bi, isum in spec_bin_sum.items():
                bin_intensity_sum[int(bi)] += isum

        if not bin_intensity_sum:
            continue

        # 3. 计算所有谱图平均后的 bin 强度，展开为 long format
        for bi in sorted(bin_intensity_sum.keys()):
            center_mz = bi * mz_bin_width
            avg_intensity = bin_intensity_sum[bi] / n_spec  # 所有谱图的平均

            if avg_intensity == 0:
                continue

            rows.append(
                {
                    "precursor_mz": pmz,
                    "avg_rt": avg_rt,
                    "n_spectra": n_spec,
                    "mz": center_mz,
                    "intensity": avg_intensity,
                }
            )

    df = pd.DataFrame(rows)
    print(
        f"构建平均谱图表完成：共 {len(df)} 行，涉及 {len(groups)} 个母离子，"
        f"bin 宽度 = {mz_bin_width}。"
    )
    return df
def save_avg_table(df, parquet_path, csv_path):
    """
    将平均谱图表同时保存为 Parquet（给程序复用）和 CSV（方便人眼查看）。
    """
    parquet_path = str(parquet_path)
    csv_path = str(csv_path)

    # 保存 Parquet（快速读写）
    df.to_parquet(parquet_path, index=False)
    print(f"已保存 Parquet 文件：{parquet_path}")

    # 保存 CSV（方便用 Excel / 文本查看）
    df.to_csv(csv_path, index=False)
    print(f"已保存 CSV 文件：{csv_path}")
def save_precursor_info(ms2_spectra, out_json_path):
    """
    将所有 MS2 谱图的母离子信息整理并保存为 JSON：
    {
        "<precursor_mz>": [
            {"scan_number": ..., "rt": ..., "index": ...},
            ...
        ],
        ...
    }
    """
    precursor_dict = defaultdict(list)
    for spec in ms2_spectra:
        prec_mz = spec.get('precursor_mz')
        if prec_mz is None:
            continue
        entry = {
            "scan_number": spec.get("scan_number"),
            "index": spec.get("index"),
            "rt": spec.get("rt")
        }
        # JSON 的 key 要用字符串
        key = f"{prec_mz:.6f}"
        precursor_dict[key].append(entry)
    
    out_json_path = Path(out_json_path)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(precursor_dict, f, ensure_ascii=False, indent=2)
    
    print(f"已保存母离子信息到: {out_json_path}")


def average_blank_groups_from_parquets(parquet_paths, out_parquet, out_csv, mz_bin_width=0.01):
    """
    打开多个 avg_ms2_*.parquet 文件，将它们合并后，
    在 (precursor_mz, mz_bin) 维度上，对所有文件的 bin 强度相加，
    再除以【所有空白文件总数】，得到“按所有文件平均”的空白谱图。

    规则：
      1) 读入每个 avg_ms2_*.parquet（每个 parquet 代表一个空白组 / 一个文件）；
      2) 用四舍五入法将 m/z 归到 bin：
           mz_bin = round(mz / mz_bin_width)
           mz_center = mz_bin * mz_bin_width
      3) 在每个文件内部，对 (precursor_mz, mz_bin) 汇总，得到该文件在该 bin 的 intensity；
      4) 在 (precursor_mz, mz_bin) 上，把所有文件的 bin intensity 相加，
         再除以【总文件数 n_all_files】：
             intensity_mean = intensity_sum / n_all_files
         没有出现该 bin 的文件，相当于该 bin 强度为 0。
      5) 不做插值。
    """

    # 1) 读取并拼接
    dfs = []
    for p in parquet_paths:
        p = Path(p)
        if not p.exists():
            print(f"找不到文件: {p}")
            continue
        df = pd.read_parquet(p)
        df = df.copy()
        df["source_file"] = p.name  # 记录来自哪个空白文件
        dfs.append(df)
        print(f"已读取: {p}, 行数 = {len(df)}")

    if not dfs:
        print("没有成功读取任何 parquet 文件，请检查路径。")
        return

    # 总文件数（只统计实际成功读取的）
    n_all_files = len(dfs)
    print(f"参与空白平均的文件总数: {n_all_files}")

    all_df = pd.concat(dfs, ignore_index=True)
    print(f"合并后总行数: {len(all_df)}")

    # 2) 计算 m/z 所在的 bin（四舍五入）
    all_df["mz_bin"] = np.rint(all_df["mz"] / mz_bin_width).astype(np.int64)
    all_df["mz_center"] = all_df["mz_bin"] * mz_bin_width

    # 3) 在每个文件内部先按 (precursor_mz, source_file, mz_bin) 汇总
    file_level = (
        all_df
        .groupby(["precursor_mz", "source_file", "mz_bin"], as_index=False)
        .agg(
            mz=("mz_center", "mean"),            # 该文件在该 bin 的中心 m/z
            intensity=("intensity", "mean"),     # 该文件在该 bin 的强度（如果有多行就再平均一下）
            # avg_rt=("avg_rt", "mean"),           # 该文件在该 bin 的平均 RT
            n_spectra_total=("n_spectra", "sum") # 该文件该 bin 内的谱图总数
        )
    )

    # 4) 跨文件：在 (precursor_mz, mz_bin) 上做“相加后除以【总文件数】”
    grouped = (
        file_level
        .groupby(["precursor_mz", "mz_bin"], as_index=False)
        .agg(
            mz=("mz", "mean"),                       # 各文件的 mz_center 再平均（基本一样）
            intensity_sum=("intensity", "sum"),      # 该 bin 上所有文件强度相加
            # avg_rt=("avg_rt", "mean"),               # 各文件 avg_rt 的平均
            n_spectra_total=("n_spectra_total", "sum"),
            n_files_present=("source_file", "nunique"),  # 实际有该 bin 的文件数（仅供参考）
        )
    )

    # ★ 关键：用所有文件总数 n_all_files 做平均
    grouped["intensity"] = grouped["intensity_sum"] / float(n_all_files)
    grouped = grouped.drop(columns=["intensity_sum"])

    # 排序、整理
    grouped = grouped.sort_values(["precursor_mz", "mz"]).reset_index(drop=True)

    print(
        f"空白组平均表行数: {len(grouped)}，"
        f"涉及 {grouped['precursor_mz'].nunique()} 个母离子；"
        f"每个 bin 强度 = 所有文件该 bin 强度相加 / 总文件数 {n_all_files}。"
    )

    # 5) 保存
    out_parquet = Path(out_parquet)
    out_csv = Path(out_csv)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    grouped.to_parquet(out_parquet, index=False)
    print(f"已保存空白组平均结果 Parquet: {out_parquet}")

    grouped.to_csv(out_csv, index=False)
    print(f"已保存空白组平均结果 CSV: {out_csv}")

def display_spectrum_info(spectra):
    """
    显示谱图信息
    """
    for i, spectrum in enumerate(spectra):
        print(f"\n=== 谱图 {i+1} ===")
        print(f"扫描号: {spectrum['scan_number']}")
        print(f"索引: {spectrum['index']}")
        print(f"保留时间: {spectrum['rt']:.4f} min")
        if spectrum['precursor_mz']:
            print(f"母离子 m/z: {spectrum['precursor_mz']:.4f}")
        
        if spectrum['mz_array'] is not None and spectrum['intensity_array'] is not None:
            print(f"数据点数: {len(spectrum['mz_array'])}")
            if len(spectrum['mz_array']) > 0:
                # 显示前5个数据点
                print("前5个数据点:")
                for j in range(min(5, len(spectrum['mz_array']))):
                    print(f"  m/z: {spectrum['mz_array'][j]:.4f}, 强度: {spectrum['intensity_array'][j]:.2f}")
                
                # 显示m/z范围和最大强度峰
                mz_min = np.min(spectrum['mz_array'])
                mz_max = np.max(spectrum['mz_array'])
                max_intensity_idx = np.argmax(spectrum['intensity_array'])
                max_mz = spectrum['mz_array'][max_intensity_idx]
                max_intensity = spectrum['intensity_array'][max_intensity_idx]
                
                print(f"m/z 范围: {mz_min:.2f} - {mz_max:.2f}")
                print(f"最强峰: m/z {max_mz:.4f}, 强度 {max_intensity:.2f}")

def list_precursor_ions(mzml_file):
    """
    列出 mzML 文件中的所有母离子信息
    """
    print(f"正在解析文件: {mzml_file}")
    
    # 解析XML
    tree = ET.parse(mzml_file)
    root = tree.getroot()
    
    # 获取命名空间
    namespace = {'mzml': 'http://psi.hupo.org/ms/mzml'}
    
    # 获取光谱列表
    spectrum_list = root.find('.//mzml:spectrumList', namespace)
    
    if spectrum_list is None:
        print("未找到光谱列表")
        return []
    
    precursor_ions = set()  # 使用集合避免重复
    
    # 遍历所有光谱
    for spectrum in spectrum_list.findall('mzml:spectrum', namespace):
        # 检查是否为二级谱图
        ms_level = None
        
        # 获取质谱级别
        for cv_param in spectrum.findall('.//mzml:cvParam', namespace):
            if cv_param.get('accession') == 'MS:1000511':  # ms level
                ms_level = int(cv_param.get('value'))
                break
        
        # 只处理二级谱图
        if ms_level != 2:
            continue
            
        # 获取前体离子信息
        precursor_list = spectrum.find('mzml:precursorList', namespace)
        if precursor_list is not None:
            for precursor in precursor_list.findall('mzml:precursor', namespace):
                selected_ion_list = precursor.find('mzml:selectedIonList', namespace)
                if selected_ion_list is not None:
                    for selected_ion in selected_ion_list.findall('mzml:selectedIon', namespace):
                        for cv_param in selected_ion.findall('mzml:cvParam', namespace):
                            if cv_param.get('accession') == 'MS:1000744':  # selected ion m/z
                                precursor_mz = float(cv_param.get('value'))
                                precursor_ions.add(round(precursor_mz, 4))  # 保留4位小数避免精度问题
    
    # 转换为排序后的列表
    sorted_precursors = sorted(list(precursor_ions))
    
    print(f"总共找到 {len(sorted_precursors)} 个不同的母离子:")
    for i, mz in enumerate(sorted_precursors):
        print(f"{i+1:4d}. {mz:.4f}")
    
    return sorted_precursors

def list_precursor_ions_with_rts(mzml_file):
    """
    列出 mzML 文件中的所有母离子及其保留时间信息
    """
    print(f"正在解析文件: {mzml_file}")
    
    # 解析XML
    tree = ET.parse(mzml_file)
    root = tree.getroot()
    
    # 获取命名空间
    namespace = {'mzml': 'http://psi.hupo.org/ms/mzml'}
    
    # 获取光谱列表
    spectrum_list = root.find('.//mzml:spectrumList', namespace)
    
    if spectrum_list is None:
        print("未找到光谱列表")
        return []
    
    precursor_info = []  # 存储母离子信息
    
    # 遍历所有光谱
    for spectrum in spectrum_list.findall('mzml:spectrum', namespace):
        # 检查是否为二级谱图
        ms_level = None
        scan_time = None
        
        # 获取质谱级别和保留时间
        for cv_param in spectrum.findall('.//mzml:cvParam', namespace):
            accession = cv_param.get('accession')
            if accession == 'MS:1000511':  # ms level
                ms_level = int(cv_param.get('value'))
            elif accession == 'MS:1000016':  # scan start time
                scan_time = float(cv_param.get('value'))
        
        # 只处理二级谱图
        if ms_level != 2:
            continue
            
        # 获取前体离子信息
        precursor_list = spectrum.find('mzml:precursorList', namespace)
        if precursor_list is not None:
            for precursor in precursor_list.findall('mzml:precursor', namespace):
                selected_ion_list = precursor.find('mzml:selectedIonList', namespace)
                if selected_ion_list is not None:
                    for selected_ion in selected_ion_list.findall('mzml:selectedIon', namespace):
                        for cv_param in selected_ion.findall('mzml:cvParam', namespace):
                            if cv_param.get('accession') == 'MS:1000744':  # selected ion m/z
                                precursor_mz = float(cv_param.get('value'))
                                precursor_info.append({
                                    'mz': round(precursor_mz, 4),
                                    'rt': scan_time,
                                    'scan_id': spectrum.get('id')
                                })
    
    # 按照m/z值排序
    precursor_info.sort(key=lambda x: x['mz'])
    
    print(f"总共找到 {len(precursor_info)} 个母离子记录:")
    print(f"{'序号':<6} {'母离子 m/z':<12} {'保留时间(分钟)':<15} {'扫描ID'}")
    print("-" * 50)
    for i, info in enumerate(precursor_info):
        print(f"{i+1:<6} {info['mz']:<12.4f} {info['rt']:<15.4f} {info['scan_id']}")
    
    return precursor_info
def get_file_info(file_path):
    """
    获取 mzML 文件基本信息
    """
    print("=== mzML 文件基本信息 ===")
    
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    namespace = {'mzml': 'http://psi.hupo.org/ms/mzml'}
    
    # 获取文件描述信息
    file_description = root.find('.//mzml:fileDescription', namespace)
    if file_description is not None:
        file_content = file_description.find('mzml:fileContent', namespace)
        if file_content is not None:
            print("文件内容类型:")
            for param in file_content.findall('mzml:cvParam', namespace):
                print(f"  - {param.get('name', '')}")
    
    # 获取仪器信息
    instrument_config_list = root.find('.//mzml:instrumentConfigurationList', namespace)
    if instrument_config_list is not None:
        print("\n仪器配置:")
        for instrument in instrument_config_list.findall('mzml:instrumentConfiguration', namespace):
            instrument_id = instrument.get('id', '')
            print(f"  配置 ID: {instrument_id}")
            
            # 查找分析器信息
            component_list = instrument.find('mzml:componentList', namespace)
            if component_list is not None:
                for analyzer in component_list.findall('mzml:analyzer', namespace):
                    for param in analyzer.findall('mzml:cvParam', namespace):
                        if param.get('accession') == 'MS:1000484':  # orbitrap
                            print(f"  分析器类型: {param.get('name', '')}")

def rt_tag(rt: float, ndigits: int = 2) -> str:
    # 11.76 -> "11p76"
    s = f"{rt:.{ndigits}f}"
    return s.replace(".", "p")

if __name__ == "__main__":

    
    # 输出文件信息
    # get_file_info(mzml_path)
    # parse_mzml_file(mzml_path)
    mzml_path = r"E:\model\singlecell\singlecell_data\27_POS_HGC_DIA_2DA_AGC250-IT10ms_4bk_30Cells.mzML"

    # # 1) 提取 RT 在 2.50–3.13 分钟之间的所有 MS2 谱图
    # rt_start = 49.43
    # rt_end = 50.06
    # print("=" * 50)
    # print(f"提取 RT 在 {rt_start}-{rt_end} min 之间的所有二级谱图")
    # ms2_window = extract_ms2_spectra_by_rt(
    #     mzml_path,
    #     rt_start,
    #     rt_end,
    #     precursor_mz=None,  # 这里先不指定某个母离子，全部拿来分组
    #     tolerance=0.01,
    # )

    
    # # 2) 构建“按母离子平均”的大表
    # df_avg = build_avg_ms2_table(ms2_window)

    # if df_avg.empty:
    #     print("在该 RT 范围内没有可用的 MS2，检查一下 RT 范围或提取代码。")
    # else:
    #     # 3. 保存为一个 Parquet + 一个 CSV
    #     base_dir = Path(r"E:\model\singlecell\singlecell_data\HGC")  # 你输出目录
    #     tag = f"{rt_tag(rt_start, 2)}_{rt_tag(rt_end, 2)}"

    #     parquet_path = base_dir / f"avg_ms2_rt{tag}.parquet"
    #     csv_path = base_dir / f"avg_ms2_rt{tag}.csv"

    #     save_avg_table(df_avg, parquet_path, csv_path)
    #     print("全部完成。")


    # ====== 这里专门做四个空白组的平均 ======
    parquet_files = [
        r"E:\model\singlecell\singlecell_data\HGC\bk4\avg_ms2_rt0p36_0p99.parquet",
        r"E:\model\singlecell\singlecell_data\HGC\bk4\avg_ms2_rt1p66_2p30.parquet",
        r"E:\model\singlecell\singlecell_data\HGC\bk4\avg_ms2_rt2p97_3p61.parquet",
        r"E:\model\singlecell\singlecell_data\HGC\bk4\avg_ms2_rt4p20_4p84.parquet",
    ]

    out_parquet = r"E:\model\singlecell\singlecell_data\HGC\bk4\avg_ms2_blank_mean.parquet"
    out_csv = r"E:\model\singlecell\singlecell_data\HGC\bk4\avg_ms2_blank_mean.csv"

    average_blank_groups_from_parquets(parquet_files, out_parquet, out_csv)

    print("全部处理完成。")

