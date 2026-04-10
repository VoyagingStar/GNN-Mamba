import os
import re
import yaml
import torch
import pandas as pd
import numpy as np

from rdkit import Chem
from torch_geometric.data import Batch

from dataset.mol_dataset import MoleculeDataset
from dataset.data_utils import mol_to_graph_data_obj_pos, np_one_hot
from GNN_mamba import GNNMambaFusion


def load_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    data_d = cfg.get("data_d", {})
    run_d = cfg.get("run_d", {})
    return data_d, run_d


def build_dataset_for_meta(data_d):
    """
    仅用于复用训练时的：
      - inst_type / adduct 映射
      - CE 预处理（mean/std 或其他策略）
      - spec_dim, meta_dim
    """
    spec_df_name = data_d.get("spec_df", "spec_df_massspecgym.pkl")
    mol_df_name = data_d.get("mol_df", "mol_df_massspecgym.pkl")

    ds = MoleculeDataset(
        spec_df=spec_df_name,
        mol_df=mol_df_name,
        **data_d
    )
    return ds


def build_model(ds: MoleculeDataset, run_d: dict, device: torch.device):
    """
    和训练阶段保持一致的模型结构 + 加载 best.pt
    """
    spec_dim, meta_dim = ds.get_data_dims()

    model = GNNMambaFusion(
        hidden=128,
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
    missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
    print(f"[INFO] Loaded checkpoint from {ckpt_path}")
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")

    model.eval()
    return model, spec_dim


def build_spec_meta_from_row(row, ds: MoleculeDataset) -> torch.Tensor:
    """
    从 CSV 行构造 spec_meta：[1, meta_dim]
    包含：CE 向量 + 仪器类型 one-hot + 加合物 one-hot
    """
    inst_type = str(row["instrument_type"])
    adduct = str(row["adduct"])
    ce_val = row["collision_energy"]

    # 1) 碰撞能量向量：复用数据集里的 ce_func
    ce_vec = ds.ce_func(ce_val)  # Tensor

    # 2) 仪器类型 one-hot
    if inst_type not in ds.inst_type2id:
        raise ValueError(f"Unknown instrument_type: {inst_type}")
    inst_idx = ds.inst_type2id[inst_type]
    inst_onehot = torch.as_tensor(
        np_one_hot(inst_idx, num_classes=ds.num_inst_type),
        dtype=torch.float32
    )

    # 3) 加合物 one-hot
    if adduct not in ds.adduct2id:
        raise ValueError(f"Unknown adduct: {adduct}")
    adduct_idx = ds.adduct2id[adduct]
    adduct_onehot = torch.as_tensor(
        np_one_hot(adduct_idx, num_classes=ds.num_adduct),
        dtype=torch.float32
    )

    spec_meta = torch.cat([ce_vec, inst_onehot, adduct_onehot], dim=0).unsqueeze(0)
    return spec_meta  # [1, meta_dim]


def build_graph_from_smiles(smiles: str, idx: int):
    """
    从 SMILES 构造 PyG Data 对象
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    data = mol_to_graph_data_obj_pos(mol, idx)
    return data


def _sanitize_filename(name: str) -> str:
    """简单清理文件名中的非法字符"""
    name = str(name)
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)  # Windows 不允许的字符
    if not name:
        name = "unnamed"
    return name


def run_inference_on_csv(
    input_csv: str,
    output_dir: str,
    config_path: str = "GNN-Mamba.yaml",
    device_str: str = None,
    id_col: str = None,
):
    """
    对 CSV 中的每一行分别进行预测，并各自保存为一个 csv 文件。

    Parameters
    ----------
    input_csv : str
        输入 CSV 路径，需包含列：
        ["smiles", "instrument_type", "collision_energy", "adduct"]
        也可以额外有 name / id / 你指定的 id_col，用来给输出命名。
    output_dir : str
        输出目录，每一行对应一个 csv 文件。
    config_path : str, optional
        配置文件路径，默认 "GNN-Mamba.yaml"
    device_str : str, optional
        "cuda" 或 "cpu"；None 时自动选择
    id_col : str, optional
        用这一列的值为当前行生成文件名；如果为 None，
        则优先使用 name 列，其次 id 列，再其次用行号 row{i}.
    """
    # 读取 config
    data_d, run_d = load_config(config_path)

    # m/z 的 bin 宽度，与训练/绘图阶段保持一致
    mz_bin_res = float(data_d["mz_bin_res"])

    # 设备
    if device_str is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # 构造数据集对象，用于映射和 dim 信息
    ds = build_dataset_for_meta(data_d)
    model, spec_dim = build_model(ds, run_d, device)

    # 读取输入 CSV
    df = pd.read_csv(input_csv)
    required_cols = ["smiles", "instrument_type", "collision_energy", "adduct"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV must contain column: {col}")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Loaded {len(df)} rows from {input_csv}")
    print(f"[INFO] Output directory: {output_dir}")

    # 准备 m/z 轴（对所有样本都一样）
    mz_axis = np.arange(spec_dim, dtype=np.float32) * mz_bin_res

    with torch.no_grad():
        for idx, row in df.iterrows():
            smiles = str(row["smiles"])

            # 决定当前行的输出文件名
            if id_col is not None and id_col in df.columns:
                base_name = row[id_col]
            elif "name" in df.columns:
                base_name = row["name"]
            elif "id" in df.columns:
                base_name = row["id"]
            else:
                base_name = f"row{idx}"
            base_name = _sanitize_filename(base_name)
            out_path = os.path.join(output_dir, f"{base_name}.csv")

            try:
                # 1) 分子图
                graph_data = build_graph_from_smiles(smiles, idx)

                # 2) spec_meta
                spec_meta = build_spec_meta_from_row(row, ds)  # [1, meta_dim]

                # 3) 打包成 Batch
                graph_data.spec_meta = spec_meta
                batch = Batch.from_data_list([graph_data]).to(device)

                # 4) 模型前向
                pred = model(batch)          # [1, spec_dim]
                pred = torch.relu(pred)      # 强度非负
                max_val = pred.amax(dim=-1, keepdim=True).clamp_min(1e-12)
                pred = pred / max_val

                pred_np = pred.squeeze(0).cpu().numpy()

                # 5) 当前行结果保存为一个 csv（两列：mz, intensity）
                out_df = pd.DataFrame({
                    "mz": mz_axis.astype(float),
                    "intensity": pred_np.astype(float),
                })
                out_df.to_csv(out_path, index=False)
                print(f"[INFO] Row {idx} saved to {out_path}")

            except Exception as e:
                print(f"[WARN] row {idx} failed: {e}")
                continue


if __name__ == "__main__":

    example_input = "input_examples.csv"
    example_output_dir = "input_examples"
    example_config = "GNN-Mamba.yaml"

    run_inference_on_csv(
        input_csv=example_input,
        output_dir=example_output_dir,
        config_path=example_config,
        device_str="cuda", 
        id_col=None,       
    )
