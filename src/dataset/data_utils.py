# borrowed from https://github.com/Roestlab/massformer/tree/master/preproc_scripts/prepare_data.py
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from torch_geometric.data import Data
import os, random
import numpy as np
import torch
from collections import defaultdict

EPS = np.finfo(np.float32).eps

# allowable multiple choice node and edge features
allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}

def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except BaseException:
        return len(l) - 1
def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
        safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
        allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
        safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
        safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
        safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
        safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
        safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
        allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
        allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
    ]
    return atom_feature

def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
    ]))

def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
        safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
        allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
        allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
    ]
    return bond_feature

def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
    ]))

def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx,
     chirality_idx,
     degree_idx,
     formal_charge_idx,
     num_h_idx,
     number_radical_e_idx,
     hybridization_idx,
     is_aromatic_idx,
     is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]}

    return feature_dict

def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx,
     bond_stereo_idx,
     is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
        'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]}

    return feature_dict

def mol2graph(mol):
    """
    Converts rdkit mol to graph Data object
    :input:  rdkit mol
    :return: graph object
    """

    if mol is None or not isinstance(mol, Chem.Mol):
        raise ValueError(f"Expected Chem.Mol object, but got {type(mol)}")

   # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype=np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2,
        # num_edges]
        edge_index = np.array(edges_list, dtype=np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges,
        # num_edge_features]
        edge_attr = np.array(edge_features_list, dtype=np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph

def graph2data(graph):
    """taken from process() in https://github.com/snap-stanford/ogb/blob/master/ogb/lsc/pcqm4mv2_pyg.py"""
    data = Data()
    assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
    assert len(graph["node_feat"]) == graph["num_nodes"]
    data.__num_nodes__ = int(graph["num_nodes"])
    data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)
    data.x = convert_to_single_emb(data.x)
    data.y = torch.Tensor([-1])  # dummy
    if data.edge_attr.shape[0] > 0:
        data.edge_attr = convert_to_single_emb(data.edge_attr)
    return data

def mol_to_graph_data_obj_pos(mol, idx):
    graph = mol2graph(mol)
    data = graph2data(graph)
    data.idx = idx
    # item = preprocess_item(data)
    return data

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def preprocess_item(item):

    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)
    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    item.x = x
    return item

def bin_func(mzs, ints, mz_max, mz_bin_res, ints_thresh, return_index):

    mzs = np.array(mzs, dtype=np.float32)
    bins = np.arange(
        mz_bin_res,
        mz_max +
        mz_bin_res,
        step=mz_bin_res).astype(
        np.float32)
    bin_idx = np.searchsorted(bins, mzs, side="right")
    if return_index:
        return bin_idx.tolist()
    else:
        ints = np.array(ints, dtype=np.float32)
        bin_spec = np.zeros([len(bins)], dtype=np.float32)
        for i in range(len(mzs)):
            if bin_idx[i] < len(bin_spec) and ints[i] >= ints_thresh:
                bin_spec[bin_idx[i]] = max(bin_spec[bin_idx[i]], ints[i])
        if np.all(bin_spec == 0.):
            print("> warning: bin_spec is all zeros!")
            bin_spec[-1] = 1.
        return bin_spec
    
def np_one_hot(input, num_classes=None):
    """ numpy wrapper for one_hot """

    th_input = torch.as_tensor(input, device="cpu")
    th_oh = torch.nn.functional.one_hot(th_input, num_classes=num_classes)
    oh = th_oh.numpy()
    return oh

def set_seed(seed: int = 666, deterministic: bool = True, verbose: bool = True):
    """
    Global random seed and deterministic settings (PyTorch/Numpy/Python).
    When deterministic=True, deterministic operations are enabled to the extent possible.
    """
    # Python / Numpy / Torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


    if verbose:
        print(f"[seed] seed={seed}, deterministic={deterministic}")

def seed_worker(worker_id: int, base_seed: int):
    """
    DataLoader worker initialization function factory.
    Each worker uses a distinct but reproducible sub-seed.
    """
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def make_loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def _human_count(n: int) -> str:
    if n < 1_000: return f"{n}"
    if n < 1_000_000: return f"{n/1e3:.2f} K"
    if n < 1_000_000_000: return f"{n/1e6:.2f} M"
    return f"{n/1e9:.2f} B"

def _human_bytes(n_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    x = float(n_bytes)
    while x >= 1024 and i < len(units) - 1:
        x /= 1024.0
        i += 1
    return f"{x:.2f} {units[i]}"

def report_model_parameters(model, logger=None, top_k: int = 12):
    """
    Statistics and printing of model parameter information; 
    displays the top_k aggregated by top-level submodules.
    Returns a dictionary for subsequent recording.
    """
    total = 0
    trainable = 0
    frozen = 0
    bytes_total = 0

    # Aggregated by top-level submodules
    by_block = defaultdict(int)

    for name, p in model.named_parameters(recurse=True):
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
        else:
            frozen += n

        try:
            bytes_total += n * p.element_size()
        except Exception:
            pass

        block = name.split('.', 1)[0]  
        by_block[block] += n

    lines = []
    lines.append("==== Model Parameters ====")
    lines.append(f"Total:      {_human_count(total)}  ({total:,})")
    lines.append(f"Trainable:  {_human_count(trainable)}  ({trainable:,})")
    lines.append(f"Frozen:     {_human_count(frozen)}  ({frozen:,})")
    if bytes_total:
        lines.append(f"Param Memory (est.): {_human_bytes(bytes_total)}")

    # Top-K blocks
    if by_block:
        lines.append("\nTop blocks by params:")
        for k, n in sorted(by_block.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            lines.append(f"  - {k:<16} {_human_count(n):>8}  ({n:,})")

    msg = "\n".join(lines)
    if logger is not None:
        try:
            logger.info("\n" + msg)
        except Exception:
            print(msg)
    else:
        print(msg)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "bytes_total": bytes_total,
        "by_block": dict(sorted(by_block.items(), key=lambda x: x[1], reverse=True)),
    }