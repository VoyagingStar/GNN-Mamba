from pathlib import Path
import logging
import sys
from tqdm import tqdm
from pathos import multiprocessing as mp
import itertools
from rdkit import Chem
import numpy as np
from rdkit.Chem import Atom
import torch
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import re
def setup_logger(save_dir, log_name="output.log", debug=False, custom_label=""):
    """Create output directory"""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    log_file = save_dir / log_name

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)

    file_handler = logging.FileHandler(log_file)

    file_handler.setLevel(level)

    # Define basic logger
    logging.basicConfig(
        level=level,
        format=custom_label + "%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            stream_handler,
            file_handler,
        ],
    )

    # configure logging at the root level of lightning
    # logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    # configure logging on module level, redirect to file
    logger = logging.getLogger("pytorch_lightning.core")
    logger.addHandler(logging.FileHandler(log_file))


def chunked_parallel(
    input_list,
    function,
    chunks=100,
    max_cpu=16,
    output_func=None,
    task_name="",
    **kwargs,
):
    """chunked_parallel.

    Args:
        input_list : list of objects to apply function
        function : Callable with 1 input and returning a single value
        chunks: number of chunks
        max_cpu: Max num cpus
        output_func: an output function that writes function output to the disk
    """
    # Adding it here fixes somessetting disrupted elsewhere

    def batch_func(list_inputs):
        outputs = []
        for i in list_inputs:
            outputs.append(function(i))
        return outputs

    list_len = len(input_list)
    if list_len == 0:
        raise ValueError('Empty list to process!')
    num_chunks = min(list_len, chunks)
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i : i + step_size] for i in range(0, len(input_list), step_size)
    ]

    
    cpus = min(mp.cpu_count(), max_cpu)
    with mp.ProcessPool(processes=cpus, **kwargs) as pool:
        iter_outputs = tqdm(pool.imap(batch_func, chunked_list), total=len(chunked_list), desc=task_name)
        if output_func is None:
            list_outputs = list(iter_outputs)
            # Unroll
            full_output = [j for i in list_outputs for j in i]
            return full_output
        else:
            output_func(itertools.chain.from_iterable(iter_outputs))

def inchikey_from_smiles(smi: str) -> str:
    """inchikey_from_smiles.

    Args:
        smi (str): smi

    Returns:
        str:
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchiKey(mol)
    
P_TBL = Chem.GetPeriodicTable()

ROUND_FACTOR = 4

ELECTRON_MASS = 0.00054858
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C",
    "N",
    "P",
    "O",
    "S",
    "Si",
    "I",
    "H",
    "Cl",
    "F",
    "Br",
    "B",
    "Se",
    "Fe",
    "Co",
    "As",
    "Na",
    "K",
]

ELEMENT_TO_GROUP = {
    "C": 4,  # group 5
    "N": 3,  # group 4
    "P": 3,
    "O": 5,  # group 6
    "S": 5,
    "Si": 4,
    "I": 6,  # group 7 / halogens
    "H": 0,
    "Cl": 6,
    "F": 6,
    "Br": 6,
    "B": 2,  # group 3
    "Se": 5,
    "Fe": 7,  # transition metals
    "Co": 7,
    "As": 3,
    "Na": 1,  # alkali metals
    "K": 1,
}
ELEMENT_GROUP_DIM = len(set(ELEMENT_TO_GROUP.values()))
ELEMENT_GROUP_VECTORS = np.eye(ELEMENT_GROUP_DIM)

# Set the exact molecular weight?
# Use this to define an element priority queue
VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]
CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)


ATOM_NUM_TO_ONEHOT = torch.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM))

# Convert to onehot
ATOM_NUM_TO_ONEHOT[VALID_ATOM_NUM, torch.arange(CHEM_ELEMENT_NUM)] = 1

# Use Monoisotopic
# VALID_MASSES = np.array([Atom(i).GetMass() for i in VALID_ELEMENTS])
VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS))
ELEMENT_VECTORS_MASS = np.hstack([ELEMENT_VECTORS, CHEM_MASSES])
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))

ELEMENT_DIM_MASS = len(ELEMENT_VECTORS_MASS[0])
ELEMENT_DIM = len(ELEMENT_VECTORS[0])

COLLISION_PE_DIM = 64
COLLISION_PE_SCALAR = 10000

SIM_PE_DIM = 64
SIM_PE_SCALAR = 10

GRAPHTYPE_LEN = 32
# Reasonable normalization vector for elements
# Estimated by max counts (+ 1 when zero)
NORM_VEC_MASS = np.array(
    [81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 1, 1471]
)

NORM_VEC = np.array([81, 19, 6, 34, 6, 6, 6, 158, 10, 17, 3, 1, 2, 1, 1, 2, 1, 1])
MAX_ATOM_CT = 160


# For helping binary conversions
BINARY_BITS = 8

# Assume 64 is the highest repeat of any 1 atom
MAX_ELEMENT_NUM = 64

# Hydrogen featurizer
MAX_H = 6

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))
element_to_position_mass = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS_MASS))
element_to_group = {k: ELEMENT_GROUP_VECTORS[v] for k, v in ELEMENT_TO_GROUP.items()}

ion2mass = {
    # positive mode
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H-H2O]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M+NH4]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M]+": 0 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    "[M+H-2H2O]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    # negative mode
    "[M-H]-": -ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M+Cl]-": ELEMENT_TO_MASS["Cl"] + ELECTRON_MASS,
    "[M-H2O-H]-": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] * 3 + ELECTRON_MASS,
    "[M-H-H2O]-": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] * 3 + ELECTRON_MASS,
    "[M-H-CO2]-": -ELEMENT_TO_MASS["C"] - ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
}

ion2onehot_pos = {
    "[M+H]+": 0,
    "[M+Na]+": 1,
    "[M+K]+": 2,
    "[M-H2O+H]+": 3,
    "[M+H-H2O]+": 3,
    "[M+H3N+H]+": 4,
    "[M+NH4]+": 4,
    "[M]+": 5,
    "[M-H4O2+H]+": 6,
    "[M+H-2H2O]+": 6,
    "[M-H]-": 7,
    "[M+Cl]-": 8,
    "[M-H2O-H]-": 9,
    "[M-H-H2O]-": 9,
    "[M-H-CO2]-": 10,
}

def rdbe_filter(cross_prod):
    """rdbe_filter.

    Args:
        cross_prod:
    """
    # Filter
    pos_els = ["C", "C", "N", "P"]
    neg_els = ["H", "Cl", "Br", "I", "F"]

    # Apply rdbe filter
    # RDBE = 1 + 0.5 * (2#C − #H +#N+#P−#Cl−#Br−#I−#F)
    rdbe_total = np.zeros(cross_prod.shape[0])
    for pos_el in pos_els:
        rdbe_total += cross_prod[:, element_to_ind[pos_el]]

    for neg_el in neg_els:
        rdbe_total -= cross_prod[:, element_to_ind[neg_el]]

    # Manage
    rdbe_total = 1 + 0.5 * rdbe_total
    filter_inds = np.argwhere(rdbe_total >= 0).flatten()
    return filter_inds

def vec_to_formula(form_vec):
    """vec_to_formula."""
    build_str = ""
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str

def get_morgan_fp(mol: Chem.Mol, nbits: int = 2048, radius=3, isbool = False) -> np.ndarray:
    """get_morgan_fp."""

    if mol is None:
        return None

    curr_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    fingerprint = np.zeros((0,), dtype=np.uint8) if not isbool else  np.zeros((0,), dtype=np.bool_)
    DataStructs.ConvertToNumpyArray(curr_fp, fingerprint)
    return fingerprint

def uncharged_formula(mol, mol_type="mol") -> str:
    """Compute uncharged formula"""
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "inchi":
        mol = Chem.MolFromInchi(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]
def form_from_smi(smi: str) -> str:
    """form_from_smi.

    Args:
        smi (str): smi

    Return:
        str
    """
    return uncharged_formula(smi, mol_type="smiles")

def formula_to_dense(chem_formula: str) -> np.ndarray:
    """formula_to_dense.

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position))
    else:
        dense_vec = np.vstack(total_onehot).sum(0)

    return dense_vec

def formula_mass(chem_formula: str) -> float:
    """get formula mass"""
    mass = 0
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        mass += ELEMENT_TO_MASS[chem_symbol] * num
    return mass
