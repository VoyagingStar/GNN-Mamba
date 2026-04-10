# Borrowed from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py

import os
import torch.utils.data as th_data
import torch as th
from rdkit.Chem import AllChem
import numpy as np
from torch_geometric.data import Data
import pandas as pd
import tqdm
import logging
import torch_geometric
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from .data_utils import allowable_features, mol_to_graph_data_obj_pos, bin_func, EPS, np_one_hot
from torch_geometric.data import Batch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainSubset(th_data.Subset):

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.indices[idx])

class MoleculeDataset(th_data.Dataset):
    def __init__(
            self, 
            spec_df, 
            mol_df, 
            **kwargs):
        """
        Adapted from qm9.py. Disabled the download functionality
        
        """

        for k, v in kwargs.items():
            setattr(self, k, v)
        assert os.path.isdir(self.data_dir), self.data_dir
        self.spec_df = pd.read_pickle(
            os.path.join(self.data_dir, spec_df))
        self.mol_df = pd.read_pickle(os.path.join(self.data_dir, mol_df))
        self.mol_df = self.mol_df.set_index(
            "mol_id", drop=False).sort_index().rename_axis(None)
        
        self._setup_spec_metadata_dicts()

        self.spec_df["collision_energy"] = pd.to_numeric(self.spec_df["collision_energy"], errors="coerce")
        col_energy_series = self.spec_df["collision_energy"].values
        self.mean_ce = float(np.nanmean(col_energy_series))
        self.std_ce  = float(np.nanstd(col_energy_series))

    def __len__(self):

        return self.spec_df.shape[0]
        
    def _setup_spec_metadata_dicts(self):
    # setup metadata
        inst_type_list = self.inst_type
        adduct_list = self.adduct
        self.inst_type2id = {
            string: i for i,
            string in enumerate(inst_type_list)}
        self.inst_id2type = {
            i: string for i,
            string in enumerate(inst_type_list)}
        self.adduct2id = {
            string: i for i,
            string in enumerate(adduct_list)}
        self.id2adduct = {
            i: string for i,
            string in enumerate(adduct_list)}
    
        self.num_inst_type = len(inst_type_list)
        self.num_adduct = len(adduct_list)
        
    def __getitem__(self, idx):

        spec_entry = self.spec_df.iloc[idx]
        mol_id = spec_entry["mol_id"]

        mol_entry = self.mol_df.loc[mol_id]      
        data = self.process_entry(spec_entry, mol_entry["smiles"], mol_entry["formula"], mol_entry["mol"])
        return data
    

    def process_entry(self, spec_entry, smiles, formula, mol):

        # initialize data with shared attributes
        spec_feats = self.get_spec_feats(spec_entry)
        data = {**spec_feats}
        data["smiles"] = smiles
        data["formula"] = formula
        data["mol_graph_data"] = mol_to_graph_data_obj_pos(mol, spec_entry["spec_id"])
        return data
    
    def get_spec_feats(self, spec_entry):

        # convert to a dense vector
        mol_id = th.tensor(spec_entry["mol_id"]).unsqueeze(0)
        spec_id = th.tensor(spec_entry["spec_id"]).unsqueeze(0)
        mzs = [peak[0] for peak in spec_entry["peaks"]]
        ints = [peak[1] for peak in spec_entry["peaks"]]

        prec_mz = spec_entry["precursor_mz"]
        prec_mz = float(prec_mz)
        prec_mz_bin = self.bin_Func([prec_mz], None, return_index=True)[0]
        prec_diff = max(mz - prec_mz for mz in mzs)
        num_peaks = len(mzs)

        bin_spec = self.bin_Func(mzs, ints)
        spec = th.as_tensor(bin_spec, dtype=th.float32).unsqueeze(0)

        col_energy = spec_entry["collision_energy"]
        col_energy_meta = self.ce_func(col_energy)
        is_missing = pd.isna(col_energy)
        if is_missing:
            col_energy = 0.0

        inst_type = spec_entry["instrument_type"]
        adduct_type = spec_entry["adduct"]
        inst_type_idx = self.inst_type2id[inst_type]
        adduct_type_idx = self.adduct2id[adduct_type]
        prec_mz_idx = th.tensor(
            min(prec_mz_bin, spec.shape[1] - 1)).unsqueeze(0)
        assert prec_mz_idx < spec.shape[1], (prec_mz_bin,
                                             prec_mz_idx, spec.shape)
        
        
        inst_type_meta = th.as_tensor(
            np_one_hot(
                inst_type_idx,
                num_classes=self.num_inst_type),
            dtype=th.float32)
        adduct_type_meta = th.as_tensor(
            np_one_hot(
                adduct_type_idx,
                num_classes=self.num_adduct),
            dtype=th.float32)
        
        spec_meta_list = [
            col_energy_meta,
            inst_type_meta,
            adduct_type_meta
            ]
        spec_meta = th.cat(spec_meta_list, dim=0).unsqueeze(0)
        spec_feats = {
            "spec": spec,
            "prec_mz": [prec_mz],
            "prec_mz_bin": [prec_mz_bin],
            "prec_diff": [prec_diff],
            "num_peaks": [num_peaks],
            "inst_type": [inst_type],
            "col_energy": [col_energy],
            "spec_meta": spec_meta
        }
        return spec_feats
    
    def bin_Func(self, mzs, ints, return_index=False):

        # assert self.ints_thresh == 0., self.ints_thresh
        return bin_func(
            mzs,
            ints,
            self.mz_max,
            self.mz_bin_res,
            self.ints_thresh,
            return_index)
    
    def ce_func(self, col_energy):

        missing = col_energy is None or (isinstance(col_energy, float) and not np.isfinite(col_energy))
        if not missing:
            try:
                ce = float(col_energy)
                if not np.isfinite(ce):
                    missing = True
            except Exception:
                missing = True

        if self.preproc_ce == "normalize":
            if missing:
                z = 0.0   
                mask = 0.0   
            else:
                z = (float(col_energy) - self.mean_ce) / (self.std_ce + EPS)
                mask = 1.0
            return th.tensor([z, mask], dtype=th.float32)
        
        elif self.preproc_ce == "quantize":
            ce_bins = np.arange(0, 161, 20)          # 0,20,...,160
            num_bins = len(ce_bins) + 1          
            unknown_idx = num_bins          
            vec = th.zeros(num_bins + 1, dtype=th.float32)

            try:
                ce = float(col_energy)
                if not np.isfinite(ce):
                    raise ValueError
                idx = int(np.digitize(ce, ce_bins, right=False))
            except Exception:
                idx = unknown_idx                 

            vec[idx] = 1.0
            return vec
        
        else:
            assert self.preproc_ce == "none", self.preproc_ce

            if missing:
                v = 0.0
                m = 0.0
            else:
                v = float(col_energy)
                m = 1.0
            return th.tensor([v, m], dtype=th.float32)
    
    def get_data_dims(self):

        data = self.__getitem__(0)
        spec_dim = data["spec"].shape[1]
        meta_dim = data["spec_meta"].shape[1]
        return spec_dim, meta_dim
    
    def get_train_dataloaders(self, run_d):
        batch_size = run_d["batch_size"]
        num_workers = run_d["num_workers"]
        pin_memory = True

        train_mask = self.spec_df["fold"] == 'train'
        all_idx = np.arange(len(self))
        train_ss = TrainSubset(self, all_idx[train_mask])
        collate_fn = self.mol_spec_collate

        if len(train_ss) > 0:
            train_dl = th_data.DataLoader(
                train_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                prefetch_factor=4,
                shuffle=True,
                drop_last=True
            )
        else:
            train_dl = None
        return train_dl
    
    def get_val_dataloaders(self, run_d):
        batch_size = run_d["batch_size"]
        num_workers = run_d["num_workers"]
        pin_memory = True

        val_mask = self.spec_df["fold"] == 'val'
        all_idx = np.arange(len(self))
        val_ss = th_data.Subset(self, all_idx[val_mask])
        collate_fn = self.mol_spec_collate

        if len(val_ss) > 0:
            val_dl = th_data.DataLoader(
                val_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                prefetch_factor=4,
                shuffle=False,
                drop_last=False
            )
        else:
            val_dl = None

        return val_dl
    
    def get_test_dataloaders(self, run_d):
        batch_size = run_d["batch_size"]
        num_workers = run_d["num_workers"]
        pin_memory = True

        test_mask = self.spec_df["fold"] == 'test'
        all_idx = np.arange(len(self))
        test_ss = th_data.Subset(self, all_idx[test_mask])
        collate_fn = self.mol_spec_collate

        if len(test_ss) > 0:
            test_dl = th_data.DataLoader(
                test_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True,
                prefetch_factor=4,
                shuffle=False,
                drop_last=False
            )
        else:
            test_dl = None

        return test_dl
    @staticmethod
    def mol_spec_collate(batch_list):
        """
        batch_list: comes from MoleculeDataset.__getitem__
        Includes:
      - 'mol_graph_data': PyG Data，含 x/edge_index/pos/... 以及 .descriptors
      - 'spec': [1, n_bins]
      - 'spec_meta': [1, meta_dim]
        """
        data_list = []
        for item in batch_list:
            d = item['mol_graph_data']
            d.spec = item['spec']
            d.spec_meta = item['spec_meta'] # [meta_dim]
            
            data_list.append(d)

        batch = Batch.from_data_list(data_list)
        return batch
    
    @staticmethod
    def to_device(batch, device):
        """
        Move the collated batch (torch_geometric.data.Batch) to GPU/CPU
        """
        if hasattr(batch, 'to'):
            return batch.to(device, non_blocking=True)
        else:
            for key, value in batch.__dict__.items():
                if isinstance(value, th.Tensor):
                    batch.__dict__[key] = value.to(device)
        return batch
    
# if __name__ == "__main__":
#     data_d = {
#         "data_dir": "data",
#         "inst_type": ["QTOF", "Orbitrap","others"],
#         "adduct": ["[M+H]+", "[M+Na]+"],
#         "mz_bin_res":0.1,
#         "mz_max":1000,
#         "preproc_ce":"normalize",
#         "ints_thresh":0
#         }
#     ds = MoleculeDataset(
#     spec_df="spec_df.pkl",
#     mol_df="mol_df.pkl",
#     **data_d)
