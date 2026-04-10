import os
import logging
import multiprocessing
import torch
from torch.optim import AdamW
import torch.nn.functional as F
from dataset.mol_dataset import MoleculeDataset
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .GNN_mamba import GNNMambaFusion
from dataset.data_utils import set_seed, report_model_parameters
import numpy as np
import pandas as pd
import pickle
import yaml

config_path = "GNN-Mamba.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
data_d = config.get('data_d', {})
run_d = config.get('run_d', {})

# === logger ===
def setup_logger():
    log_path = os.path.join(run_d["log_dir"], "train.log")
    os.makedirs(run_d["log_dir"], exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

# === Loss ===
def cosine_loss(pred, target, reduce="mean"):
 
    pred = F.relu(pred)
    target = F.relu(target)
    pred = F.normalize(pred, p=2, dim=-1, eps=1e-8)
    target = F.normalize(target, p=2, dim=-1, eps=1e-8)
    raw_cos = (pred * target).sum(dim=-1)
    loss_vec = 1.0 - raw_cos
    loss = loss_vec.mean() if reduce == "mean" else loss_vec
    metrics = {
        "cos_sim": raw_cos.clamp(0, 1).mean().item(),
        "cos_loss": float(loss.detach().cpu()),
        "neg_frac": float((raw_cos < 0).float().mean().detach().cpu()),
    }
    return raw_cos, loss, metrics

# === Evaluation metric ===
def spectrum_peak_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    thr_true_rel: float = 0.02,
    thr_pred_rel: float = 0.10,
    thr_true_abs: float = 1e-3,
    thr_pred_abs: float = 1e-3,
):
    """
    Per-sample binary "peak" metric (without tolerance, bin-by-bin hit):
    - recall = hit true valid peaks / true valid peaks
    - precision = hit predicted valid peaks / predicted valid peaks
    - f1 = 2 * P * R / (P + R)
    Returns a vector of the three metrics (shape=[B]).
    """
    pred_pos = F.relu(pred)
    tgt_pos  = F.relu(target)

    t_max = tgt_pos.amax(dim=-1, keepdim=True).clamp_min(1e-12)
    p_max = pred_pos.amax(dim=-1, keepdim=True).clamp_min(1e-12)

    t_thr = torch.maximum(thr_true_rel * t_max, torch.full_like(t_max, float(thr_true_abs)))
    p_thr = torch.maximum(thr_pred_rel * p_max, torch.full_like(p_max, float(thr_pred_abs)))

    tmask = (tgt_pos  >= t_thr)          # True valid peaks（bool, B,N）
    pmask = (pred_pos >= p_thr)          # Predicted valid peaks（bool, B,N）

    hits = (tmask & pmask).sum(dim=-1).float()     # [B]
    t_cnt = tmask.sum(dim=-1).float()              # [B]
    p_cnt = pmask.sum(dim=-1).float()              # [B]

    # Handling empty sets: no true peaks → recall=1; 
    # no predicted peaks → precision=1 (when there are also no true peaks).
    recall = torch.where(t_cnt > 0, hits / t_cnt.clamp_min(1e-12), torch.ones_like(t_cnt))
    precision = torch.where(p_cnt > 0, hits / p_cnt.clamp_min(1e-12),
                            torch.where(t_cnt > 0, torch.zeros_like(p_cnt), torch.ones_like(p_cnt)))
    f1 = torch.where((precision + recall) > 0, 2 * precision * recall / (precision + recall),
                     torch.zeros_like(precision))
    return recall, precision, f1

@torch.no_grad()
def evaluate(_model, loader, device, phase: str = "val", leave: bool = False):
    _model.eval()
    loss_sum, rec_sum, prec_sum, f1_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    pbar = tqdm(loader, desc=f"[{phase}] evaluating", unit="batch",
                leave=leave, dynamic_ncols=True)
    for batch in pbar:
        batch = MoleculeDataset.to_device(batch, device)
        pred = _model(batch)                     
        tgt = batch.spec.float()
        _ , loss, _ = cosine_loss(pred, tgt)
        bs = pred.shape[0]
        loss_sum += loss.item() * bs
        rec_vec, prec_vec, f1_vec = spectrum_peak_metrics(
            pred, tgt)
        rec_sum  += float(rec_vec.sum().cpu())
        prec_sum += float(prec_vec.sum().cpu())
        f1_sum   += float(f1_vec.sum().cpu())
        n += bs
    cos_loss = loss_sum / max(n, 1)
    cos_sim = 1.0 - cos_loss
    peak_rec = rec_sum / max(n, 1)
    peak_prec = prec_sum / max(n, 1)
    peak_f1 = f1_sum / max(n, 1)
    return {"cos_sim": cos_sim,
            "cos_loss": cos_loss,      
            "peak_rec": peak_rec,
            "peak_prec": peak_prec,
            "peak_f1": peak_f1,}

def log_epoch(epoch, train_metrics, val_metrics, test_metrics=None, improved=False):
    tag = "(best)" if improved else ""
    def _fmt(tag, m):
        s = (f"cos_loss={m['cos_loss']:.4f}, cos_sim={m['cos_sim']:.4f}")
        if 'cov' in m:
            s += f", cov={m['cov']:.4f}"
        else:
            s += (f", peak_rec={m['peak_rec']:.4f}, "
                  f"peak_prec={m['peak_prec']:.4f}, peak_f1={m['peak_f1']:.4f}")
        return s

    base = f"Epoch {epoch:03d} {tag} | " + _fmt("Train", train_metrics) + " | " + _fmt("Val", val_metrics)
    if test_metrics is not None:
        base += " | " + _fmt("Test", test_metrics)
    logger.info(base)


def build_optimizer(model, lr, wd, lr_mamba=None):
    main_params, mamba_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("mamba_enc."):
            mamba_params.append(p)
        else:
            main_params.append(p)

    groups = []
    if main_params:
        groups.append({"params": main_params, "lr": lr, "weight_decay": wd})
    if lr_mamba is not None and mamba_params:
        groups.append({"params": mamba_params, "lr": lr_mamba, "weight_decay": wd})

    return AdamW(groups, betas=(0.9, 0.999))

@torch.no_grad()
def result_prediction(
    model,
    test_loader,
    device,
    out_dir,
    spec_dim,
    mz_bin_res=0.1,
    max_batches=None,
    cos_threshold=0.8,
    file_tag=None,
    load_best: bool = True,
    ckpt_dir = None,
):
    """
    After the model training is completed, load the best checkpoint, run the test set once and output:
    - selected_{tag}.pkl : samples meeting the threshold saved individually (id / cos_sim / pred / target / meta)
    - summary_{tag}.csv : summary of IDs and cos_sim for samples passing the threshold
    - mz_axis.npy : frequency axis
    """
    os.makedirs(out_dir, exist_ok=True)

    # === 1) Load the best model (if required) ===
    if load_best:
        ckpt_root = ckpt_dir
        ckpt_path = os.path.join(ckpt_root, "best.pt")
        if os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=device)
            missing, unexpected = model.load_state_dict(state["model_state"], strict=False)
            try:
                logger.info(
                    f" loaded best checkpoint: {ckpt_path} "
                    f"(epoch={state.get('epoch')}); missing={list(missing)}, unexpected={list(unexpected)}"
                )
            except Exception:
                print(f" loaded best checkpoint: {ckpt_path} (epoch={state.get('epoch')})")
        else:
            try:
                logger.warning(f" best checkpoint not found at {ckpt_path}; using current model weights.")
            except Exception:
                print(f" best checkpoint not found at {ckpt_path}; using current model weights.")

    model.eval()

    kept = []
    total = 0
    n_seen_batches = 0

    pbar = tqdm(
        test_loader,
        desc="[test] selecting (best)",
        unit="batch",
        leave=False,
        dynamic_ncols=True
    )

    for batch in pbar:
        batch = MoleculeDataset.to_device(batch, device)

        pred_logits = model(batch)              # [B, n_bins]
        tgt = batch.spec.float()                # [B, n_bins]
        total += pred_logits.size(0)

        # Post-processing consistent with training + cosine similarity
        pred_nonneg = F.relu(pred_logits)
        tgt_nonneg  = F.relu(tgt)
        pred_norm = F.normalize(pred_nonneg, p=2, dim=-1, eps=1e-8)
        tgt_norm  = F.normalize(tgt_nonneg,  p=2, dim=-1, eps=1e-8)
        cos = (pred_norm * tgt_norm).sum(dim=-1).clamp(0, 1)  # [B]

        # Available sample identifiers / metadata
        meta = getattr(batch, "spec_meta", None)
        if meta is not None:
            meta_cpu = meta.detach().cpu().numpy() if torch.is_tensor(meta) else meta
        else:
            meta_cpu = None

        if hasattr(batch, "idx"):
            ids = batch.idx.detach().cpu().numpy().tolist() if torch.is_tensor(batch.idx) else batch.idx
        else:
            bs = pred_logits.size(0)
            ids = list(range(len(kept), len(kept) + bs))

        pred_np = pred_nonneg.detach().cpu().numpy()
        tgt_np  = tgt_nonneg.detach().cpu().numpy()
        cos_np  = cos.detach().cpu().numpy()

        # Only retain data that meets the threshold
        for i, sid in enumerate(ids):
            cos_value = float(cos_np[i])
            if cos_value >= cos_threshold:
                kept.append({
                    "id":        int(sid),
                    "cos_sim":   cos_value,
                    "pred":      pred_np[i],
                    "target":    tgt_np[i],
                    "spec_meta": (meta_cpu[i] if meta_cpu is not None else None),
                })

        n_seen_batches += 1
        if (max_batches is not None) and (n_seen_batches >= max_batches):
            break

        if total > 0:
            pbar.set_postfix(kept=len(kept), seen=total, thr=cos_threshold)

    # === 2) Write results ===
    tag = file_tag or f"cos{str(cos_threshold).replace('.', 'p')}"  # 0.8 -> cos0p8

    selected_pkl = os.path.join(out_dir, f"selected_{tag}.pkl")
    sum_csv  = os.path.join(out_dir, f"summary_{tag}.csv")
    with open(selected_pkl, "wb") as f:
        pickle.dump(kept, f)

    df = pd.DataFrame({"id": [r["id"] for r in kept],
                       "cos_sim": [r["cos_sim"] for r in kept]})
    df.to_csv(sum_csv, index=False)

    mz_axis = np.arange(spec_dim) * float(mz_bin_res)
    mz_path = os.path.join(out_dir, "mz_axis.npy")
    np.save(mz_path, mz_axis)

    try:
        logger.info(f"[selected] kept {len(kept)}/{total} samples with cos >= {cos_threshold} -> {out_dir}")
    except Exception:
        print(f"[selected] kept {len(kept)}/{total} samples with cos >= {cos_threshold} -> {out_dir}")

    return {"selected_pkl": selected_pkl, "summary_csv": sum_csv, "mz_axis": mz_path,
            "kept": len(kept), "total": total}
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("===== Run starts =====")
    os.makedirs(run_d["save_dir"], exist_ok=True)
    os.makedirs(run_d["log_dir"], exist_ok=True)
    seed = run_d["seed"]
    set_seed(seed, deterministic=True, verbose=True)

    if os.name == "nt":
        multiprocessing.freeze_support()
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    # === Construct data ===
    ds = MoleculeDataset(
        spec_df="spec_df_massspecgym.pkl",
        mol_df="mol_df_massspecgym.pkl",
        **data_d
    )
    spec_dim, meta_dim = ds.get_data_dims()
    train_loader = ds.get_train_dataloaders(run_d)
    val_loader = ds.get_val_dataloaders(run_d)
    test_loader = ds.get_test_dataloaders(run_d)

    # === Construct model：GNNMamba ===
    model = GNNMambaFusion(
        hidden=128,
        n_bins=spec_dim,
        meta_dim=meta_dim,
        use_adj_from_bond=True,
        n_mamba_layers=run_d["n_mamba_layers"],
        seg_head=run_d["use_seg_head"],
        gnn_type=run_d["gnn_type"]
    ).to(device)
    # report_model_parameters(model, logger=logger, top_k=12)

    model.train()
    lr_mamba = run_d["lr"] * float(run_d["mamba_lr_scale"])
    optimizer = build_optimizer(model, lr=run_d["lr"], wd=run_d["weight_decay"], lr_mamba=lr_mamba)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float("inf")
    best_epoch = -1
    best_test_metrics = None

    # === Training loop ===
    num_epochs = int(run_d["num_epochs"])
    for epoch in range(1, num_epochs + 1):

        # ---- Train one epoch----
        model.train()
        loss_sum, cos_sum, n = 0.0, 0.0, 0
        rec_sum, prec_sum, f1_sum = 0.0, 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = MoleculeDataset.to_device(batch, device)
            pred = model(batch)                  
            tgt = batch.spec.float()
            _ , loss, _ = cosine_loss(pred, tgt)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = pred.shape[0]
            loss_sum += float(loss.detach().cpu()) * bs

            # Peak metric
            rec_vec, prec_vec, f1_vec = spectrum_peak_metrics(
            pred, tgt)
            rec_sum  += float(rec_vec.sum().cpu())
            prec_sum += float(prec_vec.sum().cpu())
            f1_sum   += float(f1_vec.sum().cpu())
            n += bs

        cos_loss = loss_sum  / max(n, 1) 
        peak_rec = rec_sum / max(n, 1)
        peak_prec = prec_sum / max(n, 1)
        peak_f1 = f1_sum / max(n, 1)

        train_metrics = {"cos_sim": 1-cos_loss, 
                        "cos_loss": cos_loss,
                        "peak_rec": peak_rec, "peak_prec": peak_prec, "peak_f1": peak_f1}
        # ---- Validation ----
        val_metrics = evaluate(model, val_loader, device, phase="val")

        # ---- Checkpoint + Test-on-best ----
        improved = val_metrics["cos_loss"] < best_val_loss
        test_metrics = None
        if improved:
            best_val_loss = val_metrics["cos_loss"]
            best_epoch = epoch
            ckpt_path = os.path.join(run_d["save_dir"], "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            logger.info(f"New best at epoch {epoch:03d}; checkpoint saved to {ckpt_path}")
            test_metrics = evaluate(model, test_loader, device, phase="test")
            best_test_metrics = test_metrics

        # ---- LR scheduler ----
        scheduler.step(val_metrics["cos_loss"])
        # ---- Log ----
        log_epoch(epoch, train_metrics, val_metrics, test_metrics, improved=improved)

    logger.info(f"Training done. Best epoch = {best_epoch:03d}; "
                f"BestValLoss={best_val_loss:.4f}; BestTest={best_test_metrics}")
    result_prediction(
    model,
    test_loader=test_loader,
    device=device,
    out_dir=os.path.join(run_d["plot_dir"]),
    spec_dim=spec_dim,
    mz_bin_res=data_d["mz_bin_res"],
    cos_threshold=0.8,
    file_tag=f"best_epoch{best_epoch:03d}",
    load_best=True,
    ckpt_dir=run_d["save_dir"],
)
    logger.info("===== Run ends =====")

if __name__ == "__main__":
    logger = setup_logger()
    main()