
import numpy as np
import torch
from typing import Tuple, List, Optional

def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def ece_mce(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> Tuple[float, float]:
    probs = probs.detach().float()
    labels = labels.detach().long()
    conf, pred = probs.max(dim=1)
    correct = pred.eq(labels).float()

    bins = torch.linspace(0, 1, steps=n_bins+1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    mce = torch.zeros((), device=probs.device)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            acc_bin = correct[mask].mean()
            conf_bin = conf[mask].mean()
            gap = (acc_bin - conf_bin).abs()
            ece += gap * (mask.float().mean())
            mce = torch.maximum(mce, gap)
    return float(ece.item()), float(mce.item())

def _binary_auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    R_pos = ranks[y_true == 1].sum()
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    U = R_pos - n_pos*(n_pos + 1)/2.0
    auc = U / (n_pos * n_neg)
    return float(auc)

def multiclass_auroc(probs: torch.Tensor, labels: torch.Tensor, average: str = "macro") -> float:
    P = _as_numpy(probs)
    y = _as_numpy(labels).astype(np.int64)
    n_classes = P.shape[1]
    aucs: List[float] = []
    weights: List[int] = []
    for c in range(n_classes):
        scores = P[:, c]
        y_c = (y == c).astype(np.int64)
        auc = _binary_auc(scores, y_c)
        aucs.append(auc)
        weights.append(int(y_c.sum()))
    aucs = np.array(aucs, dtype=float)
    weights = np.array(weights, dtype=float)
    valid = np.isfinite(aucs)
    if not np.any(valid):
        return float("nan")
    aucs = aucs[valid]
    weights = weights[valid]
    if average == "macro":
        return float(np.mean(aucs))
    elif average == "weighted":
        w = weights / np.maximum(weights.sum(), 1.0)
        return float(np.sum(w * aucs))
    else:
        raise ValueError(f"Unsupported average='{average}'")

@torch.no_grad()
def collect_probs_labels(model, loader, device="cuda"):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        all_probs.append(torch.softmax(logits, dim=-1))
        all_labels.append(y)
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)

@torch.no_grad()
def collect_probs_labels_bayespe(model, loader, text_banks: torch.Tensor, weights: Optional[torch.Tensor] = None, device="cuda"):
    model.eval()
    all_probs, all_labels = [], []
    tb = text_banks.to(device)
    w  = None if weights is None else weights.to(device)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        probs = model.forward_bayespe_probs(x, tb, weights=w)
        all_probs.append(probs)
        all_labels.append(y)
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)
