# uncert_utils.py
import json
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = logits.softmax(dim=-1).clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)



@torch.no_grad()
def compute_predictive_entropy(model, dataloader, device='cuda', max_batches=None) -> float:
    model.eval()
    vals = []
    for b, (x, _) in enumerate(dataloader):
        if max_batches is not None and b >= max_batches: break
        x = x.to(device)
        logits = model(x)                # logits [B,C]
        vals.append(entropy_from_logits(logits).cpu())
    return float(torch.cat(vals).mean()) if vals else 0.0




# ---------- MC Dropout for epistemic/aleatoric split ----------
def _enable_dropout(m: nn.Module):
    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        m.train()   # turn on dropout at eval time




@torch.no_grad()
def mc_dropout_entropy_components(model, dataloader, T=10, device='cuda', max_batches=None):
    """
    Returns dict with:
      - pred_ent: H[ E_omega p(y|x, omega) ]
      - data_ent: E_omega[ H[ p(y|x, omega) ] ]
      - mi:       pred_ent - data_ent  (epistemic)
    """
    model.eval()
    # Gather logits over T stochastic passes
    logits_T = []
    for t in range(T):
        model.apply(_enable_dropout)
        pass_logits = []
        for b, (x, _) in enumerate(dataloader):
            if max_batches is not None and b >= max_batches: break
            x = x.to(device)
            logits = model(x)
            # pass_logits.append(logits.detach().cpu())
            pass_logits.append(logits.detach().float().cpu())  # cast to fp32 on GPU, then move to CPU
        if not pass_logits:
            return dict(pred_ent=0.0, data_ent=0.0, mi=0.0)
        logits_T.append(torch.cat(pass_logits, dim=0))  # [N,C]
    logits_T = torch.stack(logits_T, dim=0)             # [T,N,C]
    probs_T = logits_T.softmax(dim=-1).clamp_min(1e-12)
    probs_bar = probs_T.mean(dim=0)                     # [N,C]

    pred_ent = -(probs_bar * probs_bar.log()).sum(dim=-1).mean()
    data_ent = (-(probs_T * probs_T.log()).sum(dim=-1)).mean(dim=0).mean()
    mi = pred_ent - data_ent

    return dict(pred_ent=float(pred_ent), data_ent=float(data_ent), mi=float(mi))



# ---------- Prompt collection / diversity ----------

@torch.no_grad()
def collect_prompts(models, client_ids):
    return [models[c].prompt_learner.ctx_global.detach().float().flatten()
            for c in client_ids]



@torch.no_grad()
def compute_prompt_similarity(prompt_tensors):
    K = len(prompt_tensors)
    S = torch.zeros(K, K)
    for i in range(K):
        for j in range(i, K):
            s = F.cosine_similarity(prompt_tensors[i], prompt_tensors[j], dim=0)
            S[i, j] = s
            S[j, i] = s
    client_sim = S.mean(dim=1)
    sem_unc = (1.0 / client_sim.clamp_min(1e-6)).tolist()
    return S, client_sim.tolist(), sem_unc





@torch.no_grad()
def diversity_stats(prompt_tensors):
    """
    Returns:
      - mean_offdiag_cos, std_offdiag_cos
      - eff_rank (Shannon effective rank of singular values)
    """
    if len(prompt_tensors) < 2:
        return dict(mean_offdiag_cos=1.0, std_offdiag_cos=0.0, eff_rank=1.0)
    S, _, _ = compute_prompt_similarity(prompt_tensors)          # [K,K]
    off = S[~torch.eye(S.size(0), dtype=bool)]
    mean_off = float(off.mean())
    std_off = float(off.std())

    # Effective rank from singular values of (K x D) prompt matrix
    P = torch.stack(prompt_tensors, dim=0)                       # [K,D]
    U, Svals, Vh = torch.linalg.svd(P, full_matrices=False)      # Svals: [min(K,D)]
    ps = (Svals**2) / (Svals**2).sum()
    ent = -(ps * (ps + 1e-12).log()).sum()
    eff_rank = float(torch.exp(ent))
    return dict(mean_offdiag_cos=mean_off, std_offdiag_cos=std_off, eff_rank=eff_rank)





@torch.no_grad()
def prompt_delta_norm(prev_prompts, curr_prompts):
    """L2 movement of prompts per client (list of floats)."""
    assert len(prev_prompts) == len(curr_prompts)
    out = []
    for p0, p1 in zip(prev_prompts, curr_prompts):
        out.append(float(torch.norm(p1 - p0, p=2)))
    return out



@torch.no_grad()
def avg_domain_entropy_from_model(model, loader, device="cuda", max_batches=2):
    """Mean entropy of cquery/domain weights over a few batches."""
    vals = []
    model.eval()
    for b, (x, _) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x = x.to(device)
        # FedUPro client forward returns: h, txt_mean, out_dom, logits, mu, logvar, kl
        _, _, out_dom, _, _, _, _ = model(x)
        p = out_dom.softmax(dim=-1).clamp_min(1e-12)
        ent = -(p * p.log()).sum(dim=-1)      # per-sample entropy
        vals.append(ent.detach().cpu())
    if not vals:
        return 0.0
    return float(torch.cat(vals, dim=0).mean())



class UncertaintyLogger:
    def __init__(self, path: str):
        self.path = path
        self.rows = []

    @staticmethod
    def _to_jsonable_scalar(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return str(x)  # last resort, avoid crashes

    def record_round(self, round_id, client_ids, **metrics):
        row = {"round": int(round_id), "clients": list(map(int, client_ids))}
        for k, v in metrics.items():
            if isinstance(v, dict):
                # try integer keys first; fall back to strings
                try:
                    row[k] = {int(kk): self._to_jsonable_scalar(vv) for kk, vv in v.items()}
                except (ValueError, TypeError):
                    row[k] = {str(kk): self._to_jsonable_scalar(vv) for kk, vv in v.items()}
            elif isinstance(v, (list, tuple)):
                row[k] = [self._to_jsonable_scalar(x) for x in v]
            else:
                row[k] = self._to_jsonable_scalar(v)
        self.rows.append(row)
        with open(self.path, "w") as f:
            json.dump(self.rows, f, indent=2)