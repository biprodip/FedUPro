"""
FedUPro / CLIP-based federated prompt learning with:

1. Local prompt updates of a global text prompt via:
   - image–text alignment (currently CE on CLIP logits)
   - class-prompt diversity (InfoNCE-style separation)

2. A server-side pool of prompt "experts" collected from clients each round:
   - per-client prompts
   - aggregated global prompt (optional expert)

3. After training:
   - Save expert pools (per round) to disk.
   - Optionally load a saved pool and train client-local mixture-of-experts
     (MoE) weights on top of these prompts, entirely locally.

Stage A:
    Run normally → train prompts, build & save expert pools each round.

Stage B:
    Run with --load_pool → skip training, load experts, fit local MoE weights,
    and evaluate accuracy + calibration (ECE/MCE/AUROC).
"""

import os
import clip
import torch
import copy
import random
import numpy as np
import argparse
import logging
import pickle, gzip
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, ConcatDataset
from collections import defaultdict  # may be used later

# from domain import get_domainnet_dataset
from data_utils.domain_gh import get_domainnet_dataset
from promptModels.clip_base_MMD_bayes import CustomCLIP_client
from data_utils.OfficeCaltech10 import get_office_caltech10_dloader
from data_utils.data_loader_wrapper import (
    load_federated_datasets,
    print_train_samples_per_class_per_client,
)
from utils.calibration_metrics import (
    collect_probs_labels,      # standard probs
    collect_probs_labels_bayespe,  # BayesPE probs (if you use BayesPE)
    ece_mce,
    multiclass_auroc,
)
from utils.utils import make_optimizer, evala, evalaa
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from utils.clip_backbone import load_clip_model, to_pil_first

_tokenizer = _Tokenizer()


# =============================================================================
#   Distance / kernel utilities
# =============================================================================

def _pdist2(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise squared distance between rows of x and y.

    Args:
        x: [m, d]
        y: [n, d]

    Returns:
        [m, n] matrix of squared distances.
    """
    x2 = (x ** 2).sum(-1, keepdim=True)   # [m,1]
    y2 = (y ** 2).sum(-1).unsqueeze(0)    # [1,n]
    return x2 + y2 - 2.0 * x @ y.t()


def rbf_mmd2(x: torch.Tensor, y: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """Unbiased RBF-MMD^2 between two sets of samples."""
    dxx = _pdist2(x, x)
    dyy = _pdist2(y, y)
    dxy = _pdist2(x, y)
    Kxx = torch.exp(-dxx / (2 * sigma * sigma))
    Kyy = torch.exp(-dyy / (2 * sigma * sigma))
    Kxy = torch.exp(-dxy / (2 * sigma * sigma))
    m = x.size(0)
    n = y.size(0)
    return Kxx.sum() / (m * m) + Kyy.sum() / (n * n) - 2.0 * Kxy.sum() / (m * n)


# =============================================================================
#   Contrastive / alignment objectives
# =============================================================================

def align_infonce_loss(model, images, labels, stop_grad_image: bool = True, tau: float | None = None):
    """
    Contrastive image->text alignment: image vs all class texts.

    Uses model.forward_feats() for normalized features and logit scale.
    Negatives = all other classes.

    Args:
        model: CustomCLIP_client or similar with forward_feats().
        images: [B, C, H, W]
        labels: [B]
        stop_grad_image: if True, detach image features.
        tau: optional extra temperature scaling.

    Returns:
        Cross-entropy loss (scalar).
    """
    img_f, txt_f, logit_scale = model.forward_feats(images)
    if stop_grad_image:
        img_f = img_f.detach()
    logits = logit_scale * (img_f @ txt_f.t())
    if tau:
        logits = logits / tau
    return torch.nn.functional.cross_entropy(logits.float(), labels)


def align_infonce_symmetric(
    model,
    images,
    labels,
    stop_grad_image: bool = True,
    tau: float | None = None,
    lam_sym: float = 0.5,
):
    """
    Symmetric image<->text InfoNCE:
      - image->text CE
      - text->image CE (index-aligned with labels)

    lam_sym controls how much text->image contributes.
    """
    img_f, txt_f, logit_scale = model.forward_feats(images)
    if stop_grad_image:
        img_f = img_f.detach()

    logits_i2t = logit_scale * (img_f @ txt_f.t())
    if tau:
        logits_i2t = logits_i2t / tau
    loss_i2t = torch.nn.functional.cross_entropy(logits_i2t, labels)

    # text -> image (match each label's text to its own image index in the batch)
    logits_t2i = logit_scale * (txt_f[labels] @ img_f.t())
    if tau:
        logits_t2i = logits_t2i / tau
    targets_t2i = torch.arange(img_f.size(0), device=img_f.device)
    loss_t2i = torch.nn.functional.cross_entropy(logits_t2i, targets_t2i)

    return 0.5 * (loss_i2t + lam_sym * loss_t2i)


def prompt_diversity_contrastive(ctx_global: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    InfoNCE-style separation between per-class prompt vectors.

    - ctx_global: [C, n_ctx, D] or [C, D]
    - We take the mean across context tokens (if present) to get a per-class anchor.

    Returns:
        Scalar cross-entropy loss encouraging class prompts to be apart.
    """
    if ctx_global.dim() == 3:
        anchors = ctx_global.mean(dim=1)        # [C,D]
    else:
        anchors = ctx_global                    # [C,D]
    anchors = torch.nn.functional.normalize(anchors, dim=-1)

    sim = anchors @ anchors.t() / temperature   # [C,C]
    logits = sim - torch.diag_embed(sim.diag()) # mask self
    labels = torch.arange(sim.size(0), device=sim.device)
    return torch.nn.functional.cross_entropy(logits, labels)


# =============================================================================
#   Class-balanced weighting (Cui et al., "effective number")
# =============================================================================

def _class_counts_from_loader(loader: DataLoader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    with torch.no_grad():
        for _, y in loader:
            counts += torch.bincount(y, minlength=num_classes)
    return counts


def class_balanced_weights(
    loader: DataLoader,
    num_classes: int,
    beta: float = 0.9999,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Compute dataset-level class weights for THIS client."""
    counts = _class_counts_from_loader(loader, num_classes).float()
    eff_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
    w = (1.0 - beta) / torch.clamp(eff_num, min=1e-12)
    w[counts == 0] = 0.0
    w = w / (w.mean().clamp(min=1e-12))     # normalize ~ mean=1
    return w.to(device)


def cb_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    Class-balanced focal loss computed in float32 for numerical stability.
    Works even if logits are fp16.
    """
    logits_f32 = logits.float()
    weight_f32 = class_weights.float() if class_weights is not None else None

    ce = torch.nn.functional.cross_entropy(
        logits_f32, targets, weight=weight_f32, reduction="none"
    )

    with torch.no_grad():
        pt = torch.softmax(logits_f32, dim=-1).gather(
            1, targets.view(-1, 1)
        ).squeeze(1).clamp_min(1e-8)

    loss = ((1.0 - pt).pow(gamma)) * ce
    return loss.mean()


# =============================================================================
#   Expert pool save / load
# =============================================================================

def save_pool_to_pkl(server_pool, path: str | Path, meta: dict | None = None) -> None:
    """
    Save only what's needed for mixing/eval:
      - experts: list[Tensor [C, n_ctx, D]]
      - w_server (optional): learned global weights
      - meta: optional dict for reproducibility (gctx, clip_name, seed, etc.)
    """
    payload = {
        "experts": [t.cpu() for t in server_pool.experts],
        "w_server": None if server_pool.w is None else server_pool.w.cpu(),
        "meta": meta or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f)


def load_pool_from_pkl(path: str | Path, device: str | torch.device = "cuda"):
    """Load expert pool from disk."""
    with gzip.open(path, "rb") as f:
        payload = pickle.load(f)
    experts = [t.to(device) for t in payload["experts"]]
    w_server = None if payload["w_server"] is None else payload["w_server"].to(device)
    meta = payload.get("meta", {})
    return experts, w_server, meta


def save_pool_checkpoint(server_pool, base_path: str | Path, round_idx: int, meta: dict | None = None) -> None:
    """
    Save the current expert pool with a round-tagged filename.

    If base_path ends with .pkl or .pkl.gz, we strip the extension
    and append _rXXX.pkl.gz.
    """
    base = Path(base_path)
    stem = base.stem  # strips one suffix (.gz or .pkl)
    # Handle .pkl.gz double-suffix
    if base.suffix == ".gz" and stem.endswith(".pkl"):
        stem = Path(stem).stem
    out = base.parent / f"{stem}_r{round_idx:03d}.pkl.gz"
    payload_meta = dict(meta or {})
    payload_meta.update({"round": int(round_idx), "num_experts": len(server_pool.experts)})
    save_pool_to_pkl(server_pool, out, payload_meta)
    print(f"[SAVE] round {round_idx}: wrote expert pool → {out}")


# =============================================================================
#   Arg parser / setup
# =============================================================================

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--data", default="domainnet", help="dataset name (e.g., domainnet)")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_clients", default=5, type=int)
    parser.add_argument("--num_classes", default=345, type=int)
    parser.add_argument("--asyn", action="store_true")
    parser.add_argument("--round", default=50, type=int)
    parser.add_argument("--local_epochs", default=1, type=int, help="local epochs")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--learning_rate", default=1.0, type=float)
    parser.add_argument("--gctx", default=16, type=int)
    parser.add_argument("--logname", default="basedevice_gtx16")
    parser.add_argument("--datapath", default="...")
    parser.add_argument("--choose", default="rand", help="client selection: 'rand' or 'domain'")
    parser.add_argument(
        "--non_iid",
        type=str,
        required=True,
        choices=["FNLN", "FNLI"],
        help="Non-IID setting: FNLN (label+domain non-iid) or FNLI (label iid, domain non-iid)",
    )
    parser.add_argument(
        "--mix_n_experts",
        type=int,
        default=7,
        help="If >0, use only the last N experts for MoE mixing and weight-fitting",
    )
    parser.add_argument("--val_split", default=0.30, type=float)
    parser.add_argument(
        "--save_pool",
        type=str,
        default="pathological_pool.pkl.gz",
        help="Save server expert pool base path (per-round checkpoints appended)",
    )
    parser.add_argument(
        "--load_pool",
        type=str,
        default=None,
        help="If set, skip training and run Stage B: load experts from this file and evaluate local MoE",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="RN50",
        help="CLIP backbone, e.g., RN50, RN101, RN50x4, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px",
    )
    parser.add_argument(
        "--clip_cache",
        type=str,
        default=None,
        help="Optional path for CLIP download cache (download_root)",
    )
    parser.add_argument(
        "--clip_jit",
        action="store_true",
        help="Use torchscripted (JIT) CLIP if available",
    )
    return parser


parser = get_parser()
args = parser.parse_args()


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)

# Device
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
print(f"[Device] Using device: {device_str}")

# Load CLIP backbone
model, preprocess = load_clip_model(
    backbone=args.backbone,
    device=device_str,
    download_root=args.clip_cache,
    jit=args.clip_jit,
)

cfg = {
    "BackBone": args.backbone,
    "alpha": args.alpha,
    "batch_size": args.batch_size,
    "choose": args.choose,
    "learning_rate": args.learning_rate,
    "num_classes": args.num_classes,
    "gctx": args.gctx,
    "local_epochs": args.local_epochs,
    "num_clients (per domain)": args.num_clients,
    "round": args.round,
    "seed": args.seed,
}

print("[Config]")
for k in sorted(cfg):
    print(f"{k}={cfg[k]!r}")


# =============================================================================
#   Dataset
# =============================================================================
client_dataloaders, client_testloaders, out, domains, numclass = load_federated_datasets(
    args, preprocess
)
print(
    f"Length of train/test loaders: {len(client_dataloaders)}, /, {len(client_testloaders)}"
    "........................."
)


# =============================================================================
#   Local MoE helpers
# =============================================================================

def make_local_val_loader(
    client_dl: DataLoader, frac: float = 0.10, seed: int = 0, batch_size: int = 256
) -> DataLoader:
    """Build a small client-held dev split (no upload); overlaps allowed."""
    ds = client_dl.dataset
    n = len(ds)
    m = max(1, int(frac * n))
    g = np.random.default_rng(seed)
    idx = g.choice(n, size=m, replace=False)
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=True, num_workers=0)


@torch.no_grad()
def _per_expert_probs_batch(
    model,
    images: torch.Tensor,
    expert_ctxs: list[torch.Tensor],
    device: str | torch.device = "cuda",
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return stacked per-expert probabilities for a batch: [K,B,C]."""
    probs_list = []
    for ctx in expert_ctxs:
        logits = model.logits_with_ctx(images.to(device), ctx.to(device)) / temperature
        probs = torch.softmax(logits, dim=-1)  # [B,C]
        probs_list.append(probs)
    return torch.stack(probs_list, dim=0)  # [K,B,C]


def fit_local_moe_weights(
    model,
    expert_ctxs: list[torch.Tensor],
    val_loader: DataLoader,
    device: str | torch.device = "cuda",
    steps: int = 10,
    lr: float = 0.5,
    temperature: float = 1.0,
    entropy_reg: float = 0.01,
    cache_max_batches: int = 32,
) -> torch.Tensor | None:
    """
    Client-side: learn w over server experts (kept private).
    Prompts (experts) are frozen; we learn only a K-dim softmax over experts.
    """
    K = len(expert_ctxs)
    if K == 0:
        return None

    w_logits = torch.nn.Parameter(torch.zeros(K, device=device))
    opt = torch.optim.SGD([w_logits], lr=lr)

    # Cache a small slice of per-expert probs to avoid recomputing each step
    cache = []
    with torch.no_grad():
        for bi, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            probs_K = _per_expert_probs_batch(
                model, images, expert_ctxs, device=device, temperature=temperature
            )  # [K,B,C]
            cache.append((probs_K, labels))
            if bi + 1 >= cache_max_batches:
                break

    model.eval()
    for _ in range(steps):
        total_loss = 0.0
        opt.zero_grad()
        for probs_K, labels in cache:
            weights = torch.softmax(w_logits, dim=0).view(K, 1, 1)  # [K,1,1]
            p_mix = (weights * probs_K).sum(0).clamp_min(1e-12)     # [B,C]
            loss = torch.nn.functional.nll_loss(torch.log(p_mix), labels)
            if entropy_reg > 0:
                ent = -(weights.view(-1) * torch.log(weights.view(-1) + 1e-12)).sum()
                loss = loss - entropy_reg * ent
            loss.backward()
            total_loss += float(loss)
        opt.step()

    return torch.softmax(w_logits.detach(), dim=0).cpu()  # [K]


def pick_experts(experts: list[torch.Tensor], n_last: int | None):
    """Return last N experts (most recent). If n_last<=0 or None, return all."""
    if not experts:
        return experts
    if (n_last is None) or (n_last <= 0):
        return experts
    n = min(n_last, len(experts))
    return experts[-n:]


@torch.no_grad()
def local_forward_mixture(
    model,
    images: torch.Tensor,
    expert_ctxs: list[torch.Tensor],
    weights: torch.Tensor | None,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Client-side mixture forward with learned weights; returns [B,C] probs."""
    images = images.to(device)
    if (weights is None) or (len(expert_ctxs) == 0):
        logits = model(images, True)
        return torch.softmax(logits, dim=-1)

    K = len(expert_ctxs)
    w = weights.to(device).view(K, 1, 1)  # [K,1,1]
    probs_K = _per_expert_probs_batch(model, images, expert_ctxs, device=device)  # [K,B,C]
    return (w * probs_K).sum(0)


# =============================================================================
#   Calibration helpers
# =============================================================================

@torch.no_grad()
def eval_client_moe(
    model,
    loader: DataLoader,
    expert_ctxs: list[torch.Tensor],
    w_loc: torch.Tensor | None,
    device: str | torch.device = "cuda",
):
    """Evaluate a single client with a local MoE on its test loader."""
    model.eval()
    Ps, Ys = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        p = local_forward_mixture(model, x, expert_ctxs, w_loc, device=device)
        Ps.append(p)
        Ys.append(y)
    P = torch.cat(Ps, dim=0)
    Y = torch.cat(Ys, dim=0)
    return P, Y


# =============================================================================
#   Server expert pool
# =============================================================================

class ServerExpertPool:
    def __init__(self, max_experts: int = 10, include_global: bool = True):
        self.max_experts = max_experts
        self.include_global = include_global
        self.experts: list[torch.Tensor] = []  # list of [C, n_ctx, D]
        self.w: torch.Tensor | None = None     # optional learned mixture weights

    def add(self, ctx_tensor: torch.Tensor) -> None:
        """Add a new expert context: ctx_tensor [C, n_ctx, D]."""
        self.experts.append(ctx_tensor.detach().clone().cpu())
        if len(self.experts) > self.max_experts:
            self.experts.pop(0)

    def clear(self) -> None:
        """Clear all experts and any learned weights."""
        self.experts = []
        self.w = None

    @torch.no_grad()
    def _probs_from_ctx(self, model, images: torch.Tensor, ctx: torch.Tensor):
        """Single-expert forward: logits_with_ctx → probs."""
        logits = model.logits_with_ctx(images, ctx)
        return torch.softmax(logits, dim=-1)  # [B,C]

    @torch.no_grad()
    def forward_mixture(self, model, images: torch.Tensor, device: str | torch.device = "cuda"):
        """Mixture forward using self.w or uniform mixing if w is None."""
        K = len(self.experts)
        if K == 0:
            return None
        if self.w is None:
            w = torch.full((K, 1, 1), 1.0 / K, device=device)
        else:
            w = self.w.to(device).view(K, 1, 1)

        probs_list = []
        for k in range(K):
            ctx = self.experts[k].to(device)
            probs = self._probs_from_ctx(model, images.to(device), ctx)  # [B,C]
            probs_list.append(probs)
        probs_K = torch.stack(probs_list, dim=0)  # [K,B,C]
        return (w * probs_K).sum(0)               # [B,C]


# =============================================================================
#   Model init
# =============================================================================

global_model = CustomCLIP_client(out, model, args.gctx, domain_number=len(domains)).to(device)
models = [
    CustomCLIP_client(out, model, args.gctx, domain_number=len(domains)).to(device)
    for _ in range(len(client_dataloaders))
]


# =============================================================================
#   Stage B: load experts and only fit/evaluate local MoE
# =============================================================================

if args.load_pool is not None:
    # ---- STAGE B: load experts and evaluate mixing without training ----
    expert_ctxs_eval, w_server, meta = load_pool_from_pkl(args.load_pool, device=device_str)
    expert_ctxs_eval = pick_experts(expert_ctxs_eval, args.mix_n_experts)

    # Fit per-client local MoE weights on a tiny local dev split and evaluate
    client_local_w: dict[int, torch.Tensor | None] = {}
    for cl in range(len(client_dataloaders)):
        cl_val = make_local_val_loader(
            client_dataloaders[cl],
            frac=args.val_split,
            seed=args.seed,
            batch_size=256,
        )
        w_loc = fit_local_moe_weights(
            models[cl],
            expert_ctxs_eval,
            cl_val,
            device=device_str,
            steps=30,
            lr=0.5,
            temperature=0.1,
            entropy_reg=0.05,
        )
        client_local_w[cl] = w_loc

    # Accuracy (macro / micro)
    local_moe_accs, local_moe_weights = [], []
    for lid in range(len(client_dataloaders)):
        dom_id = lid // args.num_clients  # adjust if you have a mapping
        tl = client_testloaders[dom_id]
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in tl:
                images = images.to(device)
                labels = labels.to(device)
                probs = local_forward_mixture(
                    models[lid], images, expert_ctxs_eval, client_local_w[lid], device=device_str
                )
                pred = probs.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.numel()
        top1 = 100.0 * correct / max(1, total)
        local_moe_accs.append(top1)
        local_moe_weights.append(len(tl.dataset))
        print(f"[LOCAL-MOE] client {lid} top1: {top1:.2f}")

    macro_top1 = sum(local_moe_accs) / len(local_moe_accs)
    micro_top1 = sum(a * w for a, w in zip(local_moe_accs, local_moe_weights)) / sum(local_moe_weights)
    print(f"[LOCAL-MOE] mean top1 (macro): {macro_top1:.2f}")
    print(f"[LOCAL-MOE] mean top1 (micro): {micro_top1:.2f}")

    # Calibration pooled + per-client
    pooled_Ps, pooled_Ys = [], []
    for cl_id, model_cl in enumerate(models):
        dom_id = cl_id // args.num_clients
        tl = client_testloaders[dom_id]
        P, Y = eval_client_moe(
            model_cl, tl, expert_ctxs_eval, client_local_w.get(cl_id, None), device=device_str
        )
        pooled_Ps.append(P)
        pooled_Ys.append(Y)
        ece, mce = ece_mce(P, Y, n_bins=15)
        auc = multiclass_auroc(P, Y, average="macro")
        print(f"[LOCAL-MoE] client {cl_id}: ECE={ece:.4f} MCE={mce:.4f} AUROC={auc:.4f}")

    P_all = torch.cat(pooled_Ps, dim=0)
    Y_all = torch.cat(pooled_Ys, dim=0)
    ece_all, mce_all = ece_mce(P_all, Y_all, n_bins=15)
    auc_all = multiclass_auroc(P_all, Y_all, average="macro")
    print(f"[LOCAL-MoE/pooled] ECE={ece_all:.4f} MCE={mce_all:.4f} AUROC={auc_all:.4f}")

    raise SystemExit(0)


# =============================================================================
#   Stage A: federated prompt training + expert pool saving
# =============================================================================

server_pool = ServerExpertPool(max_experts=10, include_global=True)

# Sync all client weights from global model initially
for client in models[1:]:
    client.load_state_dict(models[0].state_dict())

# Freeze CLIP tower, train only prompt_learner
for m in models:
    for name, param in m.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

tot_cl = args.num_clients  # per domain

for fe in tqdm(range(args.round)):
    server_pool.clear()  # guarantees pool = only this round

    # Client selection per round
    if args.choose in ("rand", "random"):
        # random across all domain-clients
        this_round_clients = np.sort(
            np.random.choice(list(range(tot_cl * len(domains))), 6, replace=False)
        ).tolist()
    else:
        # 'domain': sample one client per domain block
        this_round_clients = [
            np.random.choice(list(range(tot_cl * k, tot_cl * k + tot_cl)), 1, replace=False)[0]
            for k in range(len(domains))
        ]

    print("------------- federated ", fe, "-th  --------------------------")

    # Broadcast global prompt to all clients after first round
    if fe > 0:
        for cl_model in models:
            cl_model.prompt_learner.load_state_dict(
                global_model.prompt_learner.state_dict(), strict=False
            )

    for cl in this_round_clients:
        local_epochs = args.local_epochs
        optimizer = make_optimizer(models[cl].prompt_learner, base_lr=0.01)

        with torch.no_grad():
            before = models[cl].prompt_learner.ctx_global.norm().item()

        for e in range(local_epochs):
            epoch_loss = 0.0

            for i, (image, label) in enumerate(client_dataloaders[cl]):
                optimizer.zero_grad()
                image = image.to(device)
                label = label.to(device)

                # hyperparams (tune lightly)
                lambda_div = 0.05

                out_cls = models[cl](image, True)
                ce = torch.nn.functional.cross_entropy(out_cls, label)

                # MI-style diversity across class prompts (push class prompts apart)
                ctx = models[cl].prompt_learner.ctx_global
                mi_div = prompt_diversity_contrastive(ctx, temperature=0.1)

                loss = ce + lambda_div * mi_div

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if e == (local_epochs - 1) or (e == 0):
                print(
                    f"Client {cl} Ep :{e+1}/{local_epochs} "
                    f"Tr_Data:{len(client_dataloaders[cl].dataset)} Tot Loss: {epoch_loss:.4f}"
                )

        with torch.no_grad():
            after = models[cl].prompt_learner.ctx_global.norm().item()
        print(f"[client {cl}] Δ||ctx_global|| = {after - before:.4e}")

        print(f"Adding client {cl} prompt to the pool.")
        server_pool.add(models[cl].prompt_learner.ctx_global)

    # Simple average aggregation of prompt_learner across selected clients
    weights = [1 / len(this_round_clients)] * len(this_round_clients)
    local_state = models[0].prompt_learner.state_dict()

    for k, cl in enumerate(this_round_clients):
        client_state = models[cl].prompt_learner.state_dict()
        for st in local_state:
            if k == 0:
                local_state[st] = client_state[st] * weights[k]
            else:
                local_state[st] += client_state[st] * weights[k]

    global_model.prompt_learner.load_state_dict(local_state, strict=False)

    print("Adding aggregated prompt to the pool.")
    server_pool.add(global_model.prompt_learner.ctx_global)

    print("Global prompt evaluation (non mixture)....")
    for te, test_loader in enumerate(client_testloaders):
        top1, topk = evalaa(global_model, test_loader)
        top1_pct = top1 * 100.0 if top1 <= 1.0 else top1
        print(f"round {fe} in test_domain_loader {te} acc: {top1_pct:.2f}")

    meta = {
        "gctx": args.gctx,
        "seed": args.seed,
        "num_classes": numclass,
        "num_experts": len(server_pool.experts),
    }
    if args.save_pool:
        save_pool_checkpoint(server_pool, args.save_pool, fe, meta=meta)
        print(f"Wrote expert pools of round {fe}")


# =============================================================================
#   Fit client-local MoE weights after training (Stage A extension)
# =============================================================================

client_local_w: dict[int, torch.Tensor | None] = {}

if len(server_pool.experts) < 2:
    # Not enough experts yet; fall back to single-model per client (no MoE).
    for cl in range(len(client_dataloaders)):
        client_local_w[cl] = None
else:
    # snapshot current pool once
    expert_ctxs_snapshot = [ctx.clone() for ctx in server_pool.experts]
    expert_ctxs_snapshot = pick_experts(expert_ctxs_snapshot, args.mix_n_experts)

    for cl in range(len(client_dataloaders)):
        cl_val = make_local_val_loader(
            client_dataloaders[cl],
            frac=0.10,
            seed=args.seed,
            batch_size=256,
        )
        w_loc = fit_local_moe_weights(
            models[cl],
            expert_ctxs_snapshot,
            cl_val,
            device=device_str,
            steps=10,
            lr=0.5,
            temperature=1.0,
            entropy_reg=0.01,
        )
        client_local_w[cl] = w_loc
        if w_loc is not None:
            print(f"[CLIENT {cl}/MoE] learned local mixture weights:", w_loc.tolist())

# ====================== LOCAL-MOE EVAL (macro/micro) =========================

local_moe_accs, local_moe_weights = [], []
expert_ctxs_eval = [ctx.clone() for ctx in server_pool.experts]
expert_ctxs_eval = pick_experts(expert_ctxs_eval, args.mix_n_experts)

for lid, dl in enumerate(client_dataloaders):
    w_loc = client_local_w.get(lid, None)
    models[lid].eval()
    correct, total = 0, 0
    tl_id = lid // args.num_clients  # per-domain
    tl = client_testloaders[tl_id]
    with torch.no_grad():
        for images, labels in tl:
            images = images.to(device)
            labels = labels.to(device)
            probs = local_forward_mixture(
                models[lid], images, expert_ctxs_eval, w_loc, device=device_str
            )
            pred = probs.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    top1 = 100.0 * correct / max(1, total)
    local_moe_accs.append(top1)
    local_moe_weights.append(len(tl.dataset))
    print(f"[LOCAL-MOE] round {fe} client {lid} top1: {top1:.2f}")

if local_moe_accs:
    macro_top1 = sum(local_moe_accs) / len(local_moe_accs)
    micro_top1 = sum(a * w for a, w in zip(local_moe_accs, local_moe_weights)) / sum(local_moe_weights)
    print(f"[LOCAL-MOE] round {fe} mean top1 (macro): {macro_top1:.2f}")
    print(f"[LOCAL-MOE] round {fe} mean top1 (micro): {micro_top1:.2f}")

# ---- One pass: per-client + pooled calibration ----
pooled_Ps, pooled_Ys = [], []

for cl_id, model_cl in enumerate(models):
    w_loc = client_local_w.get(cl_id, None)

    dom_id = cl_id // args.num_clients  # per-domain
    tl = client_testloaders[dom_id]

    P, Y = eval_client_moe(model_cl, tl, expert_ctxs_eval, w_loc, device=device_str)

    ece, mce = ece_mce(P, Y, n_bins=15)
    auc = multiclass_auroc(P, Y, average="macro")
    print(f"[LOCAL-MoE] client {cl_id}: ECE={ece:.4f} MCE={mce:.4f} AUROC={auc:.4f}")

    pooled_Ps.append(P)
    pooled_Ys.append(Y)

P_all = torch.cat(pooled_Ps, dim=0)
Y_all = torch.cat(pooled_Ys, dim=0)
ece_all, mce_all = ece_mce(P_all, Y_all, n_bins=15)
auc_all = multiclass_auroc(P_all, Y_all, average="macro")
print(f"[LOCAL-MoE/pooled] ECE={ece_all:.4f} MCE={mce_all:.4f} AUROC(macro)={auc_all:.4f}")
