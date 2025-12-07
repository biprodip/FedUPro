import os, json, time, numpy as np
import os, json, numpy as np
from typing import Optional, List, Dict

def _spearman_rank_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    rx = (rx - rx.mean()) / (rx.std() + 1e-8)
    ry = (ry - ry.mean()) / (ry.std() + 1e-8)
    return float(np.mean(rx * ry))


class _SimpleJSONLogger:
    """Fallback if uncert_utils.UncertaintyLogger isn't present."""
    def __init__(self, path="./uncert_rounds.json"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({"rounds": []}, f)

    def record_round(self, round_idx, clients, **payload):
        with open(self.path, "r") as f:
            data = json.load(f)
        data["rounds"].append({
            "round": int(round_idx),
            "ts": time.time(),
            "clients": list(map(int, clients)),
            **payload
        })
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

# Prefer your utils logger if available
# try:
#     from uncert_utils import UncertaintyLogger as _ULogger
#     logger = _ULogger(path=getattr(args, "uncert_log", "./uncert_rounds.json"))
# except Exception:
#     logger = _SimpleJSONLogger(path=getattr(args, "uncert_log", "./uncert_rounds.json"))

# Keep this outside the loop once:
try:
    top1_prev
except NameError:
    top1_prev = None




def log_correlations_and_round(
    fe: int,
    this_round_clients: List[int],
    top1_prev: Optional[float],
    top1_now: float,
    u_pred: np.ndarray,
    u_dataE: np.ndarray,
    u_mi: np.ndarray,
    delta_prompts: List[float],
    div_stats: Dict[str, float],
    client_traces: Optional[Dict[int, float]] = None,
    logger_obj=None,
    *,
    # NEW (optional):
    utility_vec: Optional[List[float]] = None,          # per-client NEXT-ROUND utility (aligned to this_round_clients)
    diversity_prev: Optional[Dict[str, float]] = None,  # diversity from PREVIOUS round
    cum_div_hist_path: str = "./div_top1_hist.json"
):
    """
    Backward-compatible. If utility_vec is None, falls back to scalar ΔTop1.
    Also logs within-round corr(prompt_move, MI) and cumulative corr(diversity_prev.eff_rank, top1_now).
    """
    # Fallback logger (identical behavior to UncertaintyLogger.record_round)
    if logger_obj is None:
        class _Simple:
            def __init__(self, path="./uncert_rounds.json"):
                self.path = path
                self.rows = []
                if os.path.exists(path):
                    try: self.rows = json.load(open(path))
                    except Exception: self.rows = []
            def record_round(self, round_id, client_ids, **metrics):
                row = {"round": int(round_id), "clients": list(map(int, client_ids))}
                for k, v in metrics.items():
                    if isinstance(v, dict):
                        row[k] = {int(kk): float(vv) for kk, vv in v.items()}
                    elif isinstance(v, (list, tuple)):
                        row[k] = [float(x) for x in v]
                    elif v is None:
                        row[k] = None
                    else:
                        row[k] = float(v)
                self.rows.append(row)
                json.dump(self.rows, open(self.path, "w"), indent=2)
        logger_obj = _Simple()

    # Build vectors aligned with this_round_clients
    vec_mi   = np.array([u_mi[cl, fe]   for cl in this_round_clients], dtype=float)
    vec_pred = np.array([u_pred[cl, fe] for cl in this_round_clients], dtype=float)
    vec_move = np.array(delta_prompts, dtype=float)

    # ΔTop1 (scalar) for backward-compat; if utility_vec is provided, prefer it
    delta_top1 = (top1_now - top1_prev) if (top1_prev is not None) else 0.0
    if utility_vec is not None:
        ut = np.asarray(utility_vec, dtype=float)
    else:
        ut = np.ones_like(vec_mi) * float(delta_top1)

    # Correlations
    r_mi_ut    = _spearman_rank_corr(vec_mi,   ut)
    r_pred_ut  = _spearman_rank_corr(vec_pred, ut)
    r_move_ut  = _spearman_rank_corr(vec_move, ut)
    r_move_mi  = _spearman_rank_corr(vec_move, vec_mi)  # within-round: prompt movement vs MI

    # Cumulative diversity→next-round accuracy correlation (uses a sidecar)
    corr_diversity_top1_next_cum = None
    if diversity_prev is not None:
        hist = []
        if os.path.exists(cum_div_hist_path):
            try: hist = json.load(open(cum_div_hist_path))
            except Exception: hist = []
        hist.append({
            "round": fe,
            "eff_rank_prev": float(diversity_prev.get("eff_rank", np.nan)),
            "mean_cos_prev": float(diversity_prev.get("mean_offdiag_cos", np.nan)),
            "top1_now": float(top1_now)
        })
        json.dump(hist, open(cum_div_hist_path, "w"), indent=2)

        eff = [h["eff_rank_prev"] for h in hist if np.isfinite(h["eff_rank_prev"])]
        t1  = [h["top1_now"]      for h in hist if np.isfinite(h["top1_now"])]
        if len(eff) >= 2 and len(t1) >= 2:
            corr_diversity_top1_next_cum = _spearman_rank_corr(eff, t1)
        else:
            corr_diversity_top1_next_cum = 0.0

    # Pack dicts for logging
    pred_ent_d = {cl: float(u_pred[cl, fe]) for cl in this_round_clients}
    data_ent_d = {cl: float(u_dataE[cl, fe]) for cl in this_round_clients}
    mi_d       = {cl: float(u_mi[cl, fe])   for cl in this_round_clients}
    move_d     = {cl: float(vec_move[i])    for i, cl in enumerate(this_round_clients)}
    trace_d    = {} if client_traces is None else {cl: float(client_traces.get(cl, 0.0)) for cl in this_round_clients}
    ut_d       = {cl: float(ut[i])          for i, cl in enumerate(this_round_clients)}

    logger_obj.record_round(
        fe, this_round_clients,
        pred_ent=pred_ent_d,
        data_ent=data_ent_d,
        mi=mi_d,
        prompt_move=move_d,
        diversity={k: float(v) for k, v in div_stats.items()},
        trace=trace_d,
        utility=ut_d,  # NEW: per-client utility used for corr
        corr_mi_utility=float(r_mi_ut),
        corr_pred_utility=float(r_pred_ut),
        corr_move_utility=float(r_move_ut),
        corr_move_mi_within=float(r_move_mi),  # NEW
        corr_diversity_top1_next_cum=float(corr_diversity_top1_next_cum) if corr_diversity_top1_next_cum is not None else None,
        delta_top1=float(delta_top1),
        top1_now=float(top1_now),
        top1_prev=float(top1_prev) if top1_prev is not None else None
    )

    return delta_top1, r_mi_ut, r_pred_ut, r_move_ut, r_move_mi, corr_diversity_top1_next_cum
