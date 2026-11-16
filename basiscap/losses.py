"""Training of the heads on field-solver labels (paper Section III-D): relative and log-space errors against net,
ground and coupling capacitances of training designs.

A batch holds the windows (tiles) of a few nets.  A cuboid that appears in several overlapping windows is owned by
them with shares that sum to 1 (``own``), so sums over windows give each net's capacitance once.

    cuboids, padding_mask   (B, N, 10), (B, N)
    net_id       (B,)        index of the window's target net in the batch, 0 .. n_nets - 1
    own          (B, N)      ownership share of each target cuboid in this window (0 for non-target rows)
    aggr_slot    (B, N)      for neighbour cuboids: index of their net among the target net's labelled aggressors, -1 = none
    y_total, y_gnd   (n_nets,)           field-solver net total and ground capacitance, fF
    y_cpl        (n_nets, A)             field-solver coupling to each labelled aggressor net, fF
    y_cpl_valid  (n_nets, A)  bool       which aggressor slots are used
"""
import torch
import torch.nn.functional as F


def aggregate_to_nets(pred, batch):
    """Window predictions -> per-net ground (n_nets,), per-aggressor coupling (n_nets, A), total (n_nets,)."""
    n_nets, A = batch["y_cpl"].shape
    net_id, own = batch["net_id"], batch["own"]
    dev = own.device
    gnd = torch.zeros(n_nets, device=dev).scatter_add_(0, net_id, (pred["c_gnd"].float() * own).sum(dim=1))
    cpl = torch.zeros(n_nets, A, device=dev)
    e = pred["cpl"]
    unassigned = torch.zeros(n_nets, device=dev)
    if len(e["b_idx"]) > 0:
        b, s, a = e["b_idx"], e["src_idx"], e["dst_idx"]
        val = e["c_cpl"].float() * own[b, s]
        slot = batch["aggr_slot"][b, a]
        ok = slot >= 0
        cpl = cpl.index_put((net_id[b][ok], slot[ok]), val[ok], accumulate=True)
        unassigned = unassigned.scatter_add_(0, net_id[b][~ok], val[~ok])
    total = gnd + cpl.sum(dim=1) + unassigned
    return gnd, cpl, total


def netlevel_loss(pred, target, floor=0.005):
    """Relative error + magnitude-weighted relative error + log-space error on targets above the floor;
    a small absolute penalty on predictions for targets below it."""
    pos = target >= floor
    zero = torch.zeros((), device=pred.device)
    mape = wmape = log_reg = zero_pen = zero
    if pos.any():
        p, t = pred[pos], target[pos]
        rel = torch.abs(p - t) / t.clamp(min=floor)
        weight = torch.clamp(t / t.median().clamp(min=0.1), min=0.3, max=20.0)   # keeps large nets prominent
        mape, wmape = rel.mean(), (weight * rel).mean()
        log_reg = F.smooth_l1_loss(torch.log1p(p.clamp(min=0.0)), torch.log1p(t), beta=0.1)
    if (~pos).any():
        zero_pen = F.smooth_l1_loss(pred[~pos], target[~pos], beta=0.05)
    return mape + 0.5 * wmape + 0.1 * log_reg + 0.1 * zero_pen


def _ramp(step, start, end, top):
    return 0.0 if step < start else top if step >= end else top * (step - start) / (end - start)


def training_loss(pred, batch, step, floor=0.005):
    """Composite loss of one step.  Returns (loss, dict of terms)."""
    gnd, cpl, total = aggregate_to_nets(pred, batch)
    valid = batch["y_cpl_valid"]
    y_cpl_sum = (batch["y_cpl"] * valid.float()).sum(dim=1)
    cpl_sum = (cpl * valid.float()).sum(dim=1)
    has_cpl = y_cpl_sum >= floor
    zero = torch.zeros((), device=gnd.device)

    l_total = netlevel_loss(total, batch["y_total"], floor)
    l_cpl_log = F.smooth_l1_loss(torch.log1p(cpl_sum[has_cpl]), torch.log1p(y_cpl_sum[has_cpl]), beta=0.1) if has_cpl.any() else zero
    l_gnd = netlevel_loss(gnd, batch["y_gnd"], floor)
    l_cpl = netlevel_loss(cpl_sum[has_cpl], y_cpl_sum[has_cpl], floor) if has_cpl.any() else zero
    l_pair = netlevel_loss(cpl[valid], batch["y_cpl"][valid], floor) if valid.any() else zero      # each (net, aggressor) pair

    # shares of ground and of each coupling in the net's capacitance
    p_sum, y_sum = gnd + cpl_sum + 1e-6, batch["y_gnd"] + y_cpl_sum + 1e-6
    l_share = F.smooth_l1_loss(gnd / p_sum, batch["y_gnd"] / y_sum, beta=0.05)
    if valid.any():
        l_share = l_share + F.smooth_l1_loss((cpl / p_sum.unsqueeze(1))[valid], (batch["y_cpl"] / y_sum.unsqueeze(1))[valid], beta=0.05)

    w_gnd, w_cpl = _ramp(step, 500, 2000, 2.0), _ramp(step, 1000, 3000, 1.5)       # warm-up of the direct magnitude terms
    loss = 3.0 * l_total + 1.0 * l_cpl_log + 1.5 * l_pair + 0.10 * l_share + w_gnd * l_gnd + w_cpl * l_cpl
    return loss, dict(total=l_total, cpl_log=l_cpl_log, pair=l_pair, share=l_share, gnd=l_gnd, cpl=l_cpl)


def param_groups(model, lr, layer_mult=1.0, pair_mult=1.0):
    """Optimizer groups: coupling head at 3x; the per-layer scales and the input layers of the ground head at
    ``layer_mult`` x (so that the layer and power-wire terms move during training); the layer-pair scale at
    3 x ``pair_mult``.  The power-wire scale has no weight decay."""
    ids = lambda ps: {id(p) for p in ps}                                              # noqa: E731
    scales = [model.gnd_layer_scale, model.gnd_fringe_scale, model.power_layer_scale]
    pair = [model.cpl_pair_scale]
    gnd_in = list(model.gnd_mlp[0].parameters()) + list(model.gnd_mlp[1].parameters())
    cpl = list(model.cpl_mlp.parameters()) + list(model.cpl_edge_proj.parameters())
    taken = ids(scales) | ids(pair) | ids(gnd_in) | ids(cpl)
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in taken]
    groups = [dict(params=rest, lr=lr, weight_decay=1e-4),
              dict(params=cpl, lr=lr * 3.0, weight_decay=1e-4),
              dict(params=scales, lr=lr * layer_mult, weight_decay=0.0),
              dict(params=pair, lr=lr * 3.0 * pair_mult, weight_decay=1e-4),
              dict(params=gnd_in, lr=lr * layer_mult, weight_decay=1e-4)]
    return [dict(g, params=[p for p in g["params"] if p.requires_grad]) for g in groups if any(p.requires_grad for p in g["params"])]


def train_heads(model, batches, steps, lr, layer_mult=1.0, pair_mult=1.0, floor=0.005, clip=1.0):
    """Minimal training loop of the heads (encoder frozen).  batches: iterator of batch dicts (see module docstring)."""
    model.freeze_encoder()
    opt = torch.optim.AdamW(param_groups(model, lr, layer_mult, pair_mult), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=1e-6)
    history = []
    for step, batch in zip(range(steps), batches):
        pred = model(batch["cuboids"], batch["padding_mask"], same_net_mask=batch.get("same_net_mask"),
                     n_tiles=batch.get("n_tiles"), endpoint_prox=batch.get("endpoint_prox"))
        loss, _ = training_loss(pred, batch, step, floor)
        if torch.isnan(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
        opt.step()
        sched.step()
        history.append(float(loss))
    return history
