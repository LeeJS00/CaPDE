"""BasisCap on random synthetic windows: forward pass, label-free pretraining step, head-training step."""
import os

import numpy as np
import torch

from basiscap import BasisCap, load_config
from basiscap import losses, pretrain, windows

CFG = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "example.yaml"))


def _model():
    torch.manual_seed(0)
    CFG.model.model_dim = 32                      # small model for the smoke test
    return BasisCap(CFG.technology, CFG.window, CFG.model)


def _batch(n=3, seed=0):
    rng = np.random.default_rng(seed)
    wins = [windows.synthetic_window(CFG.technology, rng) for _ in range(n)]
    cub, mask = windows.pad_batch([w["window"] for w in wins])
    return wins, cub, mask


def test_forward_positive():
    model = _model().eval()
    wins, cub, mask = _batch()
    with torch.no_grad():
        out = model(cub, mask)
    tgt = (cub[..., windows.ROLE] == 1.0) & ~mask
    assert torch.all(out["c_gnd"][tgt] > 0), "ground capacitance of every target cuboid must be positive"
    assert torch.all(out["c_gnd"][~tgt] == 0)
    assert len(out["cpl"]["c_cpl"]) > 0 and torch.all(out["cpl"]["c_cpl"] > 0), "couplings must be positive"
    assert torch.all(out["c_total"][tgt] >= out["c_gnd"][tgt])
    b, s, d = out["cpl"]["b_idx"], out["cpl"]["src_idx"], out["cpl"]["dst_idx"]
    assert torch.all(cub[b, s, windows.ROLE] == 1.0) and torch.all(cub[b, d, windows.ROLE] == 0.0)
    assert torch.all(cub[b, d, windows.NET_TYPE] < 0.5), "no coupling ends on a power wire"


def test_response_to_a_moved_neighbour():
    """A wire that keeps its route still changes capacitance when a neighbour moves; equal geometry gives equal values."""
    model = _model().eval()
    tech = CFG.technology
    L = tech.routing[0]
    wd = float(L["min_width"])
    boxes = np.array([[0.0, 0.0, L["z_center"], 1.0, wd, L["thickness"]], [0.0, 2 * wd, L["z_center"], 1.0, wd, L["thickness"]]], dtype=np.float32)
    far = boxes.copy()
    far[1, 1] = 6 * wd
    outs = []
    for bx in (boxes, boxes, far):
        cub, mask = windows.pad_batch([windows.make_window(bx, [1, 0], [0, 0], tech)])
        with torch.no_grad():
            outs.append(float(model(cub, mask)["cpl"]["c_cpl"].sum()))
    assert outs[0] == outs[1]
    assert outs[2] < outs[0], "coupling must fall when the neighbour moves away"


def test_pretrain_step():
    model = _model()
    _, cub, mask = _batch()
    sampler = pretrain.SurfaceSampler(CFG.technology, CFG.window.far_field_half_size)
    loss = pretrain.pretrain_loss(model, sampler, cub, mask, n_points=64)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.encoder.parameters())
    hist = pretrain.pretrain(model, sampler, [(cub, mask)], epochs=2)
    assert len(hist) == 2 and all(np.isfinite(hist))


def test_training_step():
    model = _model()
    wins, cub, mask = _batch(n=4)
    net_id = torch.tensor([0, 0, 1, 1])
    A = 4
    slot = torch.full(cub.shape[:2], -1, dtype=torch.long)
    for i, w in enumerate(wins):
        for j, name in enumerate(w["net_names"]):
            if name.startswith("net_a"):
                slot[i, j] = int(name[-1])
    batch = dict(cuboids=cub, padding_mask=mask, net_id=net_id, own=(cub[..., windows.ROLE] == 1.0).float() * (~mask).float(),
                 aggr_slot=slot, y_total=torch.tensor([1.0, 2.0]), y_gnd=torch.tensor([0.6, 1.2]),
                 y_cpl=torch.tensor([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]]), y_cpl_valid=torch.ones(2, A, dtype=torch.bool))
    pred = model(cub, mask)
    gnd, cpl, total = losses.aggregate_to_nets(pred, batch)
    assert gnd.shape == (2,) and cpl.shape == (2, A) and torch.all(total >= gnd)
    loss, terms = losses.training_loss(pred, batch, step=2500)
    assert torch.isfinite(loss) and set(terms) == {"total", "cpl_log", "pair", "share", "gnd", "cpl"}
    hist = losses.train_heads(model, iter([batch] * 3), steps=3, lr=1e-3, layer_mult=50.0)
    assert len(hist) == 3 and all(np.isfinite(hist))
    assert not any(p.requires_grad for p in model.encoder.parameters()), "the encoder stays frozen"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
