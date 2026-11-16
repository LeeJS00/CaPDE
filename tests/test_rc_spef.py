"""RC network split, capacitance placement and the SPEF writer on a synthetic net."""
import os

import numpy as np
import torch

from basiscap import BasisCap, load_config, rcnet, spef, windows

CFG = load_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "example.yaml"))
TECH = CFG.technology
LA, LB = TECH.routing[0], TECH.routing[1]


def _shapes():
    """A driver pin, 25 um of wire on the first routing layer, a via, 5 um on the second layer, a load pin."""
    w = float(LA["min_width"])
    return [dict(type="pin", layer=LA["name"], rect=(-0.1, -w / 2, 0.1, w / 2), name="u1:Z", direction="O", cell="CELL_A"),
            dict(type="wire", layer=LA["name"], start=(0.0, 0.0), end=(25.0, 0.0), width=w),
            dict(type="via", lower=LA["name"], upper=LB["name"], pos=(25.0, 0.0), width=w),
            dict(type="wire", layer=LB["name"], start=(25.0, 0.0), end=(25.0, 5.0), width=w),
            dict(type="pin", layer=LB["name"], rect=(25.0 - w / 2, 4.9, 25.0 + w / 2, 5.1), name="u2:A", direction="I", cell="CELL_B")]


def test_rc_network():
    net = rcnet.RCNetwork("n1", _shapes(), TECH, CFG.rc_network)
    assert set(net.pins) == {"u1:Z", "u2:A"}
    groups = net.components()
    assert len(groups) == 1, "every pin and node of the net is in one connected component"
    xs = sorted(x for x, y, l in net.nodes.values() if l == LA["name"] and abs(y) < 1e-6)
    assert max(np.diff(xs)) <= float(CFG.rc_network.max_segment_length) + 1e-6, "no piece exceeds the maximum length"
    assert any(abs(x - 10.0) < 1e-6 for x in xs) and any(abs(x - 20.0) < 1e-6 for x in xs), "split points at the maximum length"
    wire_r = sum(r for a, b, r in net.resistors if r > rcnet.CONTACT_RES and net.nodes.get(a, (0, 0, ""))[2] == net.nodes.get(b, (0, 0, "x"))[2] == LA["name"])
    expect = float(LA["sheet_res"]) * 25.0 / float(LA["min_width"])
    assert 0.9 * expect < wire_r <= expect, "sheet resistance x length / width (ends shortened by half a width)"
    via_r = [r for a, b, r in net.resistors if a in net.nodes and b in net.nodes and net.nodes[a][2] != net.nodes[b][2]]
    assert via_r == [float(TECH.via_between(LA["name"], LB["name"])["via_res"])]


def test_place_and_spef():
    net = rcnet.RCNetwork("n1", _shapes(), TECH, CFG.rc_network)
    caps = net.place([dict(xyz=(1.0, 0.0, LA["z_center"]), gnd=0.3, cpl={"n2": 0.1}),
                      dict(xyz=(19.0, 0.0, LA["z_center"]), gnd=0.2, cpl={"n2": 0.05, "n3": 0.02}),
                      dict(xyz=(25.0, 4.0, LB["z_center"]), gnd=0.1, cpl={})])
    assert abs(sum(c for _, c in caps["ground"]) - 0.6) < 1e-9
    nodes_n2 = {n for n, a, _ in caps["coupling"] if a == "n2"}
    assert len(nodes_n2) == 2, "couplings sit at the nearest nodes along the net"
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_smoke.spef")      # written beside the test, removed below
    try:
        spef.write_spef(path, "top", [("in", "I")], {"n1": net}, {"n1": caps})
        back = spef.read_spef(path)
    finally:
        if os.path.exists(path):
            os.remove(path)
    assert abs(back["n1"]["total"] - 0.77) < 1e-6
    assert len(back["n1"]["resistors"]) == len(net.resistors)
    assert sorted(a for _, a, _ in back["n1"]["coupling"]) == ["n2", "n2", "n3"]


def test_model_to_network():
    """Window prediction -> cuboid capacitances -> nodes."""
    torch.manual_seed(0)
    CFG.model.model_dim = 32
    model = BasisCap(TECH, CFG.window, CFG.model).eval()
    w = float(LA["min_width"])
    boxes = np.array([[12.5, 0.0, LA["z_center"], 25.0, w, LA["thickness"]], [12.5, 2 * w, LA["z_center"], 25.0, w, LA["thickness"]]], dtype=np.float32)
    cub, mask = windows.pad_batch([windows.make_window(boxes, [1, 0], [0, 0], TECH, center=(12.5, 0.0))])
    with torch.no_grad():
        pred = model(cub, mask)
    entries = rcnet.cuboid_caps(pred, 0, boxes, ["n1", "n2"])
    assert len(entries) == 1 and entries[0]["gnd"] > 0 and entries[0]["cpl"]["n2"] > 0
    caps = rcnet.RCNetwork("n1", _shapes(), TECH, CFG.rc_network).place(entries)
    assert caps["ground"] and caps["coupling"][0][1] == "n2"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
