"""Cuboid windows: the input of BasisCap (paper Section III-D).

For each net the extractor reads windows in which the net (target), its neighbours and the power wires are cuboids.
One cuboid is a row of 10 channels:

    0..2   x, y relative to the window center, z absolute (cuboid center)
    3..5   size along x, y, z
    6      shape kind: 1.0 wire, 0.5 pin
    7      role: 1.0 target net, 0.0 neighbour (aggressor)
    8      relative permittivity at the cuboid height
    9      net type: 0.0 signal, POWER (0.67) supply wire, GROUND (1.0) ground wire

A batch is (cuboids (B, N, 10), padding_mask (B, N) with True = padding).  Each window also carries the number of
windows of its net (n_tiles) and its position in that window sequence (``tile_position``), read by the ground head.
"""
import numpy as np
import torch

X, Y, Z, W, H, D, KIND, ROLE, EPS, NET_TYPE = range(10)
N_CHANNELS = 10
SIGNAL, POWER, GROUND = 0.0, 0.67, 1.0
WIRE, PIN = 1.0, 0.5


def make_window(boxes, is_target, net_type, tech, center=(0.0, 0.0), is_pin=None):
    """boxes: (n, 6) absolute cuboids (x, y, z center; w, h, d).  Returns the (n, 10) window tensor (numpy)."""
    boxes = np.asarray(boxes, dtype=np.float32)
    n = len(boxes)
    win = np.zeros((n, N_CHANNELS), dtype=np.float32)
    win[:, X] = boxes[:, 0] - center[0]
    win[:, Y] = boxes[:, 1] - center[1]
    win[:, Z] = boxes[:, 2]
    win[:, W:D + 1] = boxes[:, 3:6]
    win[:, KIND] = np.where(np.zeros(n, bool) if is_pin is None else np.asarray(is_pin, bool), PIN, WIRE)
    win[:, ROLE] = np.asarray(is_target, dtype=np.float32)
    win[:, EPS] = tech.permittivity(boxes[:, 2])
    win[:, NET_TYPE] = np.asarray(net_type, dtype=np.float32)
    return win


def tile_position(tile_index, n_tiles):
    """Position of window ``tile_index`` (0 .. n_tiles - 1) in its net's window sequence: min(i, n - 1 - i) / max(n / 2, 1),
    0 for the first and last window."""
    n = max(1, int(n_tiles))
    return float(min(tile_index, max(n - 1 - tile_index, 0)) / max(n / 2.0, 1.0))


def pad_batch(windows):
    """List of (n_i, 10) arrays -> (cuboids (B, N, 10), padding_mask (B, N)) torch tensors."""
    n = max(len(w) for w in windows)
    cub = torch.zeros(len(windows), n, N_CHANNELS)
    mask = torch.ones(len(windows), n, dtype=torch.bool)
    for i, w in enumerate(windows):
        cub[i, :len(w)] = torch.as_tensor(w)
        mask[i, :len(w)] = False
    return cub, mask


def synthetic_window(tech, rng, n_target=3, n_neighbour=8, n_power=2, half=1.0):
    """Random window on the routing layers of ``tech`` (for the tests; no layout is read).

    Returns dict(window (n, 10), abs_boxes (n, 6), net_names [n], target_net)."""
    boxes, role, ntype, names = [], [], [], []

    def wire(layer, name, target, typ):
        horizontal = rng.random() < 0.5
        length = rng.uniform(0.3, 1.5) * half
        width = float(layer.get("min_width", 0.05)) * rng.choice([1.0, 1.0, 2.0])
        cx, cy = rng.uniform(-half, half, 2)
        w, h = (length, width) if horizontal else (width, length)
        boxes.append([cx, cy, layer["z_center"], w, h, layer["thickness"]])
        role.append(1.0 if target else 0.0)
        ntype.append(typ)
        names.append(name)

    for _ in range(n_target):
        wire(tech.routing[rng.integers(len(tech.routing))], "net_t", True, SIGNAL)
    for i in range(n_neighbour):
        wire(tech.routing[rng.integers(len(tech.routing))], f"net_a{i % 4}", False, SIGNAL)
    supply = sorted(tech.power_net_names) or ["supply"]
    for i in range(n_power):
        wire(tech.routing[-1], supply[i % len(supply)], False, GROUND if i % 2 == 0 else POWER)
    boxes = np.array(boxes, dtype=np.float32)
    return dict(window=make_window(boxes, role, ntype, tech), abs_boxes=boxes, net_names=names, target_net="net_t")
