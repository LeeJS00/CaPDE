"""BasisCap: learned capacitance extractor on cuboid windows (paper Section III-D).

Ground and coupling capacitances start from closed-form bases (plate and fringe terms over facing areas, power rails
included, weighted by gap permittivity; a shielding factor on couplings), scaled per layer and per layer pair.  A network
applies a bounded multiplicative correction to each base and adds a nonnegative term; the output heads keep every value
positive.  The encoder embeds each cuboid with a perceptron and adds a fixed inverse-distance average of its neighbours.
The ground head also reads the net's tile count (log) and the window's position in the net's window sequence.

Units: micrometre, fF.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import windows as Wn
from .shielding import shield_factor

EPS_0 = 0.008854          # vacuum permittivity, fF / um
GAP_FLOOR = 0.005         # added to every gap before a division (um)
MIN_BASE = 1e-5           # coupling bases below this (fF) are dropped
N_EDGE_GEOM = 12


def bounded_exp(x, limit):
    """Bounded multiplicative correction in [exp(-limit), exp(limit)] with a nonzero gradient everywhere; 1 at x = 0."""
    return torch.exp(limit * torch.tanh(x / limit))


def _softplus_inverse(y):
    return math.log(math.expm1(y))


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class CuboidEncoder(nn.Module):
    """Perceptron on the normalized cuboid channels."""

    def __init__(self, in_dim, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))

    def forward(self, x):
        return self.net(x)


class BasisCap(nn.Module):
    """forward(cuboids, padding_mask) -> per-cuboid ground capacitance and sparse couplings (see ``forward``)."""

    def __init__(self, tech, window, model):
        super().__init__()
        dim = int(model.model_dim)
        self.limit = float(model.modifier_limit)
        self.residual_scale = float(model.residual_scale)
        self.context_r = float(window.context_radius)
        self.cutoff_r = float(window.cutoff_radius)
        self.xy_scale = float(window.xy_scale)
        self.wide_width = float(window.wide_width)

        self.encoder = CuboidEncoder(Wn.N_CHANNELS, dim)
        z_dim = dim + 4                                   # embedding + four inverse-distance context channels
        self.norm = nn.LayerNorm(z_dim)

        # heads
        self.gnd_mlp = nn.Sequential(nn.LayerNorm(z_dim + 6), nn.Linear(z_dim + 6, 64), nn.GELU(), nn.Linear(64, 1))
        self.cpl_edge_proj = nn.Linear(z_dim, 32)
        cpl_in = 32 * 3 + N_EDGE_GEOM
        self.edge_geom_norm = nn.LayerNorm(N_EDGE_GEOM)
        self.cpl_mlp = nn.Sequential(nn.LayerNorm(cpl_in), nn.Linear(cpl_in, 256), nn.GELU(), nn.Linear(256, 2))
        # label-free pretraining head: per-cuboid charge and equivalent radius (basiscap.pretrain)
        self.charge_mlp = nn.Sequential(nn.LayerNorm(z_dim), nn.Linear(z_dim, 32), nn.GELU(), nn.Linear(32, 2))

        # per-layer and per-layer-pair scales of the bases (anchors = routing-layer heights of the technology)
        self.register_buffer("layer_z", torch.tensor(tech.routing_z(), dtype=torch.float32))
        k = int(self.layer_z.numel())
        self.gnd_layer_scale = nn.Parameter(torch.zeros(k))                                   # softplus -> fF / um^2
        self.gnd_fringe_scale = nn.Parameter(torch.tensor([_logit(p) for p in tech.fringe_init()], dtype=torch.float32))
        self.power_layer_scale = nn.Parameter(torch.full((k,), _softplus_inverse(0.15)))      # softplus -> scale of the power-wire base
        self.cpl_pair_scale = nn.Parameter(torch.zeros(k, k))                                 # symmetrized, softplus
        self._init_heads()

    def _init_heads(self):
        """Heads start at the identity of the bases: multiplicative corrections 1, additive term ~ 0."""
        for mlp in (self.cpl_mlp, self.gnd_mlp, self.charge_mlp):
            nn.init.normal_(mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.gnd_mlp[-1].bias)
        nn.init.zeros_(self.charge_mlp[-1].bias)
        with torch.no_grad():
            self.cpl_mlp[-1].bias.copy_(torch.tensor([0.0, -4.0]))

    def freeze_encoder(self):
        """After pretraining: freeze the encoder (and its context normalization) and train the heads and scales."""
        self._init_heads()
        for p in list(self.encoder.parameters()) + list(self.norm.parameters()):
            p.requires_grad = False

    # ------------------------------------------------------------------------------------------------------------
    def _layer_index(self, z):
        return torch.argmin(torch.abs(z.unsqueeze(-1) - self.layer_z), dim=-1)

    def embed(self, cuboids, padding_mask):
        """Encoder: perceptron embedding of each cuboid plus fixed (not learned) inverse-distance averages over its
        neighbours.  Returns a dict of the embedding and the geometry tensors shared by the heads."""
        B, N, _ = cuboids.shape
        x = cuboids.clone()
        x[..., :3] = x[..., :3] / self.xy_scale
        x[..., 3:6] = torch.log1p(torch.clamp(cuboids[..., 3:6], min=0.0)) / 1.5
        x[..., Wn.EPS] = torch.log(torch.clamp(cuboids[..., Wn.EPS], min=1.0)) / 2.0
        feats = self.encoder(x)

        valid = (~padding_mask).float()
        centers = cuboids[..., :3].clone()
        centers[padding_mask] = 1e5
        sizes = cuboids[..., 3:6] * valid.unsqueeze(-1)
        mins, maxs = centers - sizes / 2.0, centers + sizes / 2.0
        w, h, d = sizes[..., 0], sizes[..., 1], sizes[..., 2]
        eps = torch.clamp(cuboids[..., Wn.EPS], min=1.0) * valid
        is_target = (cuboids[..., Wn.ROLE] == 1.0).float() * valid
        is_aggr = (cuboids[..., Wn.ROLE] == 0.0).float() * valid
        is_power = (cuboids[..., Wn.NET_TYPE] > 0.5).float() * valid
        area = torch.clamp(2.0 * (w * h + h * d + d * w), min=1e-6).unsqueeze(-1)

        gap = torch.clamp(torch.max(mins.unsqueeze(2) - maxs.unsqueeze(1), mins.unsqueeze(1) - maxs.unsqueeze(2)), min=0.0)
        dist = torch.norm(gap, dim=-1)                                                # surface-to-surface distance (B, N, N)
        near = ((dist < self.context_r) & ~torch.eye(N, dtype=torch.bool, device=cuboids.device).unsqueeze(0)).float()
        inv_d = near / torch.clamp(dist, min=0.1)
        inv_d2 = near / torch.clamp(dist ** 2, min=0.01)

        tgt_area = is_target.unsqueeze(-1) * area
        density = torch.log1p(torch.bmm(inv_d2, area) * 0.01)                         # all conductors around
        ambient = torch.log1p(torch.bmm(inv_d, is_aggr.unsqueeze(-1) * area) * 0.01)  # neighbours around
        own = torch.log1p(torch.bmm(inv_d, tgt_area) * 0.01)                          # target net around
        share = torch.log1p(area / (tgt_area.sum(dim=1, keepdim=True) + 1e-6))        # cuboid area / target area
        z_emb = self.norm(torch.cat([feats, density, ambient, own, share], dim=-1))

        # power wires near each cuboid (grounded conductors on which part of the ground capacitance ends); a window
        # without power-wire cuboids takes its neighbours wider than wide_width as power wires
        has_power = is_power.sum(dim=1, keepdim=True) > 0
        power_signal = torch.where(has_power, is_power, (w > self.wide_width).float() * is_aggr)
        power_density = torch.log1p(torch.bmm(inv_d, (w * h * power_signal).unsqueeze(-1)) * 0.1)
        return dict(z=z_emb, mins=mins, maxs=maxs, sizes=sizes, eps=eps, area=area, dist=dist, valid=valid, is_target=is_target,
                    is_aggr=is_aggr, is_power=is_power, power_density=power_density)

    @staticmethod
    def _pair_geometry(g, b, s, d):
        """Gaps and facing overlaps of cuboid pairs (b, s) - (b, d)."""
        s_min, s_max, d_min, d_max = g["mins"][b, s], g["maxs"][b, s], g["mins"][b, d], g["maxs"][b, d]
        gap = torch.clamp(torch.maximum(s_min, d_min) - torch.minimum(s_max, d_max), min=0.0)
        ov = torch.clamp(torch.minimum(s_max, d_max) - torch.maximum(s_min, d_min), min=0.0)
        return gap, ov

    def ground(self, cuboids, g, n_tiles=None, tile_position=None):
        """Ground capacitance per target cuboid = scaled plate / fringe base x bounded correction + power-wire terms.
        n_tiles, tile_position (B,): number of windows of the window's net and the window's position in that sequence
        (``windows.tile_position``); zeros when not given."""
        B, N, _ = cuboids.shape
        z = cuboids[..., Wn.Z]
        w, h, d = g["sizes"][..., 0], g["sizes"][..., 1], g["sizes"][..., 2]
        li = self._layer_index(z)
        base = ((w * h).clamp(min=1e-6) + torch.sigmoid(self.gnd_fringe_scale[li]) * 2.0 * (w + h) * d) * F.softplus(self.gnd_layer_scale[li])

        dtype = g["z"].dtype
        zeros = torch.zeros(B, N, 1, device=cuboids.device, dtype=dtype)
        log_n = zeros if n_tiles is None else torch.log1p(n_tiles.to(dtype)).view(B, 1, 1).expand(B, N, 1)
        pos = zeros if tile_position is None else tile_position.to(dtype).view(B, 1, 1).expand(B, N, 1)
        feats = torch.cat([g["z"], torch.log1p(g["area"]), torch.log1p(g["eps"].unsqueeze(-1)), torch.clamp(z, min=0.0).unsqueeze(-1) / 10.0,
                           g["power_density"], log_n, pos], dim=-1)
        c_gnd = base * bounded_exp(self.gnd_mlp(feats).squeeze(-1), self.limit) * g["is_target"]

        # target -> power-wire bases: part of the ground capacitance ends on the power wires
        edges = ((g["is_target"] > 0.5).unsqueeze(2) & (g["is_power"] > 0.5).unsqueeze(1) & (g["dist"] < self.cutoff_r)).nonzero(as_tuple=False)
        if len(edges) > 0:
            b, s, p = edges[:, 0], edges[:, 1], edges[:, 2]
            gap, ov = self._pair_geometry(g, b, s, p)
            dist = torch.clamp(torch.norm(gap, dim=-1), min=1e-4) + GAP_FLOOR
            a_face = torch.maximum(torch.maximum(ov[:, 0] * ov[:, 1], ov[:, 1] * ov[:, 2]), ov[:, 2] * ov[:, 0])
            length = ov.max(dim=-1)[0]
            thick = torch.minimum(g["sizes"][b, s, 2], g["sizes"][b, p, 2])
            eps = (g["eps"][b, s] + g["eps"][b, p]) / 2.0
            c_pw = EPS_0 * eps * (a_face / dist + length * torch.log1p(thick / dist) * (2.0 / math.pi))
            c_pw = c_pw * F.softplus(self.power_layer_scale[self._layer_index(z[b, s])])
            add = torch.zeros(B * N, device=cuboids.device, dtype=c_pw.dtype).index_add(0, b * N + s, c_pw)
            c_gnd = c_gnd + add.view(B, N) * g["is_target"]
        return c_gnd

    def coupling(self, cuboids, g, same_net_mask=None):
        """Sparse couplings target cuboid -> neighbour cuboid = scaled base x bounded correction + nonnegative term."""
        z = cuboids[..., Wn.Z]
        dst = (g["is_aggr"] * (1.0 - g["is_power"])) > 0.5              # power wires end ground capacitance, not couplings
        if same_net_mask is not None:                                   # other pieces of the target net cannot couple to it
            dst = dst & (same_net_mask < 0.5)
        edges = ((g["is_target"] > 0.5).unsqueeze(2) & dst.unsqueeze(1) & (g["dist"] < self.cutoff_r)).nonzero(as_tuple=False)
        empty = dict(b_idx=edges[:0, 0], src_idx=edges[:0, 0], dst_idx=edges[:0, 0], c_cpl=torch.zeros(0, device=cuboids.device),
                     base=torch.zeros(0, device=cuboids.device), shield=torch.zeros(0, device=cuboids.device))
        if len(edges) == 0:
            return empty
        b, s, a = edges[:, 0], edges[:, 1], edges[:, 2]
        gap, ov = self._pair_geometry(g, b, s, a)
        d_surf = torch.clamp(torch.norm(gap, dim=-1), min=1e-4)
        a_xy = ov[:, 0] * ov[:, 1]                                      # broadside (between layers)
        a_lat = torch.maximum(ov[:, 0] * ov[:, 2], ov[:, 1] * ov[:, 2])  # lateral (same layer)
        d_lat = torch.clamp(torch.sqrt(gap[:, 0] ** 2 + gap[:, 1] ** 2), min=1e-4)
        length = ov.max(dim=-1)[0]
        thick = torch.minimum(g["sizes"][b, s, 2], g["sizes"][b, a, 2])
        e_s, e_a = g["eps"][b, s].clamp(min=1.0), g["eps"][b, a].clamp(min=1.0)
        e_arith, e_harm = (e_s + e_a) / 2.0, 2.0 / (1.0 / e_s + 1.0 / e_a)
        cos2 = (gap[:, 2] / d_surf.clamp(min=1e-6)).clamp(0.0, 1.0) ** 2
        e_pair = e_harm * cos2 + e_arith * (1.0 - cos2)

        shield = shield_factor(cuboids, b, s, a).view(-1)
        base_lat = EPS_0 * e_arith * (a_lat / (d_lat + GAP_FLOOR)
                                      + (a_lat > 1e-6).float() * length * torch.log1p(thick / (d_lat + GAP_FLOOR)) * (2.0 / math.pi))
        base_brd = EPS_0 * e_harm * a_xy / (gap[:, 2] + GAP_FLOOR)
        base = (base_lat + base_brd) * torch.clamp(shield, min=0.01)

        keep = base > MIN_BASE                                          # the bases fix which pairs couple
        if not keep.any():
            return empty
        (b, s, a, d_surf, a_xy, a_lat, d_lat, length, thick, e_pair, shield, base_lat, base_brd, base) = (
            t[keep] for t in (b, s, a, d_surf, a_xy, a_lat, d_lat, length, thick, e_pair, shield, base_lat, base_brd, base))
        gz, perim = gap[keep][:, 2], 2.0 * ov[keep].sum(dim=-1)

        # neighbours closer to the source than this one (occlusion rank)
        d_src = g["dist"][b, s]
        signal_aggr = (g["is_aggr"] * (1.0 - g["is_power"]))[b] > 0.5
        rank = ((d_src < d_src.gather(1, a.unsqueeze(1))) & signal_aggr).sum(dim=1).float()

        geom = torch.stack([d_surf, torch.log1p(a_lat), torch.log1p(a_xy), gz, e_pair, torch.log1p(base_lat), torch.log1p(base_brd), shield,
                            torch.log1p(rank), torch.log1p(thick), d_lat, torch.log1p(perim)], dim=-1)
        geom = self.edge_geom_norm(geom.to(g["z"].dtype))
        zi, zj = self.cpl_edge_proj(g["z"][b, s]), self.cpl_edge_proj(g["z"][b, a])
        logits = self.cpl_mlp(torch.cat([zi + zj, zi * zj, torch.abs(zi - zj), geom], dim=-1))

        sym = (self.cpl_pair_scale + self.cpl_pair_scale.T) / 2.0
        pair_scale = F.softplus(sym[self._layer_index(z[b, s]), self._layer_index(z[b, a])])
        c_cpl = base * pair_scale * bounded_exp(logits[:, 0], self.limit) + F.softplus(logits[:, 1]) * self.residual_scale
        return dict(b_idx=b, src_idx=s, dst_idx=a, c_cpl=c_cpl, base=base, shield=shield)

    def forward(self, cuboids, padding_mask, same_net_mask=None, n_tiles=None, tile_position=None, pretrain=False):
        """cuboids (B, N, 10), padding_mask (B, N); same_net_mask (B, N) marks neighbour rows that belong to the target net;
        n_tiles, tile_position (B,) are the inputs of the ground head described in ``ground`` (given in training and
        extraction alike).

        Returns dict:
            c_gnd   (B, N)  ground capacitance of every target cuboid (0 elsewhere), fF
            cpl     dict of edge tensors b_idx, src_idx (target cuboid), dst_idx (neighbour cuboid), c_cpl (fF), base, shield
            c_total (B, N)  ground + sum of the cuboid's couplings
        With pretrain=True only the encoder and the charge head run: q (B, N) charge of target cuboids, radius (B, N)."""
        g = self.embed(cuboids, padding_mask)
        if pretrain:
            out = self.charge_mlp(g["z"])
            return dict(q=out[..., 0] * g["is_target"], radius=F.softplus(out[..., 1]) * 0.1 + GAP_FLOOR)
        B, N, _ = cuboids.shape
        c_gnd = self.ground(cuboids, g, n_tiles, tile_position)
        cpl = self.coupling(cuboids, g, same_net_mask)
        c_sum = torch.zeros(B * N, device=cuboids.device, dtype=torch.float32)
        if len(cpl["b_idx"]) > 0:
            c_sum = c_sum.index_add(0, cpl["b_idx"] * N + cpl["src_idx"], cpl["c_cpl"].float())
        return dict(c_gnd=c_gnd, cpl=cpl, c_total=c_gnd.float() + c_sum.view(B, N) * g["is_target"])
