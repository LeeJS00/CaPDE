"""Label-free electrostatic pretraining of the encoder (paper Section III-D).

Each target cuboid carries a charge predicted from its embedding.  The potential of these cuboid charges is evaluated
at sampled points and must meet the boundary conditions: 1 on the surface of the target net, 0 on the surfaces of the
neighbours, 0 in the far field.  No capacitance label is used.  After pretraining the encoder is frozen
(``BasisCap.freeze_encoder``).
"""
import numpy as np
import torch
import torch.nn.functional as F

from . import windows as Wn

COULOMB = 8.987     # 1 / (4 pi eps_0) in the unit system of the potential target (um, fF)


class SurfaceSampler:
    """Samples boundary-condition points of a batch of windows."""

    def __init__(self, tech, far_field_half_size, device="cpu", n_bins=2000):
        zs = [L["z_bottom"] for L in tech.layers] + [L["z_top"] for L in tech.layers]
        lo, hi = min(zs) - far_field_half_size, max(zs) + far_field_half_size
        grid = np.linspace(lo, hi, n_bins)
        self.z_bins = torch.tensor(grid, dtype=torch.float32, device=device)
        self.eps_bins = torch.tensor(np.maximum(tech.permittivity(grid), 1.0), dtype=torch.float32, device=device)
        self.half = float(far_field_half_size)

    @staticmethod
    def _surface(cuboids, select, n):
        """n points on the surfaces of the selected cuboids, drawn by surface area.  Returns (points (B, n, 3), valid (B,))."""
        B = cuboids.shape[0]
        sizes = cuboids[..., 3:6] * select.unsqueeze(-1).float()
        centers = cuboids[..., :3]
        area = 2 * (sizes[..., 0] * sizes[..., 1] + sizes[..., 0] * sizes[..., 2] + sizes[..., 1] * sizes[..., 2])
        total = area.sum(dim=1)
        valid = total > 1e-9
        prob = area / torch.where(valid, total, torch.ones_like(total)).unsqueeze(1)
        prob[~valid, 0] = 1.0
        idx = torch.multinomial(prob, n, replacement=True)
        bi = torch.arange(B, device=cuboids.device).unsqueeze(1).expand(-1, n)
        u = (torch.rand(B, n, 3, device=cuboids.device) - 0.5) * 0.96
        axis = torch.abs(u).argmax(dim=-1, keepdim=True)
        sign = torch.sign(u.gather(-1, axis))
        onehot = torch.zeros_like(u).scatter_(-1, axis, 1.0)
        u = u * (1 - onehot) + sign * 0.5 * onehot                       # project to the nearest face
        pts = centers[bi, idx] + u * sizes[bi, idx] + onehot * sign * 1e-4
        return pts, valid

    def sample(self, cuboids, padding_mask, n=512):
        """Returns points (B, n, 3), kind (B, n): 1 target surface, 0 neighbour surface, -1 far field, -2 unused; eps (B, n, 1)."""
        B = cuboids.shape[0]
        n_t = n_a = int(n * 0.4)
        n_f = n - n_t - n_a
        tgt = (cuboids[..., Wn.ROLE] == 1.0) & ~padding_mask
        agg = (cuboids[..., Wn.ROLE] == 0.0) & ~padding_mask
        p_t, v_t = self._surface(cuboids, tgt, n_t)
        p_a, v_a = self._surface(cuboids, agg, n_a)
        p_f = (torch.rand(B, n_f, 3, device=cuboids.device) * 2 - 1) * self.half
        k_t = torch.ones(B, n_t, device=cuboids.device)
        k_t[~v_t] = -2.0
        k_a = torch.zeros(B, n_a, device=cuboids.device)
        k_a[~v_a] = -2.0
        pts = torch.cat([p_t, p_a, p_f], dim=1)
        kind = torch.cat([k_t, k_a, torch.full((B, n_f), -1.0, device=cuboids.device)], dim=1)
        eps = self.eps_bins[torch.bucketize(pts[..., 2].contiguous(), self.z_bins).clamp(0, len(self.eps_bins) - 1)].unsqueeze(-1)
        return pts, kind, eps


def potential(points, eps, cuboids, padding_mask, q, radius):
    """Potential of the cuboid charges at the points; the distance to a cuboid is floored by its equivalent radius."""
    centers = cuboids[..., :3].clone()
    centers[padding_mask] = 1e5
    dist = torch.maximum(torch.cdist(points, centers), radius.unsqueeze(1))
    return torch.sum(COULOMB * q.unsqueeze(1) / (torch.clamp(eps, min=1.0) * dist), dim=2)


def pretrain_loss(model, sampler, cuboids, padding_mask, n_points=512):
    """Boundary-condition loss: V = 1 on the target net, V = 0 on neighbours, V = 0 in the far field."""
    pts, kind, eps = sampler.sample(cuboids, padding_mask, n_points)
    out = model(cuboids, padding_mask, pretrain=True)
    v = potential(pts, eps, cuboids, padding_mask, out["q"], out["radius"])
    loss = 0.0
    for value, target in ((1.0, 1.0), (0.0, 0.0), (-1.0, 0.0)):
        m = (kind == value).float()
        loss = loss + F.smooth_l1_loss(v * m, target * m, reduction="sum") / torch.clamp(m.sum(), min=1.0)
    return loss


def pretrain(model, sampler, batches, epochs=1, lr=1e-4, max_cuboids=1024, clip=5.0):
    """Minimal pretraining loop.  batches: iterable of (cuboids, padding_mask), re-iterable per epoch."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=1e-6)
    history = []
    for _ in range(epochs):
        total, n = 0.0, 0
        for cuboids, mask in batches:
            cuboids, mask = cuboids[:, :max_cuboids], mask[:, :max_cuboids]
            loss = pretrain_loss(model, sampler, cuboids, mask)
            if torch.isnan(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total, n = total + float(loss), n + 1
        sched.step()
        history.append(total / max(n, 1))
    return history
