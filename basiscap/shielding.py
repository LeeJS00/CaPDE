"""Shielding factor on a coupling base: how much of the flux between two cuboids other conductors intercept."""
import torch


@torch.no_grad()
def shield_factor(cuboids, b_idx, src_idx, dst_idx, chunk=8192):
    """Per edge (batch b, source cuboid, destination cuboid) a factor in (0, 1]; 1 = nothing in between.

    1. The corridor is the bounding box of the pair.  Every third cuboid contributes its overlap with the corridor as a
       blocked share of each of the three cross sections; shares of several blockers combine as 1 - prod(1 - a).
    2. The three cross sections are weighted by the direction from source to destination.
    3. A blocked corridor still leaks: the leak grows with the corridor's aspect ratio and with the distance of the
       first blocker hit by the center-to-center ray (slab test).
    """
    E = len(b_idx)
    if E == 0:
        return torch.empty(0, device=cuboids.device)
    B, N, _ = cuboids.shape
    step = max(512, int(chunk * (256 / max(N, 1))))
    mins = cuboids[..., :3] - cuboids[..., 3:6] / 2.0
    maxs = cuboids[..., :3] + cuboids[..., 3:6] / 2.0
    out = torch.zeros(E, device=cuboids.device)
    tiny = 1e-6
    for i in range(0, E, step):
        j = min(i + step, E)
        b, s, d = b_idx[i:j], src_idx[i:j], dst_idx[i:j]
        s_min, s_max, d_min, d_max = mins[b, s], maxs[b, s], mins[b, d], maxs[b, d]
        c_min, c_max = torch.min(s_min, d_min), torch.max(s_max, d_max)
        c_dim = torch.clamp(c_max - c_min, min=tiny)
        k_min, k_max = mins[b], maxs[b]                                              # (e, N, 3)
        overlap = torch.clamp(torch.min(c_max.unsqueeze(1), k_max) - torch.max(c_min.unsqueeze(1), k_min), min=0.0)
        k = torch.arange(N, device=cuboids.device).view(1, N).expand(j - i, N)
        other = (k != s.unsqueeze(1)) & (k != d.unsqueeze(1))
        r = overlap * other.unsqueeze(-1).float() / c_dim.unsqueeze(1)

        def union(a):
            a = torch.where(a < 1e-5, torch.zeros_like(a), a)
            return torch.clamp(1.0 - torch.exp(torch.sum(torch.log(1.0 - a + tiny), dim=1)), 0.0, 1.0)

        s_xy, s_yz, s_zx = union(r[..., 0] * r[..., 1]), union(r[..., 1] * r[..., 2]), union(r[..., 2] * r[..., 0])
        s_c, d_c = (s_min + s_max) / 2.0, (d_min + d_max) / 2.0
        delta = torch.abs(d_c - s_c)
        w = delta / torch.clamp(delta.sum(dim=1, keepdim=True), min=tiny)
        blocked = w[:, 0] * s_yz + w[:, 1] * s_zx + w[:, 2] * s_xy

        dist = torch.clamp(torch.norm(delta, dim=1), min=tiny)
        aspect = dist / torch.clamp(c_dim.max(dim=1)[0], min=tiny)
        bend = 1.0 - 1.0 / (1.0 + 0.5 * torch.log1p(aspect))
        ray = d_c - s_c
        ray_len = torch.clamp(torch.norm(ray, dim=-1), min=1e-9)
        inv = (1.0 / (ray / ray_len.unsqueeze(-1) + 1e-9)).unsqueeze(1)
        t0, t1 = (k_min - s_c.unsqueeze(1)) * inv, (k_max - s_c.unsqueeze(1)) * inv
        t_in = torch.max(torch.minimum(t0, t1), dim=-1)[0]
        t_out = torch.min(torch.maximum(t0, t1), dim=-1)[0]
        hit = (t_out > 0) & (t_in < t_out) & (t_in < ray_len.unsqueeze(-1)) & other
        first = torch.where(hit, torch.clamp(t_in, min=0.0), torch.full_like(t_in, 1e5)).min(dim=-1)[0]
        leak = torch.clamp(bend + (1.0 - torch.exp(-first / dist)) * 0.2, min=0.01, max=0.95)
        out[i:j] = 1.0 - blocked * (1.0 - leak) * 0.8
    return out
