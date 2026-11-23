"""Configuration: every technology constant comes from a YAML file (see configs/example.yaml); none is in the code."""
from types import SimpleNamespace

import numpy as np
import yaml


class Technology:
    """Layer stack view used by the model, the correction features and the RC network."""

    def __init__(self, tech):
        self.layers = [dict(L) for L in tech["layers"]]
        for i, L in enumerate(self.layers):
            L["index"] = i
            L["z_top"] = L["z_bottom"] + L["thickness"]
            L["z_center"] = L["z_bottom"] + 0.5 * L["thickness"]
        self.routing = [L for L in self.layers if L["kind"] == "routing"]
        self.vias = [L for L in self.layers if L["kind"] == "via"]
        self.by_name = {L["name"].lower(): L for L in self.layers}
        self.default_eps_r = float(tech.get("default_eps_r", 1.0))
        self.power_net_names = {str(n).lower() for n in tech.get("power_net_names", [])}

    # ---- vertical lookups ---------------------------------------------------------------------------------------
    def routing_z(self):
        """z centers of the routing layers, bottom to top (the anchors of the per-layer scales)."""
        return [L["z_center"] for L in self.routing]

    def fringe_init(self):
        """Initial sidewall share of the ground base per routing layer (0..1)."""
        return [float(L.get("fringe_init", 0.1)) for L in self.routing]

    def permittivity(self, z):
        """Relative permittivity at height z (array)."""
        z = np.atleast_1d(np.asarray(z, dtype=np.float64))
        eps = np.full(z.shape, self.default_eps_r)
        for L in self.layers:
            eps[(z >= L["z_bottom"]) & (z < L["z_top"])] = float(L["eps_r"])
        return eps

    def layer_of_z(self, z, kind=None):
        """Layer whose center is nearest to z (optionally among one kind)."""
        cand = self.layers if kind is None else [L for L in self.layers if L["kind"] == kind]
        return min(cand, key=lambda L: abs(L["z_center"] - z))

    def is_power(self, net_name):
        return str(net_name).lower() in self.power_net_names

    def via_between(self, lower, upper):
        """Via layer between two routing layers given by name (None if the stack lists none)."""
        a, b = self.by_name[lower.lower()], self.by_name[upper.lower()]
        lo, hi = (a, b) if a["index"] < b["index"] else (b, a)
        for L in self.layers[lo["index"] + 1:hi["index"]]:
            if L["kind"] == "via":
                return L
        return None


def load_config(path):
    """YAML -> namespace with .technology (Technology) and .window / .model / .training / .correction / .rc_network."""
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    cfg = SimpleNamespace(**{k: SimpleNamespace(**v) for k, v in raw.items() if k != "technology"})
    cfg.technology = Technology(raw["technology"])
    cfg.raw = raw
    return cfg
