# CaPDE and BasisCap — method code

Released for double-blind review. This repository holds the method code of the paper: the CaPDE slack estimate and
its pass / fail / defer decision, and BasisCap, the physics-guided 3D learned extractor that supplies the layout delta.

It contains code only. No layout, netlist, parasitic file, cell library, technology file, trained model or result is
included, and no technology constant is in the code: every one is a key of a configuration file
(`configs/example.yaml` shows the keys with invented placeholder values).

## Paper section / equation → file

| Paper | What | File |
|---|---|---|
| Sec. II, Eq. (1) | reference slack s\* (more pessimistic of two field-solver runs), violator, missed / false violation, coverage | `capde/estimate.py` `reference_slack`, `capde/interval.py` `counts`, `coverage` |
| Sec. III-A, Eq. (2) | CaPDE estimate = reference back-annotation + layout delta | `capde/estimate.py` `capde_estimate`, `layout_delta` |
| Sec. III-A, Eqs. (3)–(4) | error identity, error correlation and the MSE condition | `capde/estimate.py` `error_correlation`, `error_stats` |
| Sec. III-A | comparison estimators (2 × 2 family): back-annotation alone, carried offset, learned extraction alone | `capde/estimate.py` `estimator_family` |
| Sec. III-B, Eq. (5) | fixed-width split-conformal half-width, three-way decision | `capde/interval.py` `fit_half_width`, `decide`, `counts`, `rates` |
| Sec. III-C | deferral policy: default fail; candidate nets for a partial field-solver re-extraction; re-decision with a per-budget half-width | `capde/deferral.py`, `capde/interval.py` `redecide` |
| Sec. III-D | cuboid windows (net, neighbours, power wires) | `basiscap/windows.py` |
| Sec. III-D | closed-form plate / fringe bases, per-layer and layer-pair scales, bounded multiplicative correction + nonnegative term, positive outputs | `basiscap/model.py` `BasisCap.ground`, `BasisCap.coupling` |
| Sec. III-D | shielding factor on couplings | `basiscap/shielding.py` |
| Sec. III-D | encoder: perceptron + fixed inverse-distance average of neighbours | `basiscap/model.py` `CuboidEncoder`, `BasisCap.embed` |
| Sec. III-D | label-free electrostatic pretraining | `basiscap/pretrain.py` |
| Sec. III-D | heads and training losses (relative and log-space errors on net, ground and coupling capacitance) | `basiscap/losses.py` |
| Sec. III-D | final correction: per-tile gradient-boosted net-total model, pair rescaling, ground = net total − couplings, coupling symmetrization | `basiscap/correction/features.py`, `models.py`, `apply.py` |
| Sec. III-D | RC network (split at junctions, vias, pins, maximum length; one resistance constant), capacitances at the nearest nodes, SPEF writer | `basiscap/rcnet.py`, `basiscap/spef.py` |
| Sec. IV | design-cluster bootstrap, exact two-sided sign-flip test, Holm adjustment | `capde/stats.py` |

## What a user must provide

1. **Per-endpoint slack tables** (`endpoint<TAB>slack`) of a signoff timer with crosstalk analysis, for one iteration:
   `s(N_L, G(B))`, `s(N_L, S(L))`, `s(N_L, S(B))` for the CaPDE estimate; `s(N_B, G(B))`, `s(N_B, S(B))` in addition
   for the carried offset. For fitting or evaluating an interval: `s(N_L, G(L))` of training / test layouts (and a
   repeated field-solver run for the pessimistic reference).
   N = netlist, G = field-solver parasitics, S = extractor parasitics, B = reference layout, L = optimized layout.
2. **A technology description** in the form of `configs/example.yaml`: layer heights, thicknesses, permittivities, sheet
   and via resistances, power-net names. Take these from the technology files you are licensed to use.
3. **For BasisCap**: routed shapes of each net as cuboids / wires / vias / pins (from a routed layout), and field-solver
   capacitance labels of training designs.
4. **For a partial re-extraction**: your extractor's and timer's commands; `capde/deferral.py` holds a generic template
   with placeholders.

## How to run

```bash
pip install -r requirements.txt
python -m pytest -q tests/            # or: PYTHONPATH=. python tests/test_capde.py  (each test file also runs alone)
```

Estimate and decision for one layout:

```python
import numpy as np
from capde import estimate, interval, deferral

tables = [estimate.read_slack_table(p) for p in ("nl_gb.tsv", "nl_sl.tsv", "nl_sb.tsv")]
endpoints, (s_nl_gb, s_nl_sl, s_nl_sb) = estimate.align(*tables)
x = estimate.capde_estimate(s_nl_gb, s_nl_sl, s_nl_sb)                 # Eq. (2)

b = interval.fit_half_width(np.abs(x_train - sstar_train), alpha=0.05)  # on training layouts
decision = interval.labels(x, b, theta=0.0)                             # Eq. (5): pass / fail / defer
final = deferral.default_fail(decision)                                 # default deferral policy
```

BasisCap on windows (see `tests/test_basiscap.py`, `tests/test_rc_spef.py` for complete synthetic examples):

```python
from basiscap import BasisCap, load_config, pretrain, losses, rcnet, spef
cfg = load_config("configs/example.yaml")
model = BasisCap(cfg.technology, cfg.window, cfg.model)
pretrain.pretrain(model, pretrain.SurfaceSampler(cfg.technology, cfg.window.far_field_half_size), window_batches)
losses.train_heads(model, label_batches, cfg.training.steps, cfg.training.lr, cfg.training.layer_scale_lr_mult)
pred = model(cuboids, padding_mask)                                      # ground + sparse couplings, all positive
```

Extraction of a layout runs the model on the windows of every net, places the cuboid capacitances on the net's RC
network (`rcnet.cuboid_caps`, `RCNetwork.place`), applies the final correction
(`correction.apply.correct`, `coupling_threshold`, `symmetrize`) and writes the file (`spef.write_spef`). One frozen
model extracts both layouts of an iteration.

## Tests

Smoke tests run on synthetic inputs without any technology file:

- `tests/test_capde.py` — the four estimators, the half-width rank, the decision and its counts, deferral selection and
  the statistics reproduce hand-computed values.
- `tests/test_basiscap.py` — forward pass on random windows returns positive capacitances; coupling falls when a
  neighbour moves away and equal geometry gives equal values; one pretraining step and one head-training step run.
- `tests/test_correction.py` — features, tile-additive net-total fit / apply, pair models, ground remainder,
  threshold and symmetrization.
- `tests/test_rc_spef.py` — RC split, resistance, placement at nearest nodes, write / read round trip.

## Requirements

Python ≥ 3.9, `numpy`, `scipy`, `torch`, `lightgbm`, `pyyaml`, `pytest`. A signoff timer, a field solver and any
commercial extractor are external and not part of this repository.

## Layout

```
capde/        estimate.py  interval.py  deferral.py  stats.py
basiscap/     config.py  windows.py  shielding.py  model.py  pretrain.py  losses.py  rcnet.py  spef.py
              correction/  features.py  models.py  apply.py
configs/      example.yaml
tests/        test_capde.py  test_basiscap.py  test_correction.py  test_rc_spef.py
```
