# CaPDE and BasisCap — method code

Released for double-blind review. This repository holds the method code of the paper: the CaPDE slack estimate and
its pass / fail / defer decision, and BasisCap, the learned extractor with its own RC network that supplies the layout
delta.

It contains code only. No layout, netlist, parasitic file, cell library, technology file, trained model or result is
included, and no technology constant is in the code: every one is a key of a configuration file
(`configs/example.yaml` shows the keys with invented placeholder values).

## Paper section / equation → file

| Paper | What | File |
|---|---|---|
| Sec. II, Eq. (1) | field-solver slack s\* (more pessimistic of two field-solver runs), violator, missed / false violation, coverage | `capde/estimate.py` `reference_slack`, `capde/interval.py` `counts`, `coverage` |
| Sec. III-A, Eq. (2) | CaPDE estimate = reference back-annotation + layout delta | `capde/estimate.py` `capde_estimate`, `layout_delta` |
| Sec. III-A | cell part and wire part of the slack shift of a sizing iteration | `capde/estimate.py` `cell_and_wire_part` |
| Sec. III-A, Eqs. (3)–(4) | error identity, error correlation ρ and the condition for a lower MSE | `capde/estimate.py` `error_correlation` |
| Sec. III-A | three comparison estimators: back-annotation alone, carried offset, learned extraction alone | `capde/estimate.py` `back_annotation_alone`, `carried_offset`, `learned_extraction_alone`, `estimator_family` |
| Sec. III-B, Eq. (5) | fixed-width split-conformal half-width b, three-way decision | `capde/interval.py` `fit_half_width`, `decide`, `labels` |
| Sec. III-C | deferral policy: default fail; candidate nets and a share β for a partial field-solver re-extraction; re-decision with the half-width b_β | `capde/deferral.py`, `capde/interval.py` `redecide` |
| Sec. III-D | cuboid windows (net, neighbours, power wires); each window's position in its net's window sequence | `basiscap/windows.py` (`tile_position`) |
| Sec. III-D | closed-form plate / fringe bases, per-layer and layer-pair scales, bounded multiplicative correction + nonnegative term, positive outputs; the ground head also reads the net's tile count (log) and the window position | `basiscap/model.py` `BasisCap.ground`, `BasisCap.coupling` |
| Sec. III-D | shielding factor on couplings | `basiscap/shielding.py` |
| Sec. III-D | encoder: perceptron + fixed inverse-distance average of neighbours; power-wire density (in a window without power-wire cuboids, neighbours wider than `wide_width` act as power wires) | `basiscap/model.py` `CuboidEncoder`, `BasisCap.embed` |
| Sec. III-D | label-free electrostatic pretraining, then frozen encoder | `basiscap/pretrain.py`, `BasisCap.freeze_encoder` |
| Sec. III-D | head training on relative and log-space errors against net, ground and coupling capacitances | `basiscap/losses.py` |
| Sec. III-D | two-stage final correction: gradient-boosted trees per layout tile, then a pair model; ground = net total − couplings; symmetric couplings | `basiscap/correction/features.py`, `models.py`, `apply.py` |
| Sec. III-D | RC network (split at junctions, vias, pins and a maximum length; segment resistance from sheet resistance and shape with a width bias; one constant multiplies every resistance), capacitances at the nearest nodes, SPEF writer | `basiscap/rcnet.py`, `basiscap/spef.py` |
| Sec. IV-A | slack MSE, 95th-percentile slack error, decision counts and shares | `capde/estimate.py` `error_stats`, `capde/interval.py` `counts`, `rates` |
| Sec. IV-A | design-cluster bootstrap 95% CI (2,000 resamples), pooled ratios over extractor seeds, exact two-sided sign-flip test (uncorrected) | `capde/stats.py` |

## What a user must provide

1. **Per-endpoint slack tables** (`endpoint<TAB>slack`) of a signoff timer with crosstalk analysis.
   Per iteration: `s(N_L, G(B))`, `s(N_L, S(L))`, `s(N_L, S(B))` for the CaPDE estimate, and `s(N_B, G(B))`,
   `s(N_B, S(B))` in addition for the carried offset. For the interval fit and for evaluation: `s(N_L, G(L))` and
   `s(N_L, G'(L))` from two field-solver runs on L; on the interval-fit designs also `s(N_L, G'(B))` from a second
   field-solver run on B.
   N = netlist, G = field-solver parasitics, S = extractor parasitics, B = reference layout, L = optimized layout.
2. **A technology description** in the form of `configs/example.yaml`: layer heights, thicknesses, permittivities, sheet
   and via resistances, the wire width bias, power-net names. Take these from the technology files you are licensed to
   use.
3. **For BasisCap**: routed shapes of each net as cuboids / wires / vias / pins (from a routed layout), cut into windows,
   with the number of windows of each net and each window's index in that sequence; field-solver net, ground and
   coupling capacitances of the training designs.
4. **For a partial re-extraction**: the commands of your extractor and signoff timer; `capde/deferral.py` holds a
   generic template with placeholders.

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

# once, on the optimized layouts of the interval-fit designs (arrays over all their endpoints)
sstar_fit = estimate.reference_slack(fit["nl_gl"], fit["nl_gl2"])         # Eq. (1)
sb_fit = estimate.reference_slack(fit["nl_gb"], fit["nl_gb2"])            # reference at its more pessimistic run
resid = estimate.layout_delta(fit["nl_sl"], fit["nl_sb"]) - (sstar_fit - sb_fit)
b = interval.fit_half_width(np.abs(resid), alpha=0.05)                  # target coverage 0.95

decision = interval.labels(x, b, theta=0.0)                             # Eq. (5): pass / fail / defer
final = deferral.default_fail(decision)                                 # default deferral policy
```

BasisCap on windows (see `tests/test_basiscap.py`, `tests/test_rc_spef.py` for complete synthetic examples):

```python
from basiscap import BasisCap, load_config, pretrain, losses, rcnet, spef
cfg = load_config("configs/example.yaml")
model = BasisCap(cfg.technology, cfg.window, cfg.model)
pretrain.pretrain(model, pretrain.SurfaceSampler(cfg.technology, cfg.window.far_field_half_size), window_batches)
losses.train_heads(model, label_batches, cfg.training.steps, cfg.training.lr, cfg.training.layer_scale_lr_mult,
                   cfg.training.pair_scale_lr_mult, cfg.training.cap_floor)
pred = model(cuboids, padding_mask, same_net_mask, n_tiles, tile_position)   # ground + sparse couplings, all positive
```

The final correction is fit once on training designs: `correction.features.net_tables` builds the tile and pair
geometry features, `correction.models.fit_tile_total` fits the per-tile net capacitance, and
`correction.models.fit_pair_additive` and `fit_pair_model` fit the pair model. Extraction of a layout runs the model on
the windows of every net, places the cuboid capacitances on the net's RC network (`rcnet.cuboid_caps`,
`RCNetwork.place`), applies the final correction (`predict_tile_total`, `predict_pair_model`, then
`correction.apply.correct`, `coupling_threshold`, `symmetrize`) and writes the file (`spef.write_spef`). One frozen
model extracts both layouts of an iteration.

## Tests

Tests run on synthetic inputs without any technology file:

- `tests/test_capde.py` — the CaPDE estimate and the three comparison estimators, the error identity of Eq. (3), the
  half-width rank, the decision and its counts, deferral and candidate-net selection, and the statistics reproduce
  hand-computed values.
- `tests/test_basiscap.py` — forward pass on random windows returns positive capacitances; coupling falls when a
  neighbour moves away and equal geometry gives equal values; the ground head reads the tile count and window position,
  and only a neighbour wider than `wide_width` stands for a missing power wire; one pretraining step and one
  head-training step run.
- `tests/test_correction.py` — features, per-tile net capacitance fit / apply, pair model, ground remainder,
  threshold and symmetrization.
- `tests/test_rc_spef.py` — RC split, resistance, placement at nearest nodes, write / read round trip.

## Requirements

Python ≥ 3.9, `numpy`, `scipy`, `torch`, `lightgbm`, `pyyaml`, `pytest`. A signoff timer, a field solver and any other
extractor are external and not part of this repository.

## Layout

```
capde/        estimate.py  interval.py  deferral.py  stats.py
basiscap/     config.py  windows.py  shielding.py  model.py  pretrain.py  losses.py  rcnet.py  spef.py
              correction/  features.py  models.py  apply.py
configs/      example.yaml
tests/        test_capde.py  test_basiscap.py  test_correction.py  test_rc_spef.py
```
