# design
Concept design

## Twin high-wing RC plane (`twin_rc/`)

A simulation of a twin-engine, high-wing RC plane:

- **Two wing-mounted tractor motors**, each with a **2-blade propeller** (counter-rotating)
- **One rudder** on a single vertical fin
- **Two stabilators** (left and right all-moving horizontal tail halves) instead of
  a fixed stabiliser + elevator. Deflecting them together controls pitch;
  deflecting them differentially adds roll.

Built with [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox). Airfoil
polars come from NeuralFoil, and whole-aircraft forces from AeroBuildup. A vortex
lattice method cross-checks the neutral point. The geometry is exported to
[XFLR5](http://www.xflr5.tech/xflr5.htm) (and AVL), so you can repeat the
analysis there.

### Run it

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m twin_rc.run_all      # ~2 min, writes results/
```

Results: [`results/SUMMARY.md`](results/SUMMARY.md) (plots, performance,
stability, control power, one-engine-out Vmc) and `results/results.json`.

### Change the design

All dimensions, masses, the motor, the prop and the battery live in
[`twin_rc/config.py`](twin_rc/config.py). Edit them and re-run.

| Module | What it does |
|---|---|
| `config.py` | Design parameters |
| `geometry.py` | AeroSandbox airplane: wing + ailerons, two stabilator halves, fin + rudder, fuselage, nacelles, props |
| `mass.py` | Point-mass weight and CG build-up |
| `propulsion.py` | Motor (Kv / R / I0) + 2-blade prop (Ct, Cp vs J) operating-point solver |
| `analysis.py` | Trim (alpha + stabilator), stability and control derivatives, static margin, VLM cross-check, engine-out rudder / Vmc |
| `export.py` | XFLR5 XML, AVL file, airfoil `.dat` files |
| `run_all.py` | Runs everything and writes plots + summary |

### Check it in XFLR5

1. File -> Direct Foil Design: load `results/exports/clarky.dat` and `naca0009.dat`, then
   run a Batch Foil Analysis over the Reynolds range in the summary.
2. File -> Import -> Import from XML -> `results/exports/twin_rc_xflr5.xml`.
3. Run a T2 (fixed lift) analysis for the polar, and a T7 stability analysis with the
   tail tilt as the control variable for the stabilators.

XFLR5 has only one "elevator" object, so the two stabilators are exported as
one symmetric all-moving tail. The differential-stabilator and engine-out
cases are only in the AeroSandbox results.

### Limits

- Prop coefficients are fits to typical 2-blade 10x6 electric-prop data, not measured
  data for a specific prop.
- CLmax comes from 2D airfoil data run through AeroBuildup and is likely optimistic,
  so expect the real stall speed to be somewhat higher.
- The prop wash over the wing and tail isn't modelled.
