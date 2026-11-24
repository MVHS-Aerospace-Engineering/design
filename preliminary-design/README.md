# Preliminary Design Phase

The configuration is fixed here. We take `../conceptual-design/outputs/design_point.json` and validate it instead of exploring.

## What's here now

`visualize.py` builds an AeroSandbox `Airplane` from the design point, with control surfaces (ailerons, elevator, rudder) ready for stability work. It writes:

- `outputs/three_view.png` and `outputs/isometric.png`
- `outputs/airplane.html`: an interactive 3-D view (open it in a browser; it loads plotly from the internet)

It also prints a first cross-check: AeroSandbox's AeroBuildup drag vs. the conceptual drag at the loiter point.

```bash
.venv/bin/python conceptual-design/conceptual.py --evaluate   # refresh design_point.json (no search)
.venv/bin/python preliminary-design/visualize.py
```

The landing gear is drawn but not included in AeroBuildup, which has no gear model. The CG is not computed yet, so the main-gear position is a placeholder.

## Plan

1. **AeroSandbox optimization**, seeded from the design point
   - Build `asb.Airplane` from the JSON geometry (wing, tails, fuselage, nacelles).
   - Use `asb.AeroBuildup` / vortex-lattice for trimmed aerodynamics (trim drag and propwash, which conceptual design ignored).
   - Re-optimize with `asb.Opti`, keeping every conceptual-design constraint (ground rolls, endurance, span, currents).
   - If W_TO, wing area, or power differ by more than ~20% from the conceptual numbers, find out why before trusting either.
2. **Stability & control**
   - CG location from component positions; static margin 5–15%.
   - Tail sizing (replaces the Raymer tail-volume coefficients) and control-surface/servo sizing from hinge moments.
   - One-motor-out: confirm the rudder holds V_mc ≤ 1.2 V_stall, or set up a flight-controller cutoff.
3. **Testing that replaces estimates**
   - Thrust stand: motor + prop + ESC efficiency and current vs. the model.
   - Grass rolling-friction tow test.
   - Weigh printed test panels, props, wheels, and the avionics stack; measure avionics power.
