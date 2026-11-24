# MVHS Aerospace Engineering Club Aircraft Design

The overall design process of an aircraft can be split into 3 parts: conceptual design, detail design, and preliminary design.

## Conceptual Design

This is where a wide range of configurations are explored and where design concepts and top-level requirements are evaluated against one another. We will narrow down towards a single, well-balanced configuration. It is a very *fluid* process; nothing is frozen.

### Requirements & Constraints (these are not engineering descisions)

- The aircraft shall remain airborne for a minimum of 30 minutes with no payload.
- The aircraft shall be capable of taking off and landing in under 250 feet grass runways.
- The aircraft shall support full autonomous and remote-controlled operation.

**After defining the requirements and constraints, there are 2 main parts of conceputal design are sizing and synthesis. See how this is implemented in the conceptual design folder.**

### Sizing

The numerical process of calculating the specific physical magnitude of the aircraft, primarily its takeoff weight, wing area, and engine thrust or power, needed to perform a defined mission.

### Synthesis

Synthesis takes those calculated numbers and translates them into a cohesive geometric configuration, determining the actual layout, shape, and integration of all subsystems into a functional vehicle.

## Preliminary Design

This is where the aircraft starts to become **real**. We'll fix the configuration (wing planform, tail layout, propulsion, etc) from the conceptual design and vaildate designs rather than exploring them. Here is where we will use CFD (computational fluid dynamics), propulsion testing, and S&C (stability and control) analysis methods for the design. The goal is to use realistic assumptions to verify that the conceptual design will work.

## Detail Design

If all of us are satisfied with the preliminary design, we will enter the detail design phase. This focuses on the exact geometry, materials, the structural layouts, and system implementations. Every component must be designed to a manufacturable level. We sort out wiring, control systems, integration with the other subsystems, and more.

*Detail design happens in CAD, outside this repository. The code here only produces design parameters.*

## Repository Layout

```
design/
  conceptual-design/     sizing loop -> outputs/design_point.json   (see its README)
  preliminary-design/    AeroSandbox optimization + validation, seeded from design_point.json
  requirements.txt       Python packages
```

### Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python conceptual-design/conceptual.py        # ~3 min, uses all CPU cores
.venv/bin/python -m pytest conceptual-design/tests -q   # model verification checks
```
