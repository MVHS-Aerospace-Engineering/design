# twin_rc_render

C++/OpenGL renderer for the plane in `twin_rc/`. Python exports the geometry to
`scenes/twin_rc.json` (`python -m twin_rc.scene`). The renderer draws it with
shadows, movable control surfaces and spinning props, and reloads automatically
when the scene file changes.

```sh
cmake -S render -B render/build && cmake --build render/build -j
./render/build/twin_rc_render                      # interactive window
./render/build/twin_rc_render --screenshot out.png --trim --spin --view side
```

Run `--help` for every option and key binding. W/S, A/D and Q/E move the
stabilators, roll controls and rudder; T poses the plane at the simulated
cruise trim. V (or `--xray`) makes the fuselage and nacelles see-through to show
the Pi 5, flight controller, battery, motors and ESCs inside.

**CAD:** a scene part can load a mesh file instead of inline geometry, e.g.
`{"name": "Wing", "file": "cad/wing.stl", "scale": 0.001, "color": [1,1,1]}`.
STL (ASCII or binary) and OBJ are supported, and the path is relative to the
scene file. Add `control` + `hinge` to a part to make it movable.
