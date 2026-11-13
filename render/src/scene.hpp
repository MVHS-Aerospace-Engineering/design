// Scene description produced by `python -m twin_rc.scene` (format "twin_rc.scene/1").
#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "math.hpp"
#include "mesh.hpp"

namespace twr {

struct Hinge {
    Vec3 origin;
    Vec3 axis;  // positive rotation = trailing edge down (horizontal) / right (rudder)
};

struct Part {
    std::string name;
    Vec3 color{0.8f, 0.8f, 0.8f};
    Mesh mesh;
    std::string control;  // "" = fixed part; else e.g. "aileron", "rudder", "stab_L"
    float gain = 1.0f;     // multiplies the control deflection (ailerons: +1 right, -1 left)
    Hinge hinge;
    std::string source;  // "inline" or the mesh file it came from (CAD)
};

struct Prop {
    std::string name;
    Vec3 hub;
    Vec3 thrust_dir{-1, 0, 0};
    float radius = 0.127f;
    int blades = 2;
    float pitch = 0.1524f;  // m per revolution
    int spin = 1;  // right-hand sense about thrust_dir
};

struct Marker {
    std::string name;
    Vec3 position;
    Vec3 color{1, 1, 0};
};

struct SimInfo {
    bool present = false;
    float cruise_speed = 0, trim_alpha_deg = 0, trim_stab_deg = 0;
    float L_over_D = 0, static_margin = 0, v_stall = 0, vmc = 0, mass_kg = 0;
};

struct Scene {
    std::string name;
    std::string path;
    std::vector<Part> parts;
    std::vector<Prop> props;
    std::vector<Marker> markers;
    std::map<std::string, float> control_limits_deg;
    SimInfo sim;

    Vec3 bounds_min() const;
    Vec3 bounds_max() const;
    std::optional<Vec3> marker(const std::string& name) const;
};

// Throws std::runtime_error with a readable message on bad input.
Scene load_scene(const std::string& path);

}  // namespace twr
