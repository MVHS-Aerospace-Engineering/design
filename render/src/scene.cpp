#include "scene.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace twr {

using json = nlohmann::json;
namespace fs = std::filesystem;

static Vec3 vec3(const json& j) {
    if (!j.is_array() || j.size() != 3) throw std::runtime_error("expected [x, y, z], got " + j.dump());
    return {j[0].get<float>(), j[1].get<float>(), j[2].get<float>()};
}

static Mat4 mat4(const json& j) {
    if (!j.is_array() || j.size() != 16) throw std::runtime_error("transform must be 16 floats (column-major)");
    Mat4 m;
    for (int i = 0; i < 16; ++i) m.m[i] = j[i].get<float>();
    return m;
}

static Part load_part(const json& j, const fs::path& base_dir) {
    Part p;
    p.name = j.value("name", "part");
    if (j.contains("color")) p.color = vec3(j["color"]);

    if (j.contains("file")) {
        // External mesh (CAD export). Path is relative to the scene file.
        fs::path file = base_dir / j["file"].get<std::string>();
        p.mesh = load_mesh_file(file.string());
        p.source = file.string();
        Mat4 xf = Mat4::identity();
        if (j.contains("scale")) xf = scale(j["scale"].get<float>());  // e.g. 0.001 for mm
        if (j.contains("transform")) xf = mat4(j["transform"]) * xf;
        p.mesh.transform(xf);
    } else {
        const auto& v = j.at("vertices");
        const auto& idx = j.at("indices");
        if (v.size() % 3 != 0 || idx.size() % 3 != 0)
            throw std::runtime_error("part '" + p.name + "': vertices/indices not multiples of 3");
        p.mesh.positions.reserve(v.size() / 3);
        for (size_t i = 0; i < v.size(); i += 3)
            p.mesh.positions.push_back({v[i].get<float>(), v[i + 1].get<float>(), v[i + 2].get<float>()});
        p.mesh.indices = idx.get<std::vector<uint32_t>>();
        for (auto i : p.mesh.indices)
            if (i >= p.mesh.positions.size())
                throw std::runtime_error("part '" + p.name + "': index out of range");
        p.source = "inline";
    }

    p.shell = j.value("shell", false);
    p.internal = j.value("internal", false);

    if (j.contains("control") && !j["control"].is_null()) {
        p.control = j["control"].value("id", "");
        p.gain = j["control"].value("gain", 1.0f);
        const auto& h = j.at("hinge");
        p.hinge = {vec3(h.at("origin")), normalize(vec3(h.at("axis")))};
    }
    return p;
}

Scene load_scene(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open scene " + path);
    json j;
    try {
        f >> j;
    } catch (const json::exception& e) {
        throw std::runtime_error("bad JSON in " + path + ": " + e.what());
    }

    Scene s;
    s.path = path;
    s.name = j.value("name", "scene");
    fs::path base_dir = fs::path(path).parent_path();

    try {
        for (const auto& pj : j.at("parts")) s.parts.push_back(load_part(pj, base_dir));
        for (const auto& pj : j.value("props", json::array())) {
            Prop p;
            p.name = pj.value("name", "prop");
            p.hub = vec3(pj.at("hub"));
            p.thrust_dir = normalize(vec3(pj.value("thrust_dir", json::array({-1, 0, 0}))));
            p.radius = pj.value("radius", 0.127f);
            p.blades = pj.value("blades", 2);
            p.pitch = pj.value("pitch", 0.1524f);
            p.spin = pj.value("spin", 1);
            s.props.push_back(p);
        }
        for (const auto& mj : j.value("markers", json::array()))
            s.markers.push_back({mj.value("name", ""), vec3(mj.at("position")),
                                 mj.contains("color") ? vec3(mj["color"]) : Vec3{1, 1, 0}});
        for (const auto& [id, c] : j.value("controls", json::object()).items())
            s.control_limits_deg[id] = c.value("max_deg", 25.0f);
        if (j.contains("sim")) {
            const auto& sj = j["sim"];
            s.sim.present = true;
            s.sim.cruise_speed = sj.value("cruise_speed", 0.0f);
            s.sim.trim_alpha_deg = sj.value("trim_alpha_deg", 0.0f);
            s.sim.trim_stab_deg = sj.value("trim_stab_deg", 0.0f);
            s.sim.L_over_D = sj.value("L_over_D", 0.0f);
            s.sim.static_margin = sj.value("static_margin", 0.0f);
            s.sim.v_stall = sj.value("v_stall", 0.0f);
            s.sim.vmc = sj.contains("vmc") && !sj["vmc"].is_null() ? sj["vmc"].get<float>() : 0.0f;
            s.sim.mass_kg = sj.value("mass_kg", 0.0f);
        }
    } catch (const json::exception& e) {
        throw std::runtime_error("bad scene structure in " + path + ": " + e.what());
    }
    if (s.parts.empty()) throw std::runtime_error("scene has no parts: " + path);
    return s;
}

Vec3 Scene::bounds_min() const {
    Vec3 r{1e30f, 1e30f, 1e30f};
    for (const auto& p : parts) r = vmin(r, p.mesh.bounds_min());
    return r;
}

Vec3 Scene::bounds_max() const {
    Vec3 r{-1e30f, -1e30f, -1e30f};
    for (const auto& p : parts) r = vmax(r, p.mesh.bounds_max());
    return r;
}

std::optional<Vec3> Scene::marker(const std::string& name) const {
    for (const auto& m : markers)
        if (m.name == name) return m.position;
    return std::nullopt;
}

}  // namespace twr
