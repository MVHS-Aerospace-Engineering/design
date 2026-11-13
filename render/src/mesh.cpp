#include "mesh.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace twr {

void Mesh::append(const Mesh& other, const Mat4& xf) {
    auto base = static_cast<uint32_t>(positions.size());
    for (const auto& p : other.positions) positions.push_back(xf.point(p));
    for (auto i : other.indices) indices.push_back(base + i);
}

void Mesh::transform(const Mat4& xf) {
    for (auto& p : positions) p = xf.point(p);
}

Vec3 Mesh::bounds_min() const {
    Vec3 r{1e30f, 1e30f, 1e30f};
    for (const auto& p : positions) r = vmin(r, p);
    return r;
}

Vec3 Mesh::bounds_max() const {
    Vec3 r{-1e30f, -1e30f, -1e30f};
    for (const auto& p : positions) r = vmax(r, p);
    return r;
}

std::vector<Vertex> build_vertices(const Mesh& mesh, float crease_deg) {
    const size_t n_tri = mesh.indices.size() / 3;
    std::vector<Vec3> face_n(n_tri);
    std::vector<float> face_area(n_tri);
    for (size_t t = 0; t < n_tri; ++t) {
        const Vec3& a = mesh.positions[mesh.indices[3 * t]];
        const Vec3& b = mesh.positions[mesh.indices[3 * t + 1]];
        const Vec3& c = mesh.positions[mesh.indices[3 * t + 2]];
        Vec3 n = cross(b - a, c - a);
        face_area[t] = length(n);
        face_n[t] = normalize(n);
    }

    // vertex -> incident faces
    std::vector<std::vector<uint32_t>> incident(mesh.positions.size());
    for (size_t t = 0; t < n_tri; ++t)
        for (int k = 0; k < 3; ++k) incident[mesh.indices[3 * t + k]].push_back(static_cast<uint32_t>(t));

    const float cos_crease = std::cos(radians(crease_deg));
    std::vector<Vertex> out;
    out.reserve(mesh.indices.size());
    for (size_t t = 0; t < n_tri; ++t) {
        for (int k = 0; k < 3; ++k) {
            uint32_t vi = mesh.indices[3 * t + k];
            Vec3 n{0, 0, 0};
            for (uint32_t f : incident[vi])
                if (dot(face_n[f], face_n[t]) >= cos_crease) n += face_n[f] * face_area[f];
            n = length(n) > 0 ? normalize(n) : face_n[t];
            const Vec3& p = mesh.positions[vi];
            out.push_back({p.x, p.y, p.z, n.x, n.y, n.z});
        }
    }
    return out;
}

static std::string lower_ext(const std::string& path) {
    auto dot_pos = path.find_last_of('.');
    std::string ext = dot_pos == std::string::npos ? "" : path.substr(dot_pos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext;
}

Mesh load_mesh_file(const std::string& path) {
    std::string ext = lower_ext(path);
    if (ext == "stl") return load_stl(path);
    if (ext == "obj") return load_obj(path);
    throw std::runtime_error("unsupported mesh format '" + ext + "' (use .stl or .obj): " + path);
}

// Welds identical positions so STL soups get shared vertices (needed for smooth normals).
namespace {
struct Welder {
    struct Key {
        float x, y, z;
        bool operator==(const Key& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct Hash {
        size_t operator()(const Key& k) const {
            uint32_t a, b, c;
            std::memcpy(&a, &k.x, 4); std::memcpy(&b, &k.y, 4); std::memcpy(&c, &k.z, 4);
            return (size_t(a) * 73856093u) ^ (size_t(b) * 19349663u) ^ (size_t(c) * 83492791u);
        }
    };
    std::unordered_map<Key, uint32_t, Hash> map;
    Mesh mesh;
    void add(const Vec3& p) {
        Key k{p.x, p.y, p.z};
        auto it = map.find(k);
        if (it == map.end()) {
            auto id = static_cast<uint32_t>(mesh.positions.size());
            mesh.positions.push_back(p);
            map.emplace(k, id);
            mesh.indices.push_back(id);
        } else {
            mesh.indices.push_back(it->second);
        }
    }
};
}  // namespace

Mesh load_stl(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Welder w;

    bool binary = true;
    if (buf.size() >= 84) {
        uint32_t n;
        std::memcpy(&n, buf.data() + 80, 4);
        binary = (84 + size_t(n) * 50 == buf.size());
    }
    if (buf.size() < 84) binary = false;

    if (binary) {
        uint32_t n;
        std::memcpy(&n, buf.data() + 80, 4);
        const char* p = buf.data() + 84;
        for (uint32_t t = 0; t < n; ++t, p += 50) {
            for (int k = 0; k < 3; ++k) {
                float v[3];
                std::memcpy(v, p + 12 + 12 * k, 12);
                w.add({v[0], v[1], v[2]});
            }
        }
    } else {
        std::istringstream in(std::string(buf.begin(), buf.end()));
        std::string tok;
        while (in >> tok) {
            if (tok == "vertex") {
                float x, y, z;
                in >> x >> y >> z;
                w.add({x, y, z});
            }
        }
    }
    if (w.mesh.indices.size() % 3 != 0 || w.mesh.indices.empty())
        throw std::runtime_error("no triangles in STL: " + path);
    return w.mesh;
}

Mesh load_obj(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    Mesh m;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream in(line);
        std::string tag;
        in >> tag;
        if (tag == "v") {
            Vec3 p;
            in >> p.x >> p.y >> p.z;
            m.positions.push_back(p);
        } else if (tag == "f") {
            std::vector<uint32_t> poly;
            std::string corner;
            while (in >> corner) {
                long idx = std::stol(corner.substr(0, corner.find('/')));
                if (idx < 0) idx = static_cast<long>(m.positions.size()) + idx + 1;  // relative index
                poly.push_back(static_cast<uint32_t>(idx - 1));
            }
            for (size_t k = 1; k + 1 < poly.size(); ++k) {  // fan-triangulate
                m.indices.push_back(poly[0]);
                m.indices.push_back(poly[k]);
                m.indices.push_back(poly[k + 1]);
            }
        }
    }
    if (m.indices.empty()) throw std::runtime_error("no faces in OBJ: " + path);
    return m;
}

Mesh make_uv_sphere(float radius, int segments, int rings) {
    Mesh m;
    for (int r = 0; r <= rings; ++r) {
        float th = kPi * r / rings;
        for (int s = 0; s <= segments; ++s) {
            float ph = 2 * kPi * s / segments;
            m.positions.push_back({radius * std::sin(th) * std::cos(ph), radius * std::sin(th) * std::sin(ph),
                                   radius * std::cos(th)});
        }
    }
    for (int r = 0; r < rings; ++r)
        for (int s = 0; s < segments; ++s) {
            uint32_t a = r * (segments + 1) + s, b = a + segments + 1;
            m.indices.insert(m.indices.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
    return m;
}

Mesh make_cone(float radius, float height, int segments) {
    Mesh m;
    m.positions.push_back({0, 0, height});
    m.positions.push_back({0, 0, 0});
    for (int s = 0; s < segments; ++s) {
        float ph = 2 * kPi * s / segments;
        m.positions.push_back({radius * std::cos(ph), radius * std::sin(ph), 0});
    }
    for (int s = 0; s < segments; ++s) {
        uint32_t a = 2 + s, b = 2 + (s + 1) % segments;
        m.indices.insert(m.indices.end(), {0, a, b, 1, b, a});
    }
    return m;
}

Mesh make_blade(float radius, float hub_radius, float root_chord, float tip_chord, float root_twist_deg,
                float tip_twist_deg) {
    // Thin lenticular section lofted from hub to tip.
    const int n_span = 12, n_chord = 10;
    Mesh m;
    for (int i = 0; i <= n_span; ++i) {
        float t = float(i) / n_span;
        float y = hub_radius + t * (radius - hub_radius);
        // Elliptical-ish planform: widest at ~30 % span, rounded tip.
        float chord = root_chord + (tip_chord - root_chord) * t;
        chord *= (t < 0.3f ? 0.75f + t / 0.3f * 0.25f : std::sqrt(std::fmax(1 - std::pow((t - 0.3f) / 0.7f, 4.0f), 0.05f)));
        float thick = 0.10f * chord * (1 - 0.6f * t);
        float tw = radians(root_twist_deg + (tip_twist_deg - root_twist_deg) * t);
        for (int side = 0; side < 2; ++side) {
            for (int j = 0; j <= n_chord; ++j) {
                float u = float(j) / n_chord;  // 0 = LE, 1 = TE
                float x = (u - 0.35f) * chord;
                float z = (side == 0 ? 1 : -1) * thick * 2.6f * u * (1 - u) * (1 - u) * 1.5f;
                // twist about the blade's spanwise (y) axis
                float xr = x * std::cos(tw) + z * std::sin(tw);
                float zr = -x * std::sin(tw) + z * std::cos(tw);
                m.positions.push_back({xr, y, zr});
            }
        }
    }
    const uint32_t per_side = n_chord + 1, per_ring = 2 * per_side;
    for (int i = 0; i < n_span; ++i)
        for (int side = 0; side < 2; ++side)
            for (int j = 0; j < n_chord; ++j) {
                uint32_t a = i * per_ring + side * per_side + j, b = a + per_ring;
                if (side == 0)
                    m.indices.insert(m.indices.end(), {a, a + 1, b, a + 1, b + 1, b});
                else
                    m.indices.insert(m.indices.end(), {a, b, a + 1, a + 1, b, b + 1});
            }
    return m;
}

}  // namespace twr
