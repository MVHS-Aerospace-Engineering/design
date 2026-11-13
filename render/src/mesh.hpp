// CPU-side triangle meshes: loading (inline scene data, STL, OBJ) and normal
// generation. STL/OBJ loading is the hook for dropping in CAD exports later.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "math.hpp"

namespace twr {

struct Mesh {
    std::vector<Vec3> positions;
    std::vector<uint32_t> indices;  // triangles

    bool empty() const { return indices.empty(); }
    void append(const Mesh& other, const Mat4& xf);
    void transform(const Mat4& xf);
    Vec3 bounds_min() const;
    Vec3 bounds_max() const;
};

// Render-ready vertex stream: position + normal per corner (unwelded).
struct Vertex {
    float px, py, pz;
    float nx, ny, nz;
};

// Builds per-corner normals. Faces sharing a vertex are smoothed together if the
// angle between them is below `crease_deg`, so airfoil skins stay smooth while
// trailing edges and wing caps stay sharp.
std::vector<Vertex> build_vertices(const Mesh& mesh, float crease_deg = 35.0f);

// Loads .stl (ASCII or binary) or .obj, picked by file extension. Throws
// std::runtime_error on failure. Coordinates are returned as stored in the file;
// apply units/transform through the scene.
Mesh load_mesh_file(const std::string& path);
Mesh load_stl(const std::string& path);
Mesh load_obj(const std::string& path);

// Procedural primitives.
Mesh make_uv_sphere(float radius, int segments = 24, int rings = 16);
Mesh make_cone(float radius, float height, int segments = 24);  // base at z=0, apex +z
// One propeller blade along +y from the hub, chord along x, thin in z, with
// root-to-tip twist. Rotate copies of it for the other blades.
Mesh make_blade(float radius, float hub_radius, float root_chord, float tip_chord, float root_twist_deg,
                float tip_twist_deg);

}  // namespace twr
