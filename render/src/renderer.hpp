// OpenGL renderer for a twin_rc scene: shaded parts with shadow mapping,
// control-surface deflection about hinge lines, spinning props, a ground grid,
// CG / neutral-point markers and hinge-line overlays.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "gl.hpp"
#include "math.hpp"
#include "scene.hpp"

namespace twr {

struct Camera {
    Vec3 target{0.5f, 0, 0.05f};
    float yaw = radians(215), pitch = radians(22), distance = 2.4f;
    float fov = radians(40);

    Vec3 eye() const;
    Mat4 view() const;
    void set_view(const std::string& name);  // iso | front | side | top | rear
};

struct FrameState {
    Camera camera;
    std::map<std::string, float> deflection_deg;  // by control id
    float pitch_deg = 0;   // whole-airplane nose-up rotation about the CG (angle of attack)
    float prop_angle = 0;  // radians
    bool props_spinning = false;
    bool show_markers = true;
    bool show_hinges = false;
    bool show_ground = true;
};

class Renderer {
public:
    Renderer() = default;
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void init();      // needs a current GL context
    void shutdown();  // frees GL objects; call while the context is still current
    void set_scene(const Scene& scene);  // (re)uploads meshes
    // Draws into `fbo` (0 = window) at `width` x `height` pixels.
    void render(const FrameState& st, int width, int height, GLuint fbo = 0);

    const Scene& scene() const { return scene_; }
    Vec3 pivot() const { return pivot_; }  // CG if present, else bounds centre

private:
    struct GpuMesh {
        GLuint vao = 0, vbo = 0;
        GLsizei count = 0;
    };
    GpuMesh upload(const Mesh& mesh, float crease_deg = 35.0f);
    void free_mesh(GpuMesh& m);
    void free_scene_meshes();

    Mat4 plane_transform(const FrameState& st) const;
    Mat4 part_transform(const Part& p, const FrameState& st) const;
    std::vector<Mat4> blade_transforms(const Prop& p, const FrameState& st, bool& mirrored) const;
    void draw_scene_geometry(const FrameState& st, GLuint program, bool shadow_pass);
    void draw_mesh(GLuint program, const GpuMesh& m, const Mat4& model, const Vec3& color, float alpha = 1.0f,
                   float emissive = 0.0f);

    Scene scene_;
    Vec3 pivot_;
    float ground_z_ = 0;
    std::vector<GpuMesh> part_meshes_;
    GpuMesh blade_, blade_mirror_, spinner_, sphere_, disc_;
    GLuint mesh_prog_ = 0, depth_prog_ = 0, ground_prog_ = 0, sky_prog_ = 0, line_prog_ = 0;
    GLuint ground_vao_ = 0, ground_vbo_ = 0, sky_vao_ = 0, line_vao_ = 0, line_vbo_ = 0;
    GLuint shadow_fbo_ = 0, shadow_tex_ = 0;
    int shadow_size_ = 2048;
    Mat4 light_vp_;
    Vec3 light_dir_;
    bool initialised_ = false;
};

}  // namespace twr
