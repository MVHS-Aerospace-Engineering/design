#include "renderer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace twr {

// ----------------------------------------------------------------- shaders --

namespace {

const char* kMeshVS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNrm;
uniform mat4 uModel, uView, uProj, uNormal, uLightVP;
out vec3 vPos;
out vec3 vNrm;
out vec4 vLight;
void main() {
    vec4 w = uModel * vec4(aPos, 1.0);
    vPos = w.xyz;
    vNrm = mat3(uNormal) * aNrm;
    vLight = uLightVP * w;
    gl_Position = uProj * uView * w;
}
)";

const char* kShadowFn = R"(
uniform sampler2DShadow uShadow;
uniform int uUseShadow;
float shadow_factor(vec4 lp) {
    if (uUseShadow == 0) return 1.0;
    vec3 p = lp.xyz / lp.w * 0.5 + 0.5;
    if (p.z > 1.0 || p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0) return 1.0;
    vec2 texel = 1.0 / vec2(textureSize(uShadow, 0));
    float s = 0.0;
    for (int x = -2; x <= 2; ++x)
        for (int y = -2; y <= 2; ++y)
            s += texture(uShadow, vec3(p.xy + vec2(x, y) * texel, p.z - 0.0008));
    return s / 25.0;
}
)";

const char* kMeshFSBody = R"(
in vec3 vPos;
in vec3 vNrm;
in vec4 vLight;
uniform vec3 uColor, uEye, uLightDir;
uniform float uAlpha, uEmissive;
out vec4 frag;
void main() {
    vec3 base = pow(uColor, vec3(2.2));
    vec3 N = normalize(vNrm);
    vec3 V = normalize(uEye - vPos);
    if (dot(N, V) < 0.0) N = -N;  // two-sided lighting
    vec3 L = normalize(uLightDir);
    float sh = shadow_factor(vLight);
    float ndl = max(dot(N, L), 0.0);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 64.0) * 0.45;
    vec3 sky = vec3(0.55, 0.66, 0.85), ground = vec3(0.22, 0.20, 0.17);
    vec3 amb = mix(ground, sky, N.z * 0.5 + 0.5) * 0.45;
    float fill = max(dot(N, normalize(vec3(0.7, 0.6, 0.25))), 0.0) * 0.20;
    float rim = pow(1.0 - max(dot(N, V), 0.0), 4.0) * 0.12;
    vec3 c = base * (amb + fill + ndl * sh * 1.05) + vec3(spec * sh) + vec3(rim);
    c = mix(c, base, uEmissive);
    c = c / (1.0 + 0.15 * c);  // gentle highlight roll-off
    frag = vec4(pow(c, vec3(1.0 / 2.2)), uAlpha);
}
)";

const char* kDepthVS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uModel, uLightVP;
void main() { gl_Position = uLightVP * uModel * vec4(aPos, 1.0); }
)";

const char* kDepthFS = R"(#version 330 core
void main() {}
)";

const char* kGroundVS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uView, uProj, uLightVP;
out vec3 vPos;
out vec4 vLight;
void main() {
    vPos = aPos;
    vLight = uLightVP * vec4(aPos, 1.0);
    gl_Position = uProj * uView * vec4(aPos, 1.0);
}
)";

const char* kGroundFSBody = R"(
in vec3 vPos;
in vec4 vLight;
uniform vec3 uCenter;
uniform float uViewportH;
out vec4 frag;
float grid(vec2 p, float spacing, float width) {
    vec2 g = abs(fract(p / spacing - 0.5) - 0.5) / fwidth(p / spacing);
    return 1.0 - min(min(g.x, g.y) / width, 1.0);
}
void main() {
    vec2 p = vPos.xy;
    float d = length(p - uCenter.xy);
    vec3 base = vec3(0.17, 0.19, 0.21);
    float g = max(grid(p, 0.1, 1.0) * 0.35, grid(p, 1.0, 1.4));
    vec3 c = mix(base, vec3(0.42, 0.46, 0.52), g * 0.6);
    // x axis (fuselage line) and y axis hints
    float ax = 1.0 - min(abs(p.y) / fwidth(p.y) / 1.5, 1.0);
    c = mix(c, vec3(0.75, 0.30, 0.30), ax * 0.5);
    float sh = shadow_factor(vLight);
    c *= mix(0.45, 1.0, sh);
    float fade = 1.0 - smoothstep(1.0, 7.0, d);
    // Fade into exactly the sky colour behind this pixel so there is no seam.
    vec3 horizon = mix(vec3(0.20, 0.23, 0.28), vec3(0.09, 0.11, 0.15), clamp(gl_FragCoord.y / uViewportH, 0.0, 1.0));
    frag = vec4(mix(horizon, c, fade), 1.0);
}
)";

const char* kSkyVS = R"(#version 330 core
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.999, 1.0);
}
)";

const char* kSkyFS = R"(#version 330 core
in vec2 vUV;
out vec4 frag;
void main() {
    vec3 top = vec3(0.09, 0.11, 0.15), bottom = vec3(0.20, 0.23, 0.28);
    frag = vec4(mix(bottom, top, clamp(vUV.y, 0.0, 1.0)), 1.0);
}
)";

const char* kLineVS = R"(#version 330 core
layout(location = 0) in vec3 aPos;
uniform mat4 uView, uProj;
void main() { gl_Position = uProj * uView * vec4(aPos, 1.0); }
)";

const char* kLineFS = R"(#version 330 core
uniform vec3 uColor;
out vec4 frag;
void main() { frag = vec4(uColor, 1.0); }
)";

GLuint compile(GLenum type, const std::string& src) {
    GLuint s = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(s, 1, &c, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        throw std::runtime_error(std::string("shader compile failed: ") + log);
    }
    return s;
}

GLuint link(const std::string& vs, const std::string& fs) {
    GLuint p = glCreateProgram();
    GLuint v = compile(GL_VERTEX_SHADER, vs), f = compile(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        throw std::runtime_error(std::string("shader link failed: ") + log);
    }
    return p;
}

void set_mat(GLuint p, const char* name, const Mat4& m) {
    glUniformMatrix4fv(glGetUniformLocation(p, name), 1, GL_FALSE, m.data());
}
void set_vec(GLuint p, const char* name, const Vec3& v) { glUniform3f(glGetUniformLocation(p, name), v.x, v.y, v.z); }
void set_f(GLuint p, const char* name, float v) { glUniform1f(glGetUniformLocation(p, name), v); }
void set_i(GLuint p, const char* name, int v) { glUniform1i(glGetUniformLocation(p, name), v); }

Mesh make_disc(float radius, int segments = 64) {
    Mesh m;
    m.positions.push_back({0, 0, 0});
    for (int s = 0; s < segments; ++s) {
        float a = 2 * kPi * s / segments;
        m.positions.push_back({radius * std::cos(a), radius * std::sin(a), 0});
    }
    for (int s = 0; s < segments; ++s)
        m.indices.insert(m.indices.end(), {0u, uint32_t(1 + s), uint32_t(1 + (s + 1) % segments)});
    return m;
}

// Orthonormal basis whose local +z is `z`.
Mat4 basis_from_z(const Vec3& z_axis, const Vec3& origin) {
    Vec3 z = normalize(z_axis);
    Vec3 helper = std::fabs(z.z) < 0.9f ? Vec3{0, 0, 1} : Vec3{1, 0, 0};
    Vec3 x = normalize(cross(helper, z));
    Vec3 y = cross(z, x);
    Mat4 r = Mat4::identity();
    r(0, 0) = x.x; r(1, 0) = x.y; r(2, 0) = x.z;
    r(0, 1) = y.x; r(1, 1) = y.y; r(2, 1) = y.z;
    r(0, 2) = z.x; r(1, 2) = z.y; r(2, 2) = z.z;
    r(0, 3) = origin.x; r(1, 3) = origin.y; r(2, 3) = origin.z;
    return r;
}

}  // namespace

// ------------------------------------------------------------------ camera --

Vec3 Camera::eye() const {
    return target + Vec3{std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), std::sin(pitch)} * distance;
}

Mat4 Camera::view() const { return look_at(eye(), target, {0, 0, 1}); }

void Camera::set_view(const std::string& name) {
    // Nose points to -x, right wing to +y.
    if (name == "front") { yaw = radians(180); pitch = radians(4); }
    else if (name == "side") { yaw = radians(270); pitch = radians(2); }
    else if (name == "top") { yaw = radians(180); pitch = radians(89); }
    else if (name == "rear") { yaw = radians(0); pitch = radians(10); }
    else { yaw = radians(215); pitch = radians(22); }  // iso
}

// ---------------------------------------------------------------- renderer --

Renderer::~Renderer() { shutdown(); }

void Renderer::shutdown() {
    if (!initialised_) return;
    initialised_ = false;
    free_scene_meshes();
    for (GpuMesh* m : {&blade_, &blade_mirror_, &spinner_, &sphere_, &disc_}) free_mesh(*m);
    for (GLuint p : {mesh_prog_, depth_prog_, ground_prog_, sky_prog_, line_prog_}) glDeleteProgram(p);
    glDeleteVertexArrays(1, &ground_vao_);
    glDeleteBuffers(1, &ground_vbo_);
    glDeleteVertexArrays(1, &sky_vao_);
    glDeleteVertexArrays(1, &line_vao_);
    glDeleteBuffers(1, &line_vbo_);
    glDeleteFramebuffers(1, &shadow_fbo_);
    glDeleteTextures(1, &shadow_tex_);
}

void Renderer::init() {
    const std::string hdr = "#version 330 core\n";
    mesh_prog_ = link(kMeshVS, hdr + kShadowFn + kMeshFSBody);
    depth_prog_ = link(kDepthVS, kDepthFS);
    ground_prog_ = link(kGroundVS, hdr + kShadowFn + kGroundFSBody);
    sky_prog_ = link(kSkyVS, kSkyFS);
    line_prog_ = link(kLineVS, kLineFS);

    glGenVertexArrays(1, &sky_vao_);
    glGenVertexArrays(1, &ground_vao_);
    glGenBuffers(1, &ground_vbo_);
    glGenVertexArrays(1, &line_vao_);
    glGenBuffers(1, &line_vbo_);

    // Shadow map
    glGenTextures(1, &shadow_tex_);
    glBindTexture(GL_TEXTURE_2D, shadow_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, shadow_size_, shadow_size_, 0, GL_DEPTH_COMPONENT, GL_FLOAT,
                 nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
    glGenFramebuffers(1, &shadow_fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, shadow_tex_, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    sphere_ = upload(make_uv_sphere(1.0f), 80.0f);
    initialised_ = true;
}

Renderer::GpuMesh Renderer::upload(const Mesh& mesh, float crease_deg) {
    GpuMesh g;
    auto verts = build_vertices(mesh, crease_deg);
    g.count = static_cast<GLsizei>(verts.size());
    glGenVertexArrays(1, &g.vao);
    glGenBuffers(1, &g.vbo);
    glBindVertexArray(g.vao);
    glBindBuffer(GL_ARRAY_BUFFER, g.vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(verts.size() * sizeof(Vertex)), verts.data(),
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(3 * sizeof(float)));
    glBindVertexArray(0);
    return g;
}

void Renderer::free_mesh(GpuMesh& m) {
    if (m.vbo) glDeleteBuffers(1, &m.vbo);
    if (m.vao) glDeleteVertexArrays(1, &m.vao);
    m = {};
}

void Renderer::free_scene_meshes() {
    for (auto& m : part_meshes_) free_mesh(m);
    part_meshes_.clear();
}

void Renderer::set_scene(const Scene& scene) {
    scene_ = scene;
    free_scene_meshes();
    for (const auto& p : scene_.parts) part_meshes_.push_back(upload(p.mesh));

    Vec3 lo = scene_.bounds_min(), hi = scene_.bounds_max();
    pivot_ = scene_.marker("CG").value_or((lo + hi) * 0.5f);
    float span = length(hi - lo);
    ground_z_ = lo.z - 0.30f * span;

    // Props: blade mesh sized from the first prop (all props share a size here).
    for (GpuMesh* m : {&blade_, &blade_mirror_, &spinner_, &disc_}) free_mesh(*m);
    float R = scene_.props.empty() ? 0.127f : scene_.props[0].radius;
    float pitch = scene_.props.empty() ? 0.15f : scene_.props[0].pitch;
    float hub = 0.12f * R;
    float root_twist = std::atan(pitch / (2 * kPi * (hub + 0.15f * R))) * 180 / kPi;
    float tip_twist = std::atan(pitch / (2 * kPi * R)) * 180 / kPi;
    Mesh blade = make_blade(R, hub, 0.16f * R, 0.10f * R, root_twist, tip_twist);
    blade_ = upload(blade, 60.0f);
    Mesh mirrored = blade;  // opposite-hand blade for counter-rotating props
    for (auto& p : mirrored.positions) p.x = -p.x;
    for (size_t i = 0; i < mirrored.indices.size(); i += 3) std::swap(mirrored.indices[i + 1], mirrored.indices[i + 2]);
    blade_mirror_ = upload(mirrored, 60.0f);
    spinner_ = upload(make_cone(hub * 1.25f, hub * 2.6f), 50.0f);
    disc_ = upload(make_disc(R), 10.0f);

    // Ground quad
    Vec3 c = (lo + hi) * 0.5f;
    float e = 40.0f;
    float quad[] = {c.x - e, c.y - e, ground_z_, c.x + e, c.y - e, ground_z_, c.x + e, c.y + e, ground_z_,
                    c.x - e, c.y - e, ground_z_, c.x + e, c.y + e, ground_z_, c.x - e, c.y + e, ground_z_};
    glBindVertexArray(ground_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, ground_vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glBindVertexArray(0);

    // Sun: high, from front-left, so the shadow falls behind and to the right.
    light_dir_ = normalize({-0.45f, -0.55f, 1.0f});
    float r = 0.75f * span + 0.5f;
    Vec3 centre{c.x, c.y, (c.z + ground_z_) * 0.5f};
    Mat4 lv = look_at(centre + light_dir_ * (3 * r), centre, {1, 0, 0});
    light_vp_ = ortho(-r, r, -r, r, 0.1f, 6 * r) * lv;
}

Mat4 Renderer::plane_transform(const FrameState& st) const {
    // Nose-up = right-hand rotation about +y (x aft, z up).
    return rotate_about(radians(st.pitch_deg), pivot_, {0, 1, 0});
}

Mat4 Renderer::part_transform(const Part& p, const FrameState& st) const {
    Mat4 m = plane_transform(st);
    if (!p.control.empty()) {
        auto it = st.deflection_deg.find(p.control);
        float d = it == st.deflection_deg.end() ? 0.0f : it->second;
        if (d != 0.0f) m = m * rotate_about(radians(d * p.gain), p.hinge.origin, p.hinge.axis);
    }
    return m;
}

std::vector<Mat4> Renderer::blade_transforms(const Prop& p, const FrameState& st, bool& mirrored) const {
    mirrored = p.spin < 0;
    // Blade local frame: span +y, chord x, prop axis +z = thrust direction.
    Mat4 base = plane_transform(st) * basis_from_z(p.thrust_dir, p.hub);
    std::vector<Mat4> out;
    for (int k = 0; k < p.blades; ++k) {
        float a = p.spin * st.prop_angle + 2 * kPi * k / p.blades;
        out.push_back(base * rotate(a, {0, 0, 1}));
    }
    return out;
}

void Renderer::draw_mesh(GLuint program, const GpuMesh& m, const Mat4& model, const Vec3& color, float alpha,
                         float emissive) {
    set_mat(program, "uModel", model);
    if (program == mesh_prog_) {
        set_mat(program, "uNormal", normal_matrix(model));
        set_vec(program, "uColor", color);
        set_f(program, "uAlpha", alpha);
        set_f(program, "uEmissive", emissive);
    }
    glBindVertexArray(m.vao);
    glDrawArrays(GL_TRIANGLES, 0, m.count);
}

void Renderer::draw_scene_geometry(const FrameState& st, GLuint program, bool shadow_pass) {
    for (size_t i = 0; i < scene_.parts.size(); ++i)
        draw_mesh(program, part_meshes_[i], part_transform(scene_.parts[i], st), scene_.parts[i].color);

    const Vec3 prop_color{0.12f, 0.12f, 0.13f}, spinner_color{0.85f, 0.85f, 0.88f};
    for (const auto& p : scene_.props) {
        bool mirrored = false;
        Mat4 hub = plane_transform(st) * basis_from_z(p.thrust_dir, p.hub);
        draw_mesh(program, spinner_, hub, spinner_color);
        if (st.props_spinning && !shadow_pass) continue;  // drawn as a blurred disc below
        for (const auto& xf : blade_transforms(p, st, mirrored))
            draw_mesh(program, mirrored ? blade_mirror_ : blade_, xf, prop_color);
    }
}

void Renderer::render(const FrameState& st, int width, int height, GLuint fbo) {
    // ---- shadow pass
    glBindFramebuffer(GL_FRAMEBUFFER, shadow_fbo_);
    glViewport(0, 0, shadow_size_, shadow_size_);
    glClear(GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(2.0f, 4.0f);
    glUseProgram(depth_prog_);
    set_mat(depth_prog_, "uLightVP", light_vp_);
    draw_scene_geometry(st, depth_prog_, true);
    glDisable(GL_POLYGON_OFFSET_FILL);

    // ---- main pass
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, width, height);
    glClearColor(0.1f, 0.12f, 0.15f, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Mat4 view = st.camera.view();
    float dist = st.camera.distance;
    Mat4 proj = perspective(st.camera.fov, float(width) / float(std::max(height, 1)), 0.01f * dist, 60.0f + 10 * dist);
    Vec3 eye = st.camera.eye();

    glDepthMask(GL_FALSE);
    glUseProgram(sky_prog_);
    glBindVertexArray(sky_vao_);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDepthMask(GL_TRUE);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, shadow_tex_);

    if (st.show_ground) {
        glUseProgram(ground_prog_);
        set_mat(ground_prog_, "uView", view);
        set_mat(ground_prog_, "uProj", proj);
        set_mat(ground_prog_, "uLightVP", light_vp_);
        set_vec(ground_prog_, "uCenter", pivot_);
        set_f(ground_prog_, "uViewportH", float(height));
        set_i(ground_prog_, "uShadow", 0);
        set_i(ground_prog_, "uUseShadow", 1);
        glBindVertexArray(ground_vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    glUseProgram(mesh_prog_);
    set_mat(mesh_prog_, "uView", view);
    set_mat(mesh_prog_, "uProj", proj);
    set_mat(mesh_prog_, "uLightVP", light_vp_);
    set_vec(mesh_prog_, "uEye", eye);
    set_vec(mesh_prog_, "uLightDir", light_dir_);
    set_i(mesh_prog_, "uShadow", 0);
    set_i(mesh_prog_, "uUseShadow", 1);
    draw_scene_geometry(st, mesh_prog_, false);

    // Markers drawn on top of the airframe (x-ray) so the CG is visible inside the fuselage.
    if (st.show_markers && !scene_.markers.empty()) {
        glClear(GL_DEPTH_BUFFER_BIT);
        set_i(mesh_prog_, "uUseShadow", 0);
        float r = 0.012f * std::max(1.0f, length(scene_.bounds_max() - scene_.bounds_min()));
        Mat4 plane = plane_transform(st);
        for (const auto& m : scene_.markers)
            draw_mesh(mesh_prog_, sphere_, plane * translate(m.position) * scale(r), m.color, 1.0f, 0.35f);
    }

    // Spinning props: translucent discs.
    if (st.props_spinning) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
        set_i(mesh_prog_, "uUseShadow", 0);
        for (const auto& p : scene_.props)
            draw_mesh(mesh_prog_, disc_, plane_transform(st) * basis_from_z(p.thrust_dir, p.hub),
                      {0.75f, 0.78f, 0.82f}, 0.16f, 0.6f);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }

    // Hinge lines / pivot axes.
    if (st.show_hinges) {
        std::vector<float> lines;
        for (const auto& p : scene_.parts) {
            if (p.control.empty()) continue;
            Mat4 m = plane_transform(st);
            Vec3 a = m.point(p.hinge.origin - p.hinge.axis * 0.05f);
            Vec3 b = m.point(p.hinge.origin + p.hinge.axis * 0.40f);
            lines.insert(lines.end(), {a.x, a.y, a.z, b.x, b.y, b.z});
        }
        glDisable(GL_DEPTH_TEST);
        glUseProgram(line_prog_);
        set_mat(line_prog_, "uView", view);
        set_mat(line_prog_, "uProj", proj);
        set_vec(line_prog_, "uColor", {0.2f, 1.0f, 0.6f});
        glBindVertexArray(line_vao_);
        glBindBuffer(GL_ARRAY_BUFFER, line_vbo_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(lines.size() * sizeof(float)), lines.data(),
                     GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(lines.size() / 3));
        glEnable(GL_DEPTH_TEST);
    }
    glBindVertexArray(0);
}

}  // namespace twr
