// twin_rc_render: interactive viewer + headless screenshot tool for the twin RC plane.
//
//   twin_rc_render [scene.json] [options]
//
// Run with --help for the options and key bindings.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "gl.hpp"
#include "png.hpp"
#include "renderer.hpp"
#include "scene.hpp"

using namespace twr;
namespace fs = std::filesystem;

namespace {

const char* kHelp = R"(twin_rc_render - renders the twin RC plane from the Python scene export

Usage: twin_rc_render [scene.json] [options]

  scene.json            default: render/scenes/twin_rc.json (searched from the cwd upwards)
  --screenshot FILE     render one frame to FILE (PNG) without opening a window, then exit
  --width N, --height N output / window size (default 1600 x 1000)
  --view NAME           iso | front | side | top | rear (default iso)
  --zoom F              camera zoom factor (2 = twice as close), aimed at the CG
  --trim                pose the plane at the simulated cruise trim (alpha + stabilators)
  --stab DEG            collective stabilator deflection (+ = trailing edge down)
  --diff DEG            differential stabilator (+ = left TE down -> roll right)
  --aileron DEG         aileron (+ = right TE down -> roll left)
  --rudder DEG          rudder (+ = TE right -> nose right)
  --alpha DEG           pitch the whole plane nose-up by DEG about the CG
  --spin                show props spinning
  --hinges              show hinge lines / stabilator pivot axes
  --xray                see-through fuselage and nacelles (shows Pi 5, FC, battery, motors)
  --no-markers          hide CG / neutral point markers
  --no-watch            don't reload the scene when the file changes

Keys:
  mouse drag orbit, right/shift-drag pan, scroll zoom
  W/S stabilators (pitch)   A/D roll (ailerons + differential stabilators)   Q/E rudder
  T trimmed-cruise pose     P props on/off   H hinge lines   M markers   G ground   V x-ray
  1 iso 2 front 3 side 4 top 5 rear   F reframe   X screenshot   Esc quit
Controls spring back to neutral when released, like RC sticks.
)";

struct Options {
    std::string scene_path;
    std::string screenshot;
    int width = 1600, height = 1000;
    std::string view = "iso";
    float zoom = 1.0f;
    bool trim = false, spin = false, hinges = false, markers = true, watch = true, xray = false;
    float stab = 0, diff = 0, aileron = 0, rudder = 0, alpha = 0;
};

std::string find_default_scene() {
    for (fs::path dir = fs::current_path();; dir = dir.parent_path()) {
        for (const char* rel : {"render/scenes/twin_rc.json", "scenes/twin_rc.json"}) {
            fs::path p = dir / rel;
            if (fs::exists(p)) return p.string();
        }
        if (dir == dir.parent_path()) break;
    }
    return "render/scenes/twin_rc.json";
}

Options parse_args(int argc, char** argv) {
    Options o;
    auto need = [&](int& i) -> std::string {
        if (i + 1 >= argc) throw std::runtime_error(std::string("missing value after ") + argv[i]);
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { std::cout << kHelp; std::exit(0); }
        else if (a == "--screenshot") o.screenshot = need(i);
        else if (a == "--width") o.width = std::stoi(need(i));
        else if (a == "--height") o.height = std::stoi(need(i));
        else if (a == "--view") o.view = need(i);
        else if (a == "--zoom") o.zoom = std::stof(need(i));
        else if (a == "--trim") o.trim = true;
        else if (a == "--stab") o.stab = std::stof(need(i));
        else if (a == "--diff") o.diff = std::stof(need(i));
        else if (a == "--aileron") o.aileron = std::stof(need(i));
        else if (a == "--rudder") o.rudder = std::stof(need(i));
        else if (a == "--alpha") o.alpha = std::stof(need(i));
        else if (a == "--spin") o.spin = true;
        else if (a == "--hinges") o.hinges = true;
        else if (a == "--xray") o.xray = true;
        else if (a == "--no-markers") o.markers = false;
        else if (a == "--no-watch") o.watch = false;
        else if (!a.empty() && a[0] == '-') throw std::runtime_error("unknown option " + a + " (see --help)");
        else o.scene_path = a;
    }
    if (o.scene_path.empty()) o.scene_path = find_default_scene();
    return o;
}

// Pilot inputs (deg) mixed into per-surface deflections, plus the trim offsets.
struct Controls {
    float stab = 0, diff = 0, aileron = 0, rudder = 0;  // pilot
    bool trim = false;

    void apply(FrameState& st, const Scene& s) const {
        float trim_stab = (trim && s.sim.present) ? s.sim.trim_stab_deg : 0.0f;
        auto lim = [&](const char* id, float v) {
            auto it = s.control_limits_deg.find(id);
            float m = it == s.control_limits_deg.end() ? 30.0f : it->second;
            return std::clamp(v, -m, m);
        };
        st.deflection_deg["stab_L"] = lim("stab_L", trim_stab + stab + diff);
        st.deflection_deg["stab_R"] = lim("stab_R", trim_stab + stab - diff);
        st.deflection_deg["aileron"] = lim("aileron", aileron);
        st.deflection_deg["rudder"] = lim("rudder", rudder);
    }
};

void frame_camera(Camera& cam, const Renderer& r) {
    Vec3 lo = r.scene().bounds_min(), hi = r.scene().bounds_max();
    cam.target = (lo + hi) * 0.5f;
    cam.distance = 0.8f * length(hi - lo) / (2 * std::tan(cam.fov / 2));
}

// Offscreen multisampled render target, resolved for readback.
struct OffscreenTarget {
    GLuint ms_fbo = 0, ms_color = 0, ms_depth = 0, fbo = 0, color = 0;
    int w = 0, h = 0;
    OffscreenTarget(int w_, int h_) : w(w_), h(h_) {
        GLint max_samples = 0;
        glGetIntegerv(GL_MAX_SAMPLES, &max_samples);
        int samples = std::min(8, std::max(1, max_samples));
        glGenFramebuffers(1, &ms_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, ms_fbo);
        glGenRenderbuffers(1, &ms_color);
        glBindRenderbuffer(GL_RENDERBUFFER, ms_color);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA8, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, ms_color);
        glGenRenderbuffers(1, &ms_depth);
        glBindRenderbuffer(GL_RENDERBUFFER, ms_depth);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT24, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, ms_depth);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("offscreen framebuffer incomplete");
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenRenderbuffers(1, &color);
        glBindRenderbuffer(GL_RENDERBUFFER, color);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w, h);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    ~OffscreenTarget() {
        glDeleteFramebuffers(1, &ms_fbo);
        glDeleteFramebuffers(1, &fbo);
        GLuint rbs[] = {ms_color, ms_depth, color};
        glDeleteRenderbuffers(3, rbs);
    }
    std::vector<uint8_t> resolve_and_read() {
        glBindFramebuffer(GL_READ_FRAMEBUFFER, ms_fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
        glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        std::vector<uint8_t> px(size_t(w) * h * 4), flipped(px.size());
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        const size_t stride = size_t(w) * 4;
        for (int y = 0; y < h; ++y)
            std::copy_n(px.begin() + (h - 1 - y) * stride, stride, flipped.begin() + y * stride);
        for (size_t i = 3; i < flipped.size(); i += 4) flipped[i] = 255;
        return flipped;
    }
};

bool save_screenshot(Renderer& r, const FrameState& st, int w, int h, const std::string& path) {
    OffscreenTarget t(w, h);
    r.render(st, w, h, t.ms_fbo);
    auto px = t.resolve_and_read();
    if (fs::path(path).has_parent_path()) fs::create_directories(fs::path(path).parent_path());
    return write_png(path, w, h, px);
}

std::string title_for(const Scene& s, const FrameState& st, const Controls& c) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "%s  |  stab L %+.1f R %+.1f  ail %+.1f  rud %+.1f  alpha %+.1f%s",
                  s.name.c_str(), st.deflection_deg.at("stab_L"), st.deflection_deg.at("stab_R"),
                  st.deflection_deg.at("aileron"), st.deflection_deg.at("rudder"), st.pitch_deg,
                  c.trim ? "  [TRIM]" : "");
    std::string t = buf;
    if (s.sim.present) {
        std::snprintf(buf, sizeof buf, "  |  cruise %.0f m/s  L/D %.1f  SM %.0f%%  Vs %.1f  Vmc %.1f",
                      s.sim.cruise_speed, s.sim.L_over_D, s.sim.static_margin * 100, s.sim.v_stall, s.sim.vmc);
        t += buf;
    }
    return t;
}

void print_scene_info(const Scene& s) {
    size_t tris = 0, controls = 0, cad = 0, internal = 0;
    for (const auto& p : s.parts) {
        tris += p.mesh.indices.size() / 3;
        controls += !p.control.empty();
        cad += p.source != "inline";
        internal += p.internal;
    }
    std::cout << "Loaded " << s.path << ": " << s.parts.size() << " parts (" << controls << " movable, " << internal
              << " internal, " << cad << " from mesh files), " << tris << " triangles, " << s.props.size()
              << " props\n";
}

struct App {
    GLFWwindow* win = nullptr;
    Renderer renderer;
    FrameState st;
    Controls ctl;
    Options opt;
    double last_x = 0, last_y = 0;
    bool dragging = false, panning = false;
    int shot_count = 0;
    fs::file_time_type scene_mtime{};
};

void on_scroll(GLFWwindow* w, double, double dy) {
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(w));
    app->st.camera.distance = std::clamp(app->st.camera.distance * float(std::pow(0.9, dy)), 0.2f, 40.0f);
}

void on_mouse_button(GLFWwindow* w, int button, int action, int mods) {
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(w));
    bool down = action == GLFW_PRESS;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        app->dragging = down && !(mods & GLFW_MOD_SHIFT);
        app->panning = down && (mods & GLFW_MOD_SHIFT);
    } else if (button == GLFW_MOUSE_BUTTON_RIGHT || button == GLFW_MOUSE_BUTTON_MIDDLE) {
        app->panning = down;
    }
    glfwGetCursorPos(w, &app->last_x, &app->last_y);
}

void on_cursor(GLFWwindow* w, double x, double y) {
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(w));
    float dx = float(x - app->last_x), dy = float(y - app->last_y);
    app->last_x = x;
    app->last_y = y;
    Camera& c = app->st.camera;
    if (app->dragging) {
        c.yaw -= dx * 0.008f;
        c.pitch = std::clamp(c.pitch + dy * 0.008f, radians(-89), radians(89));
    } else if (app->panning) {
        Vec3 f = normalize(c.target - c.eye());
        Vec3 right = normalize(cross(f, {0, 0, 1}));
        Vec3 up = cross(right, f);
        float k = c.distance * 0.0015f;
        c.target += (right * -dx + up * dy) * k;
    }
}

void on_key(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(w));
    auto& st = app->st;
    switch (key) {
        case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, 1); break;
        case GLFW_KEY_T:
            app->ctl.trim = !app->ctl.trim;
            st.pitch_deg = (app->ctl.trim && app->renderer.scene().sim.present)
                               ? app->renderer.scene().sim.trim_alpha_deg : 0.0f;
            if (app->ctl.trim && !app->renderer.scene().sim.present)
                std::cout << "No sim results in the scene; run `python -m twin_rc.run_all` then "
                             "`python -m twin_rc.scene`.\n";
            break;
        case GLFW_KEY_P: st.props_spinning = !st.props_spinning; break;
        case GLFW_KEY_H: st.show_hinges = !st.show_hinges; break;
        case GLFW_KEY_V: st.xray = !st.xray; break;
        case GLFW_KEY_M: st.show_markers = !st.show_markers; break;
        case GLFW_KEY_G: st.show_ground = !st.show_ground; break;
        case GLFW_KEY_F: frame_camera(st.camera, app->renderer); break;
        case GLFW_KEY_1: st.camera.set_view("iso"); break;
        case GLFW_KEY_2: st.camera.set_view("front"); break;
        case GLFW_KEY_3: st.camera.set_view("side"); break;
        case GLFW_KEY_4: st.camera.set_view("top"); break;
        case GLFW_KEY_5: st.camera.set_view("rear"); break;
        case GLFW_KEY_X: {
            char name[64];
            std::snprintf(name, sizeof name, "screenshots/twin_rc_%03d.png", app->shot_count++);
            int fw, fh;
            glfwGetFramebufferSize(w, &fw, &fh);
            if (save_screenshot(app->renderer, st, fw, fh, name)) std::cout << "Saved " << name << "\n";
            break;
        }
        default: break;
    }
}

// Spring-centred stick: move toward +/-max while held, back to 0 when released.
void update_axis(GLFWwindow* w, int key_pos, int key_neg, float& v, float max, float dt) {
    float target = 0;
    if (glfwGetKey(w, key_pos) == GLFW_PRESS) target += max;
    if (glfwGetKey(w, key_neg) == GLFW_PRESS) target -= max;
    float rate = (target == 0 ? 90.0f : 60.0f) * dt;
    v += std::clamp(target - v, -rate, rate);
}

int run_window(App& app) {
    glfwWindowHint(GLFW_SAMPLES, 8);
    app.win = glfwCreateWindow(app.opt.width, app.opt.height,
                               "twin_rc_render", nullptr, nullptr);
    if (!app.win) throw std::runtime_error("could not create window (needs OpenGL 3.3+)");
    glfwMakeContextCurrent(app.win);
    glfwSwapInterval(1);
#ifndef __APPLE__
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) throw std::runtime_error("glewInit failed");
#endif
    glEnable(GL_MULTISAMPLE);
    glfwSetWindowUserPointer(app.win, &app);
    glfwSetScrollCallback(app.win, on_scroll);
    glfwSetMouseButtonCallback(app.win, on_mouse_button);
    glfwSetCursorPosCallback(app.win, on_cursor);
    glfwSetKeyCallback(app.win, on_key);

    app.renderer.init();
    Scene scene = load_scene(app.opt.scene_path);
    print_scene_info(scene);
    app.renderer.set_scene(scene);
    app.scene_mtime = fs::last_write_time(app.opt.scene_path);
    frame_camera(app.st.camera, app.renderer);
    app.st.camera.set_view(app.opt.view);
    std::cout << kHelp + std::string(kHelp).find("Keys:");

    auto last = std::chrono::steady_clock::now();
    double since_check = 0;
    std::string last_title;
    while (!glfwWindowShouldClose(app.win)) {
        glfwPollEvents();
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        update_axis(app.win, GLFW_KEY_W, GLFW_KEY_S, app.ctl.stab, 15.0f, dt);
        // D = roll right: differential stabilator positive, ailerons negative (right TE up).
        update_axis(app.win, GLFW_KEY_D, GLFW_KEY_A, app.ctl.diff, 8.0f, dt);
        app.ctl.aileron = -app.ctl.diff / 8.0f * 18.0f;
        update_axis(app.win, GLFW_KEY_E, GLFW_KEY_Q, app.ctl.rudder, 25.0f, dt);
        app.ctl.apply(app.st, app.renderer.scene());
        if (app.st.props_spinning) app.st.prop_angle += dt * 2 * kPi * 3.0f;

        // Hot reload when Python re-exports the scene.
        since_check += dt;
        if (app.opt.watch && since_check > 0.5) {
            since_check = 0;
            std::error_code ec;
            auto mt = fs::last_write_time(app.opt.scene_path, ec);
            if (!ec && mt != app.scene_mtime) {
                app.scene_mtime = mt;
                try {
                    Scene s = load_scene(app.opt.scene_path);
                    app.renderer.set_scene(s);
                    std::cout << "Reloaded. ";
                    print_scene_info(s);
                } catch (const std::exception& e) {
                    std::cerr << "Reload failed (keeping previous scene): " << e.what() << "\n";
                }
            }
        }

        std::string title = title_for(app.renderer.scene(), app.st, app.ctl);
        if (title != last_title) {
            glfwSetWindowTitle(app.win, title.c_str());
            last_title = title;
        }

        int fw, fh;
        glfwGetFramebufferSize(app.win, &fw, &fh);
        app.renderer.render(app.st, fw, fh, 0);
        glfwSwapBuffers(app.win);
    }
    return 0;
}

int run_headless(App& app) {
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    app.win = glfwCreateWindow(64, 64, "twin_rc_render", nullptr, nullptr);
    if (!app.win) throw std::runtime_error("could not create an OpenGL context (needs OpenGL 3.3+)");
    glfwMakeContextCurrent(app.win);
#ifndef __APPLE__
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) throw std::runtime_error("glewInit failed");
#endif
    app.renderer.init();
    Scene scene = load_scene(app.opt.scene_path);
    print_scene_info(scene);
    app.renderer.set_scene(scene);
    frame_camera(app.st.camera, app.renderer);
    app.st.camera.set_view(app.opt.view);
    if (app.opt.zoom != 1.0f) {
        app.st.camera.distance /= app.opt.zoom;
        app.st.camera.target = app.renderer.pivot();
    }
    app.ctl.apply(app.st, app.renderer.scene());
    if (!save_screenshot(app.renderer, app.st, app.opt.width, app.opt.height, app.opt.screenshot))
        throw std::runtime_error("could not write " + app.opt.screenshot);
    std::cout << "Saved " << app.opt.screenshot << " (" << app.opt.width << "x" << app.opt.height << ")\n";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    App app;
    try {
        app.opt = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }
    const Options& o = app.opt;
    app.ctl = {o.stab, o.diff, o.aileron, o.rudder, o.trim};
    app.st.props_spinning = o.spin;
    app.st.show_hinges = o.hinges;
    app.st.xray = o.xray;
    app.st.show_markers = o.markers;
    app.st.pitch_deg = o.alpha;

    if (!glfwInit()) {
        std::cerr << "error: glfwInit failed\n";
        return 1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);

    int rc = 0;
    try {
        if (o.trim) {
            // Trim pose needs the scene's sim block; applied after load.
            Scene s = load_scene(o.scene_path);
            if (s.sim.present) app.st.pitch_deg += s.sim.trim_alpha_deg;
            else std::cerr << "warning: --trim but the scene has no sim results\n";
        }
        rc = o.screenshot.empty() ? run_window(app) : run_headless(app);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        rc = 1;
    }
    // Renderer GL objects must be freed while the context is alive.
    app.renderer.shutdown();
    if (app.win) glfwDestroyWindow(app.win);
    glfwTerminate();
    return rc;
}
