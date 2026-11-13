// Minimal vector / matrix math for the renderer. Column-major 4x4 matrices
// (OpenGL convention): m[col * 4 + row].
#pragma once

#include <array>
#include <cmath>

namespace twr {

constexpr float kPi = 3.14159265358979323846f;
inline float radians(float deg) { return deg * kPi / 180.0f; }

struct Vec3 {
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(float s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    float operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }
};

inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline float length(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalize(const Vec3& a) {
    float l = length(a);
    return l > 1e-12f ? a / l : Vec3{0, 0, 1};
}
inline Vec3 vmin(const Vec3& a, const Vec3& b) { return {std::fmin(a.x, b.x), std::fmin(a.y, b.y), std::fmin(a.z, b.z)}; }
inline Vec3 vmax(const Vec3& a, const Vec3& b) { return {std::fmax(a.x, b.x), std::fmax(a.y, b.y), std::fmax(a.z, b.z)}; }

struct Mat4 {
    std::array<float, 16> m{};
    static Mat4 identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
        return r;
    }
    float& operator()(int row, int col) { return m[col * 4 + row]; }
    float operator()(int row, int col) const { return m[col * 4 + row]; }
    const float* data() const { return m.data(); }

    Mat4 operator*(const Mat4& b) const {
        Mat4 r;
        for (int c = 0; c < 4; ++c)
            for (int rr = 0; rr < 4; ++rr) {
                float s = 0;
                for (int k = 0; k < 4; ++k) s += (*this)(rr, k) * b(k, c);
                r(rr, c) = s;
            }
        return r;
    }
    Vec3 point(const Vec3& p) const {
        return {(*this)(0, 0) * p.x + (*this)(0, 1) * p.y + (*this)(0, 2) * p.z + (*this)(0, 3),
                (*this)(1, 0) * p.x + (*this)(1, 1) * p.y + (*this)(1, 2) * p.z + (*this)(1, 3),
                (*this)(2, 0) * p.x + (*this)(2, 1) * p.y + (*this)(2, 2) * p.z + (*this)(2, 3)};
    }
    Vec3 dir(const Vec3& p) const {
        return {(*this)(0, 0) * p.x + (*this)(0, 1) * p.y + (*this)(0, 2) * p.z,
                (*this)(1, 0) * p.x + (*this)(1, 1) * p.y + (*this)(1, 2) * p.z,
                (*this)(2, 0) * p.x + (*this)(2, 1) * p.y + (*this)(2, 2) * p.z};
    }
};

inline Mat4 translate(const Vec3& t) {
    Mat4 r = Mat4::identity();
    r(0, 3) = t.x; r(1, 3) = t.y; r(2, 3) = t.z;
    return r;
}

inline Mat4 scale(float s) {
    Mat4 r = Mat4::identity();
    r(0, 0) = r(1, 1) = r(2, 2) = s;
    return r;
}

// Right-handed rotation of `angle` radians about unit axis `a` through the origin.
inline Mat4 rotate(float angle, Vec3 a) {
    a = normalize(a);
    float c = std::cos(angle), s = std::sin(angle), t = 1 - c;
    Mat4 r = Mat4::identity();
    r(0, 0) = t * a.x * a.x + c;       r(0, 1) = t * a.x * a.y - s * a.z; r(0, 2) = t * a.x * a.z + s * a.y;
    r(1, 0) = t * a.x * a.y + s * a.z; r(1, 1) = t * a.y * a.y + c;       r(1, 2) = t * a.y * a.z - s * a.x;
    r(2, 0) = t * a.x * a.z - s * a.y; r(2, 1) = t * a.y * a.z + s * a.x; r(2, 2) = t * a.z * a.z + c;
    return r;
}

// Rotation about an axis through `origin`.
inline Mat4 rotate_about(float angle, const Vec3& origin, const Vec3& axis) {
    return translate(origin) * rotate(angle, axis) * translate(-origin);
}

inline Mat4 look_at(const Vec3& eye, const Vec3& target, const Vec3& up) {
    Vec3 f = normalize(target - eye);
    Vec3 s = normalize(cross(f, up));
    Vec3 u = cross(s, f);
    Mat4 r = Mat4::identity();
    r(0, 0) = s.x; r(0, 1) = s.y; r(0, 2) = s.z; r(0, 3) = -dot(s, eye);
    r(1, 0) = u.x; r(1, 1) = u.y; r(1, 2) = u.z; r(1, 3) = -dot(u, eye);
    r(2, 0) = -f.x; r(2, 1) = -f.y; r(2, 2) = -f.z; r(2, 3) = dot(f, eye);
    return r;
}

inline Mat4 perspective(float fovy, float aspect, float znear, float zfar) {
    float f = 1.0f / std::tan(fovy / 2);
    Mat4 r;
    r(0, 0) = f / aspect;
    r(1, 1) = f;
    r(2, 2) = (zfar + znear) / (znear - zfar);
    r(2, 3) = 2 * zfar * znear / (znear - zfar);
    r(3, 2) = -1;
    return r;
}

inline Mat4 ortho(float l, float r_, float b, float t, float n, float f) {
    Mat4 r = Mat4::identity();
    r(0, 0) = 2 / (r_ - l); r(1, 1) = 2 / (t - b); r(2, 2) = -2 / (f - n);
    r(0, 3) = -(r_ + l) / (r_ - l); r(1, 3) = -(t + b) / (t - b); r(2, 3) = -(f + n) / (f - n);
    return r;
}

// Inverse of the upper 3x3, transposed (normal matrix). Returned as a Mat4.
inline Mat4 normal_matrix(const Mat4& a) {
    float a00 = a(0, 0), a01 = a(0, 1), a02 = a(0, 2);
    float a10 = a(1, 0), a11 = a(1, 1), a12 = a(1, 2);
    float a20 = a(2, 0), a21 = a(2, 1), a22 = a(2, 2);
    float det = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
    float id = std::fabs(det) > 1e-20f ? 1.0f / det : 0.0f;
    Mat4 r = Mat4::identity();
    // (inverse)^T
    r(0, 0) = (a11 * a22 - a12 * a21) * id; r(0, 1) = (a12 * a20 - a10 * a22) * id; r(0, 2) = (a10 * a21 - a11 * a20) * id;
    r(1, 0) = (a02 * a21 - a01 * a22) * id; r(1, 1) = (a00 * a22 - a02 * a20) * id; r(1, 2) = (a01 * a20 - a00 * a21) * id;
    r(2, 0) = (a01 * a12 - a02 * a11) * id; r(2, 1) = (a02 * a10 - a00 * a12) * id; r(2, 2) = (a00 * a11 - a01 * a10) * id;
    return r;
}

}  // namespace twr
