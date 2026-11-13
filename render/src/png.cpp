#include "png.hpp"

#include <algorithm>
#include <array>
#include <fstream>

namespace twr {

namespace {

uint32_t crc32(const uint8_t* data, size_t n, uint32_t crc = 0) {
    static const auto table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return ~crc;
}

void put32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x >> 24); v.push_back(x >> 16); v.push_back(x >> 8); v.push_back(x);
}

void chunk(std::ofstream& out, const char* type, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> buf;
    put32(buf, static_cast<uint32_t>(data.size()));
    buf.insert(buf.end(), type, type + 4);
    buf.insert(buf.end(), data.begin(), data.end());
    put32(buf, crc32(buf.data() + 4, buf.size() - 4));
    out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
}

}  // namespace

bool write_png(const std::string& path, int width, int height, const std::vector<uint8_t>& rgba) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    out.write(reinterpret_cast<const char*>(sig), 8);

    std::vector<uint8_t> ihdr;
    put32(ihdr, width);
    put32(ihdr, height);
    ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0});  // 8-bit RGBA
    chunk(out, "IHDR", ihdr);

    // Raw scanlines with filter byte 0.
    std::vector<uint8_t> raw;
    const size_t stride = size_t(width) * 4;
    raw.reserve((stride + 1) * height);
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgba.begin() + y * stride, rgba.begin() + (y + 1) * stride);
    }

    // zlib stream of stored deflate blocks.
    std::vector<uint8_t> z = {0x78, 0x01};
    uint32_t a = 1, b = 0;
    for (uint8_t c : raw) { a = (a + c) % 65521; b = (b + a) % 65521; }
    for (size_t pos = 0; pos < raw.size() || pos == 0;) {
        size_t len = std::min<size_t>(65535, raw.size() - pos);
        bool last = pos + len >= raw.size();
        z.push_back(last ? 1 : 0);
        z.push_back(len & 0xFF); z.push_back(len >> 8);
        z.push_back(~len & 0xFF); z.push_back((~len >> 8) & 0xFF);
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + len);
        pos += len;
        if (last) break;
    }
    put32(z, (b << 16) | a);
    chunk(out, "IDAT", z);
    chunk(out, "IEND", {});
    return static_cast<bool>(out);
}

}  // namespace twr
