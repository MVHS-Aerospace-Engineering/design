// Dependency-free PNG writer (RGBA8, zlib "stored" blocks: no compression, but
// valid and readable everywhere).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace twr {

// `rgba` is width*height*4 bytes, top row first.
bool write_png(const std::string& path, int width, int height, const std::vector<uint8_t>& rgba);

}  // namespace twr
