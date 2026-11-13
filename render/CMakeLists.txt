cmake_minimum_required(VERSION 3.20)
project(twin_rc_render CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

# GLFW: window + GL context
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
  URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
  URL_HASH SHA256=b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9)

# nlohmann/json: scene parsing
FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d)

FetchContent_MakeAvailable(glfw json)

add_executable(twin_rc_render
  src/main.cpp
  src/mesh.cpp
  src/png.cpp
  src/renderer.cpp
  src/scene.cpp)

target_link_libraries(twin_rc_render PRIVATE glfw nlohmann_json::nlohmann_json)

if(APPLE)
  find_library(OPENGL_FRAMEWORK OpenGL REQUIRED)
  target_link_libraries(twin_rc_render PRIVATE ${OPENGL_FRAMEWORK})
else()
  # Windows / Linux: load GL functions through GLEW (pre-generated sources).
  set(glew-cmake_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  set(ONLY_LIBS ON CACHE BOOL "" FORCE)
  FetchContent_Declare(glew
    URL https://github.com/Perlmint/glew-cmake/archive/refs/tags/glew-cmake-2.2.0.tar.gz)
  FetchContent_MakeAvailable(glew)
  find_package(OpenGL REQUIRED)
  target_link_libraries(twin_rc_render PRIVATE libglew_static OpenGL::GL)
endif()

if(MSVC)
  target_compile_options(twin_rc_render PRIVATE /W3)
else()
  target_compile_options(twin_rc_render PRIVATE -Wall -Wextra)
endif()
