// OpenGL + GLFW includes. macOS ships GL 4.1 core headers; other platforms load
// functions through GLEW.
#pragma once

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <OpenGL/gl3.h>
#else
#include <GL/glew.h>
#endif

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
