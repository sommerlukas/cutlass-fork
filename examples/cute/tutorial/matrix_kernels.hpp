#pragma once

#include "matrix_helpers.hpp"

// Tiled matrix multiplication kernels, generated from a template:

#define tK 16

#if 0
#define MM 1
#define NN 1
#include "matrix_kernel_tiled.hpp"
#undef MM
#undef NN

#define MM 2
#define NN 1
#include "matrix_kernel_tiled.hpp"
#undef MM
#undef NN

#define MM 1
#define NN 2
#include "matrix_kernel_tiled.hpp"
#undef MM
#undef NN

#define MM 2
#define NN 2
#include "matrix_kernel_tiled.hpp"
#undef MM
#undef NN

#define MM 4
#define NN 2
#include "matrix_kernel_tiled.hpp"
#undef MM
#undef NN

#define MM 2
#define NN 4
#include "matrix_kernel_tiled.hpp"
#undef MM
#undef NN
#endif

#define MM 4
#define NN 4
#include "matrix_kernel_tiled.hpp"
#undef MM
#undef NN

#undef tK
