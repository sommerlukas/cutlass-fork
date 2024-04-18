#pragma once

// Tiled matrix multiplication kernels, generated from a template:

#define tK 16

#define MM 1
#define NN 1
#include "pvc_prefetch_impl.hpp"
#undef MM
#undef NN

#define MM 2
#define NN 1
#include "pvc_prefetch_impl.hpp"
#undef MM
#undef NN

#define MM 1
#define NN 2
#include "pvc_prefetch_impl.hpp"
#undef MM
#undef NN

#define MM 2
#define NN 2
#include "pvc_prefetch_impl.hpp"
#undef MM
#undef NN

#define MM 4
#define NN 2
#include "pvc_prefetch_impl.hpp"
#undef MM
#undef NN

#define MM 2
#define NN 4
#include "pvc_prefetch_impl.hpp"
#undef MM
#undef NN

#define MM 4
#define NN 4
#include "pvc_prefetch_impl.hpp"
#undef MM
#undef NN

#undef tK
