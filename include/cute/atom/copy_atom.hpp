#pragma once 

#include <upstream-cutlass/include/cute/atom/copy_atom.hpp>

#if defined(SYCL_INTEL_TARGET)
#include <cute/atom/copy_traits_xe.hpp>
#endif
