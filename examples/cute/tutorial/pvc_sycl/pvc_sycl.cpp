/*
// Copyright (c) 2019-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl.hpp>

#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../gemm_validation.hpp"
#include "pvc_prefetch.hpp"
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/tensor.hpp>

using test_clock = std::chrono::high_resolution_clock;

using sycl::ext::oneapi::bfloat16;

using dtype_a = bfloat16;
using dtype_b = bfloat16;
using dtype_c = bfloat16;
using dtype_acc = float;

bool identityData = false;
bool fixedData = false;
bool validate = true;
int testIterations = 1;
dtype_acc threshold = 0.01f;
size_t matrixSize = 4096;
#define B_VNNI

#define WARMUP_ITERATIONS 10

std::string makeTestName(const std::string &func, int tM, int tN, int tK,
                         int MM, int NN, size_t M, size_t N, size_t K) {
  std::ostringstream ret;
  ret << func;
  ret << "<tM:" << tM << "x" << MM << ", tN:" << tN << "x" << NN
      << ", tK:" << tK << ">";
  ret << " (M=" << M << ", N=" << N << ", K=" << K << ")";
  return ret.str();
}

template <typename T>
static void fill_matrix(std::vector<T> &M, size_t numRows, size_t numCols) {
  if (identityData) {
    std::generate(std::begin(M), std::end(M), [&] { return 1.0f; });
  } else if (fixedData) {
    for (size_t r = 0; r < numRows; r++) {
      for (size_t c = 0; c < numCols; c++) {
        M[r * numCols + c] = static_cast<float>(r + c);
      }
    }
  } else {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    std::generate(std::begin(M), std::end(M), [&] { return dist(rng); });
  }
}

template <typename T>
static void vnni_matrix(std::vector<T> &dst, const std::vector<T> &src,
                        size_t numRows, size_t numCols, size_t factor) {
  for (size_t r = 0; r < numRows / factor; r++) {
    for (size_t c = 0; c < numCols; c++) {
      for (size_t k = 0; k < factor; k++) {
        dst[r * numCols * factor + c * factor + k] =
            src[(r * factor + k) * numCols + c];
      }
    }
  }
}

template <typename DstT, typename SrcT>
static void compute_reference(std::vector<DstT> &C, const std::vector<SrcT> &A,
                              const std::vector<SrcT> &B, size_t M, size_t N,
                              size_t K) {
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      DstT sum = 0;
      for (size_t k = 0; k < K; k++) {
        sum = std::fma(static_cast<DstT>(A[m * K + k]),
                       static_cast<DstT>(B[k * N + n]), sum);
      }
      C[m * N + n] = sum;
    }
  }
}

template <typename T>
void check_results(size_t M, size_t N, const std::vector<T> &C,
                   const std::vector<T> &C_ref) {
  float err = 0.f;
  size_t error_cnt = 0;
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      auto index = m * N + n;
      auto localErr = std::fabs(C[index] - C_ref[index]) /
                      std::max(std::fabs(C[index]), std::fabs(C_ref[index]));
      err = std::max(localErr, err);
      if (localErr >= threshold) {
        error_cnt++;
        // std::cerr << "Error at m = " << m << ", n = " << n << ": (local error
        // "
        //           << localErr << "): Wanted " << C_ref[index] << ", got "
        //          << C[index] << std::endl;
        // return;
      }
    }
  }

  auto fail_rate = (float)error_cnt * 100 / (M * N);

  std::cout << "\n\n==== fail rate is: " << fail_rate << "% !!!\n" << std::endl;
}

inline size_t time_event(sycl::event &e) {
  // get start and end times
  cl_ulong start_time = e.template get_profiling_info<
      sycl::info::event_profiling::command_start>();

  cl_ulong end_time =
      e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  // return the delta
  return static_cast<size_t>(end_time - start_time);
}

template <int tM, int tN, int tK, int MM, int NN>
static void
go_dpas_blockread_vnni_tiled(sycl::queue queue, std::vector<dtype_acc> &c_vec,
                             sycl::buffer<dtype_a> a, sycl::buffer<dtype_b> b,
                             size_t M, size_t N, size_t K,
                             const std::vector<dtype_acc> &C_ref) {
  printf("%80s: ",
         makeTestName(__FUNCTION__, tM, tN, tK, MM, NN, M, N, K).c_str());
  fflush(stdout);

  int total_iterations = WARMUP_ITERATIONS + testIterations;
  if (tM * MM > M) {
    printf("M is too small.\n");
  } else if (tN * NN > N) {
    printf("N is too small.\n");
  } else {
    std::vector<size_t> event_times(total_iterations);

    sycl::range<2> group_range{(M + WG_SIZE_Y - 1) / WG_SIZE_Y,
                               (N + WG_SIZE_X - 1) / WG_SIZE_X};
    sycl::range<2> local_range{(WG_SIZE_Y + ITEM_SIZE_Y - 1) / ITEM_SIZE_Y,
                               (WG_SIZE_X + ITEM_SIZE_X - 1) / ITEM_SIZE_X};
    sycl::nd_range<2> nd_range(group_range * local_range, local_range);

    for (int test = 0; test < total_iterations; test++) {
      sycl::buffer c{c_vec};
      sycl::event ev;
      ev = queue.submit([&](sycl::handler &cgh) {
        sycl::accessor accA{a, cgh, sycl::read_only};
        sycl::accessor accB{b, cgh, sycl::read_only};
        sycl::accessor accC{c, cgh, sycl::write_only};

        cgh.parallel_for /*<dpas_blockread_vnni_tiled<tM, tN, tK, MM, NN>>*/ (
            nd_range,
            [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(16)]] {
              const int M = id.get_global_range(0) * ITEM_SIZE_Y;
              const int N = id.get_global_range(1) * ITEM_SIZE_X;
              const int m = id.get_group(0) * WG_SIZE_Y +
                            (get_sub_group_id() / SGS_PER_WG_X) * SG_SIZE_Y;
              const int n = id.get_group(1) * WG_SIZE_X +
                            (get_sub_group_id() % SGS_PER_WG_X) * SG_SIZE_X;

              auto A = accA.get_multi_ptr<sycl::access::decorated::yes>().get();
              auto B = accB.get_multi_ptr<sycl::access::decorated::yes>().get();
              auto C = accC.get_multi_ptr<sycl::access::decorated::yes>().get();

              using namespace cute;

              Tensor tAr = make_tensor<ushort>(Shape<_8, Int<MM>>{});
              Tensor tBr = make_tensor<uint>(Shape<_8, Int<NN>>{});
              Tensor tCr =
                  make_tensor<dtype_acc>(Shape<_8, Int<MM>, Int<NN>>{});

            auto A_copy = make_xe_2d_copy<XE_2D_LOAD>(make_tensor(make_gmem_ptr(A), make_shape(M, K)));
    auto B_copy = make_xe_2d_copy<XE_2D_LOAD>(make_tensor(make_gmem_ptr(B), make_shape(K, N)));
    auto C_copy = make_xe_2d_copy<XE_2D_SAVE>(make_tensor(make_gmem_ptr(C), make_shape(M, N)));
              // TODO: - decide on how to deal with vector types
              //       - create layouts with tiling/partitioning

              Tensor tAi = make_tensor(
                  make_inttuple_iter(m, 0),
                  make_layout(make_shape(_1{}, Int<MM>{}, K),
                              make_stride(_1{}, tM * E<0>{}, E<1>{})));
              Tensor tBi = make_tensor(
                  make_inttuple_iter(0, n),
                  make_layout(make_shape(_1{}, K, Int<NN>{}),
                              make_stride(_1{}, E<0>{}, tN * E<1>{})));
              Tensor tCi = make_tensor(
                  make_inttuple_iter(m, n),
                  make_layout(Shape<_1, Int<MM>, Int<NN>>{},
                              make_stride(_1{}, tM * E<0>{}, tN * E<1>{})));
              TiledMMA<MMA_Atom<XE_8x16x16_BF16BF16F32F32_NN>,
                       Layout<Shape<_1, _1, _1>>>
                  tiled_mma;

              int prefetch_k = 0;
#ifdef PREFETCH_DEFAULT
              for (int p = 0; p < PREFETCH_DISTANCE; p++) {
#ifdef B_VNNI
            // HELPER_NAME(btile_block_prefetch_vnni, 4, 4)
            //((ushort *)B, tN, K, N, prefetch_k, n);
#else
                HELPER_NAME(btile_block_prefetch_rowmajor, 4, 4)
                ((ushort *)B, tN, K, N, prefetch_k, n);
#endif
                HELPER_NAME(atile_block_prefetch_rowmajor, 4, 4)
                ((ushort *)A, tM, M, K, m, prefetch_k);
                prefetch_k += tK * KK;
              }
#endif

              for (int k = 0; k < K; k += tK) {
                copy(A_copy, tAi(_, _, k), tAr);
                copy(B_copy, tBi(_, k / 2, _), tBr);

#ifdef PREFETCH_DEFAULT
                if (k % ((PREFETCH_DISTANCE)*tK) == 0) {
                  for (int p = 0; p < PREFETCH_DISTANCE; p++) {
#ifdef B_VNNI
                // HELPER_NAME(btile_block_prefetch_vnni, 4, 4)
                //  ((ushort *)B, tN, K, N, prefetch_k, n);
#else
                    HELPER_NAME(btile_block_prefetch_rowmajor, 4, 4)
                    ((ushort *)B, tN, K, N, prefetch_k, n);
#endif
                    HELPER_NAME(atile_block_prefetch_rowmajor, 4, 4)
                    ((ushort *)A, tM, M, K, m, prefetch_k);
                    prefetch_k += tK * KK;
                  }
                }
#endif
                gemm(tiled_mma, tAr, tBr, tCr);
              }

              copy(C_copy, tCr, tCi);
            });
      });

      ev.wait_and_throw();
      event_times[test] = time_event(ev);
    }

    double average_event_time = 0.f;
    for (int i = WARMUP_ITERATIONS; i < total_iterations; i++) {
      average_event_time += event_times[i];
    }
    average_event_time /= (testIterations * 1e3);
    auto gops = 2.0 * M * N * K;
    printf("Average is %f microseconds (%f gops)\n", average_event_time,
           gops / (1e3 * average_event_time));

    if (validate) {
      printf("Checking results... ");
      fflush(stdout);
      check_results(M, N, c_vec, C_ref);
      printf(" done!\n");
    }
  }
}

int main(int argc, char **argv) {
  printf("Config:\n");
  printf("\tTest Iterations: %d\n", testIterations);
  printf("\tValidating data?: %s\n", validate ? "true" : "false");
  printf("\tFixed data?: %s\n", fixedData ? "true" : "false");

  sycl::queue queue{{sycl::property::queue::enable_profiling()}};

  const auto M = matrixSize;
  const auto N = matrixSize;
  const auto K = matrixSize;

  std::vector<dtype_a> A_vec(M * K);
  std::vector<dtype_b> B_vec(K * N);
  std::vector<dtype_b> Bvnni_vec(K * N);
  std::vector<dtype_acc> C_vec(M * N);
  std::vector<dtype_acc> C_ref(M * N);

  printf("Initializing source matrices...\n");
  fill_matrix(A_vec, M, K);
  fill_matrix(B_vec, K, N);

  vnni_matrix(Bvnni_vec, B_vec, K, N, 2);

  if (validate) {
    printf("Computing reference...\n");
    get_gemm_gold<dtype_a, dtype_b, dtype_acc>(
        M, N, K, mem_layout::row_major, mem_layout::row_major,
        (dtype_a *)A_vec.data(), (dtype_b *)B_vec.data(),
        (dtype_acc *)C_ref.data());
    // compute_reference(C_ref, A_vec, B_vec, M, N, K);
  }

  printf("Creating source buffers...\n");
  sycl::buffer A{A_vec};
  sycl::buffer B{B_vec};
  sycl::buffer Bvnni{Bvnni_vec};

  printf("Running tests...\n");

#if 0
  go_dpas_blockread_vnni_tiled<8, 16, 16, 1, 1>(queue, C_vec, A, Bvnni, M, N, K,
                                                C_ref);
  go_dpas_blockread_vnni_tiled<8, 16, 16, 2, 1>(queue, C_vec, A, Bvnni, M, N, K,
                                                C_ref);
  go_dpas_blockread_vnni_tiled<8, 16, 16, 1, 2>(queue, C_vec, A, Bvnni, M, N, K,
                                                C_ref);
  go_dpas_blockread_vnni_tiled<8, 16, 16, 2, 2>(queue, C_vec, A, Bvnni, M, N, K,
                                                C_ref);
  go_dpas_blockread_vnni_tiled<8, 16, 16, 4, 2>(queue, C_vec, A, Bvnni, M, N, K,
                                                C_ref);
  go_dpas_blockread_vnni_tiled<8, 16, 16, 2, 4>(queue, C_vec, A, Bvnni, M, N, K,

                                                C_ref);
#endif

#ifdef B_VNNI
  go_dpas_blockread_vnni_tiled<8, 16, 16, 4, 4>(queue, C_vec, A, Bvnni, M, N, K,
                                                C_ref);

#else
  go_dpas_blockread_vnni_tiled<8, 16, 16, 4, 4>(queue, C_vec, A, B, M, N, K,
                                                C_ref);
#endif

  printf("Done.\n");

  return 0;
}
