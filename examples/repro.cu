#include <matx.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

// Macro for checking CUDA errors following a CUDA launch or API call
#define cudaCheckError() {                                           \
    cudaError_t e = cudaGetLastError();                              \
    if (e != cudaSuccess) {                                          \
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__,     \
               cudaGetErrorString(e));                               \
        exit(EXIT_FAILURE);                                          \
    } else {                                                         \
        printf("CUDA call successful: %s:%d\n", __FILE__, __LINE__); \
    }                                                                \
}

int main() {

  int n = 10;
  int d = 8;
    
  auto in = matx::make_tensor<float>({n, d}, matx::MATX_DEVICE_MEMORY);
  (in = matx::random<float>({n, d}, matx::UNIFORM)).run();

  auto reshape_op = matx::reshape(in, {n, 1, d});
  auto norm_op = matx::vector_norm(reshape_op, {0}, matx::NormOrder::L2);

  printf("reshaped input:\n");  matx::print(reshape_op);
  printf("after norm output (should not be zeros):\n"); matx::print(norm_op);

return 0;
}
