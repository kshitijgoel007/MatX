#include <matx.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

using namespace matx;

template <typename T> void  materializeAndTime(const char* str,  T op) {

  int iters = 10;

  auto tmp = make_tensor<typename T::value_type>(Shape(op), MATX_ASYNC_DEVICE_MEMORY, 0);
  
  cudaDeviceSynchronize();

  auto start = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < iters; i++) 
    (tmp = op).run();
  
  cudaDeviceSynchronize();
  auto stop = std::chrono::high_resolution_clock::now();
        
  // Calculate the duration
  std::chrono::duration<double> duration = stop - start;

  printf("%s: \n", str);
  printf("    Shape: \n");
  for(int i = 0; i < T::Rank(); i++) {
    printf("        dim: %d: size: %d\n", i, (int) op.Size(i));
  }

  printf("    Time: %f seconds\n", float(duration.count())/(float)iters);

  if constexpr ( T::Rank() > 0) {
    auto NormOp = vector_norm(op, {(int)T::Rank()-1}, matx::NormOrder::L2);

    if constexpr (NormOp.Rank() > 0)  {

    auto tmp2 = make_tensor<typename T::value_type>(Shape(NormOp), MATX_ASYNC_DEVICE_MEMORY, 0);
  
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < iters; i++) 
      (tmp2 = NormOp).run();
  
    cudaDeviceSynchronize();
    stop = std::chrono::high_resolution_clock::now();
        
    // Calculate the duration
    duration = stop - start;
    printf("    Vector norm time: %f seconds:\n", float(duration.count())/(float)iters);
    }
  } 
}


int main() {
#if 0
int k = 5;
int n = 70000;
int m = 50;
int D = 1024;
int d = 784;

auto A = createMockAMatrixMatX(n, k, D);
auto B = createMockBMatrixMatX(n, m, D);
auto X = createMockMnistDatasetMatX(n, d);
#endif

int k = 5;
int n = 7000;
int m = 50;
int D = 1024;
int d = 784;
    
//auto A = matx::make_tensor<float>({n, 2*k}, matx::MATX_DEVICE_MEMORY);
//auto B_i = matx::make_tensor<int32_t >({2*D, m}, matx::MATX_DEVICE_MEMORY);

auto A = matx::make_tensor<int32_t>({n, 2*k}, matx::MATX_DEVICE_MEMORY);
auto B = matx::make_tensor<int32_t>({2*D, m}, matx::MATX_DEVICE_MEMORY);
auto X = matx::make_tensor<matxFp16>({n, d}, matx::MATX_DEVICE_MEMORY);

//auto AFlat = matx::make_tensor<int32_t>({n*2*k}, matx::MATX_DEVICE_MEMORY);
//auto BFlat = matx::make_tensor<int32_t>({2*D*m}, matx::MATX_DEVICE_MEMORY);


#if 0
//auto AFlat_t = matx::flatten(A_t);
auto AFlatOp = flatten(A);

// TODO slice where approrpate

//auto BBatch_t_op = matx::remap<0>(B_t, ABatchFlat_t_op);
auto BRemapAFlatOp = remap<0>(B,AFlatOp);

auto BFlatOp = flatten(BRemapAFlatOp);

//auto XBatch_t_op = matx::remap<0>(X_t, matx::flatten(BBatch_t_op));
auto XRemapBFlatOp = remap<0>(X,BFlatOp);

//auto XBatchReshaped_t_op = matx::reshape(XBatch_t_op, {batchSize, 2 * k * m, d});
auto XReshapeOp = reshape(XRemapBFlatOp, {n, 2 * k * m, d} );

//auto XSubsetReshaped_t_op = matx::reshape(XSubset_t_op, {batchSize, 1, d});
auto XSubsetReshapeOp = reshape(X, {n, 1, d});

auto XSubsetReshapedRepmatOp = matx::repmat(XSubsetReshapeOp, {1, 2 * k * m, 1});

auto YBatchOp = (XReshapeOp - XSubsetReshapedRepmatOp); // Repmat is a workaround for minusing naively incompatibhle tensor shapes
//auto YBatchNormOp = vector_norm(YBatchOp, {2}, matx::NormOrder::L2);


materializeAndTime("A", A);
materializeAndTime("B", B);
materializeAndTime("X", X);

materializeAndTime("AFlatOp", AFlatOp);
materializeAndTime("BRemapAFlatOp", BRemapAFlatOp);
materializeAndTime("BFlatOp", BFlatOp);
materializeAndTime("XRemapBFlatOp", XRemapBFlatOp);
materializeAndTime("XReshapeOp", XReshapeOp);
materializeAndTime("XSubsetReshapeOp", XSubsetReshapeOp);
materializeAndTime("XSubsetReshapedRepmatOp", XSubsetReshapedRepmatOp);
materializeAndTime("YBatchOp", YBatchOp);
//materializeAndTime("YBatchNormOp", YBatchNormOp);
#else

   auto SqrtOp = sqrt(A);
   auto NormOp = vector_norm(SqrtOp, {1}, NormOrder::L2);

   materializeAndTime("SqrtOp", SqrtOp);
   materializeAndTime("NormOp", NormOp);

#endif
cudaDeviceSynchronize();  cudaCheckError();


// TODO materialize each op here

return 0;
}
