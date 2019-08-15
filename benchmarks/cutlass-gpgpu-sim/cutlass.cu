//added by me
#include <cutlass/wmma_matrix.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/wmma_gemm_traits.h>
#include "cutlass/gemm/sgemm_traits.h"
#include "cutlass/gemm/dgemm_traits.h"
#include <gemm-test/gemm_testbed.h>
#include <gemm-test/gemm.h>

int main(int argc, char* argv[]) {

//	#include <gemm-test/wmma_tests.h>
//	#include <gemm-test/sgemm_tests.h>
//	#include <gemm-test/dgemm_tests.h>

  const int n_streams = 2;
  cudaStream_t streams[n_streams];

  for (int i = 0; i < n_streams; i++) {
    cudaStreamCreate(&(streams[i]));
  }

#ifdef SGEMM_4
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
                                   cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
  SgemmTraits4;

  run_gemm<SgemmTraits4>(2048, 1024, 64, streams[0], 1);
#endif

#ifdef WMMA_45
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
  cutlass::MatrixLayout::kColumnMajor,
  cutlass::Shape<32, 64, 64> >
  WmmaGemmTraits44;
  run_gemm<WmmaGemmTraits44>(1024, 512, 512, streams[1], 2);
#endif




  cudaError_t result;
  do
  {
    result = cudaDeviceSynchronize();
  }while(result!=cudaSuccess);
  printf("Successfully Launched\n");


}


