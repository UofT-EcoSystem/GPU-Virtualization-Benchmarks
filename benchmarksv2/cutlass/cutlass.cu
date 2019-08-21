//added by me
#include <cutlass/wmma_matrix.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/wmma_gemm_traits.h>
#include "cutlass/gemm/sgemm_traits.h"
#include "cutlass/gemm/dgemm_traits.h"
#include <gemm-test/gemm_testbed.h>
#include <gemm-test/gemm.h>

#include "interface.h"

extern bool set_and_check(int uid);

template <typename GemmTraits>
int main_templated(int argc, char* argv[], int uid) {
  if (argc < 4) {
    printf("Cutlass needs at least three arguments: x, y, z\n");
    exit(1);
  }

  int x, y, z;

  x = atoi(argv[1]);
  y = atoi(argv[2]);
  z = atoi(argv[3]);
  if (x < 1 || y < 1 || z < 1) {
    printf("Invalid argument\n");
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  bool can_exit = false;
  const int n_runs = 1;

  while (!can_exit) {
    run_gemm<GemmTraits>(x, y, z, stream, n_runs);

    cudaStreamSynchronize(stream);

    can_exit = set_and_check(uid);
  }

  return 0;
}

#ifdef CUT_SGEMM
int main_sgemm(int argc, char* argv[], int uid) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
          cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
  SgemmTraits4;
  //run_gemm<GemmTraits>(2048, 1024, 64, stream, 1);
  return main_templated<SgemmTraits4>(argc, argv, uid);
}
#endif


#ifdef CUT_WMMA
int main_wmma(int argc, char* argv[], int uid) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
          cutlass::MatrixLayout::kColumnMajor,
          cutlass::Shape<32, 64, 64> >
            WmmaGemmTraits44;
  //run_gemm<WmmaGemmTraits44>(1024, 512, 512, stream, 2);
  return main_templated<WmmaGemmTraits44>(argc, argv, uid);
}
#endif



