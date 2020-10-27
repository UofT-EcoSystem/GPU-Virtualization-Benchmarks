//added by me
#include <cutlass/wmma_matrix.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/wmma_gemm_traits.h>
#include "cutlass/gemm/sgemm_traits.h"
#include "cutlass/gemm/dgemm_traits.h"
#include <gemm-test/gemm_testbed.h>
#include <cutlass/cutlass.h>

#include "interface.h"

#include <unistd.h>
#include <vector>
#include <iostream>

extern bool set_and_check(int uid, bool start);


////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm(
    int m,
    int n,
    int k,
    int lda,
    int ldb,
    int ldc,
    cudaStream_t & stream,
    int uid,
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1),
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0)) {

  typedef cutlass::gemm::Gemm<GemmTraits_> Gemm;

  // To avoid cache effects in repeated launches
  // Create multiple copies of params
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  const int l2_bytes = deviceProp.l2CacheSize;

  // 2bytes assumed for the smallest data size fp16
  const int matrix_bytes = (m * k + n * k) * 2;
  const int num_params = (l2_bytes / matrix_bytes + 1) * 4;

  typename Gemm::Params params[num_params];

  typedef test::GemmTestbed<
          typename test::GemmTestbedTraits<
                  typename GemmTraits_::GemmConfig::ScalarA>::host_type,  // AType
          typename test::GemmTestbedTraits<
                  typename GemmTraits_::GemmConfig::ScalarB>::host_type,  // BType
          typename test::GemmTestbedTraits<
                  typename GemmTraits_::Epilogue::ScalarC>::host_type,  // CType
          typename test::GemmTestbedTraits<
                  typename GemmTraits_::Epilogue::Accumulators::Element>::host_type,  // Accumulator
          typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type  // Scalar
  > TB;
  std::vector<TB*> testbeds;

  for (int i = 0; i < num_params; i++) {
      testbeds.push_back(new TB(
              m,
              n,
              k,
              lda,
              ldb,
              ldc,
              cutlass::convert(GemmTraits_::kLayoutA),
              cutlass::convert(GemmTraits_::kLayoutB),
              stream,
              alpha,
              beta
      ));
      testbeds[i]->initialize();

      params[i].initialize(testbeds[i]->M(),
                        testbeds[i]->N(),
                        testbeds[i]->K(),
                        testbeds[i]->alpha,
                        testbeds[i]->ptr_A(),
                        testbeds[i]->lda(),
                        testbeds[i]->ptr_B(),
                        testbeds[i]->ldb(),
                        testbeds[i]->beta,
                        testbeds[i]->ptr_C_initial(),
                        testbeds[i]->ldc(),
                        testbeds[i]->ptr_computed(),
                        testbeds[i]->ldc());
  }

  // mark setup done
  cudaStreamSynchronize(stream);
  while (!set_and_check(uid, true)) {
    usleep(1);
  }

  bool can_exit = false;
  int index = 0;

  while (!can_exit) {
    Gemm::launch(params[index], stream);
    index = (index + 1) % num_params;

//    cudaStreamSynchronize(stream);
    can_exit = set_and_check(uid, false);
  }


//  cudaError_t result;
//  do
//  {
//  result = cudaDeviceSynchronize();
//  }while(result!=cudaSuccess);
//  printf("Successfully Launched\n");
//
//  int save=1;
//  int completedsuccessfully=testbed.verify_with_host(save,save);
//  if (completedsuccessfully==1){
//    printf("Result Verified\n");
//  }
//  else{
//    printf("ERROR");
//  }

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits_>
static void run_gemm(
    int m,
    int n,
    int k,
    cudaStream_t & stream,
    int uid,
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type alpha =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(1),
    typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type beta =
        typename test::GemmTestbedTraits<typename GemmTraits_::Epilogue::Scalar>::host_type(0)) {
  int lda = GemmTraits_::kLayoutA == cutlass::MatrixLayout::kColumnMajor ? m : k;
  int ldb = GemmTraits_::kLayoutB == cutlass::MatrixLayout::kColumnMajor ? k : n;
  run_gemm<GemmTraits_>(m, n, k, lda, ldb, m, stream, uid, alpha, beta);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename GemmTraits>
int main_templated(int argc, char* argv[], int uid, cudaStream_t & stream) {
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

  run_gemm<GemmTraits>(x, y, z, stream, uid);

  return 0;
}

#ifdef CUT_SGEMM
int main_sgemm(int argc, char* argv[], int uid, cudaStream_t & stream) {
  typedef cutlass::gemm::SgemmTraits<cutlass::MatrixLayout::kColumnMajor,
          cutlass::MatrixLayout::kRowMajor, cutlass::Shape<8, 128, 128> >
  SgemmTraits4;
  //run_gemm<GemmTraits>(2048, 1024, 64, stream, 1);
  return main_templated<SgemmTraits4>(argc, argv, uid, stream);
}
#endif


#ifdef CUT_WMMA
int main_wmma(int argc, char* argv[], int uid, cudaStream_t & stream) {
  typedef cutlass::gemm::WmmaGemmTraits<cutlass::MatrixLayout::kRowMajor,
          cutlass::MatrixLayout::kColumnMajor,
          cutlass::Shape<32, 64, 64> >
            WmmaGemmTraits44;
  //run_gemm<WmmaGemmTraits44>(1024, 512, 512, stream, 2);
  return main_templated<WmmaGemmTraits44>(argc, argv, uid, stream);
}
#endif



