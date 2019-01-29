
#include <unistd.h>
#include <inttypes.h>
#include <iostream>
#include <vector>


#include "launch_gemm.h"


//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/gemm.h"


// Defines cutlass::gemm::SgemmTraits, the structural components 
// for single-precision and integer GEMM
#include "cutlass/gemm/sgemm_traits.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////



// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  cudaStream_t& stream) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size.
  //
  // Note, GemmTraits<> is a generic template defined for various general matrix product
  // computations within CUTLASS. It is intended to be maximally flexible, and consequently
  // it contains numerous template arguments.
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
  //
  typedef cutlass::gemm::SgemmTraits<
    cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
    cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
    cutlass::Shape<8, 128, 128>            // threadblock tile size
  >
    GemmTraits;

  // Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
  typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

  // Construct and initialize CUTLASS GEMM parameters object.
  //
  // One of CUTLASS's design patterns is to define parameters objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //
  typename Gemm::Params params;

  int result = params.initialize(
    M,     // GEMM M dimension
    N,     // GEMM N dimension
    K,     // GEMM K dimension
    alpha, // scalar alpha
    A,     // matrix A operand
    lda,
    B,     // matrix B operand
    ldb,
    beta,  // scalar beta
    C,     // source matrix C
    ldc,
    C,     // destination matrix C (may be different memory than source C matrix)
    ldc
  );

  if (result) {
    std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
    return cudaErrorInvalidValue;
  }

  // Launch the CUTLASS GEMM kernel.
  Gemm::launch(params, stream);

  // Return any errors associated with the launch or cudaSuccess if no error.
  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers for sgemm.
__global__ void InitFloatMatrix_kernel(
  float *matrix,
  int ldm,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i + j * ldm;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    float value = float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitFloatMatrix(float *matrix, int ldm, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitFloatMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns, seed);

  return cudaGetLastError();
}

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateFloatMatrix(float **matrix, int ldm, int rows, int columns, 
                                int num_matrices, int seed = 0, bool random = true) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(float) * ldm * columns;
  size_t sizeof_matrices = sizeof_matrix * num_matrices;


  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrices);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrices);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  if (random) {
      // Initialize matrix elements to arbitrary small integers.
      result = InitFloatMatrix(*matrix, ldm, rows, columns*num_matrices, seed);
  }
  
  
  return result;
}

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    float accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Data setup
cudaError_t SetupSgemm(float_mm_info& sgemm_info) {
  cudaError_t result;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  // Determine number of matrices in the ring buffer
  int max_size = std::max(sgemm_info.nitems_A, 
                          std::max(sgemm_info.nitems_B, sgemm_info.nitems_C));
  size_t sizeof_maxmat = (max_size * sizeof(float));

  // L1 cache on Volta is 128KB
  // Avoid cache effects by allocating a circular buffer of at least that size
  sgemm_info.num_matrices = (1 << 17) > sizeof_maxmat ?
                            (1 << 17) / sizeof_maxmat: 1;

  printf("Number of matrices in the buffer: %d\n", sgemm_info.num_matrices);


  int seed = time(0);
  printf("Seed for A: %d \n", seed);
  result = AllocateFloatMatrix(&(sgemm_info.A), sgemm_info.lda, sgemm_info.M, 
                          sgemm_info.K, sgemm_info.num_matrices, seed, true);

  if (result !=  cudaSuccess) {
    return result;
  }


  seed = time(0) >> 3;
  printf("Seed for B: %d \n", seed);
  result = AllocateFloatMatrix(&(sgemm_info.B), sgemm_info.ldb, sgemm_info.K, 
                          sgemm_info.N, sgemm_info.num_matrices, seed, true);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateFloatMatrix(&(sgemm_info.C_cutlass), sgemm_info.ldc, sgemm_info.M, 
                          sgemm_info.N, sgemm_info.num_matrices, 101, false);

  if (result != cudaSuccess) {
    return result;
  }

  
  result = AllocateFloatMatrix(&(sgemm_info.C_reference), sgemm_info.ldc, sgemm_info.M, 
                            sgemm_info.N, sgemm_info.num_matrices, 101, false);

  return result;
}


// Validate kernel results
cudaError_t ValidateSgemm(float_mm_info& sgemm_info) {
  //
  // Verify.
  //
  // Launch reference GEMM
  cudaError_t result;

  int buffer_idx = 0;

  for (int i = 0; i < sgemm_info.niter; i++) {
      float* A_adj = &(sgemm_info.A[buffer_idx * sgemm_info.nitems_A]);
      float* B_adj = &(sgemm_info.B[buffer_idx * sgemm_info.nitems_B]);
      float* C_reference_adj = &(sgemm_info.C_reference[buffer_idx * sgemm_info.nitems_C]);

      result = ReferenceGemm(sgemm_info.M, sgemm_info.N, sgemm_info.K, 
                         sgemm_info.alpha, A_adj, sgemm_info.lda, B_adj, 
                         sgemm_info.ldb, sgemm_info.beta, C_reference_adj, 
                         sgemm_info.ldc);

      buffer_idx++;

      if(buffer_idx >= sgemm_info.num_matrices) buffer_idx = 0;
  }


  if (result != cudaSuccess) {
    std::cerr << "Reference GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    return result;
  }

  // Copy to host and verify equivalence.
  std::vector<float> host_cutlass(sgemm_info.nitems_C * sgemm_info.num_matrices, 0);
  std::vector<float> host_reference(sgemm_info.nitems_C * sgemm_info.num_matrices, 0);

  size_t data_size = sizeof(float) * sgemm_info.nitems_C * sgemm_info.num_matrices;

  result = cudaMemcpy(host_cutlass.data(), sgemm_info.C_cutlass, 
          data_size, cudaMemcpyDeviceToHost);


  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    return result;
  }

  result = cudaMemcpy(host_reference.data(), sgemm_info.C_reference, 
           sizeof(float) * sgemm_info.nitems_C * sgemm_info.num_matrices, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Reference GEMM results: "
      << cudaGetErrorString(result) << std::endl;

    return result;
  }

  //
  // Test for bit equivalence of results.
  //

  if (host_cutlass != host_reference) {
    std::cerr << "SGEMM CUTLASS results incorrect." << std::endl;

    

    return cudaErrorUnknown;
  }

  return cudaSuccess;

}




