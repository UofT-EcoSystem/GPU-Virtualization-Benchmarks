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
#include "cutlass/gemm/igemm_traits.h"


///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassIgemmNN(
  int M,
  int N,
  int K,
  int8_t alpha,
  int8_t const *A,
  int lda,
  int8_t const *B,
  int ldb,
  int8_t beta,
  int *C,
  int ldc,
  cudaStream_t& stream) {

  // Note, GemmTraits<> is a generic template defined for various general matrix product
  // computations within CUTLASS. It is intended to be maximally flexible, and consequently
  // it contains numerous template arguments.
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
  //
  typedef cutlass::gemm::IgemmTraits<
    cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
    cutlass::MatrixLayout::kColumnMajor  // layout of B matrix
//    cutlass::Shape<128, 128, 128>, 
//    int8_t
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

// Kernel to initialize a matrix with small integers for igemm.
__global__ void InitIntMatrix_kernel(
  int8_t *matrix,
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
    int8_t value = int8_t(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitIntMatrix(int8_t *matrix, int ldm, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitIntMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns, seed);

  return cudaGetLastError();
}



// Allocates device memory for a matrix then fills with arbitrary small integers.
// Input matrix for igemm
cudaError_t AllocateInt8Matrix(int8_t **matrix, int ldm, int rows, int columns, int num_matrices, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(int8_t) * ldm * columns;
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

  // Initialize matrix elements to arbitrary small integers.
  for (int i = 0; i < num_matrices; ++i) {
      int8_t *p_matrix = &((*matrix)[i * ldm * columns]);
      result = InitIntMatrix(p_matrix, ldm, rows, columns, seed + i);

      if (result != cudaSuccess) {
        std::cerr << "Failed to initialize matrix: "
          << cudaGetErrorString(result) << std::endl;
        return result;
      }
  }
  
  return result;
}

// Allocates device memory for a matrix then fills with arbitrary small integers.
// Output matrix for igemm
cudaError_t AllocateIntMatrix(int **matrix, int ldm, int rows, int columns, int num_matrices, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(int) * ldm * columns;
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

  return result;
}



// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  int8_t alpha,
  int8_t const *A,
  int lda,
  int8_t const *B,
  int ldb,
  int8_t beta,
  int *C,
  int ldc) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    int accumulator = 0;

    for (int k = 0; k < K; ++k) {
      accumulator += A[i + k * lda] * B[k + j * ldb];
    }

    C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
  }
}

// Reference GEMM computation.
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  int8_t alpha,
  int8_t const *A,
  int lda,
  int8_t const *B,
  int ldb,
  int8_t beta,
  int *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaGetLastError();
}

// Data setup
cudaError_t SetupIgemm(int_mm_info& igemm_info) {
  cudaError_t result;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  // Determine number of matrices in the ring buffer
  int max_size = std::max(igemm_info.nitems_A, 
                          std::max(igemm_info.nitems_B, igemm_info.nitems_C));

  igemm_info.num_matrices = (1 << 30) / (max_size * sizeof(int));
  printf("Number of matrices in the buffer: %d\n", igemm_info.num_matrices);



  int seed = time(0);
  printf("Seed for A: %d \n", seed);
  result = AllocateInt8Matrix(&(igemm_info.A), igemm_info.lda, igemm_info.M, 
                          igemm_info.K, igemm_info.num_matrices, seed);

  if (result !=  cudaSuccess) {
    return result;
  }


  seed = time(0) >> 3;
  printf("Seed for B: %d \n", seed);
  result = AllocateInt8Matrix(&(igemm_info.B), igemm_info.ldb, igemm_info.K, 
                          igemm_info.N, igemm_info.num_matrices, seed);

  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateIntMatrix(&(igemm_info.C_cutlass), igemm_info.ldc, igemm_info.M, 
                          igemm_info.N, igemm_info.num_matrices, 101);

  if (result != cudaSuccess) {
    return result;
  }

  
  result = AllocateIntMatrix(&(igemm_info.C_reference), igemm_info.ldc, igemm_info.M, 
                            igemm_info.N, igemm_info.num_matrices, 101);

  return result;
}

// Validate kernel results
cudaError_t ValidateIgemm(int_mm_info& igemm_info, int niter) {
  //
  // Verify.
  //
  // Launch reference GEMM
  cudaError_t result;

  int buffer_idx = 0;

  for (int i = 0; i < niter; i++) {
      int8_t* A_adj = &(igemm_info.A[buffer_idx * igemm_info.nitems_A]);
      int8_t* B_adj = &(igemm_info.B[buffer_idx * igemm_info.nitems_B]);
      int* C_reference_adj = &(igemm_info.C_reference[buffer_idx * igemm_info.nitems_C]);

      result = ReferenceGemm(igemm_info.M, igemm_info.N, igemm_info.K, 
                         igemm_info.alpha, A_adj, igemm_info.lda, B_adj, 
                         igemm_info.ldb, igemm_info.beta, C_reference_adj, 
                         igemm_info.ldc);


      buffer_idx++;

      if(buffer_idx >= igemm_info.num_matrices) buffer_idx = 0;
  }




  if (result != cudaSuccess) {
    std::cerr << "Reference IGEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    return result;
  }

  // Copy to host and verify equivalence.
  std::vector<int> host_cutlass(igemm_info.nitems_C * igemm_info.num_matrices, 0);
  std::vector<int> host_reference(igemm_info.nitems_C * igemm_info.num_matrices, 0);

  result = cudaMemcpy(host_cutlass.data(), igemm_info.C_cutlass, 
           sizeof(int) * igemm_info.nitems_C * igemm_info.num_matrices, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy CUTLASS IGEMM results: "
      << cudaGetErrorString(result) << std::endl;

    return result;
  }

  result = cudaMemcpy(host_reference.data(), igemm_info.C_reference, 
           sizeof(int) * igemm_info.nitems_C * igemm_info.num_matrices, cudaMemcpyDeviceToHost);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy Reference IGEMM results: "
      << cudaGetErrorString(result) << std::endl;

    return result;
  }

  //
  // Test for bit equivalence of results.
  //

  if (host_cutlass != host_reference) {
    std::cerr << "IGEMM CUTLASS results incorrect." << std::endl;

    for (int i = 0; i < 128; ++i) {
      std::cout << host_cutlass[i] << "," << host_reference[i] << std::endl;
    }


    return cudaErrorUnknown;
  }
  else {
    std::cout << "Matched!" << std::endl;

  }

  return cudaSuccess;

}



