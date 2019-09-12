/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Main entry of dense matrix-matrix multiplication kernel
 */

#ifdef PARBOIL_SGEMM

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <parboil.h>
#include <iostream>

#include <interface.h>

extern bool set_and_check(int uid, bool start);

/* 
 * Kernel of dense matrix-matrix multiplication kernel.
 * The algorithm is based on CUDA sgemm code from Vasily Volkov
 * at UC Berkeley.
 */

#define CHECK_ERROR(errorMessage) {                                    \
  cudaError_t err = cudaGetLastError();                                    \
  if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
        errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
    exit(EXIT_FAILURE);                                                  \
  }                                                                        \
}

// CML x RML = CML, baseline version, 510FLOP/s on Fermi
/* Pseudo code
   for i < M ; i += 64   // thread block.x
   for j < N; j += 16   // thread block.y
   for tx = 0; tx < 16; tx++ // thread index x; tile of M loop
   for ty = 0; ty < 4 ; ty++ // thread index y; tile of M loop

   for m < 16; m += 1;
   c[m] = 0.0f

   for k < K; k += 4   // seq

   b[ty][tx] = B[k+ty][j+tx]

   for l < 4; l +=1   // seq
   for m < 16; m +=1 // seq
   c[m] += A[i+ty*16+tx][k+l]+b[l][m]

 */

// Parameters of tile sizes
#define TILE_N 16 
#define TILE_TB_HEIGHT 8
#define TILE_M (TILE_N*TILE_TB_HEIGHT)

__global__ void mysgemmNT( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
  // Partial results 
  float c[TILE_N];
  for (int i=0; i < TILE_N; i++)
    c[i] = 0.0f;
  int mid = threadIdx.y * blockDim.x + threadIdx.x; //flattened id
  int m = blockIdx.x * TILE_M + mid;
  int n = blockIdx.y * TILE_N + threadIdx.x;
  __shared__ float b_s[TILE_TB_HEIGHT][TILE_N];
  for (int i = 0; i < k; i+=TILE_TB_HEIGHT) {
    float a; 
    b_s[threadIdx.y][threadIdx.x]=B[n + (i+threadIdx.y)*ldb];
    __syncthreads();
    for (int j = 0; j < TILE_TB_HEIGHT; j++) {
      a = A[m + (i+j)*lda];
      for (int kk = 0; kk < TILE_N; kk++)
        c[kk] += a * b_s[j][kk];

    }
    __syncthreads();
  }
  int t = ldc*blockIdx.y * TILE_N + m;
  for (int i = 0; i < TILE_N; i++) {
    C[t+i*ldc] = C[t+i*ldc] * beta + alpha * c[i];
  }
}

void regtileSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc, cudaStream_t stream )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  // In this code we assume the matrix sizes are multiple of tile size
  if ((m%TILE_M) || (n%TILE_N)) {
    std::cerr << "unsupported size of matrix. m should be multiple of " << TILE_M
      << "; n should be multiple of " << TILE_N << std::endl;
  }


  dim3 grid( m/TILE_M, n/TILE_N ), threads( TILE_N, TILE_TB_HEIGHT );
  mysgemmNT<<<grid, threads, 0, stream>>>( A, lda, B, ldb, C, ldc, k, alpha, beta);
  CHECK_ERROR("mySgemm");


}

/***************************** End kernel ************************/


// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

extern "C"
void computeGold(float *, const float*, const float*, unsigned int, unsigned int, unsigned int);

int main_sgemm (int argc, char *argv[], int uid) {

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  float *dA, *dB, *dC;
  size_t A_sz, B_sz, C_sz;
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  pb_InitializeTimerSet(&timers);

  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL))
    {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
    }
 
  /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  // load A
  readColMajorMatrixFile(params->inpFiles[0],
      matArow, matAcol, matA);
  // copy A to device memory
  A_sz = matArow*matAcol*sizeof(float);

  // load B^T
  readColMajorMatrixFile(params->inpFiles[2],
      matBcol, matBrow, matBT);

  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  B_sz = matBrow*matBcol*sizeof(float);

  // allocate space for C
  C_sz = matArow*matBcol*sizeof(float);

  // create cuda stream for this benchmark
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // CUDA memory allocation
  std::vector<float> matC(matArow*matBcol);
  cudaMalloc((void**)&dA, A_sz);
  cudaMalloc((void**)&dB, B_sz);
  cudaMalloc((void**)&dC, C_sz);

  // Copy A and B^T into device memory
  pb_SwitchToTimer( &timers, pb_TimerID_COPY );
  cudaMemcpyAsync(dA, &matA.front(), A_sz, cudaMemcpyHostToDevice, stream); 
  cudaMemcpyAsync(dB, &matBT.front(), B_sz, cudaMemcpyHostToDevice, stream); 

  pb_SwitchToTimer( &timers, pb_TimerID_KERNEL );

  set_and_check(uid, true);
  while (!set_and_check(uid, true)) {
    usleep(100);
  }

  bool can_exit = false;

  while (!can_exit) {
    // Use standard sgemm interface
    regtileSgemm('N', 'T', matArow, matBcol, matAcol, 1.0f, \
        dA, matArow, dB, matBcol, 0.0f, dC, matArow, stream);

    cudaStreamSynchronize(stream);

    can_exit = set_and_check(uid, false);
  }

  // Done launching kernel
  if (params->outFile) {
    pb_SwitchToTimer( &timers, pb_TimerID_COPY );
    cudaMemcpyAsync(&matC.front(), dC, C_sz, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    /* Write C to file */
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    writeColMajorMatrixFile(params->outFile,
	matArow, matBcol, matC); 
  }

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  double GPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_KERNEL]));
  std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/GPUtime/1e9 << std::endl;
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
  cudaStreamDestroy(stream);
  return 0;
}

#endif
