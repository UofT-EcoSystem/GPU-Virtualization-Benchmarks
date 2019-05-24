/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#include "cuda_profiler_api.h"
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>

volatile bool ready = false;
volatile bool should_stop = false;

const char * ready_fifo = "/tmp/ready"; 


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

void regtileSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
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

    /* Create CUDA start & stop events to record total elapsed time of kernel execution */
    cudaEvent_t start;
    cudaError_t error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {   
        fprintf(stderr, "Failed to create start event (error code %s)!\n", 
                                                cudaGetErrorString(error));   
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {   
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", 
                                                cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    /* End event creation */ 
    
    
    /* Write to pipe to signal the wrapper script that we are done with data setup */
    char pid[10];
    sprintf(pid, "%d", getpid());

    int fd = open(ready_fifo, O_WRONLY);
    int res = write(fd, pid, strlen(pid));
    close(fd);

    if (res > 0) printf("Parboil sgemm write success to the pipe!\n");
    /* End pipe writing */
    
    
    /* Spin until master tells me to start kernels */
    while (!ready);
    /* End spinning */
    
    
    /* Record the start event and start nvprof profiling */
    cudaProfilerStart();
    
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", 
                                                cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    /* End CUDA start records */
 

    dim3 grid( m/TILE_M, n/TILE_N ), threads( TILE_N, TILE_TB_HEIGHT );

    while(!should_stop) 
      mysgemmNT<<<grid, threads>>>( A, lda, B, ldb, C, ldc, k, alpha, beta);

    CHECK_ERROR("mySgemm");

    /* Record and wait for the stop event */
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", 
                                                cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    cudaThreadSynchronize();

    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", 
                                                            cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    
    cudaProfilerStop();
    
    /* End stop CUDA event handling */
    
    
    /* Output total elapsed time */
    float msecTotal = 0.0f;
    // !!! Important: must use this print format, the data processing 
    // script requires this information 
    error = cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Total elapsed time: %f ms\n", msecTotal);
    /* End elpased time recording */



}

