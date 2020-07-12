/***********************************************
	streamcluster_cuda.cu
	: parallelized code of streamcluster
	
	- original code from PARSEC Benchmark Suite
	- parallelization with CUDA API has been applied by
	
	Shawn Sang-Ha Lee - sl4ge@virginia.edu
	University of Virginia
	Department of Electrical and Computer Engineering
	Department of Computer Science
	
***********************************************/
#include "streamcluster_header.hpp"

#include "unistd.h"

using namespace std;

extern int set_and_check(int uid, bool start);

// AUTO-ERROR CHECK FOR ALL CUDA FUNCTIONS
#define CUDA_SAFE_CALL( call) do {										\
   cudaError err = call;												\
   if( cudaSuccess != err) {											\
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
               __FILE__, __LINE__, cudaGetErrorString( err) );			\
   exit(EXIT_FAILURE);													\
   } } while (0)

#define THREADS_PER_BLOCK 512
#define MAXBLOCKS 65536
//#define CUDATIME

//=======================================
// Euclidean Distance
//=======================================
__device__ float
d_dist(int p1, int p2, int num, int dim, float *coord_d)
{
  float retval = 0.0;
  for(int i = 0; i < dim; i++){
    float tmp = coord_d[(i*num)+p1] - coord_d[(i*num)+p2];
    retval += tmp * tmp;
  }
  return retval;
}

//=======================================
// Kernel - Compute Cost
//=======================================
__global__ void
kernel_compute_cost(int num, int dim, long x, Point *p, int K, int stride,
                    float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d)
{
  // block ID and global thread ID
  const int bid  = blockIdx.x + gridDim.x * blockIdx.y;
  const int tid = blockDim.x * bid + threadIdx.x;

  if(tid < num)
  {
    float *lower = &work_mem_d[tid*stride];

    // cost between this point and point[x]: euclidean distance multiplied by weight
    float x_cost = d_dist(tid, x, num, dim, coord_d) * p[tid].weight;

    // if computed cost is less then original (it saves), mark it as to reassign
    if ( x_cost < p[tid].cost )
    {
      switch_membership_d[tid] = 1;
      lower[K] += x_cost - p[tid].cost;
    }
      // if computed cost is larger, save the difference
    else
    {
      lower[center_table_d[p[tid].assign]] += p[tid].cost - x_cost;
    }
  }
}

//=======================================
// Free Device Memory
//=======================================
void freeDevMem(state_t & state)
{
  CUDA_SAFE_CALL( cudaFree(state.center_table_d));
  CUDA_SAFE_CALL( cudaFree(state.switch_membership_d));
  CUDA_SAFE_CALL( cudaFree(state.p));
  CUDA_SAFE_CALL( cudaFree(state.coord_d));
}


//=======================================
// pgain Entry - CUDA SETUP + CUDA CALL
//=======================================

float pgain(int x, Points* points, state_t & state, args_t & args, float z) {
  cudaError_t error;

  // size of each work_mem segment
  int stride = state.k + 1;
  // number of centers
  int K	= state.k ;
  // number of points
  int num =  points->num;
  // number of dimension
  int dim =  points->dim;

  // number of threads == number of data points
  int nThread = num;

  //=========================================
  // ALLOCATE HOST MEMORY + DATA PREPARATION
  //=========================================
  auto work_mem_h = std::unique_ptr<float[]>(new float[stride * (nThread + 1)]);

  // Only on the first iteration
  if(state.iter == 0)
  {
    state.coord_h = std::unique_ptr<float[]>(new float[num * dim]);
  }

  // build center-index table
  int count = 0;
  for( int i = 0; i < num; i++)
  {
    if( state.is_center[i] )
    {
      state.center_table[i] = count++;
    }
  }

  // Extract 'coord'
  // Only if first iteration OR coord has changed
  if(state.isCoordChanged || state.iter == 0) {
    for(int i = 0; i < dim; i++) {
      for(int j = 0; j < num; j++) {
        state.coord_h[(num * i) + j] = points->p[j].coord[i];
      }
    }
  }

  //=======================================
  // ALLOCATE GPU MEMORY
  //=======================================
  // device memory
  float *work_mem_d;

  CUDA_SAFE_CALL( cudaMalloc((void**) &work_mem_d,
                             stride * (nThread + 1) * sizeof(float)) );

  // Only on the first iteration
  if( state.iter == 0 )
  {
    CUDA_SAFE_CALL( cudaMalloc((void**) &(state.center_table_d),
                               num * sizeof(int)) );
    CUDA_SAFE_CALL( cudaMalloc((void**) &(state.switch_membership_d),
                               num * sizeof(bool)) );
    CUDA_SAFE_CALL( cudaMalloc((void**) &(state.p),
                               num * sizeof(Point)) );
    CUDA_SAFE_CALL( cudaMalloc((void**) &(state.coord_d),
                               num * dim * sizeof(float)) );
  }

  //=======================================
  // CPU-TO-GPU MEMORY COPY
  //=======================================
  // Only if first iteration OR coord has changed
  if(state.isCoordChanged || state.iter == 0)
  {
    CUDA_SAFE_CALL( cudaMemcpyAsync(state.coord_d, state.coord_h.get(),
                                    num * dim * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    args.cuda_stream) );
  }

  CUDA_SAFE_CALL( cudaMemcpyAsync(state.center_table_d,
                                  state.center_table.get(),
                                  num * sizeof(int),
                                  cudaMemcpyHostToDevice,
                                  args.cuda_stream) );
  CUDA_SAFE_CALL( cudaMemcpyAsync(state.p,  points->p.get(),
                                  num * sizeof(Point),
                                  cudaMemcpyHostToDevice,
                                  args.cuda_stream) );

  CUDA_SAFE_CALL( cudaMemsetAsync((void*) (state.switch_membership_d), 0,
                                  num * sizeof(bool),
                                  args.cuda_stream)  );

  CUDA_SAFE_CALL( cudaMemsetAsync((void*) (work_mem_d),
                                  0, stride * (nThread + 1) * sizeof(float),
                                  args.cuda_stream) );

  while (!set_and_check(args.uid, true)) {
    usleep(100);
  }

  //=======================================
  // KERNEL: CALCULATE COST
  //=======================================
  // Determine the number of thread blocks in the x- and y-dimension
  int num_blocks = (int) ((float) (num + THREADS_PER_BLOCK - 1) /
                          (float) THREADS_PER_BLOCK);
  int num_blocks_y = (int) ((float) (num_blocks + MAXBLOCKS - 1)  /
                            (float) MAXBLOCKS);
  int num_blocks_x = (int) ((float) (num_blocks+num_blocks_y - 1) /
                            (float) num_blocks_y);

  dim3 grid_size(num_blocks_x, num_blocks_y, 1);

  bool can_exit = false;

  while (!can_exit) {
    kernel_compute_cost<<<grid_size, THREADS_PER_BLOCK, 0, args.cuda_stream>>>(
            num,					// in:	# of data
            dim,					// in:	dimension of point coordinates
            x,						// in:	point to open a center at
            state.p,						// in:	data point array
            K,						// in:	number of centers
            stride,					// in:  size of each work_mem segment
            state.coord_d,				// in:	array of point coordinates
            work_mem_d,				// out:	cost and lower field array
            state.center_table_d,			// in:	center index table
            state.switch_membership_d		// out:  changes in membership
    );

    cudaStreamSynchronize(args.cuda_stream);

    can_exit = set_and_check(args.uid, false);
  }

  // error check
  error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    printf("kernel error: %s\n", cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }

  //=======================================
  // GPU-TO-CPU MEMORY COPY
  //=======================================
  CUDA_SAFE_CALL( cudaMemcpyAsync(work_mem_h.get(), work_mem_d,
                                  stride * (nThread + 1) * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  args.cuda_stream) );
  CUDA_SAFE_CALL( cudaMemcpyAsync(state.switch_membership.get(),
                                  state.switch_membership_d,
                                  num * sizeof(bool),
                                  cudaMemcpyDeviceToHost,
                                  args.cuda_stream) );

  cudaStreamSynchronize(args.cuda_stream);

  //=======================================
  // CPU (SERIAL) WORK
  //=======================================
  int number_of_centers_to_close = 0;
  float gl_cost_of_opening_x = z;
  float *gl_lower = &work_mem_h[stride * nThread];
  // compute the number of centers to close if we are to open i
  for(int i=0; i < num; i++)
  {
    if( state.is_center[i] )
    {
      float low = z;
      for( int j = 0; j < num; j++ )
      {
        low += work_mem_h[ j*stride + state.center_table[i] ];
      }

      gl_lower[state.center_table[i]] = low;

      if ( low > 0 )
      {
        ++number_of_centers_to_close;
        work_mem_h[i*stride+K] -= low;
      }
    }
    gl_cost_of_opening_x += work_mem_h[i*stride+K];
  }

  //if opening a center at x saves cost (i.e. cost is negative) do so;
  // otherwise, do nothing
  if ( gl_cost_of_opening_x < 0 )
  {
    for(int i = 0; i < num; i++)
    {
      bool close_center = gl_lower[state.center_table[points->p[i].assign]] > 0;
      if ( state.switch_membership[i] || close_center )
      {
        points->p[i].cost = dist(points->p[i], points->p[x], dim)
                            * points->p[i].weight;
        points->p[i].assign = x;
      }
    }

    for(int i = 0; i < num; i++)
    {
      if( state.is_center[i] && gl_lower[state.center_table[i]] > 0 )
      {
        state.is_center[i] = false;
      }
    }

    if( x >= 0 && x < num)
    {
      state.is_center[x] = true;
    }

    state.k = state.k + 1 - number_of_centers_to_close;
  }
  else
  {
    gl_cost_of_opening_x = 0;
  }

  //=======================================
  // DEALLOCATE GPU MEMORY
  //=======================================
  CUDA_SAFE_CALL( cudaFree(work_mem_d) );

  state.iter++;

  return -gl_cost_of_opening_x;
}
