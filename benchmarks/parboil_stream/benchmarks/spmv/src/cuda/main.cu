
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <functional>

#include "file.h"
#include "gpu_info.h"
//#include "spmv_jds.h"
//#include "jds_kernels.cu"
#include "convert_dataset.h"

#include "interface.h"


/****************** Kernels *******************/

#define WARP_BITS 5
#define WARP_SIZE 32

#include "spmv_jds.h"

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return -1; }}
 
//TEXTURE memory
texture<float,1> tex_x_float;

//constant memory
__constant__ int jds_ptr_int[5000];
__constant__ int sh_zcnt_int[5000];


__global__ void spmv_jds(float *dst_vector,
						const float *d_data,const int *d_index, const int *d_perm,
						const float *x_vec,const int *d_nzcnt,const int dim)
{
	int ix=blockIdx.x*blockDim.x+threadIdx.x;
	int warp_id=ix>>WARP_BITS;
	if(ix<dim)
	{
		float sum=0.0f;
		int	bound=sh_zcnt_int[warp_id];
		//prefetch 0
		int j=jds_ptr_int[0]+ix;  
		float d = d_data[j]; 
		int i = d_index[j];  
		float t = x_vec[i];
		
		if (bound>1)  //bound >=2
		{
			//prefetch 1
			j=jds_ptr_int[1]+ix;    
			i =  d_index[j];  
			int in;
			float dn;
			float tn;
			for(int k=2;k<bound;k++ )
			{	
				//prefetch k-1
				dn = d_data[j]; 
				//prefetch k
				j=jds_ptr_int[k]+ix;    
				in = d_index[j]; 
				//prefetch k-1
				tn = x_vec[i];
				
				//compute k-2
				sum += d*t; 
				//sweep to k
				i = in;  
				//sweep to k-1
				d = dn;
				t =tn; 
			}	
		
			//fetch last
			dn = d_data[j];
			tn = x_vec[i];
	
			//compute last-1
			sum += d*t; 
			//sweep to last
			d=dn;
			t=tn;
		}
		//compute last
		sum += d*t;  // 3 3
		
		//write out data
		dst_vector[d_perm[ix]]=sum; 
	}

}


/****************** End Kernels *******************/



/*
static int generate_vector(float *x_vector, int dim) 
{	
	srand(54321);	
	for(int i=0;i<dim;i++)
	{
		x_vector[i] = (rand() / (float) RAND_MAX);
	}
	return 0;
}
*/

int main_spmv(int argc, 
              char** argv, 
              std::function<void(const int iter, cudaStream_t & stream)> & kernel,
              std::function<void(void)> & cleanup ) 
{
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	
	printf("CUDA accelerated sparse matrix vector multiplication****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and Shengzhao Wu<wu14@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	parameters = pb_ReadParameters(&argc, argv);
	if ((parameters->inpFiles[0] == NULL) || (parameters->inpFiles[1] == NULL))
    {
      fprintf(stderr, "Expecting two input filenames\n");
      exit(-1);
    }

	
	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//parameters declaration
	int len;
	int depth;
	int dim;
	int pad=32;
	int nzcnt_len;
	
	//host memory allocation
	//matrix
	float *h_data;
	int *h_indices;
	int *h_ptr;
	int *h_perm;
	int *h_nzcnt;
	//vector
	float *h_Ax_vector;
    float *h_x_vector;
	
	//device memory allocation
	//matrix
	float *d_data;
	int *d_indices;
	int *d_ptr;
	int *d_perm;
	int *d_nzcnt;
	//vector
	float *d_Ax_vector;
    float *d_x_vector;
	
    //load matrix from files
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	//inputData(parameters->inpFiles[0], &len, &depth, &dim,&nzcnt_len,&pad,
	//    &h_data, &h_indices, &h_ptr,
	//    &h_perm, &h_nzcnt);
	int col_count;
	coo_to_jds(
		parameters->inpFiles[0], // bcsstk32.mtx, fidapm05.mtx, jgl009.mtx
		1, // row padding
		pad, // warp size, IMPORTANT: change in kernel as well
		1, // pack size
		1, // is mirrored?
		0, // binary matrix
		1, // debug level [0:2]
		&h_data, &h_ptr, &h_nzcnt, &h_indices, &h_perm,
		&col_count, &dim, &len, &nzcnt_len, &depth
	);
	

  h_Ax_vector=(float*)malloc(sizeof(float)*dim); 
  h_x_vector=(float*)malloc(sizeof(float)*dim);
  input_vec( parameters->inpFiles[1],h_x_vector,dim);

	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
	
	
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//memory allocation
	cudaMalloc((void **)&d_data, len*sizeof(float));
	cudaMalloc((void **)&d_indices, len*sizeof(int));
	cudaMalloc((void **)&d_ptr, depth*sizeof(int));
	cudaMalloc((void **)&d_perm, dim*sizeof(int));
	cudaMalloc((void **)&d_nzcnt, nzcnt_len*sizeof(int));
	cudaMalloc((void **)&d_x_vector, dim*sizeof(float));
	cudaMalloc((void **)&d_Ax_vector,dim*sizeof(float));
	cudaMemset( (void *) d_Ax_vector, 0, dim*sizeof(float));
	
	//memory copy
	cudaMemcpy(d_data, h_data, len*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indices, h_indices, len*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_perm, h_perm, dim*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_vector, h_x_vector, dim*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(jds_ptr_int, h_ptr, depth*sizeof(int));
	cudaMemcpyToSymbol(sh_zcnt_int, h_nzcnt,nzcnt_len*sizeof(int));
	
    cudaThreadSynchronize();
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	unsigned int grid;
	unsigned int block;
    compute_active_thread(&block, &grid,nzcnt_len,pad, deviceProp.major,deviceProp.minor,
					deviceProp.warpSize,deviceProp.multiProcessorCount);

	
  cudaFuncSetCacheConfig(spmv_jds, cudaFuncCachePreferL1);

  kernel = [&](const int iter, cudaStream_t & stream)
  {
    //main execution
    pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
    //for (int i= 0; i<50; i++)
    for (int i= 0; i<iter; i++)
    spmv_jds<<<grid, block, 0, stream>>>(d_Ax_vector,
            d_data,d_indices,d_perm,
          d_x_vector,d_nzcnt,dim);
                
      CUERR // check and clear any existing errors
    
    cudaThreadSynchronize();

  };


  cleanup = [&]()
  {
    pb_SwitchToTimer(&timers, pb_TimerID_COPY);
    //HtoD memory copy
    cudaMemcpy(h_Ax_vector, d_Ax_vector,dim*sizeof(float), cudaMemcpyDeviceToHost);	

    cudaThreadSynchronize();

    cudaFree(d_data);
      cudaFree(d_indices);
      cudaFree(d_ptr);
    cudaFree(d_perm);
      cudaFree(d_nzcnt);
      cudaFree(d_x_vector);
    cudaFree(d_Ax_vector);
   
    if (parameters->outFile) {
      pb_SwitchToTimer(&timers, pb_TimerID_IO);
      outputData(parameters->outFile,h_Ax_vector,dim);
      
    }
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
    
    free (h_data);
    free (h_indices);
    free (h_ptr);
    free (h_perm);
    free (h_nzcnt);
    free (h_Ax_vector);
    free (h_x_vector);
    pb_SwitchToTimer(&timers, pb_TimerID_NONE);

    pb_PrintTimerSet(&timers);
    pb_FreeParameters(parameters);


  };
	
	return 0;

}
