
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#ifdef PARBOIL_STENCIL

#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "file.h"
#include "common.h"
#include "cuerr.h"

#include "interface.h" 

__global__ void block2D_hybrid_coarsen_x(float c0,float c1,float *A0,float *Anext, int nx, int ny, int nz)
{
  //thread coarsening along x direction
  const int i = blockIdx.x*blockDim.x*2+threadIdx.x;
  const int i2= blockIdx.x*blockDim.x*2+threadIdx.x+blockDim.x;
  const int j = blockIdx.y*blockDim.y+threadIdx.y;
  const int sh_id=threadIdx.x + threadIdx.y*blockDim.x*2;
  const int sh_id2=threadIdx.x +blockDim.x+ threadIdx.y*blockDim.x*2;

  //shared memeory
  extern __shared__ float sh_A0[];
  sh_A0[sh_id]=0.0f;
  sh_A0[sh_id2]=0.0f;
  __syncthreads();

  //get available region for load and store
  const bool w_region =  i>0 && j>0 &&(i<(nx-1)) &&(j<(ny-1)) ;
  const bool w_region2 =  j>0 &&(i2<nx-1) &&(j<ny-1) ;
  const bool x_l_bound = (threadIdx.x==0);
  const bool x_h_bound = ((threadIdx.x+blockDim.x)==(blockDim.x*2-1));
  const bool y_l_bound = (threadIdx.y==0);
  const bool y_h_bound = (threadIdx.y==(blockDim.y-1));

  //register for bottom and top planes
  //because of thread coarsening, we need to doulbe registers
  float bottom=0.0f,bottom2=0.0f,top=0.0f,top2=0.0f;

  //load data for bottom and current 
  if((i<nx) &&(j<ny))
  {

    bottom=A0[Index3D (nx, ny, i, j, 0)];
    sh_A0[sh_id]=A0[Index3D (nx, ny, i, j, 1)];
  }
  if((i2<nx) &&(j<ny))
  {
    bottom2=A0[Index3D (nx, ny, i2, j, 0)];
    sh_A0[sh_id2]=A0[Index3D (nx, ny, i2, j, 1)];
  }

  __syncthreads();

  for(int k=1;k<nz-1;k++)
  {

    float a_left_right,a_up,a_down;		

    //load required data on xy planes
    //if it on shared memory, load from shared memory
    //if not, load from global memory
    if((i<nx) &&(j<ny))
      top=A0[Index3D (nx, ny, i, j, k+1)];

    if(w_region)
    {
      a_up        =y_h_bound?A0[Index3D (nx, ny, i, j+1, k )]:sh_A0[sh_id+2*blockDim.x];
      a_down      =y_l_bound?A0[Index3D (nx, ny, i, j-1, k )]:sh_A0[sh_id-2*blockDim.x];
      a_left_right=x_l_bound?A0[Index3D (nx, ny, i-1, j, k )]:sh_A0[sh_id-1];

      Anext[Index3D (nx, ny, i, j, k)] = (top + bottom + a_up + a_down + sh_A0[sh_id+1] +a_left_right)*c1
        -  sh_A0[sh_id]*c0;		
    }


    //load another block 
    if((i2<nx) &&(j<ny))
      top2=A0[Index3D (nx, ny, i2, j, k+1)];

    if(w_region2)
    {
      a_up        =y_h_bound?A0[Index3D (nx, ny, i2, j+1, k )]:sh_A0[sh_id2+2*blockDim.x];
      a_down      =y_l_bound?A0[Index3D (nx, ny, i2, j-1, k )]:sh_A0[sh_id2-2*blockDim.x];
      a_left_right=x_h_bound?A0[Index3D (nx, ny, i2+1, j, k )]:sh_A0[sh_id2+1];


      Anext[Index3D (nx, ny, i2, j, k)] = (top2 + bottom2 + a_up + a_down + a_left_right +sh_A0[sh_id2-1])*c1
        -  sh_A0[sh_id2]*c0;
    }

    //swap data
    __syncthreads();
    bottom=sh_A0[sh_id];
    sh_A0[sh_id]=top;
    bottom2=sh_A0[sh_id2];
    sh_A0[sh_id2]=top2;
    __syncthreads();
  }
}

/************************* End kernel ***************************/


extern bool set_and_check(int uid, bool start);

static int read_data(float *A0, int nx,int ny,int nz,FILE *fp) 
{	
	int s=0;
	for(int i=0;i<nz;i++)
	{
		for(int j=0;j<ny;j++)
		{
			for(int k=0;k<nx;k++)
			{
                                fread(A0+s,sizeof(float),1,fp);
				s++;
			}
		}
	}
	return 0;
}

int main_stencil(int argc, char** argv, int uid) {
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	
	printf("CUDA accelerated 7 points stencil codes****\n");
	printf("Original version by Li-Wen Chang <lchang20@illinois.edu> and I-Jui Sung<sung10@illinois.edu>\n");
	printf("This version maintained by Chris Rodrigues  ***********\n");
	parameters = pb_ReadParameters(&argc, argv);

	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	//declaration
	int nx,ny,nz;
	int size;
    int iteration;
	float c0=1.0f/6.0f;
	float c1=1.0f/6.0f/6.0f;

	if (argc<5) 
    {
      printf("Usage: probe nx ny nz tx ty t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
		  "t: the iteration time\n");
      return -1;
    }

	nx = atoi(argv[1]);
	if (nx<1)
		return -1;
	ny = atoi(argv[2]);
	if (ny<1)
		return -1;
	nz = atoi(argv[3]);
	if (nz<1)
		return -1;
	iteration = atoi(argv[4]);
	if(iteration<1)
		return -1;

	
	//host data
	float *h_A0;
	float *h_Anext;
	//device
	float *d_A0;
	float *d_Anext;

	


	size=nx*ny*nz;
	
	h_A0=(float*)malloc(sizeof(float)*size);
	h_Anext=(float*)malloc(sizeof(float)*size);
  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  FILE *fp = fopen(parameters->inpFiles[0], "rb");
	read_data(h_A0, nx,ny,nz,fp);
  fclose(fp);
	
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//memory allocation
	cudaMalloc((void **)&d_A0, size*sizeof(float));
	cudaMalloc((void **)&d_Anext, size*sizeof(float));
	cudaMemset(d_Anext,0,size*sizeof(float));

  // create cuda stream for this benchmark
  cudaStream_t stream;
  cudaStreamCreate(&stream);

	//memory copy
	cudaMemcpyAsync(d_A0, h_A0, size*sizeof(float), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_Anext, d_A0, size*sizeof(float), cudaMemcpyDeviceToDevice, stream);

	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  
	//only use tx-by-ty threads
	int tx=32;
	int ty=4;
  
	dim3 block (tx, ty, 1);
	//also change threads size maping from tx by ty to 2tx x ty
	dim3 grid ((nx+tx*2-1)/(tx*2), (ny+ty-1)/ty,1);
	int sh_size = tx*2*ty*sizeof(float);	
 
  set_and_check(uid, true);
  while (!set_and_check(uid, true)) {
    usleep(100);
  }

	//main execution
	pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);

  bool can_exit = false;

  while (!can_exit) {
//    for(int t=0;t<iteration;t++)
    {
      block2D_hybrid_coarsen_x<<<grid, block, sh_size, stream>>>(c0,c1, d_A0, d_Anext, nx, ny,  nz);
      float *d_temp = d_A0;
      d_A0 = d_Anext;
      d_Anext = d_temp;

    }
    CUERR // check and clear any existing errors
    cudaStreamSynchronize(stream);

    can_exit = set_and_check(uid, false);
  }

  float *d_temp = d_A0;
  d_A0 = d_Anext;
  d_Anext = d_temp;  
	
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	cudaMemcpyAsync(h_Anext, d_Anext,size*sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
	cudaFree(d_A0);
  cudaFree(d_Anext);
 
	if (parameters->outFile) {
		 pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Anext,nx,ny,nz);
		
	}
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
		
	free (h_A0);
	free (h_Anext);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

  cudaStreamDestroy(stream);

	return 0;

}

#endif

