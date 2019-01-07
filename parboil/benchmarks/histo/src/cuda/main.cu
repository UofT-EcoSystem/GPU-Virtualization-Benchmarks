/***************************************************************************
 *
 *            (C) Copyright 2010 The Board of Trustees of the
 *                        University of Illinois
 *                         All Rights Reserved
 *
 ***************************************************************************/


#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>


#include "cuda_profiler_api.h"
#include "util.h"

volatile bool ready = false;
volatile bool should_stop = false;
const char * ready_fifo = "/tmp/ready"; 

void start_handler(int sig) {
    ready = true;
}

void stop_handler(int sig) {
    should_stop = true;
}


__global__ void histo_prescan_kernel (
        unsigned int* input,
        int size,
        unsigned int* minmax);

__global__ void histo_main_kernel (
        uchar4 *sm_mappings,
        unsigned int num_elements,
        unsigned int sm_range_min,
        unsigned int sm_range_max,
        unsigned int histo_height,
        unsigned int histo_width,
        unsigned int *global_subhisto,
        unsigned int *global_histo,
        unsigned int *global_overflow);

__global__ void histo_intermediates_kernel (
        uint2 *input,
        unsigned int height,
        unsigned int width,
        unsigned int input_pitch,
        uchar4 *sm_mappings);

__global__ void histo_final_kernel (
        unsigned int sm_range_min,
        unsigned int sm_range_max,
        unsigned int histo_height,
        unsigned int histo_width,
        unsigned int *global_subhisto,
        unsigned int *global_histo,
        unsigned int *global_overflow,
        unsigned int *final_histo);

/******************************************************************************
* Implementation: GPU
* Details:
* in the GPU implementation of histogram, we begin by computing the span of the
* input values into the histogram. Then the histogramming computation is carried
* out by a (BLOCK_X, BLOCK_Y) sized grid, where every group of Y (same X)
* computes its own partial histogram for a part of the input, and every Y in the
* group exclusively writes to a portion of the span computed in the beginning.
* Finally, a reduction is performed to combine all the partial histograms into
* the final result.
******************************************************************************/

int main(int argc, char* argv[]) {
  if (signal(SIGUSR1, start_handler) < 0)
      perror("Signal error");

  if (signal(SIGUSR2, stop_handler) < 0)
      perror("Signal error");

  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    return -1;

  if(!parameters->inpFiles[0]){
    fputs("Input file expected\n", stderr);
    return -1;
  }
  
  char *prescans = "PreScanKernel";
  char *postpremems = "PostPreMems";
  char *intermediates = "IntermediatesKernel";
  char *mains = "MainKernel";
  char *finals = "FinalKernel";

  pb_InitializeTimerSet(&timers);
  
  pb_AddSubTimer(&timers, prescans, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, postpremems, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, intermediates, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, mains, pb_TimerID_KERNEL);
  pb_AddSubTimer(&timers, finals, pb_TimerID_KERNEL);
  
  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  int numIterations;
  if (argc >= 2){
    numIterations = atoi(argv[1]);
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    return -1;
  }

  unsigned int img_width, img_height;
  unsigned int histo_width, histo_height;

  FILE* f = fopen(parameters->inpFiles[0],"rb");
  int result = 0;

  result += fread(&img_width,    sizeof(unsigned int), 1, f);
  result += fread(&img_height,   sizeof(unsigned int), 1, f);
  result += fread(&histo_width,  sizeof(unsigned int), 1, f);
  result += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (result != 4){
    fputs("Error reading input and output dimensions from file\n", stderr);
    return -1;
  }

  unsigned int* img = (unsigned int*) malloc (img_width*img_height*sizeof(unsigned int));
  unsigned char* histo = (unsigned char*) calloc (histo_width*histo_height, sizeof(unsigned char));

  result = fread(img, sizeof(unsigned int), img_width*img_height, f);

  fclose(f);

  if (result != img_width*img_height){
    fputs("Error reading input array from file\n", stderr);
    return -1;
  }

  int even_width = ((img_width+1)/2)*2;
  unsigned int* input;
  unsigned int* ranges;
  uchar4* sm_mappings;
  unsigned int* global_subhisto;
  unsigned short* global_histo;
  unsigned int* global_overflow;
  unsigned char* final_histo;

  cudaMalloc((void**)&input           , even_width*(((img_height+UNROLL-1)/UNROLL)*UNROLL)*sizeof(unsigned int));
  cudaMalloc((void**)&ranges          , 2*sizeof(unsigned int));
  cudaMalloc((void**)&sm_mappings     , img_width*img_height*sizeof(uchar4));
  cudaMalloc((void**)&global_subhisto , BLOCK_X*img_width*histo_height*sizeof(unsigned int));
  cudaMalloc((void**)&global_histo    , img_width*histo_height*sizeof(unsigned short));
  cudaMalloc((void**)&global_overflow , img_width*histo_height*sizeof(unsigned int));
  cudaMalloc((void**)&final_histo     , img_width*histo_height*sizeof(unsigned char));

  cudaMemset(final_histo , 0 , img_width*histo_height*sizeof(unsigned char));

  for (int y=0; y < img_height; y++){
    cudaMemcpy(&(((unsigned int*)input)[y*even_width]),&img[y*img_width],img_width*sizeof(unsigned int), cudaMemcpyHostToDevice);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);


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

  if (res > 0) printf("Parboil spmv write success to the pipe!\n");
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
 


  while (!should_stop){ 
    //for (int iter = 0; iter < numIterations; iter++) {
    unsigned int ranges_h[2] = {UINT32_MAX, 0};

    cudaMemcpy(ranges,ranges_h, 2*sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    pb_SwitchToSubTimer(&timers, prescans , pb_TimerID_KERNEL);

    histo_prescan_kernel<<<dim3(PRESCAN_BLOCKS_X),dim3(PRESCAN_THREADS)>>>((unsigned int*)input, img_height*img_width, ranges);
    
    pb_SwitchToSubTimer(&timers, postpremems , pb_TimerID_KERNEL);

    cudaMemcpy(ranges_h,ranges, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaMemset(global_subhisto,0,img_width*histo_height*sizeof(unsigned int));
    
    pb_SwitchToSubTimer(&timers, intermediates, pb_TimerID_KERNEL);

    histo_intermediates_kernel<<<dim3((img_height + UNROLL-1)/UNROLL), dim3((img_width+1)/2)>>>(
                (uint2*)(input),
                (unsigned int)img_height,
                (unsigned int)img_width,
                (img_width+1)/2,
                (uchar4*)(sm_mappings)
    );
    
    pb_SwitchToSubTimer(&timers, mains, pb_TimerID_KERNEL);
    
    
    histo_main_kernel<<<dim3(BLOCK_X, ranges_h[1]-ranges_h[0]+1), dim3(THREADS)>>>(
                (uchar4*)(sm_mappings),
                img_height*img_width,
                ranges_h[0], ranges_h[1],
                histo_height, histo_width,
                (unsigned int*)(global_subhisto),
                (unsigned int*)(global_histo),
                (unsigned int*)(global_overflow)    
    );
    
    pb_SwitchToSubTimer(&timers, finals, pb_TimerID_KERNEL);
    
    histo_final_kernel<<<dim3(BLOCK_X*3), dim3(512)>>>(
                ranges_h[0], ranges_h[1],
                histo_height, histo_width,
                (unsigned int*)(global_subhisto),
                (unsigned int*)(global_histo),
                (unsigned int*)(global_overflow),
                (unsigned int*)(final_histo)
    );
  }

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


  cudaMemcpy(histo,final_histo, histo_height*histo_width*sizeof(unsigned char), cudaMemcpyDeviceToHost);

  cudaFree(input);
  cudaFree(ranges);
  cudaFree(sm_mappings);
  cudaFree(global_subhisto);
  cudaFree(global_histo);
  cudaFree(global_overflow);
  cudaFree(final_histo);

  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(img);
  free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);
  
  pb_DestroyTimerSet(&timers);

  return 0;
}
