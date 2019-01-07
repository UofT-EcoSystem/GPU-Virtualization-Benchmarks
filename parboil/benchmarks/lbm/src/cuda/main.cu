/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*############################################################################*/

#include "main.h"
#include "lbm.h"
#include <stdio.h>
#include <stdlib.h>

#include <sys/stat.h>

#include <unistd.h>
#include <signal.h>
#include <fcntl.h>

#include "cuda_profiler_api.h"
#include <cuda.h>


/*############################################################################*/
static LBM_Grid CUDA_srcGrid, CUDA_dstGrid;


/*############################################################################*/

volatile bool ready = false;
volatile bool should_stop = false;
const char * ready_fifo = "/tmp/ready"; 



void start_handler(int sig) {
    ready = true;
}

void stop_handler(int sig) {
    should_stop = true;
}


struct pb_TimerSet timers;

int main( int nArgs, char* arg[] ) {
    if (signal(SIGUSR1, start_handler) < 0)
        perror("Signal error");

    if (signal(SIGUSR2, stop_handler) < 0)
        perror("Signal error");

	MAIN_Param param;
	int t;

	pb_InitializeTimerSet(&timers);
        struct pb_Parameters* params;
        params = pb_ReadParameters(&nArgs, arg);
        

	static LBM_GridPtr TEMP_srcGrid;
	//Setup TEMP datastructures
	LBM_allocateGrid( (float**) &TEMP_srcGrid );
	MAIN_parseCommandLine( nArgs, arg, &param, params );
	MAIN_printInfo( &param );

	MAIN_initialize( &param );


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

    if (res > 0) printf("Write success to the pipe!\n");
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


	for( t = 1; t <= param.nTimeSteps; t++ ) {
                pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
		CUDA_LBM_performStreamCollide( CUDA_srcGrid, CUDA_dstGrid );
                pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
		LBM_swapGrids( &CUDA_srcGrid, &CUDA_dstGrid );

		if( (t & 63) == 0 ) {
			printf( "timestep: %i\n", t );
#if 0
			CUDA_LBM_getDeviceGrid((float**)&CUDA_srcGrid, (float**)&TEMP_srcGrid);
			LBM_showGridStatistics( *TEMP_srcGrid );
#endif
		}
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


    // Also don't care about the stop signal


	MAIN_finalize( &param );

	LBM_freeGrid( (float**) &TEMP_srcGrid );

        pb_SwitchToTimer(&timers, pb_TimerID_NONE);
        pb_PrintTimerSet(&timers);
        pb_FreeParameters(params);
	return 0;
}

/*############################################################################*/

void MAIN_parseCommandLine( int nArgs, char* arg[], MAIN_Param* param, struct pb_Parameters * params ) {
	struct stat fileStat;

	if( nArgs < 2 ) {
		printf( "syntax: lbm <time steps>\n" );
		exit( 1 );
	}

	param->nTimeSteps     = atoi( arg[1] );

	if( params->inpFiles[0] != NULL ) {
		param->obstacleFilename = params->inpFiles[0];

		if( stat( param->obstacleFilename, &fileStat ) != 0 ) {
			printf( "MAIN_parseCommandLine: cannot stat obstacle file '%s'\n",
					param->obstacleFilename );
			exit( 1 );
		}
		if( fileStat.st_size != SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z ) {
			printf( "MAIN_parseCommandLine:\n"
					"\tsize of file '%s' is %i bytes\n"
					"\texpected size is %i bytes\n",
					param->obstacleFilename, (int) fileStat.st_size,
					SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z );
			exit( 1 );
		}
	}
	else param->obstacleFilename = NULL;

        param->resultFilename = params->outFile;
}

/*############################################################################*/

void MAIN_printInfo( const MAIN_Param* param ) {
	printf( "MAIN_printInfo:\n"
			"\tgrid size      : %i x %i x %i = %.2f * 10^6 Cells\n"
			"\tnTimeSteps     : %i\n"
			"\tresult file    : %s\n"
			"\taction         : %s\n"
			"\tsimulation type: %s\n"
			"\tobstacle file  : %s\n\n",
			SIZE_X, SIZE_Y, SIZE_Z, 1e-6*SIZE_X*SIZE_Y*SIZE_Z,
			param->nTimeSteps, param->resultFilename, 
			"store", "lid-driven cavity",
			(param->obstacleFilename == NULL) ? "<none>" :
			param->obstacleFilename );
}

/*############################################################################*/

void MAIN_initialize( const MAIN_Param* param ) {
	static LBM_Grid TEMP_srcGrid, TEMP_dstGrid;

        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	//Setup TEMP datastructures
	LBM_allocateGrid( (float**) &TEMP_srcGrid );
	LBM_allocateGrid( (float**) &TEMP_dstGrid );
	LBM_initializeGrid( TEMP_srcGrid );
	LBM_initializeGrid( TEMP_dstGrid );

        pb_SwitchToTimer(&timers, pb_TimerID_IO);
	if( param->obstacleFilename != NULL ) {
		LBM_loadObstacleFile( TEMP_srcGrid, param->obstacleFilename );
		LBM_loadObstacleFile( TEMP_dstGrid, param->obstacleFilename );
	}

        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	LBM_initializeSpecialCellsForLDC( TEMP_srcGrid );
	LBM_initializeSpecialCellsForLDC( TEMP_dstGrid );

        pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//Setup DEVICE datastructures
	CUDA_LBM_allocateGrid( (float**) &CUDA_srcGrid );
	CUDA_LBM_allocateGrid( (float**) &CUDA_dstGrid );

	//Initialize DEVICE datastructures
	CUDA_LBM_initializeGrid( (float**)&CUDA_srcGrid, (float**)&TEMP_srcGrid );
	CUDA_LBM_initializeGrid( (float**)&CUDA_dstGrid, (float**)&TEMP_dstGrid );

        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	LBM_showGridStatistics( TEMP_srcGrid );

	LBM_freeGrid( (float**) &TEMP_srcGrid );
	LBM_freeGrid( (float**) &TEMP_dstGrid );
}

/*############################################################################*/

void MAIN_finalize( const MAIN_Param* param ) {
	LBM_Grid TEMP_srcGrid;

	//Setup TEMP datastructures
	LBM_allocateGrid( (float**) &TEMP_srcGrid );

        pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	CUDA_LBM_getDeviceGrid((float**)&CUDA_srcGrid, (float**)&TEMP_srcGrid);

        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	LBM_showGridStatistics( TEMP_srcGrid );

	LBM_storeVelocityField( TEMP_srcGrid, param->resultFilename, TRUE );

	LBM_freeGrid( (float**) &TEMP_srcGrid );
	CUDA_LBM_freeGrid( (float**) &CUDA_srcGrid );
	CUDA_LBM_freeGrid( (float**) &CUDA_dstGrid );
}

