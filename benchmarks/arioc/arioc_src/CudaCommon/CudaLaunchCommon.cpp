/*
  CudaLaunchCommon.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
CudaLaunchCommon::CudaLaunchCommon() : m_cudaThreadsPerBlock(8*CUDATHREADSPERWARP)
{
}

/// constructor (UINT32, UINT32)
CudaLaunchCommon::CudaLaunchCommon( UINT32 cudaThreadsPerBlock ) : m_cudaThreadsPerBlock( cudaThreadsPerBlock )
{
}

/// destructor
CudaLaunchCommon::~CudaLaunchCommon()
{
}
#pragma endregion

#pragma region protected methods
/// [protected] computeKernelGridDimensions
void CudaLaunchCommon::computeKernelGridDimensions( dim3& d3g, dim3& d3b, UINT64 nKernelThreads )
{
    /* We map CUDA threads into a 1-dimensional "grid" of blocks, each of which is mapped as a 1-dimensional
        array of threads.

       This makes things easy to assign a unique, sequential ID to each CUDA thread in a kernel:

            UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    */
    d3b = dim3( m_cudaThreadsPerBlock, 1, 1 );  // block dimensions (threads): blockDim.x=CUDA_threads_per_block; blockDim.y=1; blockDim.z=1

    UINT32 nBlocks = static_cast<UINT32>(blockdiv( nKernelThreads, m_cudaThreadsPerBlock ));

    // in CUDA compute capability 3.0 and higher, the maximum x-dimension of a grid is 2**31
    if( nBlocks > static_cast<UINT32>(_I32_MAX) )
        throw new ApplicationException( __FILE__, __LINE__, "CUDA grid dimension limit exceeded (nBlocks=%u)", nBlocks );

    UINT32 gridDim = max2( nBlocks, 1 );    // a grid dimension of zero is a CUDA launch configuration error
    d3g = dim3( gridDim, 1, 1 );            // grid dimensions (blocks): gridDim.x=nBlocks; gridDim.y=1; gridDim.z=1
}

/// [protected] waitForKernel
void CudaLaunchCommon::waitForKernel()
{
    CRVALIDATOR;

    // block this CPU thread until all CUDA device operations (kernels, memory movement, etc.) complete in all streams
    CRVALIDATE = cudaDeviceSynchronize();

    // as of CUDA 7.0, cudaDeviceSynchronize may not return an error code in all circumstances (e.g. a kernel launch grid dimension of zero)
    CRVALIDATE = cudaPeekAtLastError();
}
#pragma endregion
