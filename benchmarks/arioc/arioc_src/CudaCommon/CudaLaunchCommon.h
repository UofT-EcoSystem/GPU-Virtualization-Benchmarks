/*
  CudaLaunchCommon.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __CudaLaunchCommon__

/// <summary>
/// Class <c>CudaLaunchCommon</c> implements common functionality for implementing a CUDA kernel launch.
/// </summary>
class CudaLaunchCommon
{
    protected:
        UINT32  m_cudaThreadsPerBlock;

    protected:
        CudaLaunchCommon();
        CudaLaunchCommon( UINT32 cudaThreadsPerBlock );
        virtual ~CudaLaunchCommon();
        void computeKernelGridDimensions( dim3& d3g, dim3& d3b, UINT64 nKernelThreads );
        void waitForKernel( void );
};
