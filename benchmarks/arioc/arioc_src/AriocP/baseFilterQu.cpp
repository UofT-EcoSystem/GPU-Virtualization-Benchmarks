/*
  baseFilterQu.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"


#pragma region constructor/destructor
/// [private] constructor
baseFilterQu::baseFilterQu()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
baseFilterQu::baseFilterQu( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, INT16 At ) : m_pqb(pqb),
                                                                                                    m_pab(pqb->pab),
                                                                                                    m_pdbb(pdbb),
                                                                                                    m_At(At),
                                                                                                    m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseFilterQu", m_cudaThreadsPerBlock );
}

/// destructor
baseFilterQu::~baseFilterQu()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeGridDimensions
void baseFilterQu::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pdbb->Qu.n );
}

/// [private] method initGlobalMemory
void baseFilterQu::initGlobalMemory()
{
    /* CUDA global memory layout unchanged */
}

/// [private] method copyKernelResults
void baseFilterQu::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // block until the kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseFilterQu::resetGlobalMemory()
{
    /* CUDA global memory layout unchanged */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Eliminates concordantly-mapped pairs from a list of candidates for subsequent gapped alignment
/// </summary>
void baseFilterQu::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    // set up and run a CUDA kernel to do the work
    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    launchKernel( d3g, d3b, initSharedMemory() );
    copyKernelResults();
    
    resetGlobalMemory();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s::%s: completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
#pragma endregion
