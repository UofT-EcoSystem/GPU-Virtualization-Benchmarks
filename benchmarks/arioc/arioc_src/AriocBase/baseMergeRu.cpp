/*
  baseMergeRu.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseMergeRu::baseMergeRu()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
baseMergeRu::baseMergeRu( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, RiFlags flagRi ) : m_pqb(pqb),
                                                                                                        m_pab(pqb->pab),
                                                                                                        m_pdbb(pdbb),
                                                                                                        m_pD(NULL), m_nD(0),
                                                                                                        m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseMergeRu", m_cudaThreadsPerBlock );

    // use a flag specified by the caller to determine which buffer to use
    pdbb->flagRi = flagRi;
    switch( pdbb->flagRi )
    {
        case riDc:
            m_pD = pdbb->Dc.p;
            m_nD = pdbb->Dc.n;
            break;

        case riDm:
            m_pD = pdbb->Dm.p;
            m_nD = pdbb->Dm.n;
            break;

        case riDx:
            m_pD = pdbb->Dx.p;
            m_nD = pdbb->Dx.n;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for flagRi: %d", flagRi );
            break;
    }
}

/// destructor
baseMergeRu::~baseMergeRu()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeGridDimensions
void baseMergeRu::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nD );
}

/// [private] method initGlobalMemory
void baseMergeRu::initGlobalMemory()
{
    /* CUDA global memory layout unchanged */
}

/// [private] method copyKernelResults
void baseMergeRu::copyKernelResults()
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
void baseMergeRu::resetGlobalMemory()
{
    /* CUDA global memory layout unchanged */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Eliminates concordantly-mapped pairs from a list of candidates for subsequent windowed gapped alignment
/// </summary>
void baseMergeRu::main()
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
