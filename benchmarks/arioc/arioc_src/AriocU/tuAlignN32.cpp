/*
  tuAlignN32.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuAlignN32::tuAlignN32()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuAlignN32::tuAlignN32( QBatch* pqb, INT16 AtN ) : m_pqb(pqb),
                                                   m_pab(pqb->pab),
                                                   m_AtN(AtN),
                                                   m_ptum(AriocBase::GetTaskUnitMetrics("tuAlignN32"))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_tuAlignN32", m_cudaThreadsPerBlock );
}

/// destructor
tuAlignN32::~tuAlignN32()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void tuAlignN32::initGlobalMemory()
{
    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBn.Dc  D values with nongapped mappings for QIDs that are candidates for gapped alignment
                ...

        high:   (unused)
    */
}

/// [private] method computeGridDimensions
void tuAlignN32::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pqb->DBn.Dc.n );
}

/// [private] method copyKernelResults
void tuAlignN32::copyKernelResults()
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
void tuAlignN32::resetGlobalMemory()
{
    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBn.Dc  D values with nongapped mappings for QIDs that are candidates for gapped alignment
                ...

        high:   (unused)
    */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Flags D values mapped by the nongapped aligner for subsequent filtering of gapped-aligner D values.
/// </summary>
void tuAlignN32::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

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

    CDPrint( cdpCD3, "[%d] %s: completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
#pragma endregion
