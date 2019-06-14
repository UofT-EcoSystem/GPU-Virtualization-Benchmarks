/*
  tuComputeKMH30.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuComputeKMH30::tuComputeKMH30()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuComputeKMH30::tuComputeKMH30( QBatch* pqb ) : m_pqb(pqb),
                                                m_pab(pqb->pab),
                                                m_nSketchBits(34),
                                                m_ptum(AriocBase::GetTaskUnitMetrics( "tuComputeKMH30" ))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_tuComputeKMH30", m_cudaThreadsPerBlock );

    // look for a configured override for m_nSketchBits
    INT32 i = m_pab->paamb->Xparam.IndexOf( "KMHsketchBits" );
    if( i >= 0 )
        m_nSketchBits = static_cast<UINT32>(m_pab->paamb->Xparam.Value(i));
    if( (m_nSketchBits < 32) || (m_nSketchBits > 40) )
        throw new ApplicationException( __FILE__, __LINE__, "KMHsketchBits must be between 32 and 40" );
}

/// destructor
tuComputeKMH30::~tuComputeKMH30()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void tuComputeKMH30::initGlobalMemory()
{
    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                KMH     kmer hash values (paired with qids)
                S64     "sketch bits" (one per unmapped Q sequence)

        high:   (unallocated)
    */
}

/// [private] method computeGridDimensions
void tuComputeKMH30::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pqb->DBkmh.S64.n );
}

/// [private] method copyKernelResults
void tuComputeKMH30::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // block until the kernel completes
        CREXEC( waitForKernel() );
        
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );

        // copy S64 values to a host buffer
        m_pqb->S64.Reuse( m_pqb->DBkmh.S64.n, false );
        m_pqb->S64.n = m_pqb->DBkmh.S64.n;
        m_pqb->DBkmh.S64.CopyToHost( m_pqb->S64.p, m_pqb->S64.n );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void tuComputeKMH30::resetGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after reset:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data

            high:   (unallocated)
        */

        // discard GPU buffers allocated in tuComputeKMH10
        m_pqb->DBkmh.S64.Free();
        m_pqb->DBkmh.KMH.Free();
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Computes S64 ("sketch bits") for unmapped Q sequences
/// </summary>
void tuComputeKMH30::main()
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
