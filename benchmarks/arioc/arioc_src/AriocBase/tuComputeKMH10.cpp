/*
  tuComputeKMH10.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuComputeKMH10::tuComputeKMH10()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuComputeKMH10::tuComputeKMH10( QBatch* pqb ) : m_pqb(pqb),
                                                m_pab(pqb->pab),
                                                m_ptum(AriocBase::GetTaskUnitMetrics( "tuComputeKMH10" ))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_tuComputeKMH10", m_cudaThreadsPerBlock );
}

/// destructor
tuComputeKMH10::~tuComputeKMH10()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void tuComputeKMH10::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

             low:   Qw      Qwarps
                    Qi      interleaved Q sequence data
                    KMH     kmer hash values (paired with qids)
                    S64     "sketch bits" (one per unmapped Q sequence)

            high:   (unallocated)
        */

        // compute the number of kmer hash values per Q sequence
        m_pqb->DBkmh.stride = (m_pqb->Nmax - m_pab->KmerSize) + 1;

        // allocate a buffer to contain KMH/qid pairs (64-bit values)
        m_pqb->DBkmh.KMH.n = m_pqb->DBkmh.stride * m_pqb->DB.nQ;
        CREXEC( m_pqb->DBkmh.KMH.Alloc( cgaLow, m_pqb->DBkmh.KMH.n, true ) );
        SET_CUDAGLOBALPTR_TAG( m_pqb->DBkmh.KMH, "DBkmh.KMH" );

        // allocate a buffer to contain S64 values and fill it with null values (all bits set)
        CREXEC( m_pqb->DBkmh.S64.Alloc( cgaLow, m_pqb->DB.nQ, false ) );
        CREXEC( cudaMemset( m_pqb->DBkmh.S64.p, 0xFF, m_pqb->DB.nQ*sizeof(UINT64) ) );
        m_pqb->DBkmh.S64.n = m_pqb->DB.nQ;
        SET_CUDAGLOBALPTR_TAG( m_pqb->DBkmh.S64, "DBkmh.S64" );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method computeGridDimensions
void tuComputeKMH10::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pqb->DB.nQ );
}

/// [private] method copyKernelResults
void tuComputeKMH10::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // block until the kernel completes
        CREXEC( waitForKernel() );
        
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );

#if TODO_CHOP_WHEN_DEBUGGED
        WinGlobalPtr<UINT64> KMHxxx( m_pqb->DBkmh.KMH.Count, false );
        m_pqb->DBkmh.KMH.CopyToHost( KMHxxx.p, KMHxxx.Count );

        for( UINT32 n=0; n<m_pqb->DBkmh.KMH.n; ++n )
        {
            if( KMHxxx.p[n] == 0 )
                CDPrint( cdpCD0, "%s: QK = 0 at n = %u", __FUNCTION__, n );
        }

        CDPrint( cdpCD0, __FUNCTION__ );
#endif


    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void tuComputeKMH10::resetGlobalMemory()
{
    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                KMH     kmer hash values (paired with qids)
                S64     "sketch bits" (one per unmapped Q sequence)

        high:   (unallocated)
    */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Computes kmer hash values for unmapped Q sequences
/// </summary>
void tuComputeKMH10::main()
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
