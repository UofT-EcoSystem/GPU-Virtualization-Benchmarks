/*
  tuSetupN10.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuSetupN10::tuSetupN10()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuSetupN10::tuSetupN10( QBatch* pqb ) : m_pqb(pqb),
                                        m_pab(pqb->pab),
                                        m_kc(pqb->pab),
                                        m_nCandidates(0),
                                        m_ptum(AriocBase::GetTaskUnitMetrics( "tuSetupN10" ))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_tuSetupN10", m_cudaThreadsPerBlock );
}

/// destructor
tuSetupN10::~tuSetupN10()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void tuSetupN10::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    
            high:   nJ      per-seed J-list sizes
                    oJ      per-seed J-list offsets
        */

        /* Allocate and zero two parallel lists:
            - J-list offset
            - J-list size
           For each list:
            - the list contains one element for each seed position in each Q sequence
            - the values corresponding to a Q sequence are stored in consecutive elements in each list (i.e. they are NOT interleaved)
            - the buffer size is doubled for bsDNA alignments where both strands of each Q sequence are seeded
        */
        m_pqb->DBj.celJ = m_pqb->DB.nQ * m_kc.npos * m_kc.sps;  // number of elements in the oJ/nJ/cnJ buffers

        // J-list offsets
        CREXEC( m_pqb->DBj.oJ.Alloc( cgaHigh, m_pqb->DBj.celJ, true ) );
        SET_CUDAGLOBALPTR_TAG(m_pqb->DBj.oJ, "DBj.oJ");

        /* J-list cardinalities
        
           We use a trailing zero-valued element in the buffer so that thrust::exclusive_scan() will
            have room to compute a final value that represents the sum of the elements in the list.
        */
        CREXEC( m_pqb->DBj.nJ.Alloc( cgaHigh, m_pqb->DBj.celJ+1, true ) );
        SET_CUDAGLOBALPTR_TAG(m_pqb->DBj.nJ, "DBj.nJ");
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method computeGridDimensions
void tuSetupN10::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pqb->DB.nQ );
}

/// [private] method copyKernelResults
void tuSetupN10::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->n.CandidateQ, m_pqb->DB.nQ );

        // block until the kernel completes
        CREXEC( waitForKernel() );
        
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s: DBj.celJ=%u", __FUNCTION__, m_pqb->DBj.celJ );

        WinGlobalPtr<UINT32> nJxxx( m_pqb->DBj.nJ.Count, false );
        m_pqb->DBj.nJ.CopyToHost( nJxxx.p, nJxxx.Count );

        WinGlobalPtr<UINT64> oJxxx( m_pqb->DBj.oJ.Count, false );
        m_pqb->DBj.oJ.CopyToHost( oJxxx.p, oJxxx.Count );

        UINT32 maxnJ = 0;
        UINT32 minnJ = INT_MAX;
        UINT32 totalJ = 0;
        for( UINT32 n=0; n<m_pqb->DBj.celJ; ++n )
        {
            maxnJ = max2(maxnJ, nJxxx.p[n]);
            minnJ = min2(minnJ, nJxxx.p[n]);
            totalJ += nJxxx.p[n];

            if( nJxxx.p[n] > 1024 )
                CDPrint( cdpCD0, "%s: n=%u nJ=%u", __FUNCTION__, n, nJxxx.p[n] );


        }
        CDPrint( cdpCD0, "maxnJ=%u minnJ=%u totalJ=%u", maxnJ, minnJ, totalJ );
#endif


    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void tuSetupN10::resetGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after reset:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    
            high:   nJ      per-seed J-list sizes
                    oJ      per-seed J-list offsets
        */
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Hashes Q sequences for nongapped alignment
/// </summary>
void tuSetupN10::main()
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
