/*
  tuSetupN24.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuSetupN24::tuSetupN24()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuSetupN24::tuSetupN24( QBatch* pqb ) : m_pqb(pqb),
                                        m_pab(pqb->pab),
                                        m_seedCoverageLeftover(2),
                                        m_ptum(AriocBase::GetTaskUnitMetrics("tuSetupN24"))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_tuSetupN24", m_cudaThreadsPerBlock );

    // look for an Xparam that specifies the nongapped seed-coverage "leftover" threshold
    INT32 i = pqb->pab->paamb->Xparam.IndexOf( "seedCoverageLeftover" );
    if( i >= 0 )
        m_seedCoverageLeftover = static_cast<INT32>(pqb->pab->paamb->Xparam.Value(i));
}

/// destructor
tuSetupN24::~tuSetupN24()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void tuSetupN24::initGlobalMemory()
{
    /* CUDA global memory layout at this point:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   DBj.D   D list
                cnJ     cumulative per-Q J-list sizes
                nJ      per-Q J-list sizes
                oJ      per-seed J-list offsets
    */
}

/// [private] method computeGridDimensions
void tuSetupN24::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pqb->DBj.D.n );
}

/// [private] method copyKernelResults
void tuSetupN24::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->n.CandidateD, m_pqb->DBj.D.n );

        // block until the kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );




#if TODO_CHOP_WHEN_DEBUGGED
WinGlobalPtr<UINT64> Dxxxx( m_pqb->DBj.D.n, false );
m_pqb->DBj.D.CopyToHost( Dxxxx.p, Dxxxx.Count );
for( UINT32 n=0; n<1000; ++n )
{
    UINT64 D = Dxxxx.p[n];
    UINT32 qid = AriocDS::D::GetQid( D );
    INT16 subId = static_cast<INT16>(D >> 32) & 0x007F;
    INT32 J = static_cast<INT32>(D & 0x7fffffff);
    INT32 Jf = (D & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - J) : J;
    CDPrint( cdpCD0, "%s: %4u: 0x%016llx %u 0x%08x %d %d Jf=%d", __FUNCTION__, n, D, static_cast<INT32>(D >> 61), qid, subId, J, Jf );
}
#endif





    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void tuSetupN24::resetGlobalMemory()
{
    /* CUDA global memory layout at this point:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   DBj.D   D list
                cnJ     cumulative per-Q J-list sizes
                nJ      per-Q J-list sizes
                oJ      per-seed J-list offsets
    */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Identifies candidates for nongapped alignment based on spaced seed coverage.
/// </summary>
void tuSetupN24::main()
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
