/*
  baseLoadJs.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseLoadJs::baseLoadJs()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
/// <param name="isi">index of the current seed-interval iteration</param>
/// <param name="nJ">total number of J values for the current iteration</param>
baseLoadJs::baseLoadJs( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, UINT32 isi, UINT32 nJ ) : m_pqb(pqb),
                                                                                                             m_pab(pqb->pab),
                                                                                                             m_pdbb(pdbb),
                                                                                                             m_pRuBuffer(NULL),
                                                                                                             m_isi(isi),
                                                                                                             m_iSeedPos(0),
                                                                                                             m_nSeedPos(0),
                                                                                                             m_nJ(nJ),
                                                                                                             m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseLoadJs")),
                                                                                                             m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseLoadJs", m_cudaThreadsPerBlock );
    
    // if paired subId bits are available, use them
    m_pRuBuffer = (m_pdbb->Rx.p ? m_pdbb->Rx.p : m_pdbb->Ru.p);

    // compute the offset and length of the seed-position list for the specified seed iteration
    m_iSeedPos = m_pab->a21hs.ofsSPSI.p[m_isi];

    UINT32 maxnSeedPos = m_pab->a21hs.ofsSPSI.p[m_isi+1] - m_iSeedPos;
    m_nSeedPos = min2(static_cast<UINT32>(m_pdbb->AKP.seedsPerQ),maxnSeedPos);
}

/// destructor
baseLoadJs::~baseLoadJs()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void baseLoadJs::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      QIDs of unmapped mates
                    Ru      [optional] subId bits for QIDs of unmapped mates
                    Dc      [optional] mapped candidates with unmapped opposite mates
                    DBj.D   D values for candidates for seed-and-extend alignment
                   
            high:   Diter   J list (D values) for current iteration
                    cnJ     cumulative per-Q J-list sizes
                    nJ      per-Q J-list sizes
                    oJ      per-seed J-list offsets
                    Du      unmapped D values
        */
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method computeGridDimensions
void baseLoadJs::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nJ );
}


#if TODO_CHOP_WHEN_DEBUGGED
static int DjQIDcomparer( const void * pa, const void * pb )
{
    const UINT64 a = *reinterpret_cast<const UINT64*>(pa);
    const UINT64 b = *reinterpret_cast<const UINT64*>(pb);
    if( a == b ) return 0;
    if( a < b ) return -1;
    return 1;
}
#endif


/// [private] method copyKernelResults
void baseLoadJs::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // performance metrics
        m_ptum->AccumulateMaxPct( tumMaxUsed, m_pqb->pgi->pCGA->GetAllocatedByteCount(cgaBoth), m_pqb->pgi->pCGA->cb );
        InterlockedExchangeAdd( &m_ptum->n.CandidateD, m_nJ );

        // wait for the kernel to complete
        CREXEC( waitForKernel() );


#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s: m_isi=%u m_nJ=%u Diter.Count=%lld", __FUNCTION__, m_isi, m_nJ, m_pdbb->Diter.Count );
        WinGlobalPtr<UINT64> Diterxxx( m_pdbb->Diter.Count, false );
        m_pdbb->Diter.CopyToHost( Diterxxx.p, Diterxxx.Count );
        for( UINT32 n=0; n<min2(100,m_nJ); n++ )
            CDPrint( cdpCD0, "%s: Diter.p[%4u]=%016llx", __FUNCTION__, n, Diterxxx.p[n] );
        CDPrint( cdpCD0, __FUNCTION__ );
#endif

#if defined(TRACE_SQID)
        WinGlobalPtr<UINT64> Diterxxx( m_pdbb->Diter.Count, false );
        m_pdbb->Diter.CopyToHost( Diterxxx.p, Diterxxx.Count );
        UINT32 nNull = 0;
        for( UINT32 n=0; n<Diterxxx.Count; n++ )
        {
            UINT64 D = Diterxxx.p[n];
            if( D == _UI64_MAX )
                ++nNull;
            else
            {
                UINT32 qid = AriocDS::D::GetQid( D );
                UINT32 iw = QID_IW( qid );
                INT16 iq = QID_IQ( qid );
                Qwarp* pQw = m_pqb->QwBuffer.p + iw;
                UINT64 sqId = pQw->sqId[iq];

                if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
                {
                    INT32 subId = ((D & AriocDS::D::maskSubId) >> 32) & 0x7F;
                    INT32 pos = static_cast<INT32>(D & AriocDS::D::maskPos);
                    UINT32 j = static_cast<UINT32>(D);
                    INT32 Jf = (D & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - pos) : pos;

#if TRACE_SUBID
                if( subId == TRACE_SUBID )
#endif
                    CDPrint( cdpCD0, "%s::%s: n=%u: sqId=0x%016llx D=0x%016llx subId=%d pos=%d j=0x%08x Jf=%d",
                        m_ptum->Key, __FUNCTION__,
                        n, sqId, D, subId, pos, j, Jf );
                }
            }
        }
        CDPrint( cdpCD0, "%s: %u/%u null D values in Diter", __FUNCTION__, nNull, static_cast<UINT32>(Diterxxx.Count) );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: counting distinct QIDs in %u loaded J values:", __FUNCTION__, m_nJtotal );
    WinGlobalPtr<UINT64> Djxxx( m_nJtotal, false );
    m_pdbb->Diter.CopyToHost( Djxxx.p, Djxxx.Count );

    qsort( Djxxx.p, Djxxx.Count, sizeof(UINT64), DjQIDcomparer );

    // count distinct QIDs among the Dj values
    UINT32 nDistinctQIDs = 0;
    UINT32 nEmittedQIDs = 0;
    UINT32 prevQID = 0xFFFFFFFF;
    for( UINT32 n=0; n<m_nJtotal; ++n )
    {
        UINT64 Dj = Djxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid(Dj);

        if( (prevQID != 0xffffffff) && (qid < prevQID) ) DebugBreak();
        if( qid != prevQID )
        {
            ++nDistinctQIDs;
            prevQID = qid;

            if( ++nEmittedQIDs < 256 )
                CDPrint( cdpCD0, "%s: qid=0x%08X", __FUNCTION__, qid );
        }
    }
        
    CDPrint( cdpCD0, "%s: nDistinctQIDs=%u", __FUNCTION__, nDistinctQIDs );
#endif

        // performance metrics
    
#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s elapsed time for Launch: %dms", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_hrt.GetElapsed(false) );
#endif


        InterlockedExchangeAdd( &m_ptum->us.Launch, m_hrt.GetElapsed(true) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseLoadJs::resetGlobalMemory()
{
    /* CUDA global memory layout at this point:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Dc      [optional] mapped candidates with unmapped opposite mates
                    Qu      QIDs of unmapped mates
                    Ru      [optional] subId bits for QIDs of unmapped mates
                    D       D values for candidates for seed-and-extend alignment
                   
            high:   Diter   J list (D values) for current iteration
                    cnJ     cumulative per-Q J-list sizes
                    nJ      per-Q J-list sizes
                    oJ      per-seed J-list offsets
    */

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, m_hrt.GetElapsed(true) );
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Loads J values from the hash table (builds a list of Dj values)
/// </summary>
void baseLoadJs::main()
{
    CRVALIDATOR;

    CDPrint( cdpCD3, "[%d] %s::%s...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    // compute D values (reference position adjusted for the seed location within the Q sequence)
    launchKernelD( d3g, d3b, initSharedMemory() );

    copyKernelResults();
    resetGlobalMemory();
    
    CDPrint( cdpCD3, "[%d] %s::%s: complete", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
#pragma endregion
