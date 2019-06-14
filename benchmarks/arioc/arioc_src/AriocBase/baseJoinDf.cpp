/*
  baseJoinDf.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseJoinDf::baseJoinDf()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
baseJoinDf::baseJoinDf( const char* ptumKey, QBatch* pqb ) : m_pqb(pqb),
                                                             m_pab(pqb->pab),
                                                             m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseJoinDf")),
                                                             m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseJoinDf", m_cudaThreadsPerBlock );
}

/// destructor
baseJoinDf::~baseJoinDf()
{
}
#pragma endregion

#pragma region protected methods
/// [private] method initGlobalMemory
void baseJoinDf::initGlobalMemory()
{
    /* CUDA global memory layout is unchanged. */
}

/// [private] method computeGridDimensions
void baseJoinDf::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pqb->DBj.totalD );
}

#if TODO_CHOP_WHEN_DEBUGGED
int DfQIDcomparer( const void * pa, const void * pb )
{
    const UINT64 a = *reinterpret_cast<const UINT64*>(pa) & 0x0FFFFF0000000001;
    const UINT64 b = *reinterpret_cast<const UINT64*>(pb) & 0x0FFFFF0000000001;
    if( a == b ) return 0;
    if( a < b ) return -1;
    return 1;
}
#endif

/// [private] method copyKernelResults
void baseJoinDf::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // block until the kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->us.Launch, m_hrt.GetElapsed(true) );

#if TODO_CHOP_WHEN_DEBUGGED
        WinGlobalPtr<UINT64> Dfxxx( m_pqb->DBj.totalD, false );
        m_pqb->DBj.Df.CopyToHost( Dfxxx.p, Dfxxx.Count );
        qsort( Dfxxx.p, Dfxxx.Count, sizeof(UINT64), DfQIDcomparer );


        // count distinct QIDs among the Df values
        UINT32 nDistinctQIDs = 0;
        UINT32 prevQID = 0xFFFFFFFF;
        for( UINT32 n=0; n<m_pqb->DBj.totalD; ++n )
        {
            UINT64 Df = Dfxxx.p[n];
            UINT32 qid = static_cast<UINT32>(((Df >> 40) << 1) | (Df & 1)) & AriocDS::QID::maskQID;

            if( prevQID != 0xffffffff && qid < prevQID ) DebugBreak();
            if( qid != prevQID )
            {
                ++nDistinctQIDs;
                prevQID = qid;
            }
        }
        
        // count Df values that were flagged
        UINT32 nCandidates1 = 0;                // mate 1
        UINT32 nCandidates2 = 0;                // mate 2
        UINT32 nFlaggedQIDs = 0;
        prevQID = 0xFFFFFFFF;
        for( UINT32 n=0; n<m_pqb->DBj.totalD; ++n )
        {
            UINT64 Df = Dfxxx.p[n];
            if( Df & AriocDS::D::flagCandidate )
            {

                if( Df & 1 )
                    nCandidates2++ ;
                else
                    nCandidates1++ ;

                UINT32 qid = static_cast<UINT32>(((Df >> 40) << 1) | (Df & 1)) & AriocDS::QID::maskQID;
                if( qid != prevQID )
                {
                    ++nFlaggedQIDs;
                    prevQID = qid;
                }
            }
        }

        CDPrint( cdpCD0, "baseJoinDf::copyKernelResults: nFlaggedQIDs/nDistinctQIDs=%u/%u (%2.1f%%)", nFlaggedQIDs, nDistinctQIDs, 100.0*nFlaggedQIDs/nDistinctQIDs );
        CDPrint( cdpCD0, "baseJoinDf::copyKernelResults: nCandidates1=%u nCandidates2=%u m_pqb->DBj.totalD=%u", nCandidates1, nCandidates2, m_pqb->DBj.totalD );
#endif

#if TRACE_SQID
        // show Df values
        WinGlobalPtr<UINT64> Dfxxx( m_pqb->DBj.totalD, false );
        m_pqb->DBj.D.CopyToHost( Dfxxx.p, Dfxxx.Count );

        for( INT64 n=0; n<static_cast<UINT32>(m_pqb->DBj.totalD); ++n )
        {
            UINT64 Df = Dfxxx.p[n];
            UINT32 qid = static_cast<UINT32>(((Df >> 40) << 1) | (Df & 1)) & AriocDS::QID::maskQID;
            UINT32 iw = QID_IW(qid);
            UINT32 iq = QID_IQ(qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];

            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                UINT8 subId = (Df >> 33) & 0x7F;
                UINT32 pos = (Df >> 2) & 0x7FFFFFFF;
                char s = (Df & 2) ? 'R' : 'F';

#if TRACE_SUBID
        if( subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "%s::%s qid=0x%08X Df=0x%016llx subId=%d pos=0x%08x (%d) %c",
                                    m_ptum->Key, __FUNCTION__,
                                    qid, Df, subId, pos, pos, s );
            }
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
void baseJoinDf::resetGlobalMemory()
{
    /* CUDA global memory layout is unchanged */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Identifies candidates for paired-end alignment
/// </summary>
void baseJoinDf::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    launchKernel( d3g, d3b, initSharedMemory() );

    copyKernelResults();
    resetGlobalMemory();
    
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s::%s: completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
#pragma endregion
