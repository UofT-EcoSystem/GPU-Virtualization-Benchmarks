/*
  baseLoadJn.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseLoadJn::baseLoadJn()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="npos">number of seed positions per Q sequence</param>
/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
baseLoadJn::baseLoadJn( const char* ptumKey, QBatch* pqb, UINT32 npos ) : m_pqb(pqb),
                                                                          m_pab(pqb->pab),
                                                                          m_npos(npos),
                                                                          m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseLoadJn")),
                                                                          m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseLoadJn", m_cudaThreadsPerBlock );
}

/// destructor
baseLoadJn::~baseLoadJn()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void baseLoadJn::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:   Qw       Qwarps
                   Qi       interleaved Q sequence data
                   DBn.Qu   QIDs for D values

            high:  DBj.D    D lists
                   nJ       per-seed J-list sizes
                   oJ       per-seed J-list offsets
        */
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method computeGridDimensions
void baseLoadJn::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pqb->DBj.D.n );
}


#if TODO_CHOP_WHEN_DEBUGGED
extern int DfQIDcomparer( const void * pa, const void * pb );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
int DQIDcomparer( const void * pa, const void * pb )
{
    const UINT64 a = *reinterpret_cast<const UINT64*>(pa) & AriocDS::D::maskQID;
    const UINT64 b = *reinterpret_cast<const UINT64*>(pb) & AriocDS::D::maskQID;
    if( a == b ) return 0;
    if( a < b ) return -1;
    return 1;
}
#endif


/// [private] method copyKernelResults
void baseLoadJn::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // performance metrics
        m_ptum->AccumulateMaxPct( tumMaxUsed, m_pqb->pgi->pCGA->GetAllocatedByteCount( cgaBoth ), m_pqb->pgi->pCGA->cb );
        InterlockedExchangeAdd( &m_ptum->n.CandidateD, m_pqb->DBj.D.n );

        // wait for the kernel to complete
        CREXEC( waitForKernel() );

#if TRACE_SQID
        CDPrint( cdpCD0, "[%d] %s::%s: verifying Q sequence for sqId 0x%016llx...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    
        UINT32 qidxxx = _UI32_MAX;
        for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
        {
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            for( INT16 iq=0; iq<pQw->nQ; ++iq )
            {
                if( (pQw->sqId[iq] == TRACE_SQID) || (m_pab->pifgQ->HasPairs && (pQw->sqId[iq] ^ TRACE_SQID) == 1) )
                {
                    qidxxx = PACK_QID( iw, iq );

                    // dump the Q sequence
                    CDPrint( cdpCD0, "%s: sqId 0x%016llx <--> qid 0x%08x", __FUNCTION__, TRACE_SQID, qidxxx );
                    UINT64* pQi = m_pqb->QiBuffer.p + pQw->ofsQi + iq;
                    AriocCommon::DumpA21( pQi, pQw->N[iq], CUDATHREADSPERWARP );
                    AriocCommon::DumpA21RC( pQi, pQw->N[iq], CUDATHREADSPERWARP );

                    if( qidxxx & 1 )
                        break;
                }
            }
        }

        if( m_pab->pifgQ->HasPairs )
        {
            CDPrint( cdpCD0, "%s::%s: loaded Df values for paired-end data...", m_ptum->Key, __FUNCTION__ );

            // show Df values
            WinGlobalPtr<UINT64> Dfxxx( m_pqb->DBj.totalD, false );
            m_pqb->DBj.D.CopyToHost( Dfxxx.p, Dfxxx.Count );

            for( UINT32 n=0; n<static_cast<UINT32>(m_pqb->DBj.totalD); ++n )
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
                    CDPrint( cdpCD0, "%s::%s qid=0x%08X Df=0x%016llx subId=%d D=0x%08x (%d) %c",
                                        m_ptum->Key, __FUNCTION__,
                                        qid, Df, subId, pos, pos, s );
                }
            }
            CDPrint( cdpCD0, __FUNCTION__ );
        }

        else
        {
            CDPrint( cdpCD0, "%s::%s: loaded D values for unpaired data...", __FUNCTION__, m_ptum->Key );
            UINT32 nDforSqId = 0;

            // show D values
            WinGlobalPtr<UINT64> Dxxx( m_pqb->DBj.totalD, false );
            m_pqb->DBj.D.CopyToHost( Dxxx.p, Dxxx.Count );

            for( UINT32 n=0; n<static_cast<UINT32>(m_pqb->DBj.totalD); ++n )
            {
                UINT64 D = Dxxx.p[n];
                UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
                UINT32 iw = QID_IW(qid);
                UINT32 iq = QID_IQ(qid);
                UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

                if( sqId == TRACE_SQID )
                {
                    INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
                    INT32 pos = static_cast<INT32>(D & 0x7FFFFFFF);
                    INT32 Jf = (D & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - pos) : pos;

#if TRACE_SUBID
                    // if( subId == TRACE_SUBID )
#endif
                    {

                        CDPrint( cdpCD0, "%s::%s: %4u 0x%016llx qid=0x%08x subId=%d pos=%d Jf=%d",
                            m_ptum->Key, __FUNCTION__,
                            n, D, qid, subId, pos, Jf );
                        ++nDforSqId;
                    }
                }
            }

            CDPrint( cdpCD0, "%s: nDforSqId=%u", __FUNCTION__, nDforSqId );
        }
#endif


#if TODO_CHOP_WHEN_DEBUGGED
        WinGlobalPtr<UINT64> Dxxx( m_pqb->DBj.totalD, false );
        m_pqb->DBj.D.CopyToHost( Dxxx.p, Dxxx.Count );

        qsort( Dxxx.p, Dxxx.Count, sizeof(UINT64), DQIDcomparer );


        // count distinct QIDs among the D values
        UINT32 nDistinctQIDs = 0;
        UINT32 nEmittedQIDs = 0;
        UINT32 prevQID = 0xFFFFFFFF;
        for( UINT32 n=0; n<m_pqb->DBj.totalD; ++n )
        {
            UINT64 D = Dxxx.p[n];
            UINT32 qid = AriocDS::D::GetQid(D);

            if( prevQID != 0xffffffff && qid < prevQID ) DebugBreak();
            if( qid != prevQID )
            {
                ++nDistinctQIDs;
                prevQID = qid;

//                if( ++nEmittedQIDs < 256 )
                    CDPrint( cdpCD0, "%s: qid=0x%08X at n=%u", __FUNCTION__, qid, n );
            }
        }
        
        CDPrint( cdpCD0, "%s: nDistinctQIDs=%u", __FUNCTION__, nDistinctQIDs );

        // get the minimum and maximum QIDs
        UINT64 Df = Dxxx.p[0];
        UINT32 minQID = AriocDS::D::GetQid(Df);
        Df = Dxxx.p[Dxxx.Count-1];
        UINT32 maxQID = AriocDS::D::GetQid(Df);
        CDPrint( cdpCD0, "%s: min QID = 0x%08x  max QID = 0x%08X", __FUNCTION__, minQID, maxQID );
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
void baseLoadJn::resetGlobalMemory()
{
    /* CUDA global memory layout after reset:

        low:   Qw       Qwarps
               Qi       interleaved Q sequence data
               DBn.Qu   QIDs for D values

        high:  D        D lists
               nJ       per-seed J-list sizes
               oJ       per-seed J-list offsets
    */

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, m_hrt.GetElapsed(true) );
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Loads J values from the hash table (builds a list of Df values)
/// </summary>
void baseLoadJn::main()
{
    CRVALIDATOR;

    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    // if we're processing paired-end input...
    if( m_pab->pifgQ->HasPairs )
        launchKernelDf( d3g, d3b, initSharedMemory() );     // compute Df values (for using paired-end criteria)
    else
        launchKernelD( d3g, d3b, initSharedMemory() );      // compute D values (for counting adjacent seed hits)

    copyKernelResults();
    resetGlobalMemory();
    
    CDPrint( cdpCD3, "[%d] %s: completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
#pragma endregion
