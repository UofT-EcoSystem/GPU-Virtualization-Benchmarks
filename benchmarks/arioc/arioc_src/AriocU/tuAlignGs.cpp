/*
  tuAlignGs.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"
#include <thrust/system_error.h>

#pragma region constructor/destructor
/// [private] constructor
tuAlignGs::tuAlignGs()
{

}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuAlignGs::tuAlignGs( QBatch* pqb ) : m_pqb(pqb),
                                      m_pab(pqb->pab)
{
}

/// destructor
tuAlignGs::~tuAlignGs()
{
}
#pragma endregion

#if TODO_CHOP_WHEN_DEBUGGED
static int QIDcomparer( const void * pa, const void * pb )
{
    const UINT32 a = *reinterpret_cast<const UINT32*>(pa);
    const UINT32 b = *reinterpret_cast<const UINT32*>(pb);
    if( a == b ) return 0;
    if( a < b ) return -1;
    return 1;
}
#endif

#pragma region private methods
/// [private] method alignGs10
void tuAlignGs::alignGs10()
{
    /* CUDA global memory here:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   DBn.Dl  [optional] "leftover" D values
    */

    /* Gs10: build a list of QIDs that do not yet have the required number of mappings */
    tuAlignGs10 k10( m_pqb );
    k10.Start();
    k10.Wait();

    /* Gs11: remove nulls from the QID list */
    launchKernel11();
    resetGlobalMemory11();

#if TRACE_SQID
    // what's the mapping status for the sqId?
    WinGlobalPtr<Qwarp> Qwxxx( m_pqb->DB.Qw.n, false );
    m_pqb->DB.Qw.CopyToHost( Qwxxx.p, Qwxxx.Count );
    for( UINT32 iw=0; iw<m_pqb->DB.Qw.n; ++iw )
    {
        Qwarp* pQw = Qwxxx.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            UINT64 sqId = pQw->sqId[iq];
            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                CDPrint( cdpCD0, "%s: 0x%016llx iw=%u iq=%u nAc=%d", __FUNCTION__, sqId, iw, iq, pQw->nAc[iq] );
            }
        }
    }

    // is the specified sqId or its mate in the Qu list?
    CDPrint( cdpCD0, "%s: looking for sqId 0x%016llx in the Qu list...", __FUNCTION__, TRACE_SQID );
    WinGlobalPtr<UINT32> Quxxx( m_pqb->DBgs.Qu.n, false );
    m_pqb->DBgs.Qu.CopyToHost( Quxxx.p, Quxxx.Count );
    UINT64 minSqId = _UI64_MAX;
    UINT64 maxSqId = 0;
    for( UINT32 n=0; n<m_pqb->DBgs.Qu.n; ++n )
    {
        UINT32 qid = Quxxx.p[n];
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            CDPrint( cdpCD0, "%s: %u: 0x%016llx 0x%08x", __FUNCTION__, n, sqId, qid );
        }

        minSqId = min2(minSqId, sqId);
        maxSqId = max2(maxSqId, sqId);
    }
    CDPrint( cdpCD0, "%s: minSqId=0x%016llx maxSqId=0x%016llx", __FUNCTION__, minSqId, maxSqId );
#endif

    /* Gs12: perform seed-and-extend gapped alignment */
    baseMapGc k12( "tuAlignGs12", m_pqb, &m_pqb->DBgs, &m_pqb->HBgc );
    k12.Start();
    k12.Wait();


#if TRACE_SQID
    // is the specified sqId or its mate in the D list?
    CDPrint( cdpCD0, "%s: looking for sqId 0x%016llx in the D list...", __FUNCTION__, TRACE_SQID );
    WinGlobalPtr<UINT64> Dcxxx( m_pqb->DBgs.Dc.n, false );
    m_pqb->DBgs.Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );
    for( UINT32 n=0; n<m_pqb->DBgs.Dc.n; ++n )
    {
        UINT64 D = Dcxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid(D);
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            CDPrint( cdpCD0, "%s: %u: 0x%016llx 0x%016llx 0x%08x", __FUNCTION__, n, sqId, D, qid );

        }





    }

#endif


#if TODO_CHOP_WHEN_DEBUGGED
    // how many QIDs remain in the Dc list?
    WinGlobalPtr<UINT64> Dcyyy( m_pqb->DBgs.Dc.n, false );
    m_pqb->DBgs.Dc.CopyToHost( Dcyyy.p, Dcyyy.Count );

    WinGlobalPtr<UINT32> Qyyy( Dcyyy.Count, false );
    Qyyy.n = m_pqb->DBgs.Dc.n;
    for( UINT32 n=0; n<Qyyy.n; ++n )
        Qyyy.p[n] = AriocDS::D::GetQid(Dcyyy.p[n]);

    qsort( Qyyy.p, Qyyy.Count, sizeof(UINT32), QIDcomparer );
    
    UINT32 prevQID = 0xFFFFFFFF;
    UINT32 nQ = 0;
    for( UINT32 n=0; n<Qyyy.n; ++n )
    {
        if( Qyyy.p[n] != prevQID )
        {
            ++nQ;
            prevQID = Qyyy.p[n];
        }
    }

    CDPrint( cdpCD0, "%s: Dc.n=%u nQ=%u", __FUNCTION__, m_pqb->DBgs.Dc.n, nQ );
#endif

    /* Gs19: copy Qwarps to host memory and clean up CUDA global memory */
    launchKernel19();
    resetGlobalMemory19();
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Computes gapped alignments for unmapped reads
/// </summary>
void tuAlignGs::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    try
    {
        alignGs10();
    }
    catch( thrust::system_error& ex )
    {
        int cudaErrno = ex.code().value();
        throw new ApplicationException( __FILE__, __LINE__,
                                        "CUDA error %u (0x%08x): %s\r\nCUDA Thrust says: %s",
                                        cudaErrno, cudaErrno, ex.code().message().c_str(), ex.what() );
    }

    CDPrint( cdpCD3, "[%d] %s completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
