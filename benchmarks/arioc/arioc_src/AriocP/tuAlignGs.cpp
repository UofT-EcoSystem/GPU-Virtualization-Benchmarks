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

        high:   (unallocated)
    */

    /* Gs10: build a list of QIDs that do not yet have concordant mappings */
    tuAlignGs10 k10( m_pqb );
    k10.Start();
    k10.Wait();

    /* Gs11: remove nulls from the QID list */
    launchKernel11();

#if TRACE_SQID
    // what's the concordant-mapping status for the sqId?
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

    /* Gs12: try to map the opposite mates using seed-and-extend gapped alignment */
    baseMapGs k12( "tuAlignGs12", m_pqb, &m_pqb->DBgs, &m_pqb->HBgs );
    k12.Start();
    k12.Wait();

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: DBgs.Dc.n=%u", __FUNCTION__, m_pqb->DBgs.Dc.n );
    WinGlobalPtr<UINT64> Dcxxx( m_pqb->DBgs.Dc.n, false );
    m_pqb->DBgs.Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );
    bool isBad = false;
    for( UINT32 n=0; n<m_pqb->DBgs.Dc.n; ++n )
    {
        UINT64 Dc = Dcxxx.p[n];
        if( ((Dc >> 32) & 0x7F) > 0x20 )
        {
            CDPrint( cdpCD0, "%s: n=%u Dc=0x%016llx", __FUNCTION__, n, Dc );
            isBad = true;
        }
    }

    if( isBad )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif


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
}


/// [private] method alignGs20
void tuAlignGs::alignGs20()
{
    /* CUDA global memory here:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBgs.Qu unmapped QIDs
                DBgs.Ru subId bits for unmapped QIDs
                DBgs.Dc mapped D values without opposite-mate mappings

        high:   (unallocated)
    */

    /* Gs20: set up to do seed-and-extend alignment on the remaining unmapped Q sequences */
    initGlobalMemory20();

    /* Gs22: do seed-and-extend alignment at reference locations that are prioritized using seed coverage */
    baseMapGc k22( "tuAlignGs22", m_pqb, &m_pqb->DBgs, &m_pqb->HBgc );
    k22.Start();
    k22.Wait();
}

/// [private] method alignGs30
void tuAlignGs::alignGs30()
{
    /* CUDA global memory here:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBgs.Qu unmapped QIDs
                DBgs.Ru subId bits for unmapped QIDs
                DBgs.Dc mapped D values without opposite-mate mappings

        high:   (unallocated)
    */

#if TRACE_SQID
    WinGlobalPtr<UINT64> Dcxxx( m_pqb->DBgs.Dc.n, false );
    m_pqb->DBgs.Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );
    for( UINT32 n=0; n<m_pqb->DBgs.Dc.n; ++n )
    {
        UINT64 D = Dcxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( D );
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 J = D & 0x7FFFFFFF;
            INT32 Jf = (D & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

            CDPrint( cdpCD0, "[%d] %s: %5u: sqId=0x%016llx D=0x%016llx qid=0x%08x subId=%d J=%d Jf=%d",
                                m_pqb->pgi->deviceId, __FUNCTION__,
                                n, sqId, D, qid, subId, J, Jf );
        }
    }
#endif

    // prepare CUDA global memory for iteration
    UINT32 nDcPerIteration = initGlobalMemory30();

    if( (CDPrintFilter & cdpCD4) && (nDcPerIteration < m_pqb->DBgs.Dc.n) )
    {
        INT64 sqId0 = m_pqb->QwBuffer.p[0].sqId[0];
        CDPrint( cdpCD4, "%s: 0x%016llx (%lld) nDcPerIteration=%u Dc.n=%u",
                            __FUNCTION__, sqId0, AriocDS::SqId::GetReadId(sqId0), nDcPerIteration, m_pqb->DBgs.Dc.n );
    }

    // loop through the list of Dc values
    UINT32 nDcRemaining = m_pqb->DBgs.Dc.n;
    UINT32 iDc = 0;
    while( nDcRemaining )
    {
        // compute the actual number of D values to be processed in the current iteration
        UINT32 nDc = min2( nDcRemaining, nDcPerIteration );

        /* Gs30: set up for windowed gapped alignment; transform the Dc list into D values for the opposite (unmapped) mates */
        baseLoadRw k30( "tuAlignGs30", m_pqb, &m_pqb->DBgs, riDc, iDc, nDc );
        k30.Start();
        k30.Wait();

        /* Gs32: perform windowed gapped alignment; update the Dc list with the traceback origin for high-scoring mappings */
        baseMaxVw k32( "tuAlignGs32", m_pqb, &m_pqb->DBgs, riDc, iDc, nDc );
        k32.Start();
        k32.Wait();

        // iterate
        iDc += nDc;
        nDcRemaining -= nDc;
    }

#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<INT16> VmaxDcxx( m_pqb->DBgs.VmaxDc.n, false );
    m_pqb->DBgs.VmaxDc.CopyToHost( VmaxDcxx.p, VmaxDcxx.Count );
    UINT32 nMapped = 0;
    for( UINT32 n=0; n<m_pqb->DBgs.VmaxDc.n; ++n )
    {
        if( VmaxDcxx.p[n] )
            ++nMapped;
    }
    CDPrint( cdpCD0, "%s: nMapped=%u", __FUNCTION__, nMapped );
#endif


    /* Gs34: count the number of mapped positions */
    launchKernel34();

    /* Gs36: build a list (Dm) that contains only the mapped D values */
    initGlobalMemory36();
    launchKernel36();
    resetGlobalMemory36();
}

/// [private] method alignGs40
void tuAlignGs::alignGs40()
{
    if( m_pqb->DBgs.Dm.n )
    {
        /* Gs40: load interleaved R sequence data */
        RaiiPtr<tuBaseS> k40 = baseLoadRix::GetInstance( "tuAlignGs40", m_pqb, &m_pqb->DBgs, riDm );
        k40->Start();
        k40->Wait();

        /* Gs42: perform gapped alignment and traceback */
        baseAlignG k42( "tuAlignGs42", m_pqb, &m_pqb->DBgs, &m_pqb->HBgwc, riDm );
        k42.Start();
        k42.Wait();

        /* Gs44: count per-Q mappings (update the Qwarp buffer in GPU global memory) */
        baseCountA k44( "tuAlignGs44", m_pqb, &m_pqb->DBgs, riDm );
        k44.Start();
        k44.Wait();
    }

    /* Gs49: tidy up CUDA global memory */
    launchKernel49();
    resetGlobalMemory49();

}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Computes gapped short-read alignments for paired-end reads
/// </summary>
void tuAlignGs::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    try
    {
        alignGs10();
        alignGs20();
        alignGs30();
        alignGs40();
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
