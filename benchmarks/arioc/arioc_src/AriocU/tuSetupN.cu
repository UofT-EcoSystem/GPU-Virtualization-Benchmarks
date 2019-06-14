/*
  tuSetupN.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
   We try to localize the Thrust-dependent code in one compilation unit so as to minimize the overall compile time.
*/
#include "stdafx.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#pragma region setupN12
/// [private] method initGlobalMemory12
void tuSetupN::initGlobalMemory12()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBn.Qu  QIDs for Df values
                    
            high:   DBj.nJ  per-seed J-list sizes
                    DBj.oJ  per-seed J-list offsets
        */

        /* Allocate the QID-list buffer:
            - there is one element for each QID to be aligned
        */
        CREXEC( m_pqb->DBn.Qu.Alloc( cgaLow, m_pqb->DB.nQ, true ) );
        m_pqb->DBn.Qu.n = m_pqb->DB.nQ;
        SET_CUDAGLOBALPTR_TAG(m_pqb->DBn.Qu, "DBn.Qu");
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel12
void tuSetupN::launchKernel12()
{
    // use a thrust "transformation" API to initialize the list of QIDs
    thrust::device_ptr<UINT32> tpQu( m_pqb->DBn.Qu.p );
    thrust::sequence( epCGA, tpQu, tpQu+m_pqb->DBn.Qu.n );         // there is one 0-based QID for each Q sequence in the batch
}
#pragma endregion

#pragma region setupN14
/// [private] method initGlobalMemory14
void tuSetupN::initGlobalMemory14()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBn.Qu  QIDs for Df values
                    DBj.cnJ cumulative per-seed J-list sizes
                    
            high:   DBj.nJ  per-seed J-list sizes
                    DBj.oJ  per-seed J-list offsets
        */

        /* Allocate a buffer to contain the cumulative J-list sizes
            - We add one trailing element so that we can do an exclusive scan (see launchKernel14())
        */
        CREXEC( m_pqb->DBj.cnJ.Alloc( cgaLow,  m_pqb->DBj.celJ+1, true ) );
        SET_CUDAGLOBALPTR_TAG(m_pqb->DBj.cnJ, "DBj.cnJ");
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel14
void tuSetupN::launchKernel14()
{
    // use a thrust "reduction" API to build a list of cumulative J-list sizes
    thrust::device_ptr<UINT32> tpnJ( m_pqb->DBj.nJ.p );
    thrust::device_ptr<UINT32> tpcnJ( m_pqb->DBj.cnJ.p );

    /* The range of values to scan includes a zero following the last element in the list.  When we invoke
        thrust::exclusive_scan() we thereby obtain the sum of all the values. */
    thrust::device_ptr<UINT32> tpEnd = thrust::exclusive_scan( epCGA, tpnJ, tpnJ+m_pqb->DBj.celJ+1, tpcnJ );
    UINT32 celScan = static_cast<UINT32>(tpEnd.get() - tpcnJ.get());
    if( celScan != (m_pqb->DBj.celJ+1) )
        CDPrint( cdpCD0, "%s: celScan=%u  (celJ+1)=%u", __FUNCTION__, celScan, m_pqb->DBj.celJ+1 );

    // save the total number of J values in all J lists
    m_pqb->DBj.totalD = tpcnJ[m_pqb->DBj.celJ];

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s::%s: back from thrust exclusive scan: DBj.celJ=%u DBj.totalD=%u",
                        m_ptum->Key, __FUNCTION__,
                        m_pqb->DBj.celJ, m_pqb->DBj.totalD );
    if( m_pqb->DBj.totalD == 0 )
    {
        WinGlobalPtr<UINT32> nJxxx( m_pqb->DBj.celJ, true );
        m_pqb->DBj.nJ.CopyToHost( nJxxx.p, nJxxx.Count );
        WinGlobalPtr<UINT32> cnJxxx( m_pqb->DBj.cnJ.Count, true );
        m_pqb->DBj.cnJ.CopyToHost( cnJxxx.p, cnJxxx.Count );

        CDPrint( cdpCD0, __FUNCTION__ );
    }
#endif

#if TRACE_SQID
    CDPrint( cdpCD0, "%s::%s: loaded J-list counts...", __FUNCTION__, m_ptum->Key );

    // find the qid
    UINT32 qid = _UI32_MAX;
    for( UINT32 iw=0; (qid==_UI32_MAX) && (iw<m_pqb->QwBuffer.n); ++iw )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        for( INT16 iq=0; (qid==_UI32_MAX) && (iq<pQw->nQ); ++iq )
        {
            if( pQw->sqId[iq] == TRACE_SQID )
                qid = PACK_QID(iw,iq);
        }
    }

    if( qid == _UI32_MAX )
        CDPrint( cdpCD0, "%s: sqId 0x%016llx is not in the current batch", __FUNCTION__, TRACE_SQID );
    else
    {
        CDPrint( cdpCD0, "%s: sqId 0x%016llx <--> qid 0x%08x", __FUNCTION__, TRACE_SQID, qid );

        WinGlobalPtr<UINT64> oJxxx( m_pqb->DBj.oJ.Count, false );
        m_pqb->DBj.oJ.CopyToHost( oJxxx.p, oJxxx.Count );
        WinGlobalPtr<UINT32> nJxxx( m_pqb->DBj.nJ.Count, false );
        m_pqb->DBj.nJ.CopyToHost( nJxxx.p, nJxxx.Count );
        WinGlobalPtr<UINT32> cnJxxx( m_pqb->DBj.cnJ.Count, false );
        m_pqb->DBj.cnJ.CopyToHost( cnJxxx.p, cnJxxx.Count );

        // the offset of the first Dq in the J-list buffer is qid * npos * sps
        UINT32 npos = m_pab->a21ss.npos * m_pab->StrandsPerSeed;
        UINT32 ofsDq = qid * npos;

        // let's have a look...
        for( UINT32 o=ofsDq; o<(ofsDq+npos); ++o )
            CDPrint( cdpCD0, "%s: at offset %u: oJ=0x%016llx nJ=%u cnJ=%u", __FUNCTION__, o, oJxxx.p[o], nJxxx.p[o], cnJxxx.p[o] );
    }

    CDPrint( cdpCD0, __FUNCTION__ );
#endif
}
#pragma endregion

#pragma region setupN21
/// [private] method launchKernel21
void tuSetupN::launchKernel21()
{
    m_hrt.Restart();

    /* Verify that there is enough memory for thrust to do the sort; empirically, thrust seems to need a bit more than enough free space to copy all of the
        data being sorted */
    INT64 cbFree = m_pqb->pgi->pCGA->GetAvailableByteCount();
    INT64 cbNeeded = m_pqb->DBj.D.n * sizeof(UINT64);
    if( cbFree < cbNeeded )
        throw new ApplicationException( __FILE__, __LINE__, "%s::%s: insufficient GPU global memory for batch size %u: cbNeeded=%lld cbFree=%lld", m_ptum->Key, __FUNCTION__, m_pab->BatchSize, cbNeeded, cbFree );

    /* sort the D list */
    thrust::device_ptr<UINT64> tpD( m_pqb->DBj.D.p );
    thrust::stable_sort( epCGA, tpD, tpD+m_pqb->DBj.D.n, TSX::isLessD() );

    // performance metrics
    AriocTaskUnitMetrics* ptum = AriocBase::GetTaskUnitMetrics( "tuSetupN21" );
    InterlockedExchangeAdd( &ptum->ms.Elapsed, m_hrt.GetElapsed(false) );
    InterlockedExchangeAdd( &ptum->n.CandidateD, m_pqb->DBj.D.n );
}

/// [private] method resetGlobalMemory21
void tuSetupN::resetGlobalMemory21()
{
    /* CUDA global memory layout after reset:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   DBj.D   D list
                cnJ     cumulative per-Q J-list sizes
                nJ      per-Q J-list sizes
                oJ      per-seed J-list offsets
    */

    
    // TODO: CHOP WHEN DEBUGGED
    if( m_pqb->DBn.Qu.p == NULL )
        DebugBreak();                   // this should never happen


    // discard the list of QIDs that was used for building the D list
    m_pqb->DBn.Qu.Free();
}
#pragma endregion

#pragma region setupN22
/// [private] method launchKernel22
void tuSetupN::launchKernel22()
{
    try
    {
        /* At this point DBj.D contains a list of D values for each of the spaced seeds in each read.
            The three high-order bits are initialized to 001 (see baseLoadJn_KernelD).  Since there
            are 7 spaced seeds per read, we can use a Thrust scan operator to count (in those 3 bits)
            the number of spaced-seed "hits" per read and per reference-sequence locus.
            
           We use the number of hits to prioritize reference-sequence loci for subsequent nongapped
            alignment.  (This heuristic is only used in AriocU; it is superseded by paired-end
            heuristics in AriocP.)
        */

        m_hrt.Restart();

#if TODO_CHOP_WHEN_DEBUGGED
        if( true )
        {
            CDPrint( cdpCD0, "%s: before inclusive_scan_by_key", __FUNCTION__ );
            WinGlobalPtr<UINT64> Dxxxx( m_pqb->DBj.D.Count, false );
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
        }
#endif
        
        // use a thrust prefix-sum API to compute the number of D values for each QID
        thrust::device_ptr<UINT64> tpD( m_pqb->DBj.D.p );
        thrust::inclusive_scan_by_key( epCGA,                       // execution policy
                                       tpD, tpD+m_pqb->DBj.D.n,     // keys
                                       tpD,                         // values
                                       tpD,                         // results
                                       TSX::isEqualD(),             // compares QID, subId, and J in D values
                                       TSX::addDvalueFlags() );     // adds flag bits in D values; leaves bits 0-60 intact

#if TODO_CHOP_WHEN_DEBUGGED
        if( true )
        {
            CDPrint( cdpCD0, "%s: after inclusive_scan_by_key", __FUNCTION__ );
            WinGlobalPtr<UINT64> Dxxxx( m_pqb->DBj.D.Count, false );
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
        }
#endif

        // performance metrics
        AriocTaskUnitMetrics* ptum = AriocBase::GetTaskUnitMetrics( "tuSetupN22" );
        InterlockedExchangeAdd( &ptum->ms.Elapsed, m_hrt.GetElapsed(false) );
    }
    catch( ApplicationException* pex )
    {
        pex->SetCallerExceptionInfo( __FILE__, __LINE__, "Unable to allocate GPU memory for unduplicating D values.  Try a smaller maximum J-list size (maxJ) for gapped alignments and/or a smaller maximum batch size (batchSize)." );
        throw pex;
    }
}
#pragma endregion

#pragma region setupN26
/// [private] method initGlobalMemory26
void tuSetupN::initGlobalMemory26()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBn.Dc  candidates for nongapped alignment

            high:   DBj.D   D list
                    cnJ     cumulative per-Q J-list sizes
                    nJ      per-Q J-list sizes
                    oJ      per-seed J-list offsets
        */

        /* Allocate a buffer to contain D values that are candidates for nongapped alignment.  We assume the
            worst case (i.e., every seed location will be a candidate location) for now and adjust the buffer
            size later.
        */
        UINT32 cel = blockdiv( m_pqb->DBj.D.n, m_pab->aas.ACP.seedCoverageN );
        CREXEC( m_pqb->DBn.Dc.Alloc( cgaLow, cel, false ) );
        SET_CUDAGLOBALPTR_TAG(m_pqb->DBn.Dc, "DBn.Dc" );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel26
void tuSetupN::launchKernel26()
{
    m_hrt.Restart();

    // performance metrics
    AriocTaskUnitMetrics* ptum = AriocBase::GetTaskUnitMetrics( "tuSetupN26" );
    InterlockedExchangeAdd( &ptum->n.CandidateD, m_pqb->DBj.D.n );


#if TODO_CHOP_WHEN_DEBUGGED
    // look at the D list
    WinGlobalPtr<UINT64> Dxxx( m_pqb->DBj.D.Count, false );
    m_pqb->DBj.D.CopyToHost( Dxxx.p, Dxxx.Count );

    UINT32 nCan = 0;
    for( UINT32 x=0; x<static_cast<UINT32>(m_pqb->DBj.D.Count); ++x )
    {
        if( (Dxxx.p[x] & AriocDS::D::flagCandidate) != 0 )
            ++nCan;
    }

    CDPrint( cdpCD0, "%s: nCan=%u", __FUNCTION__, nCan );
#endif


    /* Use a Thrust "stream compaction" API to identify the candidates for nongapped alignment */
    thrust::device_ptr<UINT64> tpD( m_pqb->DBj.D.p );
    thrust::device_ptr<UINT64> tpDc( m_pqb->DBn.Dc.p );
    thrust::device_ptr<UINT64> eolDc = thrust::copy_if( epCGA, tpD, tpD+m_pqb->DBj.D.n, tpDc, TSX::isCandidateDvalue() );
    m_pqb->DBn.Dc.n = static_cast<UINT32>(eolDc.get() - tpDc.get());

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: %u/%u candidates for nongapped alignment using seedCoverageN=%d",
                        __FUNCTION__, m_pqb->DBn.Dc.n, m_pqb->DBj.D.n, m_pab->aas.ACP.seedCoverageN );
#endif

    // shrink the Dc buffer if possible
    m_pqb->DBn.Dc.Resize( m_pqb->DBn.Dc.n );

    // performance metrics
    InterlockedExchangeAdd( &ptum->ms.Launch, m_hrt.GetElapsed(false) );
}
#pragma endregion

#pragma region setupN28
/// [private] method initGlobalMemory28
void tuSetupN::initGlobalMemory28()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBn.Dc  candidates for nongapped alignment
                    DBn.Du  "leftover" D values (candidates for gapped alignment)

            high:   DBj.D   D list
                    cnJ     cumulative per-Q J-list sizes
                    nJ      per-Q J-list sizes
                    oJ      per-seed J-list offsets
        */

        // allocate a buffer for the leftovers list
        CREXEC( m_pqb->DBn.Du.Alloc( cgaLow, m_pqb->DBj.D.n, false ) );
        SET_CUDAGLOBALPTR_TAG( m_pqb->DBn.Du, "DBn.Du" );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel28
void tuSetupN::launchKernel28()
{
    m_hrt.Restart();

    // performance metrics
    AriocTaskUnitMetrics* ptum = AriocBase::GetTaskUnitMetrics( "tuSetupN28" );
    InterlockedExchangeAdd( &ptum->n.CandidateD, m_pqb->DBj.D.n );

    /* Use a Thrust "stream compaction" API to identify the leftover candidates for gapped alignment */
    thrust::device_ptr<UINT64> tpD( m_pqb->DBj.D.p );
    thrust::device_ptr<UINT64> tpDu( m_pqb->DBn.Du.p );
    thrust::device_ptr<UINT64> eolDu = thrust::copy_if( epCGA, tpD, tpD+m_pqb->DBj.D.n, tpDu, TSX::isFlagX() );
    m_pqb->DBn.Du.n = static_cast<UINT32>(eolDu.get() - tpDu.get());

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: %u/%u leftover candidates for gapped alignment", __FUNCTION__, m_pqb->DBn.Du.n, m_pqb->DBj.D.n );
#endif

    // shrink the Du buffer if possible
    m_pqb->DBn.Du.Resize( m_pqb->DBn.Du.n );

    // performance metrics
    InterlockedExchangeAdd( &ptum->ms.Launch, m_hrt.GetElapsed(false) );
}

/// [private] method resetGlobalMemory28
void tuSetupN::resetGlobalMemory28()
{
    /* CUDA global memory layout after reset:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data
                DBn.Dc  candidates for nongapped alignment
                DBn.Du  leftover candidates for gapped alignment

        high:           (unallocated)
    */

    // free buffers that are no longer needed
    m_pqb->DBj.D.Free();
    m_pqb->DBj.nJ.Free();
    m_pqb->DBj.oJ.Free();
}
#pragma endregion
