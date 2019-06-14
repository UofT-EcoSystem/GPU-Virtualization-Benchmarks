/*
  baseMapGs.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/merge.h>

#pragma region private methods
/// [private] method initSharedMemory
UINT32 baseMapGs::initSharedMemory()
{
    return 0;
}

/// [private] method accumulateJcounts
void baseMapGs::accumulateJcounts()
{
    // use a thrust "reduction" API to build a list of cumulative J-list sizes
    thrust::device_ptr<UINT32> tpnJ( m_pqb->DBj.nJ.p );
    thrust::device_ptr<UINT32> tpcnJ( m_pqb->DBj.cnJ.p );

    /* The range of values to scan includes a zero following the last element in the list, so we can also obtain
        the sum of all the values:
        
        - DBj.cnJ.Count includes the trailing 0 element
        - DBj.celJ = nQ*seedsPerQ, i.e. it excludes the trailing zero
    */
    thrust::exclusive_scan( epCGA, tpnJ, tpnJ+m_pqb->DBj.celJ+1, tpcnJ );

    // get a copy of the cumulative J-list counts into a host buffer
    if( m_cnJ.Count < m_pqb->DBj.cnJ.Count )
        m_cnJ.Realloc( m_pqb->DBj.cnJ.Count, false );
    m_pqb->DBj.cnJ.CopyToHost( m_cnJ.p, m_pqb->DBj.cnJ.Count );

    // save the total number of J values in all J lists
    m_pqb->DBj.totalD = m_cnJ.p[m_pqb->DBj.celJ];

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: DBj.totalD = %llu", __FUNCTION__, m_pqb->DBj.totalD );
#endif


#if TODO_CHOP_WHEN_DEBUGGED
    if( m_pqb->DBj.totalD == 0 )
        DebugBreak();
#endif
 
}

/// [private] method loadJforQ
void baseMapGs::loadJforQ( UINT32 iQ, UINT32 nQ, UINT32 nJ )
{
    // build a list of QID and seed position for each J value in the current iteration
    baseSetupJs setupJs( m_ptum->Key, m_pqb, m_pdbg, m_isi, iQ, nQ );
    setupJs.Start();
    setupJs.Wait();

    // load the D values (J values adjusted for seed position) for the current iteration
    baseLoadJs loadJs( m_ptum->Key, m_pqb, m_pdbg, m_isi, nJ ); 
    loadJs.Start();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->n.CandidateD, nJ );

    loadJs.Wait();

    // use a thrust "stream compaction" API to remove null D values from the D list for the current iteration
    thrust::device_ptr<UINT64> tpD( m_pdbg->Diter.p );
    thrust::device_ptr<UINT64> eolD = thrust::remove( epCGA, tpD, tpD+nJ, _UI64_MAX );
    m_pdbg->Diter.n = static_cast<UINT32>(eolD.get() - tpD.get());


#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: looking for sqId 0x%016llx in D list...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    
    UINT32 qidxxx = _UI32_MAX;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            if( pQw->sqId[iq] == TRACE_SQID )
            {
                qidxxx = PACK_QID(iw,iq);
                break;
            }
        }
    }

    if( qidxxx == _UI32_MAX )
        CDPrint( cdpCD0, "[%d] %s::%s: sqId 0x%016llx not in current batch", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    else
    {
        WinGlobalPtr<UINT64> Djyyy( m_pdbg->Diter.n, false );
        m_pdbg->Diter.CopyToHost( Djyyy.p, Djyyy.Count );

        for( UINT32 n=0; n<m_pdbg->Diter.n; ++n )
        {
            UINT64 Dj = Djyyy.p[n];
            UINT32 qid = AriocDS::D::GetQid(Dj);
            if( (qid ^ qidxxx) <= 1 )
            {
                INT8 subId = static_cast<INT8>(Dj >> 32) & 0x007F;
                INT32 J = Dj & 0x7FFFFFFF;
                INT32 Jf = (Dj & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "%s::%s: %6u qid=0x%08X Dj=0x%016llx subId=%d J=%d Jf=%d",
                                    m_ptum->Key, __FUNCTION__,
                                    n, qid, Dj, subId, J, Jf );

            }
        }
        
        // show Dc values for the opposite mate
        qidxxx ^= 1;
        WinGlobalPtr<UINT64> Dcyyy( m_pdbg->Dc.n, false );
        m_pdbg->Dc.CopyToHost( Dcyyy.p, Dcyyy.Count );

        for( UINT32 n=0; n< m_pdbg->Dc.n; ++n )
        {
            UINT64 Dc = Dcyyy.p[n];
            UINT32 qid = AriocDS::D::GetQid(Dc);
            if( (qid ^ qidxxx) <= 1 )
            {
                INT8 subId = static_cast<INT8>(Dc >> 32) & 0x007F;
                INT32 J = Dc & 0x7FFFFFFF;
                INT32 Jf = (Dc & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

#if TRACE_SUBID
        if( subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "%s::%s: %6u qid=0x%08X Dc=0x%016llx subId=%d J=%d Jf=%d",
                                    m_ptum->Key, __FUNCTION__,
                                    n, qid, Dc, subId, J, Jf );
            }
        }
    }
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: loaded %u/%u Dj values", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_pdbg->Diter.n, nJ );
#endif
}

/// [private] method combineRuForSeedInterval
void baseMapGs::combineRuForSeedInterval()
{
    if( m_pdbg->Dc.n )
    {
        // for each mapped (but not concordant) D value in the Dc list, set the corresponding subId bit in the Ru buffer
        baseMergeRu bMR( m_ptum->Key, m_pqb, m_pdbg, riDc );
        bMR.Start();
        bMR.Wait();

#if TODO_CHOP_WHEN_DEBUGGED
        WinGlobalPtr<UINT64> Dcxxx( m_pdbg->Dc.n, false );
        m_pdbg->Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );
        WinGlobalPtr<UINT32> Quxxx( m_pdbg->Qu.n, false );
        m_pdbg->Qu.CopyToHost( Quxxx.p, Quxxx.Count );
        WinGlobalPtr<UINT32> Ruxxx( m_pdbg->Ru.n, false );
        m_pdbg->Ru.CopyToHost( Ruxxx.p, Ruxxx.Count );
        for( UINT32 n=0; n<m_pdbg->Dc.n; ++n )
        {
            UINT64 D = Dcxxx.p[n];
            UINT32 qid = AriocDS::D::GetQid(D);

            bool notFound = true;
            for( UINT32 x=0; x<m_pdbg->Qu.n; ++x )
            {
                if( Quxxx.p[x] == qid )
                {
                    if( n < 256 )
                        CDPrint( cdpCD0, "%s: %u 0x%016llx 0x%08x 0x%08x", __FUNCTION__, n, D, qid, Ruxxx.p[x] );
                    notFound = false;
                    break;
                }
            }
            if( notFound )
                CDPrint( cdpCD0, "%s: %u 0x%016llx 0x%08x NOT FOUND!", __FUNCTION__, n, D, qid );
        }
        CDPrint( cdpCD0, __FUNCTION__ );
#endif

    }



    // allocate the Rx buffer
    m_pdbg->Rx.Alloc( cgaLow, m_pdbg->Ru.n, false );

    /* AND the subId bits for each pair of opposite QIDs.  The Thrust transformation function operates on pairs
        of 32-bit values, so we use type punning to keep the implementation simple.  No worries here about endianness
        (we assume we're running on an Intel machine if we're using NVidia hardware) or 64-bit alignment (the
        CudaGlobalAllocator's granularity is 128 bytes). */
    thrust::device_ptr<UINT64> tpRu( reinterpret_cast<UINT64*>(m_pdbg->Ru.p) );
    thrust::device_ptr<UINT64> tpRx( reinterpret_cast<UINT64*>(m_pdbg->Rx.p) );
    thrust::device_ptr<UINT64> tpEol = thrust::transform( epCGA, tpRu, tpRu+m_pdbg->Ru.n/2, tpRx, TSX::andHiLoUInt64() );
    m_pdbg->Rx.n = 2 * static_cast<UINT32>(tpEol.get() - tpRx.get());
    if( m_pdbg->Rx.n != m_pdbg->Ru.n )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: actual=%u expected=%u", m_pdbg->Rx.n, m_pdbg->Ru.n );
}

/// [private] method loadJforSeedInterval
void baseMapGs::loadJforSeedInterval()
{
    // accumulate the J-list counts
    accumulateJcounts();

    // performance metrics
    InterlockedIncrement( &m_ptum->n.Instances );

    /* Iterate across the list of QIDs for Q sequences whose J lists need to be loaded. */
    UINT32 nIterations = 0;
    UINT32 iQ = 0;          // index of first QID (Q sequence) for the current iteration
    UINT32 nQ = 0;          // number of QIDs for the current iteration
    UINT32 nJ = 0;          // number of J values to load for the current iteration
    INT64 nJremaining = m_pqb->DBj.totalD;
    UINT32 nQremaining = m_pdbg->Qu.n;
    while( nQremaining && nJremaining )
    {
        // compute the number of QIDs (Q sequences) and J values to process in the current iteration
        prepareIteration( nQremaining, nJremaining, iQ, nQ, nJ );

#if TODO_CHOP_WHEN_DEBUGGED
        if( nQ == 0 )
            DebugBreak();   // shouldn't happen, right?
#endif
        
        
        // set up buffers in CUDA global memory
        initGlobalMemory_LJSIIteration( nJ );

        // load D values for the Q sequences
#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: calling loadJforQ (m_isi=%u) (inner-loop iteration %u: nJ=%u)...",
                            m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                            m_isi, nIterations, nJ );
#endif
        loadJforQ( iQ, nQ, nJ );

        // consolidate the list of D values (DBj.D)
        resetGlobalMemory_LJSIIteration();

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: after resetting global memory for iteration", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
#endif

        // iterate
        nQremaining -= nQ;
        nJremaining -= nJ;
        iQ += nQ;
        nIterations++ ;
    }

    // at this point we no longer need the combined subId bits
    m_pdbg->Rx.Free();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->n.Iterations, nIterations );
}

/// [private] method mergeDu
void baseMapGs::mergeDu()
{
    /* build a consolidated Du list that includes previously-aligned D values (regardless of whether or not they mapped)
        and D values that were aligned in the current iteration */
    UINT32 cel = m_pdbg->Du.n + m_pdbg->Dx.n + 1;
    CudaGlobalPtr<UINT64> tempDu( m_pqb->pgi->pCGA );
    tempDu.Alloc( cgaLow, cel, false );

    // make a sorted copy of the values in the Dx list
    CudaGlobalPtr<UINT64> tempDx( m_pqb->pgi->pCGA );
    tempDx.Alloc( cgaLow, m_pdbg->Dx.n, false );
    m_pdbg->Dx.CopyInDevice( tempDx.p, m_pdbg->Dx.n );
    thrust::device_ptr<UINT64> ttpDx( tempDx.p );
    thrust::stable_sort( epCGA, ttpDx, ttpDx+m_pdbg->Dx.n, TSX::isLessD() );

    thrust::device_ptr<UINT64> tpDu( m_pdbg->Du.p );
    thrust::device_ptr<UINT64> ttpDu( tempDu.p );

    
#if TODO_CHOP_WHEN_DEBUGGED
        bool isSorted = thrust::is_sorted( epCGA, tpDu, tpDu+m_pdbg->Du.n, TSX::isLessD() );
        if( !isSorted ) DebugBreak();
        isSorted = thrust::is_sorted( epCGA, ttpDx, ttpDx+m_pdbg->Dx.n, TSX::isLessD() );
        if( !isSorted ) DebugBreak();
#endif

    // merge the current set of D values (in the sorted copy of Dx) with the previous list of D values (in Du)
    thrust::device_ptr<UINT64> tpEolDu = thrust::merge( epCGA, ttpDx, ttpDx+m_pdbg->Dx.n, tpDu, tpDu+m_pdbg->Du.n, ttpDu, TSX::isLessD() );
    tempDu.n = static_cast<UINT32>(tpEolDu.get() - ttpDu.get());

    // zap the temporary copy of the Dx list
    tempDx.Free();

    // reset the flags in the Du buffer
    thrust::for_each( epCGA, ttpDu, ttpDu+tempDu.n, TSX::initializeDflags(0) );

#if TODO_CHOP_IF_UNUSED
    // unduplicate
    UINT32 nBefore = tempDu.n;
    tpEolDu = thrust::unique( epCGA, ttpDu, ttpDu+tempDu.n );
    tempDu.n = static_cast<UINT32>(tpEolDu.get() - ttpDu.get());


    // is the above call to thrust::unique really needed?
    if( tempDu.n != nBefore ) DebugBreak();
#endif

    // append a null D value (all bits set)
    ttpDu[tempDu.n] = _UI64_MAX;

    // discard the old Du list
    m_pdbg->Du.Free();

    // move the Du list out of the way
    m_pdbg->Du.Alloc( cgaHigh, tempDu.n+1, false );     // (include the additional null element)
    tempDu.CopyInDevice( m_pdbg->Du.p, m_pdbg->Du.Count );
    m_pdbg->Du.n = tempDu.n;
    SET_CUDAGLOBALPTR_TAG(m_pdbg->Du,"Du");
    tempDu.Free();

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: m_isi=%d Du.n=%u", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_isi, m_pdbg->Du.n );
#endif
}

/// [private] method filterJforSeedInterval
void baseMapGs::filterJforSeedInterval( UINT32 cbSharedPerBlock )
{
    if( m_pdbg->Dc.n )
    {
        // append already-mapped opposite-mate D values
        UINT32 cel = m_pqb->DBj.D.n + m_pdbg->Dc.n;
        m_pqb->DBj.D.Resize( cel );
        m_pdbg->Dc.CopyInDevice( m_pqb->DBj.D.p+m_pqb->DBj.D.n, m_pdbg->Dc.n );
        m_pqb->DBj.D.n = cel;
    }

#if TODO_CHOP_WHEN_DEBUGGED
    thrust::device_ptr<UINT64> tpD( m_pqb->DBj.D.p );
    size_t nMappedBefore = thrust::count_if( epCGA, tpD, tpD+m_pqb->DBj.D.n, TSX::isMappedDvalue() );
#endif

    // convert the D values to Df values; zero the "candidate" flag (but keep the "mapped" flag intact)
    tuXlatToDf xlatToDf( m_pqb, m_pqb->DBj.D.p, m_pqb->DBj.D.n, m_pqb->DBj.D.p, AriocDS::D::flagCandidate, 0 );
    xlatToDf.Start();
    xlatToDf.Wait();

#if TODO_CHOP_WHEN_DEBUGGED
    size_t nMappedAfter = thrust::count_if( epCGA, tpD, tpD+m_pqb->DBj.D.n, TSX::isMappedDvalue() );
    if( nMappedBefore != nMappedAfter ) DebugBreak();
#endif

    // sort and unduplicate the Df values
    thrust::device_ptr<UINT64> tpD( m_pqb->DBj.D.p );
    thrust::stable_sort( epCGA, tpD, tpD+m_pqb->DBj.D.n, TSX::isLessDf() );
    thrust::device_ptr<UINT64> tpEol = thrust::unique( epCGA, tpD, tpD+m_pqb->DBj.D.n );
    m_pqb->DBj.D.n = static_cast<UINT32>(tpEol.get() - tpD.get());

    // set the "candidate" flag for Df values whose distance and orientation are consistent with potential concordant mappings
    m_pqb->DBj.totalD = m_pqb->DBj.D.n;
    baseJoinDf joinDf( m_ptum->Key, m_pqb );
    joinDf.Start();
    joinDf.Wait();


#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: looking for sqId 0x%016llx in Df list (DBj.D)...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    
    UINT32 qidxxx = _UI32_MAX;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            if( pQw->sqId[iq] == TRACE_SQID )
            {
                qidxxx = PACK_QID(iw,iq);
                break;
            }
        }
    }

    if( qidxxx == _UI32_MAX )
        CDPrint( cdpCD0, "[%d] %s::%s: sqId 0x%016llx not in current batch", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    else
    {
        // dump the Df values for the QID
        WinGlobalPtr<UINT64> Dxxx( m_pqb->DBj.D.n, false );
        m_pqb->DBj.D.CopyToHost( Dxxx.p, Dxxx.Count );
        for( UINT32 n=0; n<m_pqb->DBj.D.n; ++n )
        {
            UINT64 D = Dxxx.p[n];
            UINT32 qid = AriocDS::Df::GetQid(D);
            if( (qid ^ qidxxx) > 1 )
                continue;

            Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
            UINT64 sqId = pQw->sqId[QID_IQ(qid)];
            INT16 subId = static_cast<INT16>((D >> 33) & 0x7F);
            INT32 Jf = static_cast<INT32>((D >> 2)& 0x7FFFFFFF);        // (Df values contain Jf)
            INT32 J = (D & 2) ? (m_pab->M.p[subId] - 1) - Jf : Jf;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "[%d] %s::%s: DBj.D: %4u 0x%016llx qid=0x%08x subId=%d J=%d Jf=%d sqId=0x%016llx",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        n, D, qid, subId, J, Jf, sqId );
        }
    }
#endif

    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs for D values
                Ru      subId bits for QIDs
                DBgs.Dc mapped candidates with unmapped opposite mates
                DBj.D   Df list

        high:   Du      previously-processed D values
    */

    // move the D list to the high memory region
    CudaGlobalPtr<UINT64> tempD( m_pqb->pgi->pCGA );
    tempD.Alloc( cgaHigh, m_pqb->DBj.D.n, false );
    m_pqb->DBj.D.CopyInDevice( tempD.p, tempD.Count );
    tempD.n = m_pqb->DBj.D.n;
    m_pqb->DBj.D.Free();

    // copy flagged candidates to the Dx list
    m_pdbg->Dx.Alloc( cgaLow, tempD.n, false );
    thrust::device_ptr<UINT64> ttpD( tempD.p );
    thrust::device_ptr<UINT64> tpDx( m_pdbg->Dx.p );
    thrust::device_ptr<UINT64> tpEolD = thrust::copy_if( epCGA, ttpD, ttpD+tempD.n, tpDx, TSX::isUnmappedCandidateDvalue() );
    m_pdbg->Dx.n = static_cast<UINT32>(tpEolD.get() - tpDx.get());
    m_pdbg->Dx.Resize( m_pdbg->Dx.n );

    // discard the unfiltered D list
    tempD.Free();

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: DBj.totalD=%llu; Dx.n=%u",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        m_pqb->DBj.totalD, m_pdbg->Dx.n );
#endif

    // convert the Df values back to D values; leave the flags alone
    tuXlatToD xlatToD( m_pqb, m_pdbg->Dx.p, m_pdbg->Dx.n, m_pdbg->Dx.p, 0, 0 );
    xlatToD.Start();
    xlatToD.Wait();


    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs for D values
                Ru      subId bits for QIDs
                DBgs.Dc mapped candidates with unmapped opposite mates
                Dx      D values (candidates for gapped alignment)

        high:   Du      previously-processed D values
    */

    // sort the Dx list
    thrust::stable_sort( epCGA, tpDx, tpDx+m_pdbg->Dx.n, TSX::isLessD() );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: iLoop=%d Du.n=%u Dx.n=%u before filtering previously-processed D values",
                            m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                            m_isi, m_pdbg->Du.n, m_pdbg->Dx.n );
#endif
    
    if( m_pdbg->Du.n )
    {
        // avoid redoing unsuccessful alignments by removing values in the D list that are close enough to previously-evaluated D values
        exciseRedundantDvalues( m_pdbg );
    }
}

/// [private] method mapJforSeedInterval
void baseMapGs::mapJforSeedInterval()
{
    // load interleaved R sequence data for the Dx list
    RaiiPtr<tuBaseS> kLoadRi1 = baseLoadRix::GetInstance( m_ptum->Key, m_pqb, m_pdbg, riDx );
    kLoadRi1->Start();
    kLoadRi1->Wait();

    // find maximum alignment scores for seed-and-extend gapped alignment
    baseMaxV kMaxV( m_ptum->Key, m_pqb, m_pdbg, riDx );
    kMaxV.Start();
    kMaxV.Wait();


#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: looking for sqId 0x%016llx in Dx list...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    
    UINT32 qidxxx = _UI32_MAX;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            if( pQw->sqId[iq] == TRACE_SQID )
            {
                qidxxx = PACK_QID(iw,iq);
                break;
            }
        }
    }

    if( qidxxx == _UI32_MAX )
        CDPrint( cdpCD0, "[%d] %s::%s: sqId 0x%016llx not in current batch", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    else
    {
        // dump the D values for the QID
        WinGlobalPtr<UINT64> Dxxx( m_pdbg->Dx.n, false );
        m_pdbg->Dx.CopyToHost( Dxxx.p, Dxxx.Count );
        WinGlobalPtr<INT16> VmaxDxxx( m_pdbg->VmaxDx.n, false );
        m_pdbg->VmaxDx.CopyToHost( VmaxDxxx.p, VmaxDxxx.Count );
        for( UINT32 n=0; n<m_pdbg->Dx.n; ++n )
        {
            UINT64 D = Dxxx.p[n];
            INT16 Vmax = VmaxDxxx.p[n];
            UINT32 qid = AriocDS::D::GetQid(D);
            if( (qid ^ qidxxx) > 1 )
                continue;

            Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
            UINT64 sqId = pQw->sqId[QID_IQ(qid)];
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & 0x80000000) ? (m_pab->M.p[subId] - 1) - J : J;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "[%d] %s::%s: DBgs.Dx: %4u 0x%016llx qid=0x%08x subId=%d J=%d Jf=%d Vmax=%d sqId=0x%016llx",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        n, D, qid, subId, J, Jf, Vmax, sqId );
        }
    }
#endif
    
    // move the Vmax list (just created by baseMaxV) to the low memory region
    m_pdbg->VmaxDx.ShuffleHiLo();

    // build a consolidated Du list that merges previously-processed D values with the D values that were just processed in the current iteration
    mergeDu();

    // remove unmapped D values from the Dx list
    thrust::device_ptr<UINT64> tpDx( m_pdbg->Dx.p );
    thrust::device_ptr<UINT64> tpEolDx = thrust::remove_if( epCGA, tpDx, tpDx+m_pdbg->Dx.n, TSX::isUnmappedDvalue() );
    m_pdbg->Dx.n = static_cast<UINT32>(tpEolDx.get() - tpDx.get());

    // do the same for the corresponding Vmax list
    thrust::device_ptr<INT16> tpVmaxDx( m_pdbg->VmaxDx.p );
    thrust::device_ptr<INT16> tpEolVmaxDx = thrust::remove_if( epCGA, tpVmaxDx, tpVmaxDx+m_pdbg->VmaxDx.n, TSX::isZero<INT16>() );
    m_pdbg->VmaxDx.n = static_cast<UINT32>(tpEolVmaxDx.get() - tpVmaxDx.get());
    if( m_pdbg->Dx.n != m_pdbg->VmaxDx.n )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: Dx.n=%u VmaxDx=%u", m_pdbg->Dx.n, m_pdbg->VmaxDx.n );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: Dx.n=%u", __FUNCTION__, m_pdbg->Dx.n );
#endif

    // discard interleaved R sequence data
    m_pqb->DB.Ri.Free();

    if( m_pdbg->Dx.n )
    {
        // reload interleaved R sequence data for the Dx list
        RaiiPtr<tuBaseS> kLoadRi2 = baseLoadRix::GetInstance( m_ptum->Key, m_pqb, m_pdbg, riDx );
        kLoadRi2->Start();
        kLoadRi2->Wait();

        // do gapped alignment and traceback
        baseAlignG alignG( m_ptum->Key, m_pqb, m_pdbg, m_phb, riDx );
        alignG.Start();
        alignG.Wait();

        // count per-Q mappings (update the Qwarp buffer in GPU global memory)
        baseCountA bCA( m_ptum->Key, m_pqb, m_pdbg, riDx );
        bCA.Start();
        bCA.Wait();
    }

#if TODO_CHOP_WHEN_DEBUGGED
    else
        CDPrint( cdpCD0, "%s: Dx.n == 0", __FUNCTION__ );
#endif

    // discard the Vmax buffer
    m_pdbg->VmaxDx.Free();

    /* Count concordant mappings.
    
       We need a consolidated Dm list that contains
        - mapped D values from the Dx list
        - mapped D values from the Dc list (if the list is non-empty)
       We do this simply by appending the newly-mapped D values to the Dc list.
    */
    
    // allocate space for the Dm buffer
    m_pdbg->Dm.Alloc( cgaLow, m_pdbg->Dx.n+m_pdbg->Dc.n, false );

    // copy the mapped D values...
    m_pdbg->Dx.CopyInDevice( m_pdbg->Dm.p, m_pdbg->Dx.n );              // ...from the Dm buffer
    m_pdbg->Dc.CopyInDevice( m_pdbg->Dm.p+m_pdbg->Dx.n, m_pdbg->Dc.n ); // ...from the Dc buffer
    m_pdbg->Dm.n = m_pdbg->Dx.n + m_pdbg->Dc.n;

    
#if TODO_CHOP_WHEN_DEBUGGED
    // dump the D values for a specified QID
    UINT32 qidxxx = 0x000001;
    WinGlobalPtr<UINT64> Dmxxx( m_pdbg->Dm.n, false );
    m_pdbg->Dm.CopyToHost( Dmxxx.p, Dmxxx.Count );
    for( UINT32 n=0; n<m_pdbg->Dm.n; ++n )
    {
        UINT64 D = Dmxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid(D);
        if( (qid ^ qidxxx) > 1 )
            continue;
        
        INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
        INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
        INT32 Jf = (D & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

        CDPrint( cdpCD0, "[%d] %s::%s: Dm 1: %4u 0x%016llx qid=0x%08x subId=%d J=%d Jf=%d",
                            m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                            n, D, qid, subId, J, Jf );
    }
#endif


    /* baseCountC wants a list of D values (not Df values), but the list must be sorted so that adjacent D values
        represent paired mates, i.e., the list needs to be sorted in Df order!
    */

    // translate to Df format; reset the flag bits
    tuXlatToDf xDf( m_pqb, m_pdbg->Dm.p, m_pdbg->Dm.n, m_pdbg->Dm.p, AriocDS::Df::maskFlags, 0 );
    xDf.Start();
    xDf.Wait();

    // sort in Df order for the benefit of baseCountC
    thrust::device_ptr<UINT64> tpDm( m_pdbg->Dm.p );
    thrust::stable_sort( epCGA, tpDm, tpDm+m_pdbg->Dm.n, TSX::isLessDf() );

    // unduplicate
    thrust::device_ptr<UINT64> tpEolDm = thrust::unique( epCGA, tpDm, tpDm+m_pdbg->Dm.n );
    m_pdbg->Dm.n = static_cast<UINT32>(tpEolDm.get() - tpDm.get());

    // translate back to D format
    tuXlatToD xD( m_pqb, m_pdbg->Dm.p, m_pdbg->Dm.n, m_pdbg->Dm.p, 0, 0 );
    xD.Start();
    xD.Wait();
  
    // count concordant mappings in the Dm buffer
    baseCountC bCC( m_ptum->Key, m_pqb, m_pdbg, riDm, false );
    bCC.Start();
    bCC.Wait();



#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: looking for sqId 0x%016llx in Dm list...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    
    UINT32 qidzzz = _UI32_MAX;
    WinGlobalPtr<Qwarp> Qwzzz( m_pqb->DB.Qw.n, false );
    m_pqb->DB.Qw.CopyToHost( Qwzzz.p, Qwzzz.Count );
    for( UINT32 iw=0; iw<m_pqb->DB.Qw.n; ++iw )
    {
        Qwarp* pQw = Qwzzz.p + iw;
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            if( pQw->sqId[iq] == TRACE_SQID )
            {
                qidzzz = PACK_QID(iw,iq);
                break;
            }
        }
    }

    if( qidzzz == _UI32_MAX )
        CDPrint( cdpCD0, "[%d] %s::%s: sqId 0x%016llx not in current batch", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    else
    {
        // dump the D values for the QID
        WinGlobalPtr<UINT64> Dzzz( m_pdbg->Dm.n, false );
        m_pdbg->Dm.CopyToHost( Dzzz.p, Dzzz.Count );
        for( UINT32 n=0; n<m_pdbg->Dm.n; ++n )
        {
            UINT64 D = Dzzz.p[n];
            UINT32 qid = AriocDS::D::GetQid(D);
            if( (qid ^ qidzzz) > 1 )
                continue;

            UINT32 iw = QID_IW(qid);
            UINT32 iq = QID_IQ(qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & 0x80000000) ? (m_pab->M.p[subId] - 1) - J : J;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "[%d] %s::%s: DBgs.Dm: %4u 0x%016llx qid=0x%08x subId=%d J=%d Jf=%d sqId=0x%016llx nAc=%d",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        n, D, qid, subId, J, Jf, sqId, pQw->nAc[iq] );
        }
    }
#endif






    // discard buffers
    m_pdbg->Dx.Free();
}

/// [private] method pruneQu
void baseMapGs::pruneQu()
{
#if TODO_CHOP_WHEN_DEBUGGED
    UINT32 nD0 = m_pdbg->Dc.n;
    UINT32 nQ0 = m_pdbg->Qu.n;
    UINT32 nR0 = m_pdbg->Ru.n;
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    // what is the state of the flags in the Dm buffer
    UINT32 nFlagged = 0;
    UINT32 nNotFlagged = 0;
    WinGlobalPtr<UINT64> Dmxxx( m_pdbg->Dm.n, false );
    m_pdbg->Dm.CopyToHost( Dmxxx.p, Dmxxx.Count );
    for( UINT32 n=0; n<m_pdbg->Dm.n; ++n )
    {
        if( Dmxxx.p[n] & AriocDS::D::flagCandidate )
            nFlagged++ ;
        else
            nNotFlagged++ ;
    }

    CDPrint( cdpCD0, "%s: nFlagged=%u nNotFlagged=%u", __FUNCTION__, nFlagged, nNotFlagged );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    m_pqb->DB.Qw.CopyToHost( m_pqb->QwBuffer.p, m_pqb->QwBuffer.n );
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; iq+=2 )
        {
            INT16 iqo = iq + 1;
            if( pQw->nAc[iq] && pQw->nAc[iqo] ) eh?++ ;
        }
    }

    CDPrint( cdpCD0, "%s: Dm.n=%u (total mapped QIDs), nDm1=%u (concordant pairs mapped by baseMapGs)", __FUNCTION__, m_pdbg->Dm.n, m_pdbg->nDm1 );
#endif



    // allocate a temporary buffer to contain a new consolidated Dc list
    UINT32 cel = m_pdbg->Dc.n + m_pdbg->Dm.n;
    CudaGlobalPtr<UINT64> tempDc( m_pqb->pgi->pCGA );
    tempDc.Alloc( cgaHigh, cel, false );
    tempDc.n = 0;
    thrust::device_ptr<UINT64> ttpDc( tempDc.p );
    thrust::device_ptr<UINT64> tpEolDc;

    if( m_pdbg->Dc.n )
    {
        // set the "candidate" flag for the Dc values
        thrust::device_ptr<UINT64> tpDc( m_pdbg->Dc.p );
        thrust::for_each( epCGA, tpDc, tpDc+m_pdbg->Dc.n, TSX::initializeDflags(AriocDS::D::flagCandidate) );

        // reset the "candidate" flag for D values in concordantly-mapped pairs
        baseFilterD bFDc( m_ptum->Key, m_pqb, m_pdbg, riDc, m_pab->aas.ACP.AtN );
        bFDc.Start();
        bFDc.Wait();

        // copy mapped D values without concordantly-mapped mates from the current Dc list to the temporary Dc list
        tpEolDc = thrust::copy_if( epCGA, tpDc, tpDc+m_pdbg->Dc.n, ttpDc, TSX::isCandidateDvalue() );
        tempDc.n = static_cast<UINT32>(tpEolDc.get() - ttpDc.get());

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: Dc list triaged: %u concordant, %u not concordant (copied to tempDc)",
                            m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                            m_pdbg->Dc.n, tempDc.n );
#endif
    }

    // set the "candidate" flag and clear the remaining flags in the Dm list
    thrust::device_ptr<UINT64> tpDm( m_pdbg->Dm.p );
    thrust::for_each( epCGA, tpDm, tpDm+m_pdbg->Dm.n, TSX::initializeDflags(AriocDS::D::flagCandidate) );

    // reset the "candidate" flag on D values that have concordant mappings
    baseFilterD bFDm( m_ptum->Key, m_pqb, m_pdbg, riDm, m_pab->aas.ACP.AtN );
    bFDm.Start();
    bFDm.Wait();

#if TODO_CHOP_WHEN_DEBUGGED
    UINT32 prevDcn = tempDc.n;
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    // dump the first few D values
    WinGlobalPtr<UINT64> Dmxxx( m_pdbg->Dm.n, false );
    m_pdbg->Dm.CopyToHost( Dmxxx.p, Dmxxx.Count );
    for( UINT32 n=0; n<100; ++n )
    {
        UINT32 qid = AriocDS::D::GetQid(Dmxxx.p[n]);
        CDPrint( cdpCD0, "%s: 0x%016llx 0x%08x", __FUNCTION__, Dmxxx.p[n], qid );
    }
#endif


    // append mapped D values without concordantly-mapped mates from the current Dm list to the temporary Dc list
    tpEolDc = thrust::copy_if( epCGA, tpDm, tpDm+m_pdbg->Dm.n, ttpDc+tempDc.n, TSX::isCandidateDvalue() );
    tempDc.n = static_cast<UINT32>(tpEolDc.get() - ttpDc.get());

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: Dm list triaged: %u concordant, %u not concordant (copied to tempDc)",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        m_pdbg->Dm.n, tempDc.n-prevDcn );
#endif

    // discard buffers
    m_pdbg->Dm.Free();
    m_pqb->DB.Ri.Free();

    // copy the Dc list
    m_pdbg->Dc.Resize( tempDc.n );
    tempDc.CopyInDevice( m_pdbg->Dc.p, tempDc.n );
    m_pdbg->Dc.n = tempDc.n;
    tempDc.Free();

    // null concordantly-mapped QIDs in the list of unmapped QIDs
    baseFilterQu filterQu( m_ptum->Key, m_pqb, m_pdbg, m_pab->aas.ACP.AtN );
    filterQu.Start();
    filterQu.Wait();

    // remove mapped QIDs and the corresponding subId bits
    thrust::device_ptr<UINT32> tpQu( m_pdbg->Qu.p );
    thrust::device_ptr<UINT32> tpRu( m_pdbg->Ru.p );

    // remove Ru where Qu is null (all bits set)
    thrust::device_ptr<UINT32> tpEol = thrust::remove_if( epCGA, tpRu, tpRu+m_pdbg->Ru.n, tpQu, TSX::isEqualTo<UINT32>(_UI32_MAX) );
    m_pdbg->Ru.n = static_cast<UINT32>(tpEol.get() - tpRu.get());

    // remove Qu where null (all bits set)
    tpEol = thrust::remove( epCGA, tpQu, tpQu+m_pdbg->Qu.n, _UI32_MAX );
    m_pdbg->Qu.n = static_cast<UINT32>(tpEol.get() - tpQu.get());
    if( m_pdbg->Qu.n != m_pdbg->Ru.n )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: Qu.n=%u Ru.n=%u", m_pdbg->Qu.n, m_pdbg->Ru.n );


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: Dc %u/%u, Qu %u/%u, Ru %u/%u",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        m_pdbg->Dc.n, nD0, m_pdbg->Qu.n, nQ0, m_pdbg->Ru.n, nR0 );
#endif


#if TODO_CHOP_WHEN_DEBUGGED
    // take a look at the Qu list
    WinGlobalPtr<UINT32> Quxxx( m_pqb->DBgs.Qu.n, false );
    m_pqb->DBgs.Qu.CopyToHost( Quxxx.p, Quxxx.Count );
    
    m_pqb->DB.Qw.CopyToHost( m_pqb->QwBuffer.p, m_pqb->QwBuffer.n );
    for( UINT32 n=0; n<100; ++n )
    {
        UINT32 qid = Quxxx.p[n];
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        CDPrint( cdpCD0, "[%d] %s: %3u:  0x%016llx 0x%08x %u %u %d+%d",
                            m_pqb->pgi->deviceId, __FUNCTION__,
                            n, pQw->sqId[iq], qid, iw, iq, pQw->nAc[iq], pQw->nAc[iq^1] );
    }
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s: DBgs.Qu.n=%u/%u", m_pqb->pgi->deviceId, __FUNCTION__, m_pqb->DBgs.Qu.n, m_pqb->DB.nQ );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    // look at the Qwarps to see where we are with concordant mappings
    UINT32 nQconcordant = 0;
    WinGlobalPtr<Qwarp> Qwxxx( m_pqb->DB.Qw.n, false );
    m_pqb->DB.Qw.CopyToHost( Qwxxx.p, Qwxxx.Count );
    for( UINT32 iw=0; iw<m_pqb->DB.Qw.n; ++iw )
    {
        Qwarp* pQw = Qwxxx.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; iq++ )
        {
            if( pQw->nAc[iq] ) nQconcordant++ ;
        }
    }
    CDPrint( cdpCD0, "%s: nQconcordant=%u/%u", __FUNCTION__, nQconcordant, m_pqb->DB.nQ );
    CDPrint( cdpCD0, __FUNCTION__ );

#endif

}
#pragma endregion
