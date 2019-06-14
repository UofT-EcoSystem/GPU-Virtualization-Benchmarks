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
#include <thrust/unique.h>


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
        CREXEC( m_pqb->DBj.cnJ.Alloc( cgaLow, m_pqb->DBj.celJ+1, true ) );
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

    /* sort the Df list */
    thrust::device_ptr<UINT64> tpDf( m_pqb->DBj.D.p );
    thrust::stable_sort( epCGA, tpDf, tpDf+m_pqb->DBj.D.n, TSX::isLessDf() );

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

        high:   Df      Df list
                cnJ     cumulative per-Q J-list sizes
                nJ      per-Q J-list sizes
                oJ      per-seed J-list offsets
                ...     ...
    */

    
#if TODO_CHOP_WHEN_DEBUGGED
    if( m_pqb->DBn.Qu.p == NULL )
        DebugBreak();                   // this should never happen
#endif


    // discard the list of QIDs that was used for building the Df list
    m_pqb->DBn.Qu.Free();
}
#pragma endregion

#pragma region setupN22
/// [private] method launchKernel22
void tuSetupN::launchKernel22()
{
    try
    {
        m_hrt.Restart();

        // use a thrust "stream compaction" API to remove duplicate Df values from the list
        thrust::device_ptr<UINT64> tpDf( m_pqb->DBj.D.p );
        thrust::device_ptr<UINT64> tpEnd = thrust::unique( epCGA, tpDf, tpDf+m_pqb->DBj.totalD );
        m_pqb->DBj.totalD = static_cast<UINT32>(tpEnd.get() - tpDf.get());

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

#pragma region setupN31
/// [private] method launchKernel31
void tuSetupN::launchKernel31()
{
    // count the number of candidates
    thrust::device_ptr<UINT64> tpDf( m_pqb->DBj.D.p );
    m_nCandidates = static_cast<UINT32>( thrust::count_if( epCGA, tpDf, tpDf+m_pqb->DBj.totalD, TSX::isCandidateDvalue() ) );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: back from thrust count_if! (nCandidates=%u)", __FUNCTION__, m_nCandidates );
#endif
}
#pragma endregion

#pragma region setupN32
/// [private] method initGlobalMemory32
void tuSetupN::initGlobalMemory32()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Dc      Dc list (candidates for paired alignment)
                    Dx      Dx list (non-candidates for paired alignment) (nongapped alignment only)

            high:   Df      Df list
                    cnJ     cumulative per-Q J-list sizes
                    nJ      per-Q J-list sizes
                    oJ      per-seed J-list offsets
                    ...     ...
        */

        /* Allocate the Dc-list buffers:
            - Dx: there is one element for each Df value that is not flagged as an alignment candidate
            - Dc: there is one element for each Df value that is flagged as an alignment candidate
        */
        m_pqb->DBn.Dc.n = m_nCandidates;
        CREXEC( m_pqb->DBn.Dc.Alloc( cgaLow, m_pqb->DBn.Dc.n, true ) );
        SET_CUDAGLOBALPTR_TAG( m_pqb->DBn.Dc, "DBn.Dc" );

        m_pqb->DBn.Dx.n = static_cast<UINT32>(m_pqb->DBj.totalD - m_nCandidates);
        CREXEC( m_pqb->DBn.Dx.Alloc( cgaLow, m_pqb->DBn.Dx.n, true ) );
        SET_CUDAGLOBALPTR_TAG( m_pqb->DBn.Dx, "DBn.Dx" );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel32
void tuSetupN::launchKernel32()
{
#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "tuSetupN[%d]::launchKernel32 starts...", m_pqb->pgi->deviceId );
#endif

    m_hrt.Restart();

    /* Use a thrust "stream compaction" API to triage the Df values according to whether or not
        they meet the configured paired-end criteria. */

    // copy flagged candidate Df values to the Dc buffer
    thrust::device_ptr<UINT64> tpDf( m_pqb->DBj.D.p );
    thrust::device_ptr<UINT64> tpDc( m_pqb->DBn.Dc.p );
    thrust::device_ptr<UINT64> peol = thrust::copy_if( epCGA, tpDf, tpDf+m_pqb->DBj.totalD, tpDc, TSX::isCandidateDvalue() );
    if( m_nCandidates != static_cast<UINT32>(peol.get() - tpDc.get()) )
        throw new ApplicationException( __FILE__, __LINE__, "[%d] %s::%s: inconsistent list count: actual=%u expected=%u", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, static_cast<UINT32>(peol.get() - tpDf.get()), m_nCandidates );

    // copy unflagged Df values to the Dx buffer
    thrust::device_ptr<UINT64> tpDx( m_pqb->DBn.Dx.p );
    peol = thrust::copy_if( epCGA, tpDf, tpDf+m_pqb->DBj.totalD, tpDx, TSX::isNotCandidateDvalue() );
    UINT32 nUnpairedCandidates = static_cast<UINT32>(peol.get() - tpDx.get());
    if( nUnpairedCandidates != m_pqb->DBn.Dx.n )
        throw new ApplicationException( __FILE__, __LINE__, "[%d] %s::%s: inconsistent list count: actual=%u expected=%u", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_pqb->DBn.Dx.n, nUnpairedCandidates );

    // performance metrics
    AriocTaskUnitMetrics* ptum = AriocBase::GetTaskUnitMetrics( "tuSetupN32" );
    InterlockedExchangeAdd( &ptum->ms.Elapsed, m_hrt.GetElapsed(false) );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: m_nCandidates=%u nUnpairedCandidates=%u", __FUNCTION__, m_nCandidates, nUnpairedCandidates );
#endif

 
#if TRACE_SQID
    // show Df values for a specified sqId
    WinGlobalPtr<UINT64> Dcxxx( m_nCandidates, true );
    m_pqb->DBn.Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );


    UINT32 qidFrom = AriocDS::Df::GetQid( Dcxxx.p[0] );
    Qwarp* pQwFrom = m_pqb->QwBuffer.p + QID_IW(qidFrom);
    UINT32 qidTo = AriocDS::Df::GetQid( Dcxxx.p[Dcxxx.Count-1] );
    Qwarp* pQwTo = m_pqb->QwBuffer.p + QID_IW(qidTo);
    CDPrint( cdpCD0, "%s: sqId 0x%016llx - 0x%016llx", __FUNCTION__, pQwFrom->sqId[QID_IQ(qidFrom)], pQwTo->sqId[QID_IQ(qidTo)] );

    for( UINT32 n=0; n<static_cast<UINT32>(Dcxxx.Count); ++n )
    {
        UINT64 Df = Dcxxx.p[n];
        UINT32 qid = AriocDS::Df::GetQid( Df );
        Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
        if( (pQw->sqId[QID_IQ(qid)] | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            UINT8 subId = (Df >> 33) & 0x7F;
            UINT32 pos = (Df >> 2) & 0x7FFFFFFF;
            char s = (Df & 2) ? 'R' : 'F';

            CDPrint( cdpCD0, "%s: qid=0x%08X Df=0x%016llx subId=%d pos=0x%08x (%d) %c", __FUNCTION__, qid, Df, subId, pos, pos, s );
        }
    }
    CDPrint( cdpCD0, __FUNCTION__ );
#endif

}

/// [private] method resetGlobalMemory32
void tuSetupN::resetGlobalMemory32()
{
    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Dc      Dc list (candidates for paired alignment)
                Dx      Dx list (non-candidates for paired alignment) (nongapped alignment only)

        high:   ...     ...
    */

    // discard all of the lists in the high region of the global allocation
    m_pqb->DBj.D.Free();
    m_pqb->DBj.cnJ.Free();
    m_pqb->DBj.nJ.Free();
    m_pqb->DBj.oJ.Free();
}
#pragma endregion

#pragma region setupN41
/// [private] method initGlobalMemory41
void tuSetupN::initGlobalMemory41()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Dc      Dc list (candidates for paired alignment)
                    Dx      Dx list (non-candidates for paired alignment) (nongapped alignment only)

            high:   m_Dx    translated Dx list
        */

        // allocate a new buffer for the Dx list
        CREXEC( m_Dx.Alloc( cgaHigh, m_pqb->DBn.Dx.n, false ) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel41
void tuSetupN::launchKernel41()
{

#if TRACE_SQID
    // copy Df values to a host buffer
    WinGlobalPtr<UINT64> Dffff( m_pqb->DBn.Dx.n, false );
    m_Dx.CopyToHost( Dffff.p, Dffff.Count );
#endif


    // translate Df-formatted values to D-formatted values in the Dx buffer; reset the D-value flags
    tuXlatToD xlatDx( m_pqb, m_Dx.p, m_pqb->DBn.Dx.n, m_pqb->DBn.Dx.p, AriocDS::D::maskFlags, 0 );
    xlatDx.Start();
    xlatDx.Wait();

    m_Dx.n = m_pqb->DBn.Dx.n;



#if TRACE_SQID
    // copy the D values to a host buffer
    WinGlobalPtr<UINT64> Dxxxx( m_pqb->DBn.Dx.n, false );
    m_Dx.CopyToHost( Dxxxx.p, Dxxxx.Count );

    for( UINT32 n=0; n<static_cast<UINT32>(Dxxxx.Count); ++n )
    {
        UINT64 Dx = Dxxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( Dx );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
            
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>(Dx >> 32) & 0x007F;
            INT32 J = static_cast<INT32>(Dx & 0x7FFFFFFF);
            INT32 Jf = static_cast<INT32>(Dffff.p[n] & 0x7FFFFFFF);

#if TRACE_SUBID
        if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "%s: %3d: Dx=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d Jf=%d", __FUNCTION__, n, Dx, sqId, qid, subId, J, Jf );
        }
    }

#endif

}

/// [private] method resetGlobalMemory41
void tuSetupN::resetGlobalMemory41()
{
    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Dc      Dc list (paired candidates for nongapped alignment)

        high:   Dx      Dx list (unpaired candidates for nongapped alignment)
    */

    // update the Dx-list reference and release the old Dx list
    m_Dx.Swap( &m_pqb->DBn.Dx );
    m_Dx.Free();
    SET_CUDAGLOBALPTR_TAG( m_pqb->DBn.Dx, "DBn.Dx" );
}
#pragma endregion
