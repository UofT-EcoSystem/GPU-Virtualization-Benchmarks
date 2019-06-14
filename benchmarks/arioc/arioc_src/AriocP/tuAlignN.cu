/*
  tuAlignN.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
   We try to localize the Thrust-dependent code in one compilation unit so as to minimize the overall compile time.
*/
#include "stdafx.h"
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>


#pragma region alignN21
/// [private] method launchKernel21
void tuAlignN::launchKernel21()
{
    // count the number of mapped candidates
    thrust::device_ptr<UINT64> tpDc( m_pqb->DBn.Dc.p );
    m_pqb->DBn.nDm1 = static_cast<UINT32>( thrust::count_if( epCGA, tpDc, tpDc+m_pqb->DBn.Dc.n, TSX::isMappedDvalue() ) );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: back from thrust count_if: m_pqb->DBn.Dc.n=%u m_pqb->DBn.nDm1=%u", __FUNCTION__, m_pqb->DBn.Dc.n, m_pqb->DBn.nDm1 );
#endif
}
#pragma endregion

#pragma region alignN22
/// [private] method initGlobalMemory22
void tuAlignN::initGlobalMemory22()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Dc      Dc list (candidates for nongapped alignment)

            high:   Dm      Dm list (candidates with nongapped mappings)
        */

        /* Allocate the Dm-list buffer:
            - there is one element for each Dc value that is flagged as having been mapped by the nongapped aligner
        */
        CREXEC( m_pqb->DBn.Dm.Alloc( cgaHigh, m_pqb->DBn.nDm1, false ) );
        SET_CUDAGLOBALPTR_TAG( m_pqb->DBn.Dm, "DBn.Dm" );
        m_pqb->DBn.Dm.n = m_pqb->DBn.nDm1;

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "tuAlignN[%d]::initGlobalMemory22: %lld bytes unused", m_pqb->pgi->deviceId, m_pqb->pgi->pCGA->GetAvailableByteCount() );
#endif
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel22
void tuAlignN::launchKernel22()
{
    // use a thrust "stream compaction" API to build a list (Dm) that contains only the mapped Dc values
    thrust::device_ptr<UINT64> tpDc( m_pqb->DBn.Dc.p );
    thrust::device_ptr<UINT64> tpDm( m_pqb->DBn.Dm.p );
    thrust::device_ptr<UINT64> eol = thrust::copy_if( epCGA, tpDc, tpDc+m_pqb->DBn.Dc.n, tpDm, TSX::isMappedDvalue() );

    m_pqb->DBn.Dm.n = static_cast<UINT32>(eol.get() - tpDm.get());
    if( m_pqb->DBn.Dm.n != m_pqb->DBn.nDm1 )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: actual=%u expected=%u", m_pqb->DBn.Dm.n, m_pqb->DBn.nDm1 );

#if TRACE_SQID
        // look for a specific sqId
        WinGlobalPtr<UINT64> Dmxxx( m_pqb->DBn.nDm1, false );
        m_pqb->DBn.Dm.CopyToHost( Dmxxx.p, Dmxxx.Count );
        
        //for( UINT32 n=0; n<100; ++n )
        for( UINT32 n=0; n<static_cast<UINT32>(Dmxxx.Count); ++n )
        {
            UINT64 Dm = Dmxxx.p[n];
            UINT32 qid = static_cast<UINT32>(Dm >> 39) & AriocDS::QID::maskQID;
            UINT32 iw = QID_IW(qid);
            UINT32 iq = QID_IQ(qid);
            UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                INT8 subId = (Dm >> 32) & 0x7F;
                INT8 strand = (Dm >> 31) & 1;
                INT32 Jf = Dm & 0x7FFFFFFF;
                if( strand )
                {
                    UINT32 M = m_pab->M.p[subId];
                    Jf = (M - 1) - Jf;
                }
                CDPrint( cdpCD0, "%s: qid=0x%08x sqId=0x%016llx Dm=0x%016llx subId=%d strand=%d Jf=%d", __FUNCTION__, qid, sqId, Dm, subId, strand, Jf );
            }
        }
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dmxxx( m_pqb->DBn.Dm.n, false );
    m_pqb->DBn.Dm.CopyToHost( Dmxxx.p, Dmxxx.Count );

    for( UINT32 n=0; n<100; ++n )
    {
        UINT64 Dm = Dmxxx.p[n];
        INT16 subId = static_cast<INT16>((Dm & AriocDS::D::maskSubId) >> 32);
        INT32 pos = Dm & 0x7FFFFFFF;
        if( Dm & AriocDS::D::maskStrand )
            pos = (m_pab->M.p[subId] - 1) - pos;

        CDPrint( cdpCD0, "tuAlignN::launchKernel22: %4u 0x%016llx qid=0x%08llx subId=%d J=%d",
                            n, Dm, (Dm>>39)&0x007FFFFF, subId, pos );
    }
#endif

        
#if TRACE_SQID
    WinGlobalPtr<UINT64> Dmyyy( m_pqb->DBn.Dm.n, false );
    m_pqb->DBn.Dm.CopyToHost( Dmyyy.p, Dmyyy.Count );

    for( UINT32 n=0; n<static_cast<UINT32>(Dmyyy.Count); ++n )
    {
        UINT64 Dm = Dmyyy.p[n];
        UINT32 qid = AriocDS::D::GetQid( Dm );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
            
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>(Dm >> 32) & 0x007F;
            INT32 J = static_cast<INT32>(Dm & 0x7FFFFFFF);
            CDPrint( cdpCD0, "tuAlignN::launchKernel22: %3d: Dm=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d", n, Dm, sqId, qid, subId, J );
        }
    }

    CDPrint( cdpCD0, "tuAlignN::launchKernel22" );
#endif
}

/// [private] method copyKernelResults22
void tuAlignN::copyKernelResults22()
{
    CRVALIDATOR;

    m_hrt.Restart();

    // (re)allocate a host buffer to contain the list of mapped Q sequences
    m_pqb->HBn.Dm.Reuse( m_pqb->DBn.nDm1, false );

    // copy the list of mapped Q sequences to a host buffer
    CREXEC( m_pqb->DBn.Dm.CopyToHost( m_pqb->HBn.Dm.p, m_pqb->DBn.nDm1 ) );
    m_pqb->HBn.Dm.n = m_pqb->DBn.nDm1;

    // performance metrics
    InterlockedExchangeAdd( &AriocBase::aam.ms.XferMappings, m_hrt.GetElapsed(false) );
}

/// [private] method resetGlobalMemory22
void tuAlignN::resetGlobalMemory22()
{
    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   DBn.Dm
                DBn.Dx
    */

    // discard the Dc (candidates for nongapped alignment) list
    m_pqb->DBn.Dc.Free();
}
#pragma endregion

#pragma region alignN31
/// [private] method launchKernel31
void tuAlignN::launchKernel31()
{
    /* flag every mapped paired D value as a candidate for windowed gapped alignment; the flag will be
        reset later for concordant mappings */
    thrust::device_ptr<UINT64> tpDm( m_pqb->DBn.Dm.p );
    thrust::for_each( tpDm, tpDm+m_pqb->DBn.Dm.n, TSX::initializeDflags(AriocDS::D::flagCandidate) );

    // flag every mapped unpaired D value as a candidate for windowed gapped alignment
    thrust::device_ptr<UINT64> tpDx( m_pqb->DBn.Dx.p );
    thrust::for_each( tpDx, tpDx+m_pqb->DBn.Dx.n, TSX::initializeDflags(AriocDS::D::flagCandidate) );
}
#pragma endregion

#pragma region alignN47
/// [private] method launchKernel47
void tuAlignN::launchKernel47()
{
    if( m_pqb->DBn.Dm.n )
    {
        // count the number of candidates for windowed gapped alignment in the Dm list
        thrust::device_ptr<UINT64> tpDm( m_pqb->DBn.Dm.p );
        m_pqb->DBgw.nDc1 = static_cast<UINT32>(thrust::count_if( epCGA, tpDm, tpDm+m_pqb->DBn.Dm.n, TSX::isCandidateDvalue() ));
    }
    else
        m_pqb->DBgw.nDc1 = 0;

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "tuAlignN::launchKernel47: back from thrust count_if! (%u/%u candidates remain after filtering)", m_pqb->DBgw.nDc1, m_pqb->DBn.nDm1 );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dmxxx(m_pqb->DBn.Dm.Count, false);
    m_pqb->DBn.Dm.CopyToHost( Dmxxx.p, Dmxxx.Count );

    UINT32 nx = 0;
    for( UINT32 n=0; n<m_pqb->DBn.Dm.n; ++n )
    {
        if( (Dmxxx.p[n] & AriocDS::D:flagCandidate) != 0 )
            nx++ ;
    }

    CDPrint( cdpCD0, "nx = %u", nx );
#endif

}
#pragma endregion

#pragma region alignN48
/// [private] method initGlobalMemory48
void tuAlignN::initGlobalMemory48()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBgw.Dc Dc list (candidates for windowed gapped alignment)

            high:   Dm      Dm list (D values with nongapped mappings)
                    Dx      Dx list (non-candidates for paired nongapped alignment)
        */


        /* Allocate the Dc-list buffer:
            - there is one element in the buffer for each candidate for windowed gapped alignment
        */
        CREXEC( m_pqb->DBgw.Dc.Alloc( cgaLow, m_pqb->DBgw.nDc1, false ) );
        SET_CUDAGLOBALPTR_TAG( m_pqb->DBgw.Dc, "DBgw.Dc" );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel48
void tuAlignN::launchKernel48()
{
    if( m_pqb->DBgw.nDc1 )
    {
        thrust::device_ptr<UINT64> tpDm( m_pqb->DBn.Dm.p );
        thrust::device_ptr<UINT64> tpDc( m_pqb->DBgw.Dc.p );

        /* use a thrust "stream compaction" API to build a list (DBgw.Dc) of the D values for mapped mates whose
            opposite mate is unmapped */
        thrust::device_ptr<UINT64> eol = thrust::copy_if( epCGA, tpDm, tpDm+m_pqb->DBn.nDm1, tpDc, TSX::isCandidateDvalue() );
        m_pqb->DBgw.Dc.n = static_cast<UINT32>(eol.get() - tpDc.get());
        if( m_pqb->DBgw.Dc.n != m_pqb->DBgw.nDc1 )
            throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: actual=%u expected=%u", m_pqb->DBgw.Dc.n, m_pqb->DBgw.nDc1 );
    }
    else
        m_pqb->DBgw.Dc.n = 0;

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: back from thrust copy_if: m_pqb->DBgw.Dc.n=%u", __FUNCTION__, m_pqb->DBgw.Dc.n );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dmxxx( m_pqb->DBgw.nDc1, false );
    m_pqb->DBn.Dm.CopyToHost( Dmxxx.p, m_pqb->DBgw.nDc1 );

    for( UINT32 n=0; n<100; ++n )
    {
        UINT64 Dm = Dmxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( Dm );
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];
        CDPrint( cdpCD0, "tuAlignN::launchKernel48: qid=0x%08x sqId=0x%016llx Dm=0x%016llx", qid, sqId, Dm );
    }
    CDPrint( cdpCD0, "tuAlignN::launchKernel48" );
#endif


#if TRACE_SQID
    WinGlobalPtr<UINT64> Dcxxx( m_pqb->DBgw.nDc1, false );
    m_pqb->DBgw.Dc.CopyToHost( Dcxxx.p, m_pqb->DBgw.nDc1 );

    for( UINT32 n=0; n<m_pqb->DBgw.nDc1; ++n )
    {
        UINT64 Dc = Dcxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( Dc );
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            CDPrint( cdpCD0, "%s: %u mappings with no opposite mappings...", __FUNCTION__, m_pqb->DBgw.nDc1 );
            CDPrint( cdpCD0, "%s: qid=0x%08x sqId=0x%016llx Dc=0x%016llx", __FUNCTION__, qid, sqId, Dc );
        }
    }
#endif




}

/// [private] method copyKernelResults48
void tuAlignN::copyKernelResults48()
{

#if TRACE_SQID
        WinGlobalPtr<UINT64> Dcxxx(m_pqb->DBgw.Dc.n, false);
        m_pqb->DBgw.Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );

        //for( INT64 n=0; n<100; ++n )
        for( UINT32 n=0; n<static_cast<UINT32>(Dcxxx.Count); ++n )
        {
            UINT64 Dc = Dcxxx.p[n];
            UINT32 qid = AriocDS::D::GetQid( Dc );
            UINT32 iw = QID_IW(qid);
            INT16 iq = QID_IQ(qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];
            
            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                INT16 subId = static_cast<INT16>(Dc >> 32) & 0x007F;
                INT32 J = static_cast<INT32>(Dc & 0x7FFFFFFF);
                CDPrint( cdpCD0, "%s: DBgw.Dc (candidates for windowed gapped alignment):", __FUNCTION__ );
                CDPrint( cdpCD0, "%s: %3d: Dc=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d", __FUNCTION__, n, Dc, sqId, qid, subId, J );
            }
        }
#endif


}

/// [private] method resetGlobalMemory48
void tuAlignN::resetGlobalMemory48()
{
    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBgw.Dc Dc list (paired candidates for windowed gapped alignment)

        high:   Dx      Dx list (unpaired candidates for windowed gapped alignment)
    */

    // discard the list of mapped candidates
    m_pqb->DBn.Dm.Free();
}
#pragma endregion

#pragma region alignN49
/// [private] method launchKernel49
void tuAlignN::launchKernel49()
{
    if( m_pqb->DBn.Dx.n )
    {
        /* Use a thrust "stream reduction" API to remove all D values in the Dx list that are not flagged as candidates for
            alignment; the Dx list now contains unpaired candidates (i.e., excluded by user-specified criteria for TLEN and
            orientation) that do not belong to reads that already have at least the minimum number of nongapped concordant
            alignments. */
        thrust::device_ptr<UINT64> tpDx( m_pqb->DBn.Dx.p );
        thrust::device_ptr<UINT64> peol = thrust::remove_if( epCGA, tpDx, tpDx+m_pqb->DBn.Dx.n, TSX::isNotCandidateDvalue() );

        m_pqb->DBn.Dx.n = static_cast<UINT32>(peol.get() - tpDx.get());
    }

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: DBn.Dx.n=%u", __FUNCTION__, m_pqb->DBn.Dx.n );
#endif
}
#pragma endregion

#pragma region alignN66
/// [private] method launchKernel66
void tuAlignN::launchKernel66()
{
    if( m_pqb->DBn.Dx.n )
    {
        // count the number of Dx-list values that were mapped by the nongapped aligner
        thrust::device_ptr<UINT64> tpDx( m_pqb->DBn.Dx.p );
        m_pqb->DBn.nDm2 = static_cast<UINT32>(thrust::count_if( epCGA, tpDx, tpDx+m_pqb->DBn.Dx.n, TSX::isMappedDvalue() ));

        /* grow the list of candidates for windowed gapped alignment; the Dc buffer is the most recently allocated
            buffer in the global memory allocation, so this is a trivial operation */
        UINT32 nCandidatesGwTotal = m_pqb->DBgw.nDc1 + m_pqb->DBn.nDm2;
        m_pqb->DBgw.Dc.Resize( nCandidatesGwTotal );

        // use a thrust "stream reduction" API to extract the mapped D values from the Dx list and append them to the Dc list
        thrust::device_ptr<UINT64> tpDc( m_pqb->DBgw.Dc.p );
        thrust::device_ptr<UINT64> peol = thrust::copy_if( epCGA, tpDx, tpDx+m_pqb->DBn.Dx.n,
                                                           tpDc+m_pqb->DBgw.Dc.n,
                                                           TSX::isMappedDvalue() );

        // verify the new buffer size
        m_pqb->DBgw.Dc.n = static_cast<UINT32>(peol.get() - tpDc.get());
        if( m_pqb->DBgw.Dc.n != nCandidatesGwTotal )
            throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: actual=%u expected=%u", m_pqb->DBgw.Dc.n, nCandidatesGwTotal );
    }
    else
        m_pqb->DBn.nDm2 = 0;

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: DBgw.Dc.n = %u+%u = %u", __FUNCTION__, m_pqb->DBgw.nDc1, m_pqb->DBn.nDm2, m_pqb->DBgw.Dc.n );
#endif


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: mapped paired N candidates (DBn.nDm1) =    %7u", __FUNCTION__, m_pqb->DBn.nDm1 );
    CDPrint( cdpCD0, "%s: mapped paired Gw candidates (DBgw.nDc1) =  %7u", __FUNCTION__, m_pqb->DBgw.nDc1 );
    CDPrint( cdpCD0, "%s: mapped unpaired Gw candidates (DBn.nDm2) = %7u", __FUNCTION__, m_pqb->DBn.nDm2 );
    CDPrint( cdpCD0, "%s: total candidates for Gw (DBgw.Dc.n) =      %7u", __FUNCTION__, m_pqb->DBgw.Dc.n );
#endif
}

/// [private] method copyKernelResults66
void tuAlignN::copyKernelResults66()
{
    CRVALIDATOR;

    m_hrt.Restart();

    // (re)allocate a host buffer to contain the consolidated list of nongapped mappings from the Dm and Dx lists
    m_pqb->HBn.nMapped = m_pqb->DBn.nDm1 +          // mapped paired D values (see launchKernel21)
                         m_pqb->DBn.nDm2;           // mapped unpaired D values (see launchKernel66)
    m_pqb->HBn.Dm.Reuse( m_pqb->HBn.nMapped, false );

    // append the list of Dx list mappings to the host buffer
    UINT64* phDm = m_pqb->HBn.Dm.p + m_pqb->DBn.nDm1;           // destination buffer pointer (host)
    UINT64* pdDxu = m_pqb->DBgw.Dc.p + m_pqb->DBgw.nDc1;        // source buffer pointer (device)
    CREXEC( cudaMemcpy( phDm, pdDxu, m_pqb->DBn.nDm2*sizeof(UINT64), cudaMemcpyDeviceToHost ) );
    m_pqb->HBn.Dm.n = m_pqb->HBn.nMapped;

    // performance metrics
    InterlockedExchangeAdd( &AriocBase::aam.ms.XferMappings, m_hrt.GetElapsed(false) );


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: HBn.Dm.n = %u+%u = %u mapped", __FUNCTION__, m_pqb->DBn.nDm1, m_pqb->DBn.nDm2, m_pqb->HBn.Dm.n );
#endif

#if TRACE_SQID
    WinGlobalPtr<UINT64> Dcxxx( m_pqb->DBgw.Dc.n, false );
    m_pqb->DBgw.Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );

     // look for a specific sqId
    for( UINT32 n=0; n<m_pqb->DBgw.Dc.n; ++n )
    {
        UINT64 Dc = Dcxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( Dc );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
            
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>(Dc >> 32) & 0x007F;
            INT32 J = static_cast<INT32>(Dc & 0x7FFFFFFF);
            INT32 Jo = static_cast<UINT32>(m_pab->M.p[subId] - 1) - J;

            CDPrint( cdpCD0, "%s: %3d: Dc=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d Jo=%d", __FUNCTION__, n, Dc, sqId, qid, subId, J, Jo );
        }
    }
#endif
}

/// [private] method resetGlobalMemory66
void tuAlignN::resetGlobalMemory66()
{
    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Dc      Dc list (consolidated list of candidates for subsequent alignment)

        high:   (unallocated)
    */

    // discard the list of mapped unpaired candidates
    m_pqb->DBn.Dx.Free();
}
#pragma endregion
