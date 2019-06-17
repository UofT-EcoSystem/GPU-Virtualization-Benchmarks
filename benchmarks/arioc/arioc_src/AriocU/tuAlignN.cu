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
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/copy.h>

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
                    DBn.Dc  candidates for nongapped alignment
                    DBn.Du  leftover candidates for gapped alignment
                    DBn.Dm  mapped D values

            high:   (unused)
        */

        /* Allocate the Dm-list buffer:
            - there is one element for each Dc value (mapped or not) */
        CREXEC( m_pqb->DBn.Dm.Alloc( cgaLow, m_pqb->DBn.Dc.n, false ) );
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
    thrust::device_ptr<UINT64> eolDm = thrust::copy_if( epCGA, tpDc, tpDc+m_pqb->DBn.Dc.n, tpDm, TSX::isMappedDvalue() );
    m_pqb->DBn.Dm.n = static_cast<UINT32>(eolDm.get() - tpDm.get());

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: %u/%u mapped D values", __FUNCTION__, m_pqb->DBn.Dm.n, m_pqb->DBn.Dc.n );


    WinGlobalPtr<UINT64> Dmxxx( m_pqb->DBn.Dm.n, false );
    m_pqb->DBn.Dm.CopyToHost( Dmxxx.p, Dmxxx.Count );

    for( UINT32 n=0; n<100; ++n )
    {
        UINT64 Dm = Dmxxx.p[n];
        INT16 subId = static_cast<INT16>((Dm & AriocDS::D::maskSubId) >> 32);
        INT32 pos = Dm & 0x7FFFFFFF;
        if( Dm & AriocDS::D::maskStrand )
            pos = (m_pab->M.p[subId] - 1) - pos;

        CDPrint( cdpCD0, "%s: %4u 0x%016llx qid=0x%08llx subId=%d J=%d",
                            __FUNCTION__, n, Dm, (Dm>>39)&0x007FFFFF, subId, pos );
    }
#endif
}

/// [private] method copyKernelResults22
void tuAlignN::copyKernelResults22()
{
    CRVALIDATOR;

    m_hrt.Restart();

    // (re)allocate a host buffer to contain the list of mapped Q sequences
    m_pqb->HBn.Dm.Reuse( m_pqb->DBn.Dm.n, false );

    // copy the list of mapped Q sequences to a host buffer
    CREXEC( m_pqb->DBn.Dm.CopyToHost( m_pqb->HBn.Dm.p, m_pqb->DBn.Dm.n ) );
    m_pqb->HBn.Dm.n = m_pqb->DBn.Dm.n;


#if TRACE_SQID
    for( UINT32 n=0; n<m_pqb->HBn.Dm.n; ++n )
    {
        UINT64 Dm = m_pqb->HBn.Dm.p[n];
        UINT32 qid = AriocDS::D::GetQid( Dm );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
            
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>(Dm >> 32) & 0x007F;
            INT8 strand = ((Dm & AriocDS::D::maskStrand) != 0);
            INT32 J = Dm & 0x7FFFFFFF;
            INT32 Jf = strand ? (m_pab->M.p[subId]-1) - J : J;
            CDPrint( cdpCD0, "%s: qid=0x%08x sqId=0x%016llx Dm=0x%016llx subId=%d strand=%d J=%d Jf=%d", __FUNCTION__, qid, sqId, Dm, subId, strand, J, Jf );
        }
    }

    CDPrint( cdpCD0, __FUNCTION__ );
#endif


    // performance metrics
    InterlockedExchangeAdd( &AriocBase::aam.ms.XferMappings, m_hrt.GetElapsed(false) );
}

/// [private] method resetGlobalMemory22
void tuAlignN::resetGlobalMemory22()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after reset:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBn.Dc  candidates for nongapped alignment

            high:   (unused)
        */

        // discard the Dm list
        m_pqb->DBn.Dm.Free();
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}
#pragma endregion

#pragma region alignN34
/// [private] method initGlobalMemory34
void tuAlignN::initGlobalMemory34()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:
            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBn.Dc  candidates for nongapped alignment (now re-flagged in tuAlignN32)
                    DBn.Du  leftover candidates for gapped alignment (excluded by insufficient seed overlap)

            high:   DBn.Dl  leftover D values (failed nongapped alignment)
        */

        /* Allocate space for all of the previously-identified leftover D values, plus all of the unmapped D values.
            None of the mapped D values will become leftovers, so the maximum number of elements in the Dl buffer can
            exclude those mapped D values, that is, the maximum size of the buffer is:
        
                (# leftover candidates in the Du buffer) + (# candidates for nongapped alignment) - (# mapped candidates)
        */
        UINT32 cel = m_pqb->DBn.Du.n + (m_pqb->DBn.Dc.n - m_pqb->HBn.Dm.n);
        CREXEC( m_pqb->DBn.Dl.Alloc( cgaHigh, cel, false ) );
        m_pqb->DBn.Dl.n = cel;
        SET_CUDAGLOBALPTR_TAG( m_pqb->DBn.Dl, "DBn.Dl" );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel34
void tuAlignN::launchKernel34()
{
    /* At this point DBn.Dc contains D values whose flags are set as follows (see tuAlign32):
    
        - flagCandidate: set if the Q sequence does not yet have the required number of mappings
        - flagMapped: set if a nongapped mapping was found for the D value

       We use these flags to identify the set of D values at which a nongapped mapping was not found.  These
        D values nevertheless represent likely candidates for gapped alignment because (like the "leftover" D values
        for which too few seeds overlap to make them candidates for nongapped alignment) the fact that multiple
        spaced seeds overlap suggests that only the number of mismatches in the read prevented it from having a
        successful nongapped alignment.
    
       Here we use a Thrust "stream compaction" API to copy these unmapped D values.  We are copying D values that
        meet the following criteria:

            - "candidate" flag is set (there are fewer than the required number of mappings for the read), AND
            - "mapped" flag is not set (no nongapped mapping for the D value)
    */

    // copy unmapped candidate values to a temporary buffer
    CudaGlobalPtr<UINT64> tempDl( m_pqb->pgi->pCGA );
    tempDl.Alloc( cgaLow, m_pqb->DBn.Dc.n-m_pqb->HBn.Dm.n, false );
    tempDl.n = static_cast<UINT32>(tempDl.Count);

    /* Use a Thrust stream-compaction method to extract the unmapped D values. */
    thrust::device_ptr<UINT64> tpDc( m_pqb->DBn.Dc.p );
    thrust::device_ptr<UINT64> ttpDl( tempDl.p );
    thrust::device_ptr<UINT64> eolDl = thrust::copy_if( epCGA, tpDc, tpDc+m_pqb->DBn.Dc.n, ttpDl, TSX::isUnmappedCandidateDvalue() );
    UINT32 nDl = static_cast<UINT32>(eolDl.get() - ttpDl.get());
    if( nDl > tempDl.n )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent number of candidates: expected %u, actual %u", tempDl.n, nDl );
    tempDl.n = nDl;

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: %u/%u candidates failed nongapped alignment", __FUNCTION__, tempDl.n, m_pqb->DBn.Dc.n );
#endif

    /* Merge the two sets of "leftovers" for subsequent gapped alignment:
        - D values that were previously flagged as "leftovers"
        - D values that failed nongapped alignment

       The result goes into DBn.Dl.
    */
    thrust::device_ptr<UINT64> tpDu(m_pqb->DBn.Du.p);
    thrust::device_ptr<UINT64> tpDl(m_pqb->DBn.Dl.p);

    // thrust::merge is stable, i.e., the result is ordered
    eolDl = thrust::merge( epCGA, tpDu, tpDu+m_pqb->DBn.Du.n, ttpDl, ttpDl+tempDl.n, tpDl, TSX::isLessD() );
    nDl = static_cast<UINT32>(eolDl.get()-tpDl.get());
    if( nDl > m_pqb->DBn.Dl.n )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent number of leftover D values: expected %u, actual %u", m_pqb->DBn.Dl.n, nDl );
    m_pqb->DBn.Dl.n = nDl;

    // discard the temporary buffer
    tempDl.Free();

#if TRACE_SQID
    // examine the Dc list
    CDPrint( cdpCD0, "[%d] %s: DBn.Dc.n=%u", m_pqb->pgi->deviceId, __FUNCTION__, m_pqb->DBn.Dc.n );

    WinGlobalPtr<UINT64> Dcxxx( m_pqb->DBn.Dc.n, false );
    m_pqb->DBn.Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );

    for( UINT32 n=0; n<m_pqb->DBn.Dc.n; ++n )
    {
        UINT64 Dc = Dcxxx.p[n];
        UINT32 qid = static_cast<UINT32>(Dc >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((Dc & AriocDS::D::maskSubId) >> 32);
            INT32 pos = static_cast<INT32>(Dc & 0x7FFFFFFF);
            INT32 Jf = (Dc & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - pos) : pos;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "%s: DBn.Dc[%u]: Dc=0x%016llx qid=0x%08x subId=%d pos=%d Jf=%d",
                            __FUNCTION__, n,
                            Dc, qid, subId, pos, Jf );
        }
    }

    // examine the Dl list
    CDPrint( cdpCD0, "[%d] %s: DBn.Dl.n=%u", m_pqb->pgi->deviceId, __FUNCTION__, m_pqb->DBn.Dl.n );

    WinGlobalPtr<UINT64> Dlxxx( m_pqb->DBn.Dl.n, false );
    m_pqb->DBn.Dl.CopyToHost( Dlxxx.p, Dlxxx.Count );

    for( UINT32 n=0; n<m_pqb->DBn.Dl.n; ++n )
    {
        UINT64 Dl = Dlxxx.p[n];
        UINT32 qid = static_cast<UINT32>(Dl >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((Dl & AriocDS::D::maskSubId) >> 32);
            INT32 pos = static_cast<INT32>(Dl & 0x7FFFFFFF);
            INT32 Jf = (Dl & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - pos) : pos;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "%s: DBn.Dl[%u]: Dl=0x%016llx qid=0x%08x subId=%d pos=%d Jf=%d",
                            __FUNCTION__, n,
                            Dl, qid, subId, pos, Jf );
        }
    }

    CDPrint( cdpCD0, __FUNCTION__ );
#endif

}

/// [private] method resetGlobalMemory34
void tuAlignN::resetGlobalMemory34()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after reset:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data

            high:   DBn.Dl  "leftover" D values [not allocated if no "leftover" values are available]
        */

        CREXEC( m_pqb->DBn.Du.Free() );
        CREXEC( m_pqb->DBn.Dc.Free() );

        if( m_pqb->DBn.Dl.n == 0 )
        {
            CREXEC( m_pqb->DBn.Dl.Free() );
        }
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}
#pragma endregion
