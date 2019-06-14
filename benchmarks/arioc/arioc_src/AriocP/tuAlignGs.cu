/*
  tuAlignGs.cu

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
#include <thrust/remove.h>


// TODO: CHOP WHEN DEBUGGED
#include <thrust/sort.h>



#if TODO_CHOP_ASAP
#include <thrust/sort.h>
#endif


#pragma region alignGs11
/// [private] method launchKernel11
void tuAlignGs::launchKernel11()
{
    // remove nulls (all bits set) from the list of unmapped QIDs
    thrust::device_ptr<UINT32> tpQu( m_pqb->DBgs.Qu.p );
    thrust::device_ptr<UINT32> tpEol = thrust::remove_if( epCGA, tpQu, tpQu+m_pqb->DB.nQ, TSX::isEqualTo<UINT32>(_UI32_MAX) );
    m_pqb->DBgs.Qu.n = static_cast<UINT32>(tpEol.get() - tpQu.get());


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
    UINT32 nQconcordant = 0;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;

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

#pragma region alignGs20
/// [private] method initGlobalMemory20
void tuAlignGs::initGlobalMemory20()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBgs.Qu unmapped QIDs
                    DBgs.Ru subId bits for unmapped QIDs (null)
                    DBgs.Dc mapped D values

            high:   (unallocated)
        */

        // discard the previously-mapped D values in the Dc list
        m_pqb->DBgs.Dc.Resize( 1 );
        m_pqb->DBgs.Dc.n = 0;

        // null the subId bits
        CREXEC( cudaMemset( m_pqb->DBgs.Ru.p, 0xFF, m_pqb->DBgs.Ru.cb ) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}
#pragma endregion

#pragma region alignGs30
/// [private] method initGlobalMemory30
UINT32 tuAlignGs::initGlobalMemory30()
{
    CRVALIDATOR;

    /* CUDA global memory after initialization:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBgs.Qu unmapped QIDs                           TODO: IS THIS CORRECT?  (see tuAlignGs30)
                DBgs.Ru subId bits for unmapped QIDs
                DBgs.Dc mapped D values without opposite-mate mappings

        high:   VmaxDc  Vmax for Q sequences
    */

    try
    {
        /* Allocate a chunk of memory to contain Vmax for each Q sequence. */
        CREXEC( m_pqb->DBgs.VmaxDc.Alloc( cgaHigh, m_pqb->DBgs.Dc.n, true ) );
        m_pqb->DBgs.VmaxDc.n = m_pqb->DBgs.Dc.n;
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }

    /* Compute the number of Dc values that can be handled in available GPU global memory.

        Interleaved R values are loaded in 2-dimensional blocks whose dimensions are Mrw (the number
        of R-sequence symbols in the "window") and CUDATHREADSPERWARP.  To keep things as simple as
        possible, we therefore iterate through the Dc values in chunks of CUDATHREADSPERWARP, so that
        the 2-dimensional blocks are full in each iteration.

        For computing Smith-Waterman scores, we need a buffer to contain the F and V values for the
        scoring matrix.
    */
    INT64 celMr = blockdiv( m_pqb->Mrw, 21 );           // max 64-bit elements needed to represent Mr for one Q sequence
    INT64 cbRi = celMr * sizeof( UINT64 );              // Ri bytes per Q sequence
    INT64 cbFV = sizeof( UINT32 ) * m_pqb->Mrw;         // FV bytes per Q sequence
    INT64 cbPerDc = cbRi + cbFV;                        // total bytes per Q sequence

    // compute the maximum number of Dc values per iteration
    UINT32 nDcPerIteration = static_cast<UINT32>(m_pqb->pgi->pCGA->GetAvailableByteCount() / cbPerDc);

    /* Round down to the nearest multiple of CUDATHREADSPERWARP so that...
        - the start offset in each iteration is a multiple of CUDATHREADSPERWARP
        - the Ri buffer is allocated in chunks of CUDATHREADSPERWARP
    */
    nDcPerIteration &= static_cast<UINT32>(-CUDATHREADSPERWARP);

    // return the number of D values to process in each iteration
    return nDcPerIteration;
}
#pragma endregion

#pragma region alignGs34
/// [private] method launchKernel34
void tuAlignGs::launchKernel34()
{
    // count the number of candidates in the Dc list that were mapped by the windowed gapped aligner
    thrust::device_ptr<UINT64> tpDc( m_pqb->DBgs.Dc.p );
    m_pqb->HBgwc.nMapped = static_cast<UINT32>( thrust::count_if( epCGA, tpDc, tpDc+m_pqb->DBgs.Dc.n, TSX::isMappedDvalue() ) );
}
#pragma endregion

#pragma region alignGs36
/// [private] method initGlobalMemory36
void tuAlignGs::initGlobalMemory36()
{
    CRVALIDATOR;

    /* CUDA global memory layout after initialization:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBgs.Qu unmapped QIDs
                DBgs.Ru subId bits for unmapped QIDs
                DBgs.Dc mapped D values without opposite-mate mappings

        high:   VmaxDm  Vmax for the Dm list
                Dm      D values mapped by the windowed gapped aligner
                VmaxDc  Vmax for the Dc list
    */
    try
    {
        /* allocate a buffer to contain the D values mapped by the windowed gapped aligner */
        CREXEC( m_pqb->DBgs.Dm.Alloc( cgaHigh, m_pqb->HBgwc.nMapped, false ) );

        /* allocate a buffer to contain the corresponding Vmax values */
        CREXEC( m_pqb->DBgs.VmaxDm.Alloc( cgaHigh, m_pqb->HBgwc.nMapped, false ) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel36
void tuAlignGs::launchKernel36()
{
    /* use a thrust "stream compaction" API to extract the mapped values from the Dc list */
    thrust::device_ptr<UINT64> tpDc( m_pqb->DBgs.Dc.p );
    thrust::device_ptr<UINT64> tpDm( m_pqb->DBgs.Dm.p );
    thrust::device_ptr<UINT64> eolD = thrust::copy_if( epCGA, tpDc, tpDc+m_pqb->DBgs.Dc.n, tpDm, TSX::isMappedDvalue() );
    m_pqb->DBgs.Dm.n = static_cast<UINT32>(eolD.get() - tpDm.get());
    if( m_pqb->DBgs.Dm.n != m_pqb->HBgwc.nMapped )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: actual=%u expected=%u", m_pqb->DBgs.Dm.n, m_pqb->HBgwc.nMapped );

    /* do the same for the corresponding Vmax values */
    thrust::device_ptr<INT16> tpVmaxDc( m_pqb->DBgs.VmaxDc.p );
    thrust::device_ptr<INT16> tpVmaxDm( m_pqb->DBgs.VmaxDm.p );
    thrust::device_ptr<INT16> tpEolV = thrust::copy_if( epCGA, tpVmaxDc, tpVmaxDc+m_pqb->DBgs.VmaxDc.n, tpVmaxDm, TSX::isNonzero<INT16>() );
    m_pqb->DBgs.VmaxDm.n = static_cast<UINT32>( tpEolV.get() - tpVmaxDm.get() );
    if( m_pqb->DBgs.VmaxDm.n != m_pqb->HBgwc.nMapped )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: actual=%u expected=%u", m_pqb->DBgs.VmaxDm.n, m_pqb->HBgwc.nMapped );


#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dcxxx( m_pqb->DBgs.Dc.n, false );
    m_pqb->DBgs.Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );
    WinGlobalPtr<UINT64> Dmxxx( m_pqb->DBgs.Dm.n, false );
    m_pqb->DBgs.Dm.CopyToHost( Dmxxx.p, Dmxxx.Count );
    WinGlobalPtr<INT16> Vcxxx( m_pqb->DBgs.VmaxDc.n, false );
    m_pqb->DBgs.VmaxDc.CopyToHost( Vcxxx.p, Vcxxx.Count );
    WinGlobalPtr<INT16> Vmxxx( m_pqb->DBgs.VmaxDm.n, false );
    m_pqb->DBgs.VmaxDm.CopyToHost( Vmxxx.p, Vmxxx.Count );

    UINT32 n1 = 0;      // Dc/VmaxDc (maybe mapped, maybe not)
    UINT32 n2 = 0;      // Dm/VmaxDm (all mapped)
    while( n1 < 350 )
    {
        UINT64 Dc = Dcxxx.p[n1];
        if( Dc & AriocDS::D::flagMapped )
        {
            UINT64 Dm = Dmxxx.p[n2];
            CDPrint( cdpCD0, "%s: Dc[%u]=0x%016llx Dm[%u]=0x%016llx (Vc=%d Vm=%d)", __FUNCTION__,
                                n1, Dc, n2, Dm, Vcxxx.p[n1], Vmxxx.p[n2] );
            n2++ ;
        }

        n1++; 
    }

    CDPrint( cdpCD0, __FUNCTION__ );
#endif
}

/// [private] method resetGlobalMemory36
void tuAlignGs::resetGlobalMemory36()
{
    CRVALIDATOR;

    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   VmaxDm  Vmax for the Dm list
                Dm      D values mapped by the windowed gapped aligner
                VmaxDc  Vmax for the Dc list
    */
    try
    {
        // free the buffers that are no longer needed
        m_pqb->DBgs.Dx.Free();
        m_pqb->DBgs.Ru.Free();
        m_pqb->DBgs.Qu.Free();
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}
#pragma endregion

#pragma region alignGs49
/// [private] method launchKernel49
void tuAlignGs::launchKernel49()
{
    CRVALIDATOR;

    m_hrt.Restart();

    // copy the updated Qwarp buffer
    CREXEC( m_pqb->DB.Qw.CopyToHost( m_pqb->QwBuffer.p, m_pqb->QwBuffer.n ) );

    // performance metrics
    InterlockedExchangeAdd( &AriocBase::aam.ms.XferMappings, m_hrt.GetElapsed(false) );
}

/// [private] method resetGlobalMemory49
void tuAlignGs::resetGlobalMemory49()
{
    CRVALIDATOR;

    /* CUDA global memory layout after reset:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   (unallocated)
    */
    try
    {
        // discard the Ri, D, and Vmax buffers
        m_pqb->DB.Ri.Free();
        m_pqb->DBgs.VmaxDm.Free();
        m_pqb->DBgs.Dm.Free();
        m_pqb->DBgs.VmaxDc.Free();
        m_pqb->DBgs.Dc.Free();
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}
#pragma endregion
