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
        CDPrint( cdpCD0, "[%d] %s: %3u:  0x%016llx 0x%08x %u %u %d",
                            m_pqb->pgi->deviceId, __FUNCTION__,
                            n, pQw->sqId[iq], qid, iw, iq, pQw->nAn[iq] );
    }

    for( UINT32 n=0; n<m_pqb->DBgs.Qu.n; ++n )
    {
        UINT32 qid = Quxxx.p[n];
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        if( pQw->nAn[iq] >= static_cast<UINT32>(m_pab->aas.ACP.AtN) )
            CDPrint( cdpCD0, "[%d] %s: %3u:  0x%016llx 0x%08x %u %u %d ***",
                                m_pqb->pgi->deviceId, __FUNCTION__,
                                n, pQw->sqId[iq], qid, iw, iq, pQw->nAn[iq] );
    }

#endif

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s: DBgs.Qu.n=%u/%u", m_pqb->pgi->deviceId, __FUNCTION__, m_pqb->DBgs.Qu.n, m_pqb->DB.nQ );
#endif

}

/// [private] method resetGlobalMemory11
void tuAlignGs::resetGlobalMemory11()
{
    // if all of the Q sequences have a sufficient number of mappings, free any Dl buffer allocation
    if( m_pqb->DBgs.Qu.n == 0 )
        m_pqb->DBn.Dl.Free();
}
#pragma endregion

#pragma region alignGs19
/// [private] method launchKernel19
void tuAlignGs::launchKernel19()
{
    CRVALIDATOR;

    m_hrt.Restart();

    // copy the updated Qwarp buffer
    CREXEC( m_pqb->DB.Qw.CopyToHost( m_pqb->QwBuffer.p, m_pqb->QwBuffer.n ) );

    // performance metrics
    InterlockedExchangeAdd( &AriocBase::aam.ms.XferMappings, m_hrt.GetElapsed(false) );
}

/// [private] method resetGlobalMemory19
void tuAlignGs::resetGlobalMemory19()
{
    CRVALIDATOR;

    /* CUDA global memory layout after reset:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   (unallocated)
    */

    // discard the Qu buffer
    CREXEC( m_pqb->DBgs.Qu.Free() );

    // discard the Dl buffer
    CREXEC( m_pqb->DBn.Dl.Free() );
}
#pragma endregion
