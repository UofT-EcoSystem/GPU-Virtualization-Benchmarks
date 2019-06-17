/*
  tuAlignGwn.cu

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

#pragma region alignGw00

#if TODO_CHOP_IF_UNUSED
/// [private] method initGlobalMemory00
void tuAlignGwn::initGlobalMemory00()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

             low:   Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      DBgs.Qu QIDs of both mates in unmapped pairs (i.e. all mates)

            high:           (unallocated)
        */

        UINT32 celQu = m_pqb->DB.Qw.n * CUDATHREADSPERWARP;
        CREXEC( m_pqb->DBgs.Qu.Alloc( cgaLow, celQu, true ) );
   }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method launchKernel00
void tuAlignGwn::launchKernel00()
{
    WinGlobalPtr<UINT32> Quh( m_pqb->DBgs.Qu.Count, false );
    for( UINT32 qid=0; qid<static_cast<UINT32>(m_pqb->DBgs.Qu.Count); ++qid )
        Quh.p[qid] = qid;

    m_pqb->DBgs.Qu.CopyToDevice( Quh.p, m_pqb->DB.nQ );
    m_pqb->DBgs.Qu.n = m_pqb->DB.nQ;

#if TODO_FIGURE_OUT_WHY_THIS_DOESNT_WORK
    /* use thrust to initialize a QID for each Q sequence in the current batch; the QIDs are just binary values
        in a 0-based monotonically-increasing sequence) */
    thrust::device_ptr<UINT32> tpQu( m_pqb->DBgs.Qu.p );
    thrust::counting_iterator<UINT32> tiQID( 0 );
    thrust::device_ptr<UINT32> eol = thrust::copy( epCGA, tiQID, tiQID+m_pqb->DBgs.Qu.n, tpQu ); <-- what is the value of Qu.n here?
    m_pqb->DBgs.Qu.n = m_pqb->DB.nQ;

    UINT32 nUnpaired = static_cast<UINT32>(eol.get() - tpQu.get());
    CDPrint( cdpCD0, "tuAlignGwn::initGlobalMemory00: nUnpaired=%u", nUnpaired );


    WinGlobalPtr<UINT32> Quxxx(m_pqb->DBgs.Qu.n,false);
    m_pqb->DBgs.Qu.CopyToHost( Quxxx.p, Quxxx.Count );
#endif
}
#endif

/// [private] method resetGlobalMemory00
void tuAlignGwn::resetGlobalMemory00()
{
    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data

        high:           (unallocated)
    */

    // (there are no candidates for windowed gapped alignment)
    m_pqb->DBgw.Dc.Free();
}

#pragma endregion

#pragma region alignGwn11
/// [private] method launchKernel11
void tuAlignGwn::launchKernel11()
{
#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: before reduce_by_key:", __FUNCTION__ );
    WinGlobalPtr<UINT32> Quxxx( m_pqb->DBgw.Qu.n, false );
    m_pqb->DBgw.Qu.CopyToHost( Quxxx.p, Quxxx.Count );
    WinGlobalPtr<UINT32> Ruxxx( m_pqb->DBgw.Ru.n, false );
    m_pqb->DBgw.Ru.CopyToHost( Ruxxx.p, Ruxxx.Count );

    for( UINT32 n=0; n<1000; n++ )
        CDPrint( cdpCD0, "%s: 0x%08x 0x%08x", __FUNCTION__, Quxxx.p[n], Ruxxx.p[n] );
#endif

    // copy Qu and Ru to temporary buffers (thrust::reduce_by_key requires non-overlapping input and output buffers)
    CudaGlobalPtr<UINT32> tempQu( m_pqb->pgi->pCGA );
    tempQu.Alloc( cgaLow, m_pqb->DBgw.Qu.n, false );
    m_pqb->DBgw.Qu.CopyInDevice( tempQu.p, tempQu.Count );
    CudaGlobalPtr<UINT32> tempRu( m_pqb->pgi->pCGA );
    tempRu.Alloc( cgaLow, m_pqb->DBgw.Ru.n, false );
    m_pqb->DBgw.Ru.CopyInDevice( tempRu.p, tempRu.Count );

    // sort the QIDs
    thrust::device_ptr<UINT32> ttpQu( tempQu.p );
    thrust::device_ptr<UINT32> ttpRu( tempRu.p );
    thrust::sort_by_key( epCGA, ttpQu, ttpQu+m_pqb->DBgw.Qu.n, ttpRu, thrust::less<UINT32>() );

    // use a thrust "stream compaction" API to unduplicate the QIDs and compute a bitwise OR on the subId bits for each QID
    thrust::device_ptr<UINT32> tpQu( m_pqb->DBgw.Qu.p );
    thrust::device_ptr<UINT32> tpRu( m_pqb->DBgw.Ru.p );
    thrust::pair< thrust::device_ptr<UINT32>,thrust::device_ptr<UINT32> > eolQR = thrust::reduce_by_key( epCGA, ttpQu, ttpQu+m_pqb->DBgw.Qu.n,
                                                                                                         ttpRu,
                                                                                                         tpQu,
                                                                                                         tpRu,
                                                                                                         thrust::equal_to<UINT32>(),
                                                                                                         thrust::bit_or<UINT32>()
                                                                                                       );     
    m_pqb->DBgw.Qu.n = static_cast<UINT32>( eolQR.first.get() - tpQu.get() );
    m_pqb->DBgw.Ru.n = static_cast<UINT32>( eolQR.second.get() - tpRu.get() );
    if( m_pqb->DBgw.Qu.n != m_pqb->DBgw.Ru.n )
        throw new ApplicationException( __FILE__, __LINE__, "reduce_by_key failed: DBgw.Qu.n=%u DBgw.Ru.n=%u", m_pqb->DBgw.Qu.n, m_pqb->DBgw.Ru.n );
    


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: DBgw.Qu.n = %u/%u", __FUNCTION__, m_pqb->DBgw.Qu.n, m_pqb->DB.Qw.n*CUDATHREADSPERWARP );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: after reduce_by_key:", __FUNCTION__ );

    //WinGlobalPtr<UINT32> Quxxx( m_pqb->DBgw.Qu.n, false );
    m_pqb->DBgw.Qu.CopyToHost( Quxxx.p, Quxxx.Count );
    //WinGlobalPtr<UINT32> Ruxxx( m_pqb->DBgw.Ru.n, false );
    m_pqb->DBgw.Ru.CopyToHost( Ruxxx.p, Ruxxx.Count );

    for( UINT32 n=0; n<1000; n++ )
        CDPrint( cdpCD0, "%s: 0x%08x 0x%08x", __FUNCTION__, Quxxx.p[n], Ruxxx.p[n] );
#endif


#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s: looking for sqId 0x%016llx in Qu list...", m_pqb->pgi->deviceId, __FUNCTION__, TRACE_SQID );
    
    WinGlobalPtr<UINT32> Quzzz( m_pqb->DBgw.Qu.n, false );
    m_pqb->DBgw.Qu.CopyToHost( Quzzz.p, Quzzz.Count );

    bool bFound = false;
    for( UINT32 n=0; n<m_pqb->DBgw.Qu.n; ++n )
    {
        UINT32 qid = Quzzz.p[n];
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        if( (pQw->sqId[iq] | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            CDPrint( cdpCD0, "[%d] %s: %u: qid=0x%08x sqId=0x%016llx", m_pqb->pgi->deviceId, __FUNCTION__, n, qid, pQw->sqId[iq] );
            bFound = true;
        }
    }
    
    if( !bFound )
        CDPrint( cdpCD0, "[%d] %s: sqId=0x%016llx not in Qu list!", m_pqb->pgi->deviceId, __FUNCTION__, TRACE_SQID );

#endif


    // free the temporary buffers
    tempRu.Free();
    tempQu.Free();
}
#pragma endregion

#pragma region alignGwn19
/// [private] method resetGlobalMemory19
void tuAlignGwn::resetGlobalMemory19()
{
    /* CUDA global memory after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   (unallocated)
    */
    
    m_pqb->DBgw.Ru.Free();
    m_pqb->DBgw.Qu.Free();
    m_pqb->DBgw.Dc.Free();
}
#pragma endregion

