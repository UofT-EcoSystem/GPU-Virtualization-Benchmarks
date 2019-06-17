/*
  tuAlignGwn.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuAlignGwn::tuAlignGwn()
{

}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuAlignGwn::tuAlignGwn( QBatch* pqb ) : m_pqb(pqb), m_pab(pqb->pab)
{
}

/// destructor
tuAlignGwn::~tuAlignGwn()
{
}
#pragma endregion

#pragma region alignGwn00
/// [private] method alignGwn00
void tuAlignGwn::alignGwn00()
{
    Qwarp* pQw = m_pqb->QwBuffer.p + (m_pqb->QwBuffer.n - 1);
    CDPrint( cdpCD0, "[%d] %s: no nongapped mappings and no windowed gapped mappings",
                        m_pqb->pgi->deviceId, __FUNCTION__ );

    CDPrint( cdpCD0, "[%d] %s: batch contains %u reads (sqId 0x%016llx-0x%016llx, readId %lld-%lld)",
                        m_pqb->pgi->deviceId, __FUNCTION__,
                        (m_pqb->QwBuffer.n-1)*CUDATHREADSPERWARP + pQw->nQ,
                        m_pqb->QwBuffer.p[0].sqId[0],
                        pQw->sqId[pQw->nQ-1],
                        AriocDS::SqId::GetReadId( m_pqb->QwBuffer.p[0].sqId[0] ),
                        AriocDS::SqId::GetReadId( pQw->sqId[pQw->nQ-1] ) );

    resetGlobalMemory00();
}
#pragma endregion

#pragma region alignGwn10
/// [private] method alignGwn10
void tuAlignGwn::alignGwn10()
{
    /* CUDA global memory here:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBgw.Dc mapped candidates with unmapped opposite mates

        high:   (unallocated)
    */



#if TODO_CHOP_WHEN_DEBUGGED
    // dump the D values for a specified QID
    UINT32 qidyyy = 0x000001;
    WinGlobalPtr<UINT64> Dmyyy( m_pqb->DBgw.Dc.n, false );
    m_pqb->DBgw.Dc.CopyToHost( Dmyyy.p, Dmyyy.Count );
    for( UINT32 n=0; n<m_pqb->DBgw.Dc.n; ++n )
    {
        UINT64 D = Dmyyy.p[n];
        UINT32 qid = AriocDS::D::GetQid(D);
        if( (qid ^ qidyyy) > 1 )
            continue;
        
        Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
        UINT64 sqId = pQw->sqId[QID_IQ(qid)];
        INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
        INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
        INT32 Jf = (D & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

        CDPrint( cdpCD0, "[%d] %s: Dc: %4u 0x%016llx qid=0x%08x subId=%d J=%d Jf=%d sqId=0x%016llx",
                            m_pqb->pgi->deviceId, __FUNCTION__,
                            n, D, qid, subId, J, Jf, sqId );
    }
#endif



    /* Gw10: build a list of QIDs and subIds for the opposite mates; set the "exclude" bit on previously-mapped candidates */
    tuAlignGw10 k10( "tuAlignGwn10", m_pqb, &m_pqb->DBgw, riDc );
    k10.Start();
    k10.Wait();

    /* Gw11: sort and unduplicate the QIDs; reduce the subIds into bitmaps */
    launchKernel11();

    /* Gw12: try to map the opposite mates using seed-and-extend gapped alignment and subIds to filter the J lists */
    baseMapGw k12( "tuAlignGwn12", m_pqb, &m_pqb->DBgw, &m_pqb->HBgwn );
    k12.Start();
    k12.Wait();

    /* Gw19: clean up global-memory buffers */
    resetGlobalMemory19();
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Computes gapped short-read alignments for pairs with one unmapped mate, using the mapped mate as an anchor
/// </summary>
void tuAlignGwn::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    if( (m_pab->a21ss.IsNull()) || (m_pqb->HBn.nMapped == 0) )
        alignGwn00();
    else
        alignGwn10();

    CDPrint( cdpCD3, "[%d] %s: completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
