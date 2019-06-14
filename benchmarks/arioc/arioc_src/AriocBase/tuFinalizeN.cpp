/*
  tuFinalizeN.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuFinalizeN::tuFinalizeN()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuFinalizeN::tuFinalizeN( QBatch* pqb ) : m_pqb(pqb),
                                          m_pab(pqb->pab),
                                          m_ptum(AriocBase::GetTaskUnitMetrics("tuFinalizeN"))
{
}

/// destructor
tuFinalizeN::~tuFinalizeN()
{
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Builds BRLEAs for reportable mappings
/// </summary>
void tuFinalizeN::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    // allocate and zero a host buffer to contain BRLEAs for the mapped Q sequences.
    UINT32 celBRLEA = m_pqb->HBn.Dm.n * m_pqb->celBRLEAperQ;
    m_pqb->HBn.BRLEA.Reuse( celBRLEA, false );
    memset( m_pqb->HBn.BRLEA.p, 0, m_pqb->HBn.BRLEA.cb );
    m_pqb->HBn.BRLEA.n = celBRLEA;

    if( celBRLEA )
    {
        // split the work into concurrent partitions
        UINT32 nParts = min2( static_cast<UINT32>(m_pab->nLPs), m_pqb->HBn.Dm.n );
        m_pi.Realloc( nParts, true );
        UINT32 iDm = 0;
        for( UINT32 n=0; n<nParts; ++n )
        {
            // initialize the partition-info struct
            m_pi.p[n].iDm = iDm;
            m_pi.p[n].nDm = blockdiv( m_pqb->HBn.Dm.n-iDm, nParts-n );

            // compute the index of the first Dm value in the subsequent partition
            iDm += m_pi.p[n].nDm;
        }

        // build BRLEAs concurrently in each partition
        WinGlobalPtr<tuFinalizeN1*> finalizeN1( nParts, true );
        for( UINT32 n=0; n<nParts; ++n )
        {
            finalizeN1.p[n] = new tuFinalizeN1( m_pqb, m_pi.p+n );
            finalizeN1.p[n]->Start();
        }

        // wait for the worker threads to exit
        for( UINT32 n=0; n<nParts; ++n )
            finalizeN1.p[n]->Wait( INFINITE );

        // destruct the tuFinalizeN1 instances
        for( UINT32 n=0; n<nParts; ++n )
            delete finalizeN1.p[n];

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->n.MappedD, m_pqb->HBn.Dm.n );

        // record the total number of mappings
        m_pqb->HBn.nMapped = m_pqb->HBn.BRLEA.n / m_pqb->celBRLEAperQ;
    }

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.Elapsed, m_hrt.GetElapsed(false) );

#if TRACE_SQID
    // DEBUG: look for bad BRLEAs
    for( UINT32 n=0; n<(m_pqb->HBn.BRLEA.n/m_pqb->celBRLEAperQ); ++n )
    {
        // point to the first BRLEA byte
        BRLEAheader* pBH = reinterpret_cast<BRLEAheader*>(m_pqb->HBn.BRLEA.p+n*m_pqb->celBRLEAperQ);

        UINT32 iw = QID_IW(pBH->qid);
        INT16 iq = QID_IQ(pBH->qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
        
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            CDPrint( cdpCD0, "%s: sqId=0x%016llx subId=%d J=0x%08x cb=%d", __FUNCTION__, sqId, pBH->subId, pBH->J, pBH->cb );

        if( pBH->cb == 0 )
            continue;

        BRLEAbyte* p = reinterpret_cast<BRLEAbyte*>(pBH + 1);
        if( p->bbType != bbMatch )      // always start with a match
        {
            CDPrint( cdpCD0, "%s: n=%u BRLEA does not start with a match! (qid=0x%08x sqId=0x%016llx cb=%d subId=%d J=0x%08x Il=%d Ir=%d)", __FUNCTION__, n, pBH->qid, sqId, pBH->cb, pBH->subId, pBH->J, pBH->Il, pBH->Ir );
            CDPrint( cdpCD0, __FUNCTION__ );
        }

    }
#endif


    CDPrint( cdpCD3, "[%d] %s: completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
