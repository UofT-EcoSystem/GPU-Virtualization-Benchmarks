/*
  tuClassify.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
 */
#include "stdafx.h"

#pragma region constructors and destructor
/// default constructor
tuClassify::tuClassify()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuClassify::tuClassify( QBatch* pqb ) : m_pqb( pqb ), m_pab( pqb->pab ), m_qid(0)
{
}

/// destructor
tuClassify::~tuClassify()
{
}
#pragma endregion

/// [protected] method computeEditDistance
INT16 tuClassify::computeEditDistance( BRLEAheader* pBH, INT16 N )
{
    // each soft-clipped symbol counts as one "edit" (i.e., an "insert" into R)
    INT16 ed = pBH->Il +            // 0-based offset of the first mapped symbol in Q
               (N - (pBH->Ir + 1)); // (length of Q) - ((0-based offset of the last mapped symbol in Q) + 1)

    // point to the first BRLEA byte
    BRLEAbyte* p = reinterpret_cast<BRLEAbyte*>(pBH + 1);

    // compute the loop limit
    BRLEAbyte* pLimit = p + pBH->cb;

#ifdef _DEBUG
    if( p->bbType != bbMatch )      // always start with a match
    {
        //DebugBreak();
        UINT32 iw = QID_IW( pBH->qid );
        INT16 iq = QID_IQ( pBH->qid );
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
        CDPrint( cdpCD0, "%s (%s): BRLEA does not start with a match! (qid=0x%08x sqId=0x%016llx)", __FUNCTION__, __FILE__, pBH->qid, sqId );
    }
#endif

    // read the first BRLEA byte (which, by definition, always has bbMatch type)
    BRLEAbyteType bbTypePrev = bbMatch;
    INT16 cbRun = p->bbRunLength;

    // loop through the remaining BRLEA bytes
    while( ++p < pLimit )
    {
        // get the current BRLEA byte type
        BRLEAbyteType bbType = static_cast<BRLEAbyteType>(p->bbType);

        // if we have a new BRLEA byte type ...
        if( bbType != bbTypePrev )
        {
            // conditionally add the accumulated run length to the total edit distance
            if( bbTypePrev != bbMatch )
                ed += cbRun;

            // start again to accumulate the run length
            cbRun = p->bbRunLength;
            bbTypePrev = bbType;
        }

        else
            cbRun = (cbRun << 6) + p->bbRunLength;  // accumulate the current run length
    }

#ifdef _DEBUG
    if( bbTypePrev != bbMatch )     // always end with a match
    {
        UINT32 iw = QID_IW( pBH->qid );
        INT16 iq = QID_IQ( pBH->qid );
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
        INT32 pos = (pBH->J << 1) >> 1;

        CDPrint( cdpCD0, "%s: qid=0x%08x iw=%u iq=%d sqId=0x%016llx subId=%u J=0x%04x pos=%d", __FUNCTION__, pBH->qid, iw, iq, sqId, pBH->subId, pBH->J, pos );
        //DebugBreak();
    }
#endif

    return ed;
}

/// [protected] method appendQAI
QAI* tuClassify::appendQAI( BRLEAheader* pBH, INT16 N, QAIflags qaiFlags )
{
    /* Copy alignment info for the specified mapping into the QAI buffer.

    We compute the 0-based offset of the leftmost aligned symbol (relative to the start of the forward strand) as follows:

    - If the alignment is on the forward strand, the J value in the BRLEAheader is all we need.

    ············+-------------------------+···················>     F (forward)
    0           Jfwd                                          M

    ····+-------------------------+·······>
    0                                     N

    - If the alignment is on the reverse-complement strand, we need to subtract the number of R symbols spanned by
    the alignment:

    ············+-------------------------+···················>     F (forward)
    0           Jfwd                                          M

    <···········+-------------------------+····················     R (reverse complement)
    M            Jrev                      J                  0

    <······+-------------------------+········
    N                                        0


    The transformation is thus:

    Jrev = J + Ma - 1
    Jfwd = (M - 1) - Jrev
    = (M - 1) - (J + Ma - 1)
    = M - (J + Ma)

    - Note that Jfwd can be the same value for different values of J due to indels and/or soft clipping.
    */

#if TODO_CHOP_WHEN_DEBUGGED
    if( pBH->Ma != computeMa( pBH ) )
        DebugBreak();
#endif

#if TRACE_SQID
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(pBH->qid);
    UINT32 iq = QID_IQ(pBH->qid);
    if( (pQw->sqId[iq] | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        CDPrint( cdpCD0, "%s: subId=%u pos=0x%08x (%d)", __FUNCTION__, pBH->subId, pBH->J, pBH->J & 0x7FFFFFFF );


    if( (pBH->subId == 1) && ((pBH->J & 0x7FFFFFFF) == 242589685) )
        CDPrint( cdpCD0, __FUNCTION__ );

#endif

    // find the next free element in the QAI buffer for the current Q sequence
    QAIReference* pqair = m_pqb->ArBuffer.p + QID_ID(pBH->qid);
    QAI* pQAI = m_pqb->AaBuffer.p + (pqair->ofsQAI + pqair->nAq);

    // fill the QAI struct
    pQAI->pBH = pBH;
    pQAI->qid = QID_ID(pBH->qid);
    pQAI->N = N;
    pQAI->subId = pBH->subId;
    pQAI->editDistance = computeEditDistance( pBH, N );
    pQAI->nTBO = pBH->nTBO;

    if( pBH->J & 0x80000000 )
    {
        pQAI->Jf = m_pab->M.p[pBH->subId] - ((pBH->J & 0x7FFFFFFF) + pQAI->pBH->Ma);
        pQAI->flags = static_cast<QAIflags>(qaiFlags | qaiRCr);
    }
    else
    {
        pQAI->Jf = pBH->J;
        pQAI->flags = qaiFlags;
    }

    if( pBH->qid & AriocDS::QID::maskRC )
        pQAI->flags = static_cast<QAIflags>(pQAI->flags | qaiRCq);

#ifdef _DEBUG
    if( pQAI->Jf & 0x80000000 )
        DebugBreak();
#endif

    // update the number of QAI structs referenced by the current QAIReference instance
    pqair->nAq++;

    // return a reference to the QAI struct for the mapped mate
    return pQAI;
}
