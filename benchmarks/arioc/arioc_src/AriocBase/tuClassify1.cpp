/*
  tuClassify1.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuClassify1::tuClassify1()
{
}

/// constructor (QBatch, pQAIComparer, UINT32*, UINT32)
tuClassify1::tuClassify1( QBatch* pqb, pQAIComparer pfn, volatile UINT32* pqid ) : m_pqb( pqb ),
                                                                                   m_pab(pqb->pab),
                                                                                   m_pQAIComparer(pfn),
                                                                                   m_pqid(pqid)
{
}

/// destructor
tuClassify1::~tuClassify1()
{
}
#pragma endregion

#pragma region private methods
/// [private] method exciseDuplicateMappings
UINT32 tuClassify1::exciseDuplicateMappings( QAIReference* pqair, QAI* pqai )
{
#if TODO_CHOP_WHEN_DEBUGGED
    Qwarp* pQwx = m_pqb->QwBuffer.p + QID_IW(pqai->qid);
    //if( (pQwx->sqId[QID_IQ(pqai->qid)] & 0xFFFFFFFFFFFFFFFC) == 0x00000418000155c0 )
    //{
        for( INT16 n=pqair->nAq-1; n>=0; --n )
        {
            CDPrint( cdpCD0, "%s: n=%d flags=0x%04x subId=%d Jf=%d V=%d", __FUNCTION__, n, 
                                pqai[n].flags, pqai[n].subId, pqai[n].Jf, pqai[n].pBH->V );
        }
        CDPrint( cdpCD0, __FUNCTION__ );
    //}
#endif

    // do nothing if the list does not contain at least two QAIs
    if( pqair->nAq < 2 )
        return 0;

    // traverse the QAI list from end to start
    UINT32 nAdup = 0;

    // point to the last QAI in the specified list
    QAI* pQAIn = pqai + pqair->nAq - 1;

    // iterate from the second-to-last QAI through the first in the list
    for( QAI* pQAIn1=pQAIn-1; pQAIn1>=pqai; --pQAIn1 )
    {
        /* Adjacent mappings are duplicates if:
            - they map to the same subunit ID
            - they map to the same R strand
            - they derive from the same Q strand
            - the distance between their reference-sequence positions is within a configured limit
        */

        // subunit ID
        if( pQAIn->subId != pQAIn1->subId )
        {
            pQAIn = pQAIn1;
            continue;
        }

        // R-sequence strand
        UINT16 mappedToRC1 = (((pQAIn->flags >> qaiShrRC) ^ pQAIn->flags) & qaiRCr) != 0;
        UINT16 mappedToRC2 = (((pQAIn1->flags >> qaiShrRC) ^ pQAIn1->flags) & qaiRCr) != 0;
        if( mappedToRC1 != mappedToRC2 )
        {
            pQAIn = pQAIn1;
            continue;
        }

        // upstream positions
        INT32 diff = abs(static_cast<INT32>(pQAIn->Jf)-static_cast<INT32>(pQAIn1->Jf));
        if( diff >= static_cast<INT32>(m_pab->aas.ACP.minPosSep) )
        {
            pQAIn = pQAIn1;
            continue;
        }

        // downstream positions
        diff = abs(static_cast<INT32>(pQAIn->Jf+pQAIn->pBH->Ma)-static_cast<INT32>(pQAIn1->Jf+pQAIn1->pBH->Ma));
        if( diff >= static_cast<INT32>(m_pab->aas.ACP.minPosSep) )
        {
            pQAIn = pQAIn1;
            continue;
        }


#if TODO_CHOP
        if( diff >= static_cast<INT32>(m_pab->aas.ACP.minPosSep) )
        {
            /* At this point the upstream positions are distinct, but we also want the downstream positions
                to be distinct in order to be certain that we don't have a duplicate.
                
               We accept two closely-mapped reads as duplicates if the upstream positions */

            /* Set an absolute limit onIf we assume the two reads have the same length, the upstream positions must be within about twice
                the maximum acceptable distance in order for the downstream positions to be acceptably close together.
                This is obviously affected by indels and different read lengths.  But the idea is to try not tostring (which they should if they are duplicates), then the downstream
                posif the upstream positions are not "reasonably close", assume we have two distinct mappings
                ...
                */
            if( diff > static_cast<INT32>(2*m_pab->aas.ACP.minPosSep) )
                continue;

            // downstream positions
            diff = abs(static_cast<INT32>(pqai[n].Jf+pqai[n].pBH->Ma)-static_cast<INT32>(pqai[n-1].Jf+pqai[n-1].pBH->Ma));
            if( diff >= static_cast<INT32>(m_pab->aas.ACP.minPosSep) )
                continue;
        }
#endif



        /* At this point we have two mappings to the same location on the forward strand (Jf),
            but the mappings may still differ (different V scores, different indels):
        */
        bool useNthMapping = false;
        bool combineFlags = false;

        if( pQAIn->pBH->V == pQAIn1->pBH->V )
        {
            // Q-sequence strand
            if( (pQAIn->flags ^ pQAIn1->flags) & qaiRCq )
            {
                /* set a flag to indicate that the nth mapping has the same POS and V but derives from
                    a different Q-sequence strand than the (n-1)th mapping */
                pQAIn->flags = static_cast<QAIflags>(pQAIn->flags | qaiRCdup);
                pQAIn = pQAIn1;
                continue;
            }

            /* At this point the V scores and Q-sequence strand are the same, so we use the mapping with
                the longest substring of contiguous matches. */
            tuClassify1::computeMaximumContiguousMatches( pQAIn );
            tuClassify1::computeMaximumContiguousMatches( pQAIn1 );

            if( pQAIn->mcm >= pQAIn1->mcm )
            {
                useNthMapping = true;
                if( pQAIn->mcm == pQAIn1->mcm )
                    combineFlags = true;
            }
        }
        else
        {
            /* At this point the V scores differ, so we use the mapping with the larger V score. */
            useNthMapping = (pQAIn->pBH->V >= pQAIn1->pBH->V);
        }

        if( useNthMapping )
        {
            // use the nth mapping; excise the n-1th mapping
            if( combineFlags )
                pQAIn->flags = static_cast<QAIflags>((pQAIn1->flags & qaiMapped) | pQAIn->flags);

            pQAIn1->flags = static_cast<QAIflags>(pQAIn1->flags | qaiIgnoreDup);
        }
        else
        {
            // use the n-1th mapping; excise the nth mapping
            if( combineFlags )
                pQAIn1->flags = static_cast<QAIflags>(pQAIn1->flags | (pQAIn->flags & qaiMapped));

            pQAIn->flags = static_cast<QAIflags>(pQAIn->flags | qaiIgnoreDup);
            pQAIn = pQAIn1;
        }

        nAdup++ ;       // count duplicates
    }

    // excise the flagged mappings
    UINT32 nAq = 0;
    QAI* pTo = pqai;
    QAI* pFrom = pqai;
    UINT32 nInRun = 0;
    for( UINT32 n=0; n<pqair->nAq; ++n )
    {
        // if the nth QAI is a duplicate...
        if( pqai[n].flags & qaiIgnoreDup )
        {
            if( nInRun )
            {
                // copy the current run of unduplicated QAIs
                memmove( pTo, pFrom, nInRun*sizeof(QAI) );
                nAq += nInRun;

                // point past the newly-appended run
                pTo += nInRun;

                // reset
                nInRun = 0;
            }
        }
        else
        {
            if( nInRun )
                ++nInRun;       // extend the current run
            else
            {
                // start a new run of non-duplicate QAIs
                pFrom = pqai + n;
                nInRun = 1;
            }
        }
    }

    // append the trailing run if any
    if( nInRun )
    {
        memmove( pTo, pFrom, nInRun*sizeof(QAI) );
        nAq += nInRun;
    }

#if TODO_CHOP_WHEN_DEBUGGED
    //if( (pQwx->sqId[QID_IQ(pqai->qid)] & 0xFFFFFFFFFFFFFFFC) == 0x00000418000155c0 )
    //{
        CDPrint( cdpCD0, "%s: after excising duplicates, nAq=%d:", __FUNCTION__, pqair->nAq );
        for( INT16 n=pqair->nAq-1; n>=0; --n )
        {
            CDPrint( cdpCD0, "%s: n=%d flags=0x%04x subId=%d Jf=%d V=%d", __FUNCTION__, n, 
                                pqai[n].flags, pqai[n].subId, pqai[n].Jf, pqai[n].pBH->V );
        }
        CDPrint( cdpCD0, "%s: nAdup=%d", __FUNCTION__, nAdup );
    //}
#endif

#ifdef _DEBUG
    // sanity check
    if( pqair->nAq != (nAq + nAdup) )
        DebugBreak();
#endif

    // update the total number of (unduplicated) mappings
    pqair->nAq = nAq;

    // return the total number of duplicate mappings
    return nAdup;
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Sorts and unduplicates mappings.
/// </summary>
void tuClassify1::main()
{
    UINT32 nAdup = 0;
    UINT32 qid = InterlockedExchangeAdd( m_pqid, 1 );
    while( qid < m_pqb->DB.nQ )
    {
        // sort the mappings for the Q sequence by subId and Jf
        QAIReference* pqair = m_pqb->ArBuffer.p + qid;
        QAI* pqai = m_pqb->AaBuffer.p + pqair->ofsQAI;
        qsort( pqai, pqair->nAq, sizeof(QAI), m_pQAIComparer );

        // excise duplicate mappings
        nAdup += exciseDuplicateMappings( pqair, pqai );

#if TRACE_SQID
        Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
        UINT64 sqId = pQw->sqId[QID_IQ(qid)];
        if( ( sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            RaiiCriticalSection<tuClassify1> rcs;

            CDPrint( cdpCD0, "%s: final list for sqId=0x%016llx...", __FUNCTION__, sqId );

            for( UINT32 n=0; n<pqair->nAq; ++n )
                CDPrint( cdpCD0, "%s: sqId=0x%016llx paqi[%d]=%d/%u",
                                        __FUNCTION__, sqId,
                                        n, pqai[n].subId, pqai[n].Jf );

            CDPrint( cdpCD0, __FUNCTION__ );
        }
#endif

        for( UINT32 n=0; n<pqair->nAq; ++n )
        {
            /* compute the number of contiguous matches for the mappings in the list; we need to do this for all
                of the mappings because we don't yet know which ones will eventually be reported */
            tuClassify1::computeMaximumContiguousMatches( pqai+n );

            // associate the number of unduplicated mappings with each QAI instance
            pqai[n].nAu = pqair->nAq;
        }

        // get another QID
        qid = InterlockedExchangeAdd( m_pqid, 1 );
    }

    // performance metrics
    InterlockedExchangeAdd( &AriocBase::aam.n.DuplicateMappings, nAdup );

// TODO: CHOP    CDPrint( cdpCD0, "[%d] %s complete:  %dms", m_pqb->pgi->deviceId, __FUNCTION__, hrt.GetElapsed(false) );
}
#pragma endregion

#pragma region public methods
/// [public static] method computeMaximumContiguousMatches
void tuClassify1::computeMaximumContiguousMatches( QAI* pQAI )
{
    // do nothing if the maximum number of contiguous matches has already been computed
    if( pQAI->mcm )
        return;

    // point to the BRLEA string
    BRLEAbyte* pbb = reinterpret_cast<BRLEAbyte*>(pQAI->pBH+1);
    BRLEAbyte* pbbLimit = pbb + pQAI->pBH->cb;

    // accumulate the total number of reference locations covered by seeds
    UINT16 currentRunLength = 0;
    while( pbb < pbbLimit )
    {
        // accumulate the maximum length of a run of matches
        if( pbb->bbType == bbMatch )
            currentRunLength = (currentRunLength << 6) + pbb->bbRunLength;
        else
        {
            // save the current run length if it exceeds the previous maximum
            if( currentRunLength > pQAI->mcm )
                pQAI->mcm = currentRunLength;

            // reset the current run length
            currentRunLength = 0;
        }

        // iterate
        pbb++ ;
    }

    // save the current run length if it exceeds the previous maximum
    if( currentRunLength > pQAI->mcm )
        pQAI->mcm = currentRunLength;
}
#pragma endregion
