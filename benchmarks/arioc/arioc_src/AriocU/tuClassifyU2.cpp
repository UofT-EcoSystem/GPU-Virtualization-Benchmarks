/*
  tuClassifyU2.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuClassifyU2::tuClassifyU2()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="ppi">partition info</param>
tuClassifyU2::tuClassifyU2( QBatch* pqb, volatile UINT32* pqid ) : baseMAPQ( pqb->pab ),
                                                                   m_pqb(pqb),
                                                                   m_pab(pqb->pab),
                                                                   m_pqid( pqid )
{
}

/// destructor
tuClassifyU2::~tuClassifyU2()
{
}
#pragma endregion

#pragma region private methods
/// [private static] method QAIComparer
int tuClassifyU2::QAIComparer( const void* a, const void* b )
{
    const QAI* pa = reinterpret_cast<const QAI*>(a);
    const QAI* pb = reinterpret_cast<const QAI*>(b);

    // order by the largest number of contiguous matches in the mapping
    return static_cast<int>(pb->mcm) - static_cast<int>(pa->mcm);
}

/// [private] method computeMappingQualities
void tuClassifyU2::computeMappingQualities( QAIReference* pqair, UINT32 nLimit )
{
    baseMAPQ::Vinfo mvi( m_pab );

    // accumulate mapping counts for the mappings
    QAI* pqai = m_pqb->AaBuffer.p + pqair->ofsQAI;
    for( UINT32 n=0; n<pqair->nAq; ++n )
    {
        mvi.TrackV( pqai );
        pqai++ ;
    }

    mvi.FinalizeV();

    // compute mapping quality for the mappings that will eventually be written
    pqai = m_pqb->AaBuffer.p + pqair->ofsQAI;
    for( UINT32 n=0; n<nLimit; ++n )
    {
#if TODO_CHOP_WHEN_DEBUGGED
        INT32 iw = QID_IW(pqai->qid);
        INT32 iq = QID_IQ(pqai->qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        if( pQw->sqId[iq] == 0x000002040000007C )
            CDPrint( cdpCD0, __FUNCTION__ );
#endif

        (this->*m_computeMAPQ)( pqai+n, mvi );
    }

    /* For the primary (i.e., "best") mapping:
        - set a flag
        - record the second-best Vmax
    */
    pqai = m_pqb->AaBuffer.p + pqair->ofsQAI;
    pqai->flags = static_cast<QAIflags>(pqai->flags | qaiBest);
    pqai->Vsec = (mvi.nVmax1 >= 2) ? mvi.Vmax1 : mvi.Vmax2;
}

/// [private] method prioritizeMappings
void tuClassifyU2::prioritizeMappings()
{
    UINT32 qid = InterlockedExchangeAdd( m_pqid, 1 );
    while( qid < m_pqb->DB.nQ )
    {
        // find the mappings for the iq'th Q sequence in the iw'th Qwarp
        QAIReference* pqair = m_pqb->ArBuffer.p + qid;

        if( pqair->nAq >= 2 )
        {
            /* We want to discover whether there are multiple mappings with the same best V score.  If so,
                we will apply other criteria to decide the "best" mapping. */

            // traverse the list of mappings (already sorted by V and unduplicated) to count the number of "best" V scores
            UINT32 nBestV = 1;
            QAI* pqai = m_pqb->AaBuffer.p + pqair->ofsQAI;
            for( ; nBestV<pqair->nAq; ++nBestV )
            {
                if( pqai[nBestV-1].pBH->V > pqai[nBestV].pBH->V )
                    break;
            }

            if( nBestV > 1 )
            {
                // prioritize mappings with the largest span of contiguous matches
                for( UINT32 n=0; n<nBestV; ++n )
                    tuClassify1::computeMaximumContiguousMatches( pqai+n );

                qsort( pqai, nBestV, sizeof(QAI), &tuClassifyU2::QAIComparer );
            }
        }

        qid = InterlockedExchangeAdd( m_pqid, 1 );
    }
}

/// [private] method filterMappings
void tuClassifyU2::filterMappings()
{
    UINT32 qid = InterlockedExchangeAdd( m_pqid, 1 );
    while( qid < m_pqb->DB.nQ )
    {
        QAIReference* pqair = m_pqb->ArBuffer.p + qid;


#if TODO_CHOP_WHEN_DEBUGGED
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        if( pQw->sqId[iq] == 0x000002040000019F )
            CDPrint( cdpCD0, __FUNCTION__ );
#endif






        if( pqair->nAq )
        {
            // limit the number of mappings to report
            UINT32 nLimit = min2(pqair->nAq,static_cast<UINT32>(m_pab->aas.ACP.maxAg));

            // compute mapping qualities (MAPQ) for the reportable mappings
            computeMappingQualities( pqair, nLimit );

            // performance metrics
            if( pqair->nAq > 1 )
                InterlockedIncrement( &AriocBase::aam.n.u.nReads_Mapped2 );
            else
                InterlockedIncrement( &AriocBase::aam.n.u.nReads_Mapped1 );

            // save the number of mappings to report
            pqair->nAq = nLimit;
        }

        qid = InterlockedExchangeAdd( m_pqid, 1 );
    }

}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Filter mappings for reporting and compute mapping qualities (MAPQ)
/// </summary>
void tuClassifyU2::main()
{

#if TODO_CHOP_IF_UNUSED
    prioritizeMappings();
#endif


    filterMappings();
}
#pragma endregion


