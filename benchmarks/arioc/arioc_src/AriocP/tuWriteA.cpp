/*
  tuWriteA.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuWriteA::tuWriteA()
{
}

/// <summary>
/// Writes paired-end alignment results.
/// </summary>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="parb">a reference to a <c>baseARowBuilder</c> instance</param>
/// <param name="parw">a reference to a list of <c>baseARowWriter</c> references</param>
/// <param name="tumKey">task-unit metrics key string</param>
/// <param name="wantCounts">indicates whether to count mappings by category</param>
tuWriteA::tuWriteA( QBatch* pqb, baseARowBuilder* parb, WinGlobalPtr<baseARowWriter*>* parw, const char* tumKey, bool wantCounts ) :
                        m_pqb(pqb),
                        m_parb(parb),
                        m_parw(parw),
                        m_paiU(), m_qaiU0(), m_qaiU1(),
                        m_wantCounts(wantCounts),
                        m_ptum(AriocBase::GetTaskUnitMetrics(tumKey))
{
}

/// destructor
tuWriteA::~tuWriteA()
{
}
#pragma endregion

#pragma region private methods
/// [private] method buildTempQAI
PAI* tuWriteA::buildTempQAI( PAI* pPAI, Qwarp* pQw )
{
    // make a copy of the pair alignment info
    m_paiU = *pPAI;

    // create temporary QAI instances that can be used to reference the Q sequence data and metadata
    if( pPAI->pQAI1 == NULL )
    {
        m_paiU.pQAI1 = &m_qaiU0;
        m_qaiU0.N = pQw->N[m_paiU.iq];
        m_qaiU0.qid = PACK_QID(m_paiU.iw, m_paiU.iq);
        m_qaiU0.flags = static_cast<QAIflags>(((pPAI->arf & arfPrimary) ? qaiBest : qaiNone) | ((pQw->sqId[pPAI->iq] & AriocDS::SqId::MaskMateId) ? qaiParity : qaiNone));
    }

    if( pPAI->pQAI2 == NULL )
    {
        m_paiU.pQAI2 = &m_qaiU1;
        m_qaiU1.N = pQw->N[m_paiU.iq+1];
        m_qaiU1.qid = PACK_QID( m_paiU.iw, m_paiU.iq+1 );
        m_qaiU1.flags = static_cast<QAIflags>(((pPAI->arf & arfPrimary) ? qaiBest : qaiNone) | ((pQw->sqId[pPAI->iq+1] & AriocDS::SqId::MaskMateId) ? qaiParity : qaiNone));
    }

    return &m_paiU;
}

/// [private] method emitPaired
void tuWriteA::emitPaired( PairWriteCounts& pwc )
{
    // traverse the list of paired-end reads
    PAI* pPAI = m_pqb->ApBuffer.p;
    for( UINT32 n=0; n<m_pqb->ApBuffer.n; ++n )
    {
        if( pPAI->arf & (arfWriteMate1|arfWriteMate2) )
        {
            // point to the Qwarp that contains the pair of reads
            Qwarp* pQw = m_pqb->QwBuffer.p + pPAI->iw;
            INT64 sqId = pQw->sqId[pPAI->iq];

            // write the pair
            switch( pPAI->arf & arfMaskReport )
            {
                case arfConcordant:     // two mapped mates representing a concordant paired-end mapping
                    pwc.nConcordantRows += m_parb->WriteRowPc( m_parw->p[arfConcordant], sqId, pPAI );
                    break;

                case arfDiscordant:     // two mapped mates representing a discordant paired-end mapping
                    pwc.nDiscordantRows += m_parb->WriteRowPd( m_parw->p[arfDiscordant], sqId, pPAI );
                    break;

                case arfRejected:       // two mapped mates, but not a concordant or discordant mapping
                    pwc.nRejectedRows += m_parb->WriteRowPr( m_parw->p[arfRejected], sqId, buildTempQAI( pPAI, pQw ) );
                    break;

                case arfUnmapped:       // one or two unmapped mates
                    pwc.nUnmappedRows += m_parb->WriteRowPu( m_parw->p[arfUnmapped], sqId, buildTempQAI( pPAI, pQw ) );
                    break;

                default:
                    throw new ApplicationException( __FILE__, __LINE__, "unexpected alignment reporting flags = 0x%02x", pPAI->arf & arfMaskReport );
            }
        }

        // point to the next pair
        pPAI++ ;
    }

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD4, "%s (oft=%d) is about to flush output buffers...", __FUNCTION__, m_wtp.oft );
#endif

    // write remaining alignments for the current batch
    m_parw->p[arfConcordant]->Flush( m_pqb->pgi->deviceOrdinal );
    m_parw->p[arfDiscordant]->Flush( m_pqb->pgi->deviceOrdinal );
    m_parw->p[arfRejected]->Flush( m_pqb->pgi->deviceOrdinal );
    m_parw->p[arfUnmapped]->Flush( m_pqb->pgi->deviceOrdinal );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD4, "%s (oft=%d) just flushed output buffers", __FUNCTION__, m_wtp.oft );
#endif    
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Write paired alignment results
/// </summary>
void tuWriteA::main()
{
    PairWriteCounts pwc ={ 0 };

    CDPrint( cdpCD3, "[%d] %s (%s)...", m_pqb->pgi->deviceId, __FUNCTION__, m_ptum->Key );

    if( m_parw->n ) 
        emitPaired( pwc );

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.Elapsed, m_hrt.GetElapsed(false) );
    if( m_wantCounts )
    {
        InterlockedExchangeAdd( &AriocBase::aam.sam.p.nRowsConcordant, pwc.nConcordantRows );
        InterlockedExchangeAdd( &AriocBase::aam.sam.p.nRowsDiscordant, pwc.nDiscordantRows );
        InterlockedExchangeAdd( &AriocBase::aam.sam.p.nRowsRejected, pwc.nRejectedRows );
        InterlockedExchangeAdd( &AriocBase::aam.sam.p.nRowsUnmapped, pwc.nUnmappedRows );
    }

    CDPrint( cdpCD3, "[%d] %s (%s) completed", m_pqb->pgi->deviceId, __FUNCTION__, m_ptum->Key );
}
#pragma endregion
