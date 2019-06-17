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
    // TODO: CHOP WHEN DEBUGGED
    DebugBreak();
}

/// <summary>
/// Writes unpaired alignment results.
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
                        m_qaiU(),
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
QAI* tuWriteA::buildTempQAI( Qwarp* pQw, UINT32 iw, INT16 iq )
{
    // create a temporary QAI instance that can be used to reference the Q sequence data and metadata
    m_qaiU.N = pQw->N[iq];
    m_qaiU.qid = PACK_QID(iw, iq);

    return &m_qaiU;
}

/// [private] method emitUnpaired
void tuWriteA::emitUnpaired( RowWriteCounts& rwc )
{
    // traverse the list of Qwarps in the current batch
    Qwarp* pQw = m_pqb->QwBuffer.p;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        // traverse the list of Q sequences in the Qwarp
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            // write the mappings for the iq'th Q sequence in the Qwarp
            UINT32 qid = AriocDS::QID::Pack(iw,iq);
            QAIReference* pqair = m_pqb->ArBuffer.p + qid;

            INT64 sqId = pQw->sqId[iq];

#if TODO_CHOP_WHEN_DEBUGGED
            if( sqId == 0x000002040000009b )
                CDPrint( cdpCD0, __FUNCTION__ );
#endif
            QAI* pQAI = m_pqb->AaBuffer.p + pqair->ofsQAI;
            if( pqair->nAq )
            {
                // write the mappings for the iq'th Q sequence
                for( UINT32 n=0; n<pqair->nAq; ++n )
                {
                    m_parb->WriteRowUm( m_parw->p[arfMapped], sqId, pQAI );
                    pQAI++ ;
                }

                if( pqair->nAq > 1 )
                    rwc.nMapped2 += pqair->nAq;
                else
                    rwc.nMapped1++ ;
            }
            else
            {
                buildTempQAI( pQw, iw, iq );
                rwc.nUnmapped += m_parb->WriteRowUu( m_parw->p[arfUnmapped], sqId, &m_qaiU );
            }
        }

        // point to the next Qwarp
        pQw++ ;
    }

    // write remaining alignments for the current batch
    m_parw->p[arfMapped]->Flush( m_pqb->pgi->deviceOrdinal );
    m_parw->p[arfUnmapped]->Flush( m_pqb->pgi->deviceOrdinal );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD4, "[%d] %s: just flushed output buffers", m_pqb->pgi->deviceId, __FUNCTION__ );
#endif    
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Writes unpaired alignment results.
/// </summary>
void tuWriteA::main()
{
    RowWriteCounts rwc = { 0 };

    CDPrint( cdpCD3, "[%d] %s (%s)...", m_pqb->pgi->deviceId, __FUNCTION__, m_ptum->Key );

    if( m_parw->n ) 
        emitUnpaired( rwc );

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.Elapsed, m_hrt.GetElapsed( false ) );
    if( m_wantCounts )
    {
        InterlockedExchangeAdd( &AriocBase::aam.sam.u.nReads_Mapped1, rwc.nMapped1 );
        InterlockedExchangeAdd( &AriocBase::aam.sam.u.nReads_Mapped2, rwc.nMapped2 );
        InterlockedExchangeAdd( &AriocBase::aam.sam.u.nReads_Unmapped, rwc.nUnmapped );
    }

    CDPrint( cdpCD3, "[%d] %s (%s) completed", m_pqb->pgi->deviceId, __FUNCTION__, m_ptum->Key );
}
#pragma endregion
