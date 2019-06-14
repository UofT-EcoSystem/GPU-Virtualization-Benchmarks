/*
  QReaderU.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
   There is one instance of this class per GPU for each Q-sequence input file.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
QReaderU::QReaderU()
{
}

/// constructor
QReaderU::QReaderU( QBatch* pqb, INT16 iPart ) : baseQReader(iPart)
{
    QFI->pfi = pqb->pfiQ;
    openQfile( &m_fileQ, QFI );
    openMfile( &fileMm, MFIm, QFI );
    openMfile( &fileMq, MFIq, QFI );
    initQFB( m_QFB, QFI, pqb->QwarpLimit );
}

/// destructor
QReaderU::~QReaderU()
{
    m_fileQ.Close();
    fileMm.Close();
    fileMq.Close();
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Reads Q-sequence data (unpaired reads) from disk
/// </summary>
bool QReaderU::LoadQ( QBatch* pqb )
{
    /* determine whether we have reached the end of the current data partition */
    if( QFI->eod )
    {
        // return false to indicate that there is no more data available
        return false;
    }

    // reset the Qwarp buffer
    Qwarp* pQw = pqb->QwBuffer.p;
    memset( pQw, 0, pqb->QwBuffer.cb );
    pqb->QwBuffer.n = 0;    // total number of Qwarps for the current batch
    pqb->DB.nQ = 0;         // total number of Q sequences for the current batch
    pqb->Nmax = 0;          // maximum Q sequence length

    // reset the Qi (interleaved Q sequence data) buffer
    memset( pqb->QiBuffer.p, 0, pqb->QiBuffer.cb );
    pqb->QiBuffer.n = 0;

    INT16 iq = 0;           // index into the current Qwarp
    UINT32 celQi = 0;       // total number of elements in the interleaved Q sequence buffer

    // loop while bytes remain in the Q file input buffer
    while( !QFI->eod )
    {
        /* At this point we expect a DataRowHeader. */
        InputFileGroup::DataRowHeader drh = *reinterpret_cast<InputFileGroup::DataRowHeader*>(loadFromQfile( &m_fileQ, QFI, m_QFB, sizeof(InputFileGroup::DataRowHeader) ));

        // get the expected number of bytes of binary data
        UINT64* pQ = reinterpret_cast<UINT64*>(loadFromQfile( &m_fileQ, QFI, m_QFB, drh.cb ));

        // if we need a new Qwarp ...
        if( iq == CUDATHREADSPERWARP )
        {
            pQw++;                  // point to the next available Qwarp in the buffer
            pQw->ofsQi = celQi;     // reference the start of this Qwarp's interleaved Q sequence data
            iq = 0;                 // reset the index into the Qwarp
        }

        const INT64 readId = static_cast<INT64>(AriocDS::SqId::GetReadId( drh.sqId ));
        if( (readId >= QFI->pfi->ReadIdFrom) && (readId <= QFI->pfi->ReadIdTo) )
        {
            // copy the iq'th read into the iw'th Qwarp
            copyToQwarp( pQw, iq++, pqb->QiBuffer.p, &drh, pQ );

            // if we have filled the Qwarp ...
            if( iq == CUDATHREADSPERWARP )
            {
                if( pQw->Nmax > pqb->Nmax )                 // track the maximum Q sequence length
                    pqb->Nmax = pQw->Nmax;
                pQw->nQ = CUDATHREADSPERWARP;               // track the total number of Q sequences in the Qwarp
                pQw->wcgsc = pqb->pab->aas.ComputeWorstCaseGapSpaceCount( pQw->Nmax );
                UINT32 celCol = blockdiv( pQw->Nmax, 21 );  // number of 64-bit elements in each "column" of interleaved Q sequence data
                celQi += (celCol * CUDATHREADSPERWARP);     // track the total number of elements in the Qi buffer
                pqb->QwBuffer.n++;                          // track the total number of Qwarps

                // fall out of the loop if we have reached the maximum number of Qwarps for the current batch
                if( pqb->QwBuffer.n == pqb->QwarpLimit )
                    break;
            }
        }
    }

    /* At this point either...
        - all of the Qwarps for the batch have been initialized, or
        - there are no more Q sequences to be read in the current file partition
    */

    // track the total number of Q sequences in the batch
    pqb->DB.nQ = pqb->QwBuffer.n * CUDATHREADSPERWARP;

    /* The last Qwarp is either
        - empty (iq == 0)
        - full (iq == 32)
        - partially filled (iq between 1 and 31)
       If the last Qwarp is partially filled, we finalize it here.
    */
    if( iq & 0x1F )
    {
        if( pQw->Nmax > pqb->Nmax )                 // track the maximum Q sequence length
            pqb->Nmax = pQw->Nmax;
        pQw->nQ = iq;                               // track the total number of Q sequences in the Qwarp
        pQw->wcgsc = pqb->pab->aas.ComputeWorstCaseGapSpaceCount( pQw->Nmax );
        UINT32 celCol = blockdiv( pQw->Nmax, 21 );  // number of 64-bit elements in each "column" of interleaved Q sequence data
        celQi += (celCol * CUDATHREADSPERWARP);     // point to the interleaved Q sequence data for the next Q sequence
        pqb->QwBuffer.n++;                          // track the total number of Qwarps in the batch
        pqb->DB.nQ += iq;                           // track the total number of Q sequences in the batch
    }


#if TODO_CHOP_WHEN_DEBUGGED
    if( pqb->QwBuffer.n )
    {
        pQw = pqb->QwBuffer.p + (pqb->QwBuffer.n-1);
        UINT64 readIdFinal = AriocDS::SqId::GetReadId( pQw->sqId[pQw->nQ-1] );
        CDPrint( cdpCD0, "%s: pqb->QwBuffer.n=%u pqb->DB.nQ=%u final readId = %lld", __FUNCTION__, pqb->QwBuffer.n, pqb->DB.nQ, readIdFinal );
    }
    else
        CDPrint( cdpCD0, "%s: batch contains no Q sequences", __FUNCTION__ );
#endif


    const bool rval = (pqb->DB.nQ != 0);
    if( rval )
    {
        // compute the size of one BRLEA
        pqb->celBRLEAperQ = blockdiv( sizeof(BRLEAheader)+pqb->pab->aas.ComputeWorstCaseBRLEAsize( pqb->Nmax ), sizeof(UINT32) );

        // save the number of 64-bit values in the interleaved Q buffer
        pqb->QiBuffer.n = celQi;

        // performance metrics: track the overall total number of Q sequences read by this QReaderU (baseQReader) instance
        m_nQ += pqb->DB.nQ;
    }

    // performance metrics
    InterlockedExchangeAdd( &AriocBase::aam.ms.BuildQwarps, m_hrt.GetElapsed(false) );

    // return true to indicate that at least one Qwarp in the QBatch instance contains input data
    return rval;
}
#pragma endregion
