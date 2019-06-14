/*
  KMHBuilderPaired.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Reports kmer-hashed paired-end read sequences in SQL Server binary format
/// </summary>
KMHBuilderPaired::KMHBuilderPaired( QBatch* pqb ) : KMHBuilderBase(pqb)
{
}

/// [public] destructor
KMHBuilderPaired::~KMHBuilderPaired()
{
}
#pragma endregion

#pragma region virtual method implementations
/// [protected] method emitBRLEQ
UINT32 KMHBuilderPaired::emitBRLEQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // compute an index into the list of metadata offsets (see tuLoadM::readMetadata)
    UINT32 iMate = pQAI->qid & 1;
    UINT32 iMetadata = pQAI->qid / 2;

    return emitBRLEQfr( pbuf, pQAI, iMate, iMetadata );
}

/// [protected] method emitRGID
UINT32 KMHBuilderPaired::emitRGID( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // compute an index into the list of metadata offsets (see tuLoadM::readMetadata)
    UINT32 iMate = pQAI->qid & 1;
    UINT32 iMetadata = pQAI->qid / 2;

    UINT8 rgId = getRGIDfromReadMetadata( pQAI, iMate, iMetadata );
    return intToSBF<UINT8>( pbuf, rgId );
}

/// [public] method WriteRowPc
INT64 KMHBuilderPaired::WriteRowPc( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    // if either mate is flagged as the "primary" mapping, emit both mates; otherwise, emit nothing
    if( (pPAI->pQAI1->flags | pPAI->pQAI2->flags) & qaiBest )
    {
        if( pw->IsActive )
        {
            // lock the output buffer
            char* p0 = pw->Lock( m_pqb->pgi->deviceOrdinal, m_cbRow );
            char* p = p0;

            // set up the mapping category bits
            m_mapCat = mcConcordant;

            // save a reference to the pair alignment info
            m_pPAI = pPAI;

            /* mate 1 */
            m_pQAIx = pPAI->pQAI2;                          // save a reference to the opposite mate's alignment info
            sqId = setSqIdPairBit( sqId, pPAI->pQAI1 );     // set the pair ID bit in the sqId
            sqId = setSqIdSecBit( sqId, pPAI );             // set the secondary mapping bit in the sqId

            // emit the KMH fields
            for( INT16 f = 0; f < m_nEmitFields; ++f )
                p += (this->*(m_emitField[f]))(p, sqId, pPAI->pQAI1);     // emit the f'th field

            /* mate 2 */
            m_pQAIx = pPAI->pQAI1;                          // save a reference to the opposite mate's alignment info
            sqId = setSqIdPairBit( sqId, pPAI->pQAI2 );     // set the pair ID bit in the sqId

            // emit the KMH fields
            for( INT16 f = 0; f < m_nEmitFields; ++f )
                p += (this->*(m_emitField[f]))(p, sqId, pPAI->pQAI2);     // emit the f'th field

            // compute the total number of bytes required for the rows
            UINT32 cb = static_cast<UINT32>(p - p0);

            // release the output buffer
            pw->Release( m_pqb->pgi->deviceOrdinal, cb );
        }

        // return the number of rows emitted (even to the bitbucket)
        return 2;
    }

    // return the number of rows emitted
    return 0;
}

/// [public] method WriteRowPd
INT64 KMHBuilderPaired::WriteRowPd( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    if( pw->IsActive )
    {
        // lock the output buffer
        char* p0 = pw->Lock( m_pqb->pgi->deviceOrdinal, m_cbRow );
        char* p = p0;

        // set up the mapping category bits
        m_mapCat = mcDiscordant;

        /* mate 1 */
        m_pQAIx = pPAI->pQAI2;                          // save a reference to the opposite mate's alignment info
        sqId = setSqIdPairBit( sqId, pPAI->pQAI1 );     // set the pair ID bit in the sqId
        sqId = setSqIdSecBit( sqId, pPAI );             // set the secondary mapping bit in the sqId

    #if TODO_CHOP_IF_UNUSED
        // if necessary, compute a reverse BRLEA
        if( pPAI->pQAI1->flags & qaiRCr )
            reverseBRLEA( pPAI->pQAI1->pBH );
    #endif

        // emit the KMH fields
        for( INT16 f=0; f<m_nEmitFields; ++f )
            p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI1 );     // emit the f'th field

        /* mate 2 */
        m_pQAIx = pPAI->pQAI1;                          // save a reference to the opposite mate's alignment info
        sqId = setSqIdPairBit( sqId, pPAI->pQAI2 );     // set the pair ID bit in the sqId

        // emit the KMH fields
        for( INT16 f=0; f<m_nEmitFields; ++f )
            p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI2 );     // emit the f'th field

        // compute the total number of bytes required for the rows
        UINT32 cb = static_cast<UINT32>(p - p0);

        // release the output buffer
        pw->Release( m_pqb->pgi->deviceOrdinal, cb );
    }

    // return the number of rows written
    return 2;
}

/// [public] method writeMate1Unpaired
char* KMHBuilderPaired::writeMate1Unpaired( char* p, INT64 sqId, PAI* pPAI )
{
    /* Emit KMH info for mate 1 where there is no concordant or discordant paired-end mapping. */

    m_pQAIx = pPAI->pQAI2;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI1 );     // set the pair ID bit in the sqId

    // set up the mapping category bits
    if ((pPAI->arf & (arfMate2Unmapped | arfWriteMate2)) == (arfMate2Unmapped | arfWriteMate2))
        m_mapCat |= mcOppMateUnmapped;

    // emit the KMH fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI1 );     // emit the f'th field

    return p;
}

/// [public] method writeMate2Unpaired
char* KMHBuilderPaired::writeMate2Unpaired( char* p, INT64 sqId, PAI* pPAI )
{
    /* Emit KMH info for mate 2 where there is no concordant or discordant paired-end mapping. */

    m_pQAIx = pPAI->pQAI1;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI2 );     // set the pair ID bit in the sqId

    // set up the mapping category bits
    if ((pPAI->arf & (arfMate1Unmapped | arfWriteMate1)) == (arfMate1Unmapped | arfWriteMate1))
        m_mapCat |= mcOppMateUnmapped;

    // emit the KMH fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI2 );     // emit the f'th field

    return p;
}

/// [public] method writeMatesUnpaired
INT64 KMHBuilderPaired::writeMatesUnpaired( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    /* Emit a row for a mate if the "write" flag is set, and one of the following is true:
        - the current pair of mates has one mapped and one unmapped mate, and the mapped mate
            is flagged as the "primary" mapping
        - both mates are unmapped
    */
    INT64 nRowsEmitted = 0;
    if( ((pPAI->pQAI1->flags | pPAI->pQAI2->flags) & qaiBest) ||            // one mate mapped, one unmapped
        (((pPAI->pQAI1->flags | pPAI->pQAI2->flags) & qaiMapped) == 0) )    // both mates unmapped
    {
        if( pw->IsActive )
        {
            // lock the output buffer
            char* p0 = pw->Lock( m_pqb->pgi->deviceOrdinal, m_cbRow );
            char* p = p0;

            /* mate 1 */
            if( pPAI->arf & arfWriteMate1 )
            {
                m_mapCat = (pPAI->arf & arfMate1Unmapped) ? mcUnmapped : mcNull;
                p = writeMate1Unpaired( p, sqId, pPAI );
            }
        
            /* mate 2 */
            if( pPAI->arf & arfWriteMate2 )
            {
                m_mapCat = (pPAI->arf & arfMate2Unmapped) ? mcUnmapped : mcNull;
                p = writeMate2Unpaired( p, sqId, pPAI );
            }

            // compute the total number of bytes required for the rows
            UINT32 cb = static_cast<UINT32>(p - p0);

            // release the output buffer
            pw->Release( m_pqb->pgi->deviceOrdinal, cb );
        }

        // compute the number of rows emitted (even to the bitbucket)
        if( pPAI->arf & arfWriteMate1 )
            ++nRowsEmitted;
        if( pPAI->arf & arfWriteMate2 )
            ++nRowsEmitted;
    }

    return nRowsEmitted;
}

/// [public] method WriteRowPr
INT64 KMHBuilderPaired::WriteRowPr( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    // set up the mapping category bits
    m_mapCat = mcRejected;

    return writeMatesUnpaired( pw, sqId, pPAI );
}

/// [public] method WriteRowPu
INT64 KMHBuilderPaired::WriteRowPu( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    // reset the mapping category bits; they will be set in writeMatesUnpaired()
    m_mapCat = mcNull;

    return writeMatesUnpaired( pw, sqId, pPAI );
}
#pragma endregion
