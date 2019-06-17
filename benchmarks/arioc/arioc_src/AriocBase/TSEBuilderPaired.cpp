/*
  TSEBuilderPaired.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma warning ( push )
#pragma warning ( disable : 4996 )		// don't nag us about strcpy being "unsafe"

#pragma region constructors and destructor
/// <summary>
/// Reports paired-end alignments for the Terabase Search Engine in SQL Server binary format
/// </summary>
TSEBuilderPaired::TSEBuilderPaired( QBatch* pqb ) : TSEBuilderBase(pqb)
{
}

/// [public] destructor
TSEBuilderPaired::~TSEBuilderPaired()
{
}
#pragma endregion

#pragma region private methods
/// [private] method writeMate1Unpaired
char* TSEBuilderPaired::writeMate1Unpaired( char* p, INT64 sqId, PAI* pPAI )
{
    /* Emit a mapping for mate 1 where there is no concordant or discordant paired-end mapping. */

    m_pQAIx = pPAI->pQAI2;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI1 );     // set the pair ID bit in the sqId
    sqId = setSqIdSecBit( sqId, pPAI->pQAI1 );      // set the secondary mapping bit in the sqId

    // set up the SAM flag bits
    m_flag = sfPaired;
    m_flag |= (pPAI->pQAI1->flags & qaiParity) ? sfPair1 : sfPair0;
    if( (pPAI->pQAI1->flags & qaiBest) == 0 )
        m_flag |= sfSecondary;

    // we're emitting a mapping for the current mate, so if there is no paired mapping, the opposite mate must be the one that is unmapped
    if( pPAI->arf & arfUnmapped )
    {
        m_flag |= sfOppMateUnmapped;
        m_mapCat |= mcOppMateUnmapped;
    }

#if TODO_CHOP_IF_UNUSED
    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI1->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI1->pBH );
#endif

    // emit the TSE fields
    for( INT16 f = 0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))(p, sqId, pPAI->pQAI1);     // emit the f'th field

    return p;
}

/// [private] method writeMate2Unpaired
char* TSEBuilderPaired::writeMate2Unpaired( char* p, INT64 sqId, PAI* pPAI )
{
    /* Emit a mapping for mate 2 where there is no concordant or discordant paired-end mapping. */

    m_pQAIx = pPAI->pQAI1;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI2 );     // set the pair ID bit in the sqId
    sqId = setSqIdSecBit( sqId, pPAI->pQAI2 );      // set the secondary mapping bit in the sqId

    // set up the SAM flag bits
    m_flag = sfPaired;
    m_flag |= (pPAI->pQAI2->flags & qaiParity) ? sfPair1 : sfPair0;
    if( (pPAI->pQAI2->flags & qaiBest) == 0 )
        m_flag |= sfSecondary;

    // we're emitting a mapping for the current mate, so if there is no paired mapping, the opposite mate must be the one that is unmapped
    if( pPAI->arf & arfUnmapped )
    {
        m_flag |= sfOppMateUnmapped;
        m_mapCat |= mcOppMateUnmapped;
    }

#if TODO_CHOP_IF_UNUSED
    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI2->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI2->pBH );
#endif

    // emit the TSE fields
    for( INT16 f = 0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))(p, sqId, pPAI->pQAI2);     // emit the f'th field

    return p;
}

/// [private] method writeMatesUnpaired
INT64 TSEBuilderPaired::writeMatesUnpaired( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    INT64 nRowsEmitted = 0;

    m_fragmentLength = 0;

    // lock the output buffer
    char* p0 = pw->Lock( m_pqb->pgi->deviceOrdinal, m_cbRow );
    char* p = p0;

    /* mate 1 */
    if( pPAI->arf & arfWriteMate1 )
    {
        if( pPAI->arf & arfMate1Unmapped )
        {
            m_pQAIx = pPAI->pQAI2;                          // save a reference to the opposite mate's alignment info (which may be null)
            sqId = setSqIdPairBit( sqId, pPAI->pQAI1 );     // set the pair ID bit in the sqId

            // build a row of TSE data for the unmapped read
            TSE_FIXED_FIELDS* pi = reinterpret_cast<TSE_FIXED_FIELDS*>(p);
            memset( pi, 0, sizeof( TSE_FIXED_FIELDS ) );

            // sequence ID
            pi->sqId = sqId;

            // set up the SAM flag bits
            pi->FLAG = sfMateUnmapped | sfPaired;
            pi->FLAG |= (pPAI->pQAI1->flags & qaiParity) ? sfPair1 : sfPair0;
            if( (pPAI->arf & (arfMate2Unmapped|arfWriteMate2)) == (arfMate2Unmapped|arfWriteMate2) )
            {
                pi->FLAG |= sfOppMateUnmapped;
                pi->mapCat = mcUnmapped | mcOppMateUnmapped;
            }
            else
                pi->mapCat = mcUnmapped;

            // MAPQ
            pi->MAPQ = m_pab->aas.ACP.mapqDefault;

            // rgId
            pi->rgId = getRGIDfromReadMetadata( pPAI->pQAI1, pPAI->pQAI1->qid&1, pPAI->pQAI1->qid/2 );

            // sqLen
            pi->sqLen = pPAI->pQAI1->N;

            // hash64
            pi->hash64 = computeA21Hash64( pPAI->pQAI1 );

            // emit base quality scores
            p = reinterpret_cast<char*>(&pi->minBQS);
            p += emitBRLEQ( p, sqId, pPAI->pQAI1 );

            // emit sequence
            p += emitBRLEE( p, sqId, pPAI->pQAI1 );
            p += emitSEQ( p, sqId, pPAI->pQAI1 );
        }
        else
            p = writeMate1Unpaired( p, sqId, pPAI );

        nRowsEmitted++;
    }

    /* mate 2 */
    if( pPAI->arf & arfWriteMate2 )
    {
        if( pPAI->arf & arfMate2Unmapped )
        {
            m_pQAIx = pPAI->pQAI1;                          // save a reference to the opposite mate's alignment info (which may be null)
            sqId = setSqIdPairBit( sqId, pPAI->pQAI2 );     // set the pair ID bit in the sqId

            // build a row of TSE data for the unmapped read
            TSE_FIXED_FIELDS* pi = reinterpret_cast<TSE_FIXED_FIELDS*>(p);
            memset( pi, 0, sizeof( TSE_FIXED_FIELDS ) );

            // sequence ID
            pi->sqId = sqId;

            // set up the SAM flag bits
            pi->FLAG = sfMateUnmapped | sfPaired;
            pi->FLAG |= (pPAI->pQAI2->flags & qaiParity) ? sfPair1 : sfPair0;
            if( (pPAI->arf & (arfMate1Unmapped|arfWriteMate1)) == (arfMate1Unmapped|arfWriteMate1) )
            {
                pi->FLAG |= sfOppMateUnmapped;
                pi->mapCat = mcUnmapped | mcOppMateUnmapped;
            }
            else
                pi->mapCat = mcUnmapped;

            // MAPQ
            pi->MAPQ = m_pab->aas.ACP.mapqDefault;

            // rgId
            pi->rgId = getRGIDfromReadMetadata( pPAI->pQAI2, pPAI->pQAI2->qid&1, pPAI->pQAI2->qid/2 );

            // sqLen
            pi->sqLen = pPAI->pQAI2->N;

            // hash64
            pi->hash64 = computeA21Hash64( pPAI->pQAI2 );

            // emit base quality scores
            p = reinterpret_cast<char*>(&pi->minBQS);
            p += emitBRLEQ( p, sqId, pPAI->pQAI2 );

            // emit sequence
            p += emitBRLEE( p, sqId, pPAI->pQAI2 );
            p += emitSEQ( p, sqId, pPAI->pQAI2 );
        }
        else
            p = writeMate2Unpaired( p, sqId, pPAI );

        nRowsEmitted++;
    }

    // compute the total number of bytes required for the rows
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    return nRowsEmitted;
}
#pragma endregion

#pragma region virtual method implementations
/// [protected] method emitFLAG
UINT32 TSEBuilderPaired::emitFLAG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // set the reverse-complement flag bits pertaining to the specified mapping
    if( ((pQAI->flags >> qaiShrRC) ^ pQAI->flags) & qaiRCr )
        m_flag |= sfRC;

    // set the reverse-complement flag bit pertaining to the opposite-end mapping
    if( ((m_pQAIx->flags >> qaiShrRC) ^ m_pQAIx->flags) & qaiRCr )
        m_flag |= sfOppRC;

    return intToSBF<UINT16>( pbuf, m_flag );
}

/// [protected] method emitBRLEQ
UINT32 TSEBuilderPaired::emitBRLEQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // compute an index into the list of metadata offsets (see tuLoadM::readMetadata)
    UINT32 iMate = pQAI->qid & 1;
    UINT32 iMetadata = pQAI->qid / 2;

    return emitBRLEQfr( pbuf, pQAI, iMate, iMetadata );
}

/// [protected] method emitRGID
UINT32 TSEBuilderPaired::emitRGID( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // compute an index into the list of metadata offsets (see tuLoadM::readMetadata)
    UINT32 iMate = pQAI->qid & 1;
    UINT32 iMetadata = pQAI->qid / 2;

    UINT8 rgId = getRGIDfromReadMetadata( pQAI, iMate, iMetadata );
    return intToSBF<UINT8>( pbuf, rgId );
}

/// [public] method WriteRowPc
INT64 TSEBuilderPaired::WriteRowPc( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
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

    // set up the SAM flag bits
    m_flag = sfPairMapped | sfPaired;
    m_flag |= (pPAI->pQAI1->flags & qaiParity) ? sfPair1 : sfPair0;
    if( !(pPAI->arf & arfPrimary) )
        m_flag |= sfSecondary;

    // the sign of the fragment length (TLEN) depends on which end is leftmost
    m_fragmentLength = (pPAI->pQAI1->Jf <= pPAI->pQAI2->Jf) ? static_cast<INT32>(pPAI->FragmentLength) : -static_cast<INT32>(pPAI->FragmentLength);

    // emit the TSE fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI1 );     // emit the f'th field

    /* mate 2 */
    m_pQAIx = pPAI->pQAI1;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI2 );     // set the pair ID bit in the sqId

    // set up the SAM flag bits
    m_flag &= sfSecondary;      // reset all bits but the "secondary alignment" bit
    m_flag |= sfPairMapped | sfPaired;
    m_flag |= (pPAI->pQAI2->flags & qaiParity) ? sfPair1 : sfPair0;

    // the sign of the fragment length (TLEN) depends on which end is leftmost
    m_fragmentLength = (pPAI->pQAI2->Jf <= pPAI->pQAI1->Jf) ? static_cast<INT32>(pPAI->FragmentLength) : -static_cast<INT32>(pPAI->FragmentLength);

    // emit the TSE fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI2 );     // emit the f'th field

    // compute the total number of bytes required for the rows
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows written
    return 2;
}

/// [public] method WriteRowPd
INT64 TSEBuilderPaired::WriteRowPd( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
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

    // set up the SAM flag bits
    m_flag = sfPaired;    // (sfPairMapped (0x02) is not set)
    m_flag |= (pPAI->pQAI1->flags & qaiParity) ? sfPair1 : sfPair0;
    if( !(pPAI->arf & arfPrimary) )
        m_flag |= sfSecondary;

#if TODO_CHOP_IF_UNUSED
    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI1->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI1->pBH );
#endif

    if( pPAI->pQAI1->subId == pPAI->pQAI2->subId )
    {
        // the sign of the fragment length (TLEN) depends on which end is leftmost
        m_fragmentLength = (pPAI->pQAI1->Jf <= pPAI->pQAI2->Jf) ? static_cast<INT32>(pPAI->FragmentLength) : -static_cast<INT32>(pPAI->FragmentLength);
    }
    else
    {
        // there is no reportable fragment length when the ends map to different reference sequences
        m_fragmentLength = 0;
    }

    // emit the TSE fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI1 );     // emit the f'th field

    /* mate 2 */
    m_pQAIx = pPAI->pQAI1;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI2 );     // set the pair ID bit in the sqId

    // set up the SAM flag bits
    m_flag &= sfSecondary;      // reset all bits but the "secondary alignment" bit
    m_flag |= sfPaired;
    m_flag |= (pPAI->pQAI2->flags & qaiParity) ? sfPair1 : sfPair0;

#if TODO_CHOP_IF_UNUSED
    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI2->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI2->pBH );
#endif

    if( pPAI->pQAI1->subId == pPAI->pQAI2->subId )
    {
        // the sign of the fragment length (TLEN) depends on which end is leftmost
        m_fragmentLength = (pPAI->pQAI2->Jf <= pPAI->pQAI1->Jf) ? static_cast<INT32>(pPAI->FragmentLength) : -static_cast<INT32>(pPAI->FragmentLength);
    }

    // emit the TSE fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI2 );     // emit the f'th field

    // compute the total number of bytes required for the rows
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows written
    return 2;
}

/// [public] method WriteRowPr
INT64 TSEBuilderPaired::WriteRowPr( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    // set up the mapping category bits
    m_mapCat = mcRejected;

    return writeMatesUnpaired( pw, sqId, pPAI );
}

/// [public] method WriteRowPu
INT64 TSEBuilderPaired::WriteRowPu( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    // reset the mapping category bits; they will be set in writeMatesUnpaired()
    m_mapCat = mcNull;

    return writeMatesUnpaired( pw, sqId, pPAI );
}
#pragma endregion
#pragma warning ( pop )
