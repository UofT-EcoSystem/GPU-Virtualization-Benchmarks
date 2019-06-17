/*
  SBFBuilderPaired.cpp

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
/// Reports paired-end alignments in SQL Server binary format
/// </summary>
SBFBuilderPaired::SBFBuilderPaired( QBatch* pqb ) : SBFBuilderBase(pqb)
{
    /* initialize invariant fields for unmapped SAM records */
    m_cbInvariantFields = static_cast<INT32>( sizeof m_invariantFields );
    memset( &m_invariantFields, 0, m_cbInvariantFields );
    m_invariantFields.rname = -1;
    m_invariantFields.mapq = pqb->pab->aas.ACP.mapqDefault;
    m_invariantFields.cigar.cch = 1;
    m_invariantFields.cigar.value = '*';
    m_invariantFields.rnext = -1;
}

/// [public] destructor
SBFBuilderPaired::~SBFBuilderPaired()
{
}
#pragma endregion

#pragma region private methods
/// [private] method emitYS
UINT32 SBFBuilderPaired::emitYS( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return 0;

#if TODO_IF_NEEDED
    return intToSBF<INT16>( pbuf, pQAI->pBH->V );
#endif
}
#pragma endregion

#pragma region virtual method implementations
/// [protected] method emitQNAME
UINT32 SBFBuilderPaired::emitQNAME( char* pbuf, INT64 sqId, QAI* pQAI )
{
    const UINT32 iMate = pQAI->qid & 1;     // 0: mate 1; 1: mate 2
    const UINT32 iMetadata = pQAI->qid / 2;

    char* p = pbuf + sizeof(INT16);
    INT16 cb = emitQNAMEpu( p, pQAI, iMate, iMetadata );
    *reinterpret_cast<INT16*>(pbuf) = cb;
    return cb + sizeof(INT16);
}

/// [protected] method emitFLAG
UINT32 SBFBuilderPaired::emitFLAG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // set the reverse-complement flag bits pertaining to the specified mapping
    if( ((pQAI->flags >> qaiShrRC) ^ pQAI->flags) & qaiRCr )
        m_flag |= sfRC;

    // set the reverse-complement flag bit pertaining to the opposite-end mapping
    if( ((m_pQAIx->flags >> qaiShrRC) ^ m_pQAIx->flags) & qaiRCr )
        m_flag |= sfOppRC;

    return intToSBF<UINT16>( pbuf, m_flag );
}

/// [protected] method emitRNEXT
UINT32 SBFBuilderPaired::emitRNEXT( char* pbuf, INT64 sqId, QAI* pQAI )
{
    INT32 rnext = (m_pQAIx->flags & qaiMapped) ? m_pQAIx->subId : -1;
    return intToSBF<INT16>( pbuf, rnext );
}

/// [protected] method emitPNEXT
UINT32 SBFBuilderPaired::emitPNEXT( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // report the 1-based position relative to the start of the forward reference strand, or -1 if the opposite mate is unmapped
    INT32 pnext = (m_pQAIx->flags & qaiMapped) ? m_pQAIx->Jf+1 : 0;
    return intToSBF<INT32>( pbuf, pnext );
}

/// [protected] method emitTLEN
UINT32 SBFBuilderPaired::emitTLEN( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT32>( pbuf, m_fragmentLength );
}

/// [protected] method emitQUAL
UINT32 SBFBuilderPaired::emitQUAL( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = pbuf + sizeof(INT16);

    // compute an index into the list of metadata offsets (see tuLoadM::readMetadata)
    UINT32 iMate = pQAI->qid & 1;
    UINT32 iMetadata = pQAI->qid / 2;

    INT16 cb = emitQUALfr( p, pQAI, iMate, iMetadata );
    *reinterpret_cast<INT16*>(pbuf) = cb;
    return cb + sizeof(INT16);
}

/// [protected] method emitRG
UINT32 SBFBuilderPaired::emitRG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // if there is no read-group info, emit an empty string
    if( m_pab->paamb->RGMgr.OfsRG.n == 0 )
    {
        *reinterpret_cast<INT16*>(pbuf) = 0;
        return sizeof(INT16);
    }

    // compute an index into the list of metadata offsets (see tuLoadM::readMetadata)
    UINT32 iMate = pQAI->qid & 1;
    UINT32 iMetadata = pQAI->qid / 2;

    char* p = pbuf + sizeof(INT16);
    INT16 cb = emitRGID( p, pQAI, iMate, iMetadata );
    *reinterpret_cast<INT16*>(pbuf) = cb;
    return cb + sizeof(INT16);
}

/// [public] method WriteRowPc
INT64 SBFBuilderPaired::WriteRowPc( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    m_arf = pPAI->arf;

    // lock the output buffer
    char* p0 = pw->Lock( m_pqb->pgi->deviceOrdinal, m_cbRow );
    char* p = p0;

    /* mate 1 */
    m_pQAIx = pPAI->pQAI2;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI1 );     // set the pair ID bit in the sqId
    sqId = setSqIdSecBit( sqId, pPAI );             // set the secondary mapping bit in the sqId

    // set up the SAM flag bits
    m_flag = sfPairMapped | sfPaired;
    m_flag |= (pPAI->pQAI1->flags & qaiParity) ? sfPair1 : sfPair0;
    if( !(pPAI->arf & arfPrimary) )
        m_flag |= sfSecondary;

    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI1->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI1->pBH );

    // the sign of the fragment length (TLEN) depends on which end is leftmost
    m_fragmentLength = (pPAI->pQAI1->Jf <= pPAI->pQAI2->Jf) ? static_cast<INT32>(pPAI->FragmentLength) : -static_cast<INT32>(pPAI->FragmentLength);

    // emit the SBF fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI1 );     // emit the f'th field
    p += emitYS( p, sqId, pPAI->pQAI2 );

    /* mate 2 */
    m_pQAIx = pPAI->pQAI1;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI2 );     // set the pair ID bit in the sqId

    // set up the SAM flag bits
    m_flag &= sfSecondary;      // reset all bits but the "secondary alignment" bit
    m_flag |= sfPairMapped | sfPaired;
    m_flag |= (pPAI->pQAI2->flags & qaiParity) ? sfPair1 : sfPair0;

    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI2->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI2->pBH );

    // the sign of the fragment length (TLEN) depends on which end is leftmost
    m_fragmentLength = (pPAI->pQAI2->Jf <= pPAI->pQAI1->Jf) ? static_cast<INT32>(pPAI->FragmentLength) : -static_cast<INT32>(pPAI->FragmentLength);

    // emit the SBF fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI2 );     // emit the f'th field
    p += emitYS( p, sqId, pPAI->pQAI1 );

    // compute the total number of bytes required for the rows
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows written
    return 2;
}

/// [public] method WriteRowPd
INT64 SBFBuilderPaired::WriteRowPd( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    m_arf = pPAI->arf;

    // lock the output buffer
    char* p0 = pw->Lock( m_pqb->pgi->deviceOrdinal, m_cbRow );
    char* p = p0;

    /* mate 1 */
    m_pQAIx = pPAI->pQAI2;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI1 );     // set the pair ID bit in the sqId
    sqId = setSqIdSecBit( sqId, pPAI );             // set the secondary mapping bit in the sqId

    // set up the SAM flag bits
    m_flag = sfPaired;    // (sfPairMapped (0x02) is not set)
    m_flag |= (pPAI->pQAI1->flags & qaiParity) ? sfPair1 : sfPair0;
    if( !(pPAI->arf & arfPrimary) )
        m_flag |= sfSecondary;

    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI1->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI1->pBH );

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

    // emit the SBF fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI1 );     // emit the f'th field
    p += emitYS( p, sqId, pPAI->pQAI2 );

    /* mate 2 */
    m_pQAIx = pPAI->pQAI1;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI2 );     // set the pair ID bit in the sqId

    // set up the SAM flag bits
    m_flag &= sfSecondary;      // reset all bits but the "secondary alignment" bit
    m_flag |= sfPaired;
    m_flag |= (pPAI->pQAI2->flags & qaiParity) ? sfPair1 : sfPair0;

    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI2->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI2->pBH );

    if( pPAI->pQAI1->subId == pPAI->pQAI2->subId )
    {
        // the sign of the fragment length (TLEN) depends on which end is leftmost
        m_fragmentLength = (pPAI->pQAI2->Jf <= pPAI->pQAI1->Jf) ? static_cast<INT32>(pPAI->FragmentLength) : -static_cast<INT32>(pPAI->FragmentLength);
    }

    // emit the SBF fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI2 );     // emit the f'th field
    p += emitYS( p, sqId, pPAI->pQAI1 );

    // compute the total number of bytes required for the rows
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows written
    return 2;
}

/// [public] method writeMate1Unpaired
char* SBFBuilderPaired::writeMate1Unpaired( char* p, INT64 sqId, PAI* pPAI )
{
    m_pQAIx = pPAI->pQAI2;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI1 );     // set the pair ID bit in the sqId
    sqId = setSqIdSecBit( sqId, pPAI->pQAI1 );      // set the secondary mapping bit in the sqId

    // set up the SAM flag bits
    m_flag = sfPaired;
    m_flag |= (pPAI->pQAI1->flags & qaiParity) ? sfPair1 : sfPair0;
    if( (pPAI->pQAI1->flags & qaiBest) == 0 )
        m_flag |= sfSecondary;

    // if the pair is unmapped, the opposite mate must be the one that is unmapped
    if( pPAI->arf & arfUnmapped )
        m_flag |= sfOppMateUnmapped;

    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI1->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI1->pBH );

    // emit the SBF fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI1 );     // emit the f'th field

     return p;
}

/// [public] method writeMate2Unpaired
char* SBFBuilderPaired::writeMate2Unpaired( char* p, INT64 sqId, PAI* pPAI )
{
    m_pQAIx = pPAI->pQAI1;                          // save a reference to the opposite mate's alignment info
    sqId = setSqIdPairBit( sqId, pPAI->pQAI2 );     // set the pair ID bit in the sqId
    sqId = setSqIdSecBit( sqId, pPAI->pQAI2 );      // set the secondary mapping bit in the sqId

    // set up the SAM flag bits
    m_flag = sfPaired;
    m_flag |= (pPAI->pQAI2->flags & qaiParity) ? sfPair1 : sfPair0;
    if( (pPAI->pQAI2->flags & qaiBest) == 0 )
        m_flag |= sfSecondary;

    // if the pair is unmapped, the opposite mate must be the one that is unmapped
    if( pPAI->arf & arfUnmapped )
        m_flag |= sfOppMateUnmapped;

    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI2->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI2->pBH );

    // emit the SBF fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI2 );     // emit the f'th field

    return p;
}

/// [public] method writeMatesUnpaired
INT64 SBFBuilderPaired::writeMatesUnpaired( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    INT64 nRowsEmitted = 0;

    m_arf = pPAI->arf;
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

            // set up the SAM flag bits
            m_flag = sfMateUnmapped | sfPaired;
            m_flag |= (pPAI->pQAI1->flags & qaiParity) ? sfPair1 : sfPair0;
            if( pPAI->arf & arfMate2Unmapped )
                m_flag |= sfOppMateUnmapped;

            /* Build a row of SAM data for the unmapped read; only the following fields contain variable data:
                - QNAME
                - FLAG
                - SEQ
                - QUAL
                - id

                The other required fields are hardwired (see the constructor); the optional fields are zeroed here.
            */
            p += emitQNAME( p, sqId, pPAI->pQAI1 );
            p += emitFLAG( p, sqId, pPAI->pQAI1 );
            memcpy( p, &m_invariantFields, m_cbInvariantFields );
            if( !(pPAI->arf & arfMate2Unmapped) )
            {
                reinterpret_cast<SBFIF*>(p)->rnext = m_pQAIx->subId;
                reinterpret_cast<SBFIF*>(p)->pnext = m_pQAIx->Jf + 1;
            }
            p += m_cbInvariantFields;
            p += emitSEQ( p, sqId, pPAI->pQAI1 );
            p += emitQUAL( p, sqId, pPAI->pQAI1 );
            p += emitid( p, sqId, pPAI->pQAI1 );
            p += emitNullOptionalFields( p );
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

            // set up the SAM flag bits
            m_flag = sfMateUnmapped | sfPaired;
            m_flag |= (pPAI->pQAI2->flags & qaiParity) ? sfPair1 : sfPair0;
            if( pPAI->arf & arfMate1Unmapped )
                m_flag |= sfOppMateUnmapped;

            // build a row of SAM data for the unmapped read
            p += emitQNAME( p, sqId, pPAI->pQAI2 );
            p += emitFLAG( p, sqId, pPAI->pQAI2 );
            memcpy( p, &m_invariantFields, m_cbInvariantFields );
            if( !(pPAI->arf & arfMate1Unmapped) )
            {
                reinterpret_cast<SBFIF*>(p)->rnext = m_pQAIx->subId;
                reinterpret_cast<SBFIF*>(p)->pnext = m_pQAIx->Jf + 1;
            }
            p += m_cbInvariantFields;
            p += emitSEQ( p, sqId, pPAI->pQAI2 );
            p += emitQUAL( p, sqId, pPAI->pQAI2 );
            p += emitid( p, sqId, pPAI->pQAI2 );
            p += emitNullOptionalFields( p );
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

/// [public] method WriteRowPr
INT64 SBFBuilderPaired::WriteRowPr( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    return writeMatesUnpaired( pw, sqId, pPAI );
}

/// [public] method WriteRowPu
INT64 SBFBuilderPaired::WriteRowPu( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    return writeMatesUnpaired( pw, sqId, pPAI );
}
#pragma endregion
#pragma warning ( pop )
