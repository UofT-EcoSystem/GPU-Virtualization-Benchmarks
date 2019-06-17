/*
  SAMBuilderPaired.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma warning ( push )
#pragma warning( disable:4996 )     // (don't nag us about strcpy being "unsafe")

#pragma region constructors and destructor
/// <summary>
/// Implements SAM-formatted file output for paired-end reads
/// </summary>
SAMBuilderPaired::SAMBuilderPaired( QBatch* pqb ) : SAMBuilderBase(pqb)
{
    /* build a string of invariant fields for unmapped SAM pairs where one mate is unmapped:
        - RNAME:    *
        - POS:      0
        - MAPQ:     0 (by default; may be overridden in .cfg file)
        - CIGAR:    *
    */
    m_cbInvariantFields1 = sprintf_s( m_invariantFields1, sizeof m_invariantFields1, "*\t0\t%u\t*\t", pqb->pab->aas.ACP.mapqDefault );

    /* build a string of invariant fields for unmapped SAM pairs where both mates are unmapped:
        - RNAME:    *
        - POS:      0
        - MAPQ:     0 (by default; may be overridden in .cfg file)
        - CIGAR:    *
        - RNEXT:    *
        - PNEXT:    0
        - TLEN:     0
    */
    m_cbInvariantFields2 = sprintf_s( m_invariantFields2, sizeof m_invariantFields2, "*\t0\t%u\t*\t*\t0\t0\t", pqb->pab->aas.ACP.mapqDefault );
}

/// [public] destructor
SAMBuilderPaired::~SAMBuilderPaired()
{
}
#pragma endregion

#pragma region private methods
/// [private] method emitYS
UINT32 SAMBuilderPaired::emitYS( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = strcpy( pbuf, "YS:i:" ) + 5;
    p += u32ToString( p, pQAI->pBH->V );
    *(p++) = '\t';
    return static_cast<INT32>(p - pbuf);
}
#pragma endregion

#pragma region virtual method implementations
/// [protected] method emitQNAME
UINT32 SAMBuilderPaired::emitQNAME( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // compute an index into the list of metadata offsets (see tuLoadM::readMetadata)
    UINT32 iMate = pQAI->qid & 1;
    UINT32 iMetadata = pQAI->qid / 2;
    UINT32 cb = emitQNAMEpu( pbuf, pQAI, iMate, iMetadata );
    pbuf[cb] = '\t';
    return cb + 1;
}

/// [protected] method emitFLAG
UINT32 SAMBuilderPaired::emitFLAG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // set the reverse-complement flag bits pertaining to the specified mapping
    if( ((pQAI->flags >> qaiShrRC) ^ pQAI->flags) & qaiRCr )
        m_flag |= sfRC;

    // set the reverse-complement flag bit pertaining to the opposite-end mapping
    if( ((m_pQAIx->flags >> qaiShrRC) ^ m_pQAIx->flags) & qaiRCr )
        m_flag |= sfOppRC;

    char* p = pbuf + u32ToString( pbuf, m_flag );
    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitRNEXT
UINT32 SAMBuilderPaired::emitRNEXT( char* pbuf, INT64 sqId, QAI* pQAI )
{
    if( m_pQAIx->flags & qaiMapped )
    {
        if( pQAI->subId == m_pQAIx->subId )
        {
            strcpy( pbuf, "=\t" );
            return 2;
        }

        /* At this point the mates map to two different R sequence subunits. */
        return emitSubIdAsRNAME( pbuf, m_pQAIx->subId );
    }

    /* At this point the other end is unmapped. */

    // report RNEXT = *
    *reinterpret_cast<UINT16*>(pbuf) = *reinterpret_cast<UINT16*>(const_cast<char*>("*\t"));
    return 2;
}

/// [protected] method emitPNEXT
UINT32 SAMBuilderPaired::emitPNEXT( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // if the other end mapped ...
    if( m_pQAIx->flags & qaiMapped )
    {
        // report the 1-based position relative to the start of the forward reference strand
        char* p = pbuf + u32ToString( pbuf, m_pQAIx->Jf+1 );
        *(p++) = '\t';
        return static_cast<UINT32>(p - pbuf);
    }

    /* At this point the other end is unmapped. */

    // report PNEXT = 0
    *reinterpret_cast<UINT16*>(pbuf) = *reinterpret_cast<UINT16*>(const_cast<char*>("0\t"));
    return 2;
}

/// [protected] method emitTLEN
UINT32 SAMBuilderPaired::emitTLEN( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = pbuf + i32ToString( pbuf, m_fragmentLength );
    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitQUAL
UINT32 SAMBuilderPaired::emitQUAL( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // compute an index into the list of metadata offsets (see tuLoadM::readMetadata)
    UINT32 iMate = pQAI->qid & 1;
    UINT32 iMetadata = pQAI->qid / 2;

    UINT32 cb = emitQUALfr( pbuf, pQAI, iMate, iMetadata );
    pbuf[cb] = '\t';    // emit the trailing tab separator
    return cb + 1;
}

/// [protected] method emitMQ
UINT32 SAMBuilderPaired::emitMQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = strcpy( pbuf, "MQ:i:" ) + 5;
    UINT32 oppMapq = static_cast<UINT32>((m_flag & sfSecondary) ? m_pQAIx->mapQp2 : m_pQAIx->mapQ);
    p += u32ToString( p, oppMapq );
    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitRG
UINT32 SAMBuilderPaired::emitRG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // do nothing if there is no read-group info
    if( m_pab->paamb->RGMgr.OfsRG.n == 0 )
        return 0;

    strcpy( pbuf, "RG:Z:" );
    char* p = pbuf + 5;         // 5: strlen("RG:Z:")

    // compute an index into the list of metadata offsets (see tuLoadM::readMetadata)
    UINT32 iMate = pQAI->qid & 1;
    UINT32 iMetadata = pQAI->qid / 2;
    UINT32 cb = emitRGID( p, pQAI, iMate, iMetadata );
    if( cb == 0 )
        return 0;
    
    p[cb] = '\t';
    return cb + 6;      // 6: strlen("RG:Z:") + terminal tab
}

/// [protected] method emitXM
UINT32 SAMBuilderPaired::emitXM( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // do nothing if we are not doing bsDNA alignments
    if( !m_baseConvCT )
        return 0;

    strcpy( pbuf, "XM:Z:" );
    char* p = pbuf + 5;         // 5: strlen("XM:Z:")

    UINT32 cb = emitXMfr( p, pQAI );
    if( cb == 0 )
        return 0;
    
    p[cb] = '\t';
    return cb + 6;      // 6: strlen("XM:Z:") + terminal tab
}

/// [protected] method emitXG
UINT32 SAMBuilderPaired::emitXG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // do nothing if we are not doing bsDNA alignments
    if( !m_baseConvCT )
        return 0;

    if( pQAI->flags & qaiRCr )
        strcpy( pbuf, "XG:Z:GA\t" );
    else
        strcpy( pbuf, "XG:Z:CT\t" );

    return 8;         // 8: strlen("XG:Z:??\t")
}

/// [protected] method emitXR
UINT32 SAMBuilderPaired::emitXR( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // do nothing if we are not doing bsDNA alignments
    if( !m_baseConvCT )
        return 0;

    if( pQAI->flags & qaiRCq )
        strcpy( pbuf, "XR:Z:GA\t" );
    else
        strcpy( pbuf, "XR:Z:CT\t" );

    return 8;         // 8: strlen("XR:Z:??\t")
}

/// [public] method WriteRowPc
INT64 SAMBuilderPaired::WriteRowPc( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    m_YT = "YT:Z:CP\t";     // CP: concordant pair
    m_arf = pPAI->arf;

    /* Lock the output buffer.

       We ensure that both mates of a pair are written as adjacent rows.  We do not ensure that secondary mappings
        for the same pair are adjacent to their primary mappings, which would require a call to baseARowWriter->Lock
        across two calls to WriteRowPc...
    */
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

    // emit the SAM fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI1 );     // emit the f'th field
    p += emitYS( p, sqId, pPAI->pQAI2 );

    // replace the terminal tab with a Unix-style newline (i.e. an end-of-line character)
    p[-1] = '\n';

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

    // emit the SAM fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI2 );     // emit the f'th field
    p += emitYS( p, sqId, pPAI->pQAI1 );

    // replace the terminal tab with a Unix-style newline (i.e. an end-of-line character)
    p[-1] = '\n';

    // compute the total number of bytes required for the rows
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows written
    return 2;
}

/// [public] method WriteRowPd
INT64 SAMBuilderPaired::WriteRowPd( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    m_YT = "YT:Z:DP\t";     // DP: discordant pair
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

    // emit the SAM fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI1 );     // emit the f'th field
    p += emitYS( p, sqId, pPAI->pQAI2 );

    // replace the terminal tab with a Unix-style newline (i.e. an end-of-line character)
    p[-1] = '\n';

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

    // emit the SAM fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI2 );     // emit the f'th field
    p += emitYS( p, sqId, pPAI->pQAI1 );

    // replace the terminal tab with a Unix-style newline (i.e. an end-of-line character)
    p[-1] = '\n';

    // compute the total number of bytes required for the rows
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows written
    return 2;
}

/// [public] method writeMate1Unpaired
char* SAMBuilderPaired::writeMate1Unpaired( char* p, INT64 sqId, PAI* pPAI )
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
    if( (pPAI->arf & arfUnmapped) || ((m_pQAIx->flags & qaiMapped) == 0) )
        m_flag |= sfOppMateUnmapped;

    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI1->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI1->pBH );

    // emit the SAM fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI1 );     // emit the f'th field

    // replace the terminal tab with a Unix-style newline (i.e. an end-of-line character)
    p[-1] = '\n';

    return p;
}

/// [public] method writeMate2Unpaired
char* SAMBuilderPaired::writeMate2Unpaired( char* p, INT64 sqId, PAI* pPAI )
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
    if( (pPAI->arf & arfUnmapped) || ((m_pQAIx->flags & qaiMapped) == 0) )
        m_flag |= sfOppMateUnmapped;

    // if necessary, compute a reverse BRLEA
    if( pPAI->pQAI2->flags & qaiRCr )
        reverseBRLEA( pPAI->pQAI2->pBH );

    // emit the SAM fields
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pPAI->pQAI2 );     // emit the f'th field
     
    // replace the terminal tab with a Unix-style newline (i.e. an end-of-line character)
    p[-1] = '\n';

    return p;
}

/// [public] method writeMatesUnpaired
INT64 SAMBuilderPaired::writeMatesUnpaired( baseARowWriter* pw, INT64 sqId, PAI* pPAI, const char* pYT )
{
    INT64 nRowsEmitted = 0;

    m_arf = pPAI->arf;
    m_YT = pYT;
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
            if( (pPAI->pQAI1->flags & qaiBest) == 0 )
                m_flag |= sfSecondary;

            // build a row of SAM data for the unmapped read
            p += emitQNAME( p, sqId, pPAI->pQAI1 );
            p += emitFLAG( p, sqId, pPAI->pQAI1 );
            if( pPAI->arf & arfMate2Unmapped )
                p = strcpy( p, m_invariantFields2 ) + m_cbInvariantFields2;
            else
            {
                p = strcpy( p, m_invariantFields1 ) + m_cbInvariantFields1;
                p += emitRNEXT( p, sqId, pPAI->pQAI1 );
                p += emitPNEXT( p, sqId, pPAI->pQAI1 );
                p += emitTLEN( p, sqId, pPAI->pQAI1 );
            }
            p += emitSEQ( p, sqId, pPAI->pQAI1 );
            p += emitQUAL( p, sqId, pPAI->pQAI1 );
            p += emitYT( p, sqId, pPAI->pQAI1 );
            p += emitMQ( p, sqId, pPAI->pQAI1 );
            p += emitid( p, sqId, pPAI->pQAI1 );

            // replace the terminal tab with a Unix-style newline (i.e. an end-of-line character)
            p[-1] = '\n';
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
            if( (pPAI->pQAI2->flags & qaiBest) == 0 )
                m_flag |= sfSecondary;

            // build a row of SAM data for the unmapped read
            p += emitQNAME( p, sqId, pPAI->pQAI2 );
            p += emitFLAG( p, sqId, pPAI->pQAI2 );
            if( pPAI->arf & arfMate1Unmapped )
                p = strcpy( p, m_invariantFields2 ) + m_cbInvariantFields2;
            else
            {
                p = strcpy( p, m_invariantFields1 ) + m_cbInvariantFields1;
                p += emitRNEXT( p, sqId, pPAI->pQAI2 );
                p += emitPNEXT( p, sqId, pPAI->pQAI2 );
                p += emitTLEN( p, sqId, pPAI->pQAI2 );
            }
            p += emitSEQ( p, sqId, pPAI->pQAI2 );
            p += emitQUAL( p, sqId, pPAI->pQAI2 );
            p += emitYT( p, sqId, pPAI->pQAI2 );
            p += emitMQ( p, sqId, pPAI->pQAI2 );
            p += emitid( p, sqId, pPAI->pQAI2 );
     
            // replace the terminal tab with a Unix-style newline (i.e. an end-of-line character)
            p[-1] = '\n';
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
INT64 SAMBuilderPaired::WriteRowPr( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    return writeMatesUnpaired( pw, sqId, pPAI, "YT:Z:RP\t" );   // RP: rejected pair (both mates mapped but did not meet criteria for concordant or discordant mapping)
}

/// [public] method WriteRowPu
INT64 SAMBuilderPaired::WriteRowPu( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    return writeMatesUnpaired( pw, sqId, pPAI, "YT:Z:UP\t" );   // UP: unaligned pair (one or both mates did not map)
}
#pragma endregion
#pragma warning ( pop )
