/*
  TSEBuilderUnpaired.cpp

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
/// Reports unpaired alignments for the Terabase Search Engine in SQL Server binary format
/// </summary>
TSEBuilderUnpaired::TSEBuilderUnpaired( QBatch* pqb ) : TSEBuilderBase(pqb)
{
}

/// [public] destructor
TSEBuilderUnpaired::~TSEBuilderUnpaired()
{
}
#pragma endregion

#pragma region virtual method implementations
/// [protected] method emitFLAG
UINT32 TSEBuilderUnpaired::emitFLAG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // (called only for mapped reads)
    UINT16 flag = (((pQAI->flags >> qaiShrRC) ^ pQAI->flags) & qaiRCr) ? sfRC : sfNone;
    if( 0 == (pQAI->flags & qaiBest) )
        flag |= sfSecondary;

    return intToSBF<UINT16>( pbuf, flag );
}

/// [protected] method emitBRLEQ
UINT32 TSEBuilderUnpaired::emitBRLEQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return emitBRLEQfr( pbuf, pQAI, 0, pQAI->qid );
}

/// [protected] method emitRGID
UINT32 TSEBuilderUnpaired::emitRGID( char* pbuf, INT64 sqId, QAI* pQAI )
{
    UINT8 rgId = getRGIDfromReadMetadata( pQAI, 0, pQAI->qid );
    return intToSBF<UINT8>( pbuf, rgId );
}

/// [public] method WriteRowUm
INT64 TSEBuilderUnpaired::WriteRowUm( baseARowWriter* pw, INT64 sqId, QAI* pQAI )
{
#if TODO_CHOP_IF_UNUSED
    if( pQAI->flags & qaiRCr )
        reverseBRLEA( pQAI->pBH );
#endif

    sqId = setSqIdSecBit( sqId, pQAI );

    char* p0 = pw->Lock( m_pqb->pgi->deviceOrdinal, m_cbRow );
    char* p = p0;
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pQAI );     // emit the f'th field

    // compute the total number of bytes required for the row
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows emitted
    return 1;
}

/// [public] method WriteRowUu
INT64 TSEBuilderUnpaired::WriteRowUu( baseARowWriter* pw, INT64 sqId, QAI* pQAI )
{
    char* p0 = pw->Lock( m_pqb->pgi->deviceOrdinal, m_cbRow );
    char* p = p0;

    // build a row of TSE data for the unmapped read
    TSE_FIXED_FIELDS* pi = reinterpret_cast<TSE_FIXED_FIELDS*>(p);
    memset( pi, 0, sizeof( TSE_FIXED_FIELDS ) );

    // sequence ID
    pi->sqId = sqId;

    // FLAG
    pi->FLAG = sfReadUnmapped;

    // MAPQ
    pi->MAPQ = m_pab->aas.ACP.mapqDefault;

    // rgId
    pi->rgId = getRGIDfromReadMetadata( pQAI, 0, pQAI->qid );

    // sqLen
    pi->sqLen = pQAI->N;

    // hash64
    pi->hash64 = computeA21Hash64( pQAI );

    // emit base quality scores
    p = reinterpret_cast<char*>(&pi->minBQS);
    p += emitBRLEQ( p, sqId, pQAI );

    // emit sequence
    p += emitBRLEE( p, sqId, pQAI );
    p += emitSEQ( p, sqId, pQAI );

    // compute the total number of bytes required for the row
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows emitted
    return 1;
}
#pragma warning ( pop )
#pragma endregion
