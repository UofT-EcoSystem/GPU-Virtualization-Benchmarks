/*
  SBFBuilderUnpaired.cpp

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
/// Reports unpaired alignments in SQL Server binary data import/export format
/// </summary>
SBFBuilderUnpaired::SBFBuilderUnpaired( QBatch* pqb ) : SBFBuilderBase(pqb)
{
    /* initialize invariant fields for unmapped SAM records */
    m_cbInvariantFields = static_cast<INT32>( sizeof m_invariantFields );
    memset( &m_invariantFields, 0, m_cbInvariantFields );
    m_invariantFields.flag = sfReadUnmapped;
    m_invariantFields.rname = -1;
    m_invariantFields.mapq = pqb->pab->aas.ACP.mapqDefault;
    m_invariantFields.cigar.cch = 1;
    m_invariantFields.cigar.value = '*';
    m_invariantFields.rnext = -1;
}

/// [public] destructor
SBFBuilderUnpaired::~SBFBuilderUnpaired()
{
}
#pragma endregion

#pragma region virtual method implementations
/// [protected] method emitQNAME
UINT32 SBFBuilderUnpaired::emitQNAME( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = pbuf + sizeof(INT16);
    INT16 cb = emitQNAMEpu( p, pQAI, 0, pQAI->qid );
    *reinterpret_cast<INT16*>(p) = cb;
    return cb + sizeof(INT16);
}

/// [protected] method emitFLAG
UINT32 SBFBuilderUnpaired::emitFLAG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // (called only for mapped reads)
    UINT16 flag = (((pQAI->flags >> qaiShrRC) ^ pQAI->flags) & qaiRCr) ? sfRC : sfNone;
    if( 0 == (pQAI->flags & qaiBest) )
        flag |= sfSecondary;

    return intToSBF<UINT16>( pbuf, flag );
}

/// [protected] method emitRNEXT
UINT32 SBFBuilderUnpaired::emitRNEXT( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT16>( pbuf, -1 );
}

/// [protected] method emitPNEXT
UINT32 SBFBuilderUnpaired::emitPNEXT( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT32>( pbuf, 0 );
}

/// [protected] method emitTLEN
UINT32 SBFBuilderUnpaired::emitTLEN( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT32>( pbuf, 0 );
}

/// [protected] method emitQUAL
UINT32 SBFBuilderUnpaired::emitQUAL( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = pbuf + sizeof(INT16);
    UINT32 cb = emitQUALfr( p, pQAI, 0, pQAI->qid );
    *reinterpret_cast<INT16*>(pbuf) = cb;
    return cb + sizeof(INT16);
}

/// [protected] method emitRG
UINT32 SBFBuilderUnpaired::emitRG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // if there is no read-group info, emit an empty string
    if( m_pab->paamb->RGMgr.OfsRG.n == 0 )
    {
        *reinterpret_cast<INT16*>(pbuf) = 0;
        return sizeof(INT16);
    }

    char* p = pbuf + sizeof(INT16);
    INT16 cb = emitRGID( p, pQAI, 0, pQAI->qid );
    *reinterpret_cast<INT16*>(pbuf) = cb;
    return cb + sizeof(INT16);
}

/// [public] method WriteRowUm
INT64 SBFBuilderUnpaired::WriteRowUm( baseARowWriter* pw, INT64 sqId, QAI* pQAI )
{
    if( pQAI->flags & qaiRCr )
        reverseBRLEA( pQAI->pBH );

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
INT64 SBFBuilderUnpaired::WriteRowUu( baseARowWriter* pw, INT64 sqId, QAI* pQAI )
{
    /* Fill a local buffer with a row of SAM data for an unmapped read; only the following fields contain variable data:
        - QNAME
        - SEQ
        - QUAL
        - id

       The other fields are hardwired (see the constructor).
    */
    char* p0 = pw->Lock( m_pqb->pgi->deviceOrdinal, m_cbRow );
    char* p = p0 + emitQNAME( p0, sqId, pQAI );
    memcpy( p, &m_invariantFields, m_cbInvariantFields );
    p += m_cbInvariantFields;
    p += emitSEQ( p, sqId, pQAI );
    p += emitQUAL( p, sqId, pQAI );
    p += emitid( p, sqId, pQAI );
    p += emitNullOptionalFields( p );

    // compute the total number of bytes required for the row
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows emitted
    return 1;
}

#pragma warning ( pop )
#pragma endregion
