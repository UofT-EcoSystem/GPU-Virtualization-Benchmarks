/*
  SAMBuilderUnpaired.cpp

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
/// Builds SAM alignment records for unpaired reads
/// </summary>
SAMBuilderUnpaired::SAMBuilderUnpaired( QBatch* pqb ) : SAMBuilderBase(pqb)
{
    /* build a string of invariant fields for unmapped SAM records:
        - FLAG      4
        - RNAME:    *
        - POS:      0
        - MAPQ:     0 (by default; may be overridden in .cfg file)
        - CIGAR:    *
        - RNEXT:    *
        - PNEXT:    0
        - TLEN:     0
    */
    m_cbInvariantFields2 = sprintf_s( m_invariantFields2, sizeof m_invariantFields2, "4\t*\t0\t%u\t*\t*\t0\t0\t", pqb->pab->aas.ACP.mapqDefault );
}

/// [public] destructor
SAMBuilderUnpaired::~SAMBuilderUnpaired()
{
}
#pragma endregion

#pragma region virtual method implementations
/// [protected] method emitQNAME
UINT32 SAMBuilderUnpaired::emitQNAME( char* pbuf, INT64 sqId, QAI* pQAI )
{
    INT16 cb = emitQNAMEpu( pbuf, pQAI, 0, pQAI->qid );
    pbuf[cb] = '\t';
    return cb + 1;
}

/// [protected] method emitFLAG
UINT32 SAMBuilderUnpaired::emitFLAG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // (called only for mapped reads)
    m_flag = (((pQAI->flags >> qaiShrRC) ^ pQAI->flags) & qaiRCr) ? sfRC : sfNone;
    if( 0 == (pQAI->flags & qaiBest) )
        m_flag |= sfSecondary;
    char* p = pbuf + u32ToString( pbuf, m_flag );
    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitRNEXT
UINT32 SAMBuilderUnpaired::emitRNEXT( char* pbuf, INT64 sqId, QAI* pQAI )
{
    strcpy( pbuf, "*\t" );  // (no "next segment" for unpaired reads)
    return 2;
}

/// [protected] method emitPNEXT
UINT32 SAMBuilderUnpaired::emitPNEXT( char* pbuf, INT64 sqId, QAI* pQAI )
{
    strcpy( pbuf, "0\t" );  // (no "next segment" for unpaired reads)
    return 2;
}

/// [protected] method emitTLEN
UINT32 SAMBuilderUnpaired::emitTLEN( char* pbuf, INT64 sqId, QAI* pQAI )
{
    strcpy( pbuf, "0\t" );  // (no "template length" for unpaired reads)
    return 2;
}

/// [protected] method emitQUAL
UINT32 SAMBuilderUnpaired::emitQUAL( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // compute an index into the list of metadata offsets (see tuLoadM::readMetadata)
    UINT32 iMetadata = pQAI->qid;

    UINT32 cb = emitQUALfr( pbuf, pQAI, 0, iMetadata );
    pbuf[cb] = '\t';    // emit the trailing tab separator
    return cb + 1;
}

/// [protected] method emitMQ
UINT32 SAMBuilderUnpaired::emitMQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // (there is no opposite mate)
    return 0;
}

/// [protected] method emitRG
UINT32 SAMBuilderUnpaired::emitRG( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // do nothing if there is no read-group info
    if( m_pab->paamb->RGMgr.OfsRG.n == 0 )
        return 0;

    strcpy( pbuf, "RG:Z:" );
    char* p = pbuf + 5;         // 5: strlen("RG:Z:")

    UINT32 cb = emitRGID( p, pQAI, 0, pQAI->qid );
    if( cb == 0 )
        return 0;
    
    p[cb] = '\t';
    return cb + 6;      // 6: strlen("RG:Z:") + terminal tab
}

/// [protected] method emitXM
UINT32 SAMBuilderUnpaired::emitXM( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // do nothing if we are not doing bsDNA alignments
    if( !m_baseConvCT )
        return 0;

    strcpy( pbuf, "XM:Z:" );
    char* p = pbuf + 5;         // 5: strlen("XM:Z:")

#if TODO_CHOP_WHEN_DEBUGGED
    if( sqId == 0x0000040800000024 )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif

    UINT32 cb = emitXMfr( p, pQAI );
    if( cb == 0 )
        return 0;
    
    p[cb] = '\t';
    return cb + 6;      // 6: strlen("XM:Z:") + terminal tab
}

/// [protected] method emitXG
UINT32 SAMBuilderUnpaired::emitXG( char* pbuf, INT64 sqId, QAI* pQAI )
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
UINT32 SAMBuilderUnpaired::emitXR( char* pbuf, INT64 sqId, QAI* pQAI )
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

/// [public] method WriteRowUm
INT64 SAMBuilderUnpaired::WriteRowUm( baseARowWriter* pw, INT64 sqId, QAI* pQAI )
{
    if( pQAI->flags & qaiRCr )
        reverseBRLEA( pQAI->pBH );

    sqId = setSqIdSecBit( sqId, pQAI );

    char* p0 = pw->Lock( m_pqb->pgi->deviceOrdinal, m_cbRow );
    char* p = p0;
    for( INT16 f=0; f<m_nEmitFields; ++f )
        p += (this->*(m_emitField[f]))( p, sqId, pQAI );     // emit the f'th field

    // replace the terminal tab with a Unix-style newline (i.e. an end-of-line character)
    p[-1] = '\n';

    // compute the total number of bytes required for the row
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows emitted
    return 1;
}

/// [public] method WriteRowUu
INT64 SAMBuilderUnpaired::WriteRowUu( baseARowWriter* pw, INT64 sqId, QAI* pQAI )
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
    p = strcpy( p, m_invariantFields2 ) + m_cbInvariantFields2;
    p += emitSEQ( p, sqId, pQAI );
    p += emitQUAL( p, sqId, pQAI );
    p += emitid( p, sqId, pQAI );

    // replace the terminal tab with a Unix-style newline (i.e. an end-of-line character)
    p[-1] = '\n';

    // compute the total number of bytes required for the row
    UINT32 cb = static_cast<UINT32>(p - p0);

    // release the output buffer
    pw->Release( m_pqb->pgi->deviceOrdinal, cb );

    // return the number of rows emitted
    return 1;
}

#pragma warning ( pop )
#pragma endregion
