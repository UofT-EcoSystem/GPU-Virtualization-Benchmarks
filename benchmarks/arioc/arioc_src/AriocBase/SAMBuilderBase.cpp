/*
  SAMBuilderBase.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variables
SAMBuilderBase::mfnEmitField SAMBuilderBase::m_emitField[] = { &SAMBuilderBase::emitQNAME,
                                                               &SAMBuilderBase::emitFLAG,
                                                               &SAMBuilderBase::emitRNAME,
                                                               &SAMBuilderBase::emitPOS,
                                                               &SAMBuilderBase::emitMAPQ,
                                                               &SAMBuilderBase::emitCIGAR,
                                                               &SAMBuilderBase::emitRNEXT,
                                                               &SAMBuilderBase::emitPNEXT,
                                                               &SAMBuilderBase::emitTLEN,
                                                               &SAMBuilderBase::emitSEQ,
                                                               &SAMBuilderBase::emitQUAL,
                                                               &SAMBuilderBase::emitAS,
                                                               &SAMBuilderBase::emitXS,
                                                               &SAMBuilderBase::emitNM,
                                                               &SAMBuilderBase::emitMD,
                                                               &SAMBuilderBase::emitYT,
                                                               &SAMBuilderBase::emitMQ,
                                                               &SAMBuilderBase::emitNA,
                                                               &SAMBuilderBase::emitNB,
                                                               &SAMBuilderBase::emitRG,
                                                               &SAMBuilderBase::emitXM,     // fields for CT-converted reads only
                                                               &SAMBuilderBase::emitXR,
                                                               &SAMBuilderBase::emitXG,
                                                               &SAMBuilderBase::emitid,     // fields for Arioc trace only
                                                               &SAMBuilderBase::emitaf,
                                                               &SAMBuilderBase::emitqf
                                                             };
#pragma endregion

#pragma region constructors and destructor
/// <summary>
/// Implements base functionality for paired and unpaired SAM-formatted file output
/// </summary>
SAMBuilderBase::SAMBuilderBase( QBatch* pqb ) : SAMFormatBase(pqb, arraysize(m_emitField), false),
                                                m_arf(arfNone),
                                                m_YT(NULL),
                                                m_baseConvCT(pqb->pab->a21ss.baseConvert == A21SpacedSeed::bcCT)
{
    // compute the amount of buffer space required to build each SAM record
    m_cbRow = 256 +         // QNAME: maximum field size = 255 per SAM spec
              6 +           // FLAG: decimal representation of UINT16
              256 +         // RNAME: (worst-case guess)
              12 +          // POS: (9 digits per SAM spec, plus two extra digits just in case)
              4 +           // MAPQ: 3-digit decimal string
              pqb->Nmax +   // CIGAR: worst-case guess
              256 +         // RNEXT: (worst-case guess)
              12 +          // PNEXT: (see POS)
              12 +          // TLEN: (see POS)
              pqb->Nmax +   // SEQ: (worst-case guess)
              pqb->Nmax +   // QUAL: (worst-case guess)
              512;          // optional fields, tabs (separators), end-of-line terminator

    if( m_baseConvCT )
        m_cbRow += pqb->Nmax;   // XM (emitted for bsDNA alignments)
    
    if( pqb->pab->pifgQ->HasPairs )
        m_cbRow *= 2;       // double the buffer-space estimate if we're writing paired-end reads

    // compute the size of the reverse BRLEA buffer
    INT32 cbBRLEA = pqb->pab->aas.ComputeWorstCaseBRLEAsize( pqb->Nmax );

    // allocate the buffer
    m_revBRLEA.Realloc( cbBRLEA, false );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: m_revBRLEA.p = 0x%016llx cbBRLEA=%d", __FUNCTION__, m_revBRLEA.p, cbBRLEA );
#endif

    // set a flag to control whether optional "trace" fields are emitted
    INT32 i = pqb->pab->paamb->Xparam.IndexOf( "emitTraceFields" );
    if( i >= 0 )
        m_emitA21TraceFields = (pqb->pab->paamb->Xparam.Value(i) != 0);
}

/// [public] destructor
SAMBuilderBase::~SAMBuilderBase()
{
}
#pragma endregion

#pragma region protected methods
#pragma warning ( push )
#pragma warning( disable:4996 )     // (don't nag us about strcpy being "unsafe")

/// [protected] method emitRNAME
UINT32 SAMBuilderBase::emitRNAME( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return emitSubIdAsRNAME( pbuf, pQAI->subId );
}

/// [protected] method emitPOS
UINT32 SAMBuilderBase::emitPOS( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = pbuf + u32ToString( pbuf, pQAI->Jf+1 );       // (report the 1-based position relative to the start of the forward reference strand)
    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitMAPQ
UINT32 SAMBuilderBase::emitMAPQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    UINT8 mapQ = (sqId & AriocDS::SqId::MaskSecBit) ? pQAI->mapQp2 : pQAI->mapQ;
    char* p = pbuf + u32ToString( pbuf, mapQ );
    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitCIGAR
UINT32 SAMBuilderBase::emitCIGAR( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // emit the CIGAR string
    INT16 cch = (pQAI->flags & qaiRCr) ? emitCIGARr( pbuf, pQAI ) : emitCIGARf( pbuf, pQAI );

    // emit a tab character
    pbuf[cch++] = '\t';

    return cch;
}

/// [protected] method emitSEQ
UINT32 SAMBuilderBase::emitSEQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // emit the sequence data
    INT16 cch;
    switch( pQAI->flags & (qaiRCr|qaiRCq) )
    {
        case qaiRCr:
        case qaiRCq:
            cch = stringizeSEQr( pbuf, pQAI );
            break;

        default:
            cch = stringizeSEQf( pbuf, pQAI );
            break;
    }

    // emit a tab character
    pbuf[cch++] = '\t';

    return cch;
}

/// [protected] method emitid
UINT32 SAMBuilderBase::emitid( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // emit nothing if the user-specified flag is false
    if( !m_emitA21TraceFields )
        return 0;

    char* p = strcpy( pbuf, "id:H:" ) + 5;

    // write the sqId as a 16-digit hexadecimal string
    p += uintToHexString( p, sqId );
    *p = '\t';
    return 22;      // 5-digit SAM prefix plus 16 hexadecimal digits plus one tab character
}

/// [protected] method emitAS
UINT32 SAMBuilderBase::emitAS( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = strcpy( pbuf, "AS:i:" ) + 5;
    p += u32ToString( p, pQAI->pBH->V );
    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitXS
UINT32 SAMBuilderBase::emitXS( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // emit nothing if there is no second-best V score
    if( pQAI->Vsec <= 0 )
        return 0;

    char* p = strcpy( pbuf, "XS:i:" ) + 5;
    p += u32ToString( p, pQAI->Vsec );
    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitNM
UINT32 SAMBuilderBase::emitNM( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = strcpy( pbuf, "NM:i:" ) + 5;
    p += u32ToString( p, pQAI->editDistance );
    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitMD
UINT32 SAMBuilderBase::emitMD( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // write the SAM field prefix
    strcpy( pbuf, "MD:Z:" );
    char* const p = pbuf + 5;

    UINT32 cb = emitMDfr( p, pQAI );
    p[cb] = '\t';
    return 6 + cb;
}

/// [protected] method emitYT
UINT32 SAMBuilderBase::emitYT( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // do nothing if there is no applicable YT value (e.g., unpaired reads regardless of whether they map)
    if( m_YT == NULL )
        return 0;

    strcpy( pbuf, m_YT );
    return 8;
}

/// [protected] method emitNA
UINT32 SAMBuilderBase::emitNA( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // write the SAM field prefix
    strcpy( pbuf, "NA:i:" );
    char* p = pbuf + 5;

    /* write the number of unduplicated mappings found by the aligner (which may, of course, exceed
        the number of mappings reported) */
    p += u32ToString( p, pQAI->nAu );

    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitNB
UINT32 SAMBuilderBase::emitNB( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // write the SAM field prefix
    strcpy( pbuf, "NB:i:" );
    char* p = pbuf + 5;

    // write the number of Smith-Waterman traceback origins (highest score in SW scoring matrix) found by the gapped aligner
    p += u32ToString( p, pQAI->nTBO );

    *(p++) = '\t';
    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitaf
UINT32 SAMBuilderBase::emitaf( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // emit nothing if the user-specified flag is false
    if( !m_emitA21TraceFields )
        return 0;

    if( m_arf == arfNone )
        return 0;

    char* p = strcpy( pbuf, "af:H:" ) + 5;
    p += uintToHexString<UINT16>( p, m_arf );
    *p = '\t';
    return 10;
}

/// [protected] method emitqf
UINT32 SAMBuilderBase::emitqf( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // emit nothing if the user-specified flag is false
    if( !m_emitA21TraceFields )
        return 0;

    char* p = strcpy( pbuf, "qf:H:" ) + 5;
    p += uintToHexString<UINT16>( p, pQAI->flags );
    *p = '\t';
    return 10;
}
#pragma warning ( pop )
#pragma endregion
