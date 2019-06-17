/*
  SBFBuilderBase.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variables
SBFBuilderBase::mfnEmitField SBFBuilderBase::m_emitField[] = { &SBFBuilderBase::emitQNAME,
                                                               &SBFBuilderBase::emitFLAG,
                                                               &SBFBuilderBase::emitRNAME,
                                                               &SBFBuilderBase::emitPOS,
                                                               &SBFBuilderBase::emitMAPQ,
                                                               &SBFBuilderBase::emitCIGAR,
                                                               &SBFBuilderBase::emitRNEXT,
                                                               &SBFBuilderBase::emitPNEXT,
                                                               &SBFBuilderBase::emitTLEN,
                                                               &SBFBuilderBase::emitSEQ,
                                                               &SBFBuilderBase::emitQUAL,
                                                               &SBFBuilderBase::emitid,
                                                               &SBFBuilderBase::emitAS,
                                                               &SBFBuilderBase::emitXS,
                                                               &SBFBuilderBase::emitNM,
                                                               &SBFBuilderBase::emitMD,
                                                               &SBFBuilderBase::emitNA,
                                                               &SBFBuilderBase::emitNB,
                                                               &SBFBuilderBase::emitRG,
                                                               &SBFBuilderBase::emitaf,
                                                               &SBFBuilderBase::emitqf
                                                             };
#pragma endregion

#pragma region constructors and destructor
/// <summary>
/// Implements functionality for reporting alignments with SAM-like fields in SQL Server binary data import/export format
/// </summary>
SBFBuilderBase::SBFBuilderBase( QBatch* pqb ) : SAMFormatBase(pqb, arraysize(m_emitField), true),
                                                m_arf(arfNone),
                                                m_YT(NULL),
                                                m_cbInvariantFields(0)
{
    /* Compute the amount of buffer space required to build each SBF row.  A worst-case (maximum) size is
        assumed for each field.

       Some integer values (e.g., MAPQ, id) represent unsigned values or bitmaps even though MSSQL treats them as signed values.

       We don't use SQL null values in any column.  There might be some space to be saved by using SPARSE column definitions if
        we had lots of nulls, but overall it's probably not worth it.
    */

    // sanity check
    if( pqb->Nmax > 8000 )
        throw new ApplicationException( __FILE__, __LINE__, "cannot represent sequence of length %d in SBF (maximum length = 8000)", pqb->Nmax );
            
    INT16 maxcb = min2(8000,pqb->Nmax) + 2;     // maximum sequence length plus 2-byte length prefix        
            
    m_cbRow = 257 +     // QNAME varchar(255)   not null,   -- 2-byte length prefix followed by ASCII string (not null terminated)
              2 +       // FLAG  smallint      not null,    -- 16-bit integer
              2 +       // RNAME smallint      not null,
              4 +       // POS   int           not null,
              1 +       // MAPQ  tinyint       not null,
              maxcb +   // CIGAR varchar(8000) not null,
              2 +       // RNEXT smallint      not null,
              4 +       // PNEXT int           not null,
              4 +       // TLEN  int           not null,
              maxcb +   // SEQ   varchar(8000) not null,
              maxcb +   // QUAL  varchar(8000) not null,
              8 +       // id    bigint        not null,    -- SqId
              2 +       // AS    smallint      not null,    -- alignment score
              2 +       // XS    smallint      not null,    -- best AS for alternative mapping
              2 +       // NM    smallint      not null,    -- Levenshtein edit distance
              maxcb +   // MD    varchar(8000) not null,
              2 +       // NA    smallint      not null,    -- number of unduplicated mappings
              2 +       // NB    smallint      not null,    -- number of traceback origins
              257 +     // RG    varchar(255)  not null,
              2 +       // af    smallint      not null,    -- AlignmentResultFlags
              2;        // qf    smallint      not null     -- QAIflags
                        // YS    smallint      not null     -- TODO: PUT THIS BACK?? best AS for opposite mate (paired-end only)

    
    if( pqb->pab->pifgQ->HasPairs )
        m_cbRow *= 2;       // double the buffer-space estimate if we're writing paired-end reads

    // compute the size of the reverse BRLEA buffer
    INT32 cbBRLEA = pqb->pab->aas.ComputeWorstCaseBRLEAsize( pqb->Nmax );

    // allocate the buffer
    m_revBRLEA.Realloc( cbBRLEA, false );

        // set a flag to control whether optional "trace" fields are emitted
    INT32 i = pqb->pab->paamb->Xparam.IndexOf( "emitTraceFields" );
    if( i >= 0 )
        m_emitA21TraceFields = (pqb->pab->paamb->Xparam.Value(i) != 0);
}

/// [public] destructor
SBFBuilderBase::~SBFBuilderBase()
{
}
#pragma endregion

#pragma region protected methods
#pragma warning ( push )
#pragma warning( disable:4996 )     // (don't nag us about strcpy being "unsafe")

/// [protected] method emitRNAME
UINT32 SBFBuilderBase::emitRNAME( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT16>( pbuf, pQAI->subId );
}

/// [protected] method emitPOS
UINT32 SBFBuilderBase::emitPOS( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT32>( pbuf, pQAI->Jf+1 );
}

/// [protected] method emitMAPQ
UINT32 SBFBuilderBase::emitMAPQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<UINT8>( pbuf, pQAI->mapQ );
}

/// [protected] method emitCIGAR
UINT32 SBFBuilderBase::emitCIGAR( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = pbuf + sizeof(INT16);
    INT16 cb = (pQAI->flags & qaiRCr) ? emitCIGARr( p, pQAI ) : emitCIGARf( p, pQAI );
    *reinterpret_cast<INT16*>(pbuf) = cb;
    return cb + sizeof(INT16);
}

/// [protected] method emitSEQ
UINT32 SBFBuilderBase::emitSEQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = pbuf + sizeof(INT16);
    INT16 cb = (pQAI->flags & qaiRCr) ? stringizeSEQr( p, pQAI ) : stringizeSEQf( p, pQAI );
    *reinterpret_cast<INT16*>(pbuf) = cb;
    return cb + sizeof(INT16);
}

/// [protected] method emitid
UINT32 SBFBuilderBase::emitid( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // emit nothing if the user-specified flag is false
    if( !m_emitA21TraceFields )
        return 0;

    return intToSBF<UINT64>( pbuf, sqId );
}

/// [protected] method emitAS
UINT32 SBFBuilderBase::emitAS( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT16>( pbuf, pQAI->pBH->V );
}

/// [protected] method emitXS
UINT32 SBFBuilderBase::emitXS( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // emit zero if there is no second-best V score
    if( pQAI->Vsec <= 0 )
        return intToSBF<INT16>( pbuf, 0 );

    return intToSBF<INT16>( pbuf, pQAI->Vsec );
}

/// [protected] method emitNM
UINT32 SBFBuilderBase::emitNM( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT16>( pbuf, pQAI->editDistance );
}

/// [protected] method emitMD
UINT32 SBFBuilderBase::emitMD( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = pbuf + sizeof(INT16);
    INT16 cb = emitMDfr( p, pQAI );
    *reinterpret_cast<INT16*>(pbuf) = cb;
    return cb + sizeof(INT16);
}

/// [protected] method emitNA
UINT32 SBFBuilderBase::emitNA( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT16>( pbuf, pQAI->nAu );
}

/// [protected] method emitNB
UINT32 SBFBuilderBase::emitNB( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT16>( pbuf, pQAI->nTBO );
}

/// [protected] method emitaf
UINT32 SBFBuilderBase::emitaf( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // emit nothing if the user-specified flag is false
    if( !m_emitA21TraceFields )
        return 0;

    return intToSBF<UINT16>( pbuf, m_arf );
}

/// [protected] method emitqf
UINT32 SBFBuilderBase::emitqf( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // emit nothing if the user-specified flag is false
    if( !m_emitA21TraceFields )
        return 0;

    return intToSBF<UINT16>( pbuf, pQAI->flags );
}

/// [protected] emitNullFields
UINT32 SBFBuilderBase::emitNullOptionalFields( char* pbuf )
{
    // emit optional fields as null values
    static const UINT32 cb = 2 +    // AS   smallint      not null,     -- alignment score
                             2 +    // XS   smallint      not null,     -- best AS for alternative mapping
                             2 +    // NM   smallint      not null,     -- Levenshtein edit distance
                             2 +    // MD   varchar(8000) not null,     -- (empty MD string)
                             2 +    // NA   smallint      not null,     -- number of unduplicated mappings
                             2 +    // NB   smallint      not null,     -- number of traceback origins
                             2 +    // af   smallint      not null,     -- AlignmentResultFlags
                             2;     // qf   smallint      not null      -- QAIflags

    memset( pbuf, 0, cb );
    return cb;
}
#pragma warning ( pop )
#pragma endregion
