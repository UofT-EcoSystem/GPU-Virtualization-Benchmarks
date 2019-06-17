/*
  TSEBuilderBase.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __TSEBuilderBase__

/// <summary>
/// Class <c>TSEBuilderBase</c> is the base implementation for reporting alignments for the Terabase Search Engine in SQL Server binary data import/export format
/// </summary>
class TSEBuilderBase : public TSEFormatBase
{
    protected:
        typedef UINT32 (TSEBuilderBase::*mfnEmitField)(char* pbuf, INT64 sqId, QAI* pQAI);

    private:
        typedef void (TSEBuilderBase::*mfnEmitBRLEE)( char*& p, INT32 runLength, const UINT64* pQ, INT32& i );

    protected:
        static mfnEmitField     m_emitField[];

#pragma pack(push, 1)
        struct TSE_FIXED_FIELDS      // fixed-size fields
        {
            UINT64  sqId;       // bigint      8 bytes
            UINT16  FLAG;       // smallint    2 bytes
            UINT8   mapCat;     // tinyint     1 byte
            UINT8   MAPQ;       // tinyint     1 byte
            UINT8   rgId;       // tinyint     1 byte
            UINT8   subId;      // tinyint     1 byte
            INT32   J;          // int         4 bytes
            INT32   POS;        // int         4 bytes
            INT16   spanR;      // smallint    2 bytes
            INT16   sqLen;      // smallint    2 bytes
            UINT64  hash64;     // bigint      8 bytes
            INT16   V;          // smallint    2 bytes
            INT16   NM;         // smallint    2 bytes
            INT16   NA;         // smallint    2 bytes
            UINT8   minBQS;     // tinyint     1 byte
            UINT8   maxBQS;     // tinyint     1 byte
            INT16   BRLEQ;      // varbinary   2 bytes (plus variable-length RLE quality scores)
            INT16   BRLEE;      // varbinary   2 bytes (plus variable-length RLE sequence)
            INT16   SEQ;        // varchar     2 bytes (plus variable-length character string)
        };
#pragma pack(pop)

    private:
        static mfnEmitBRLEE     m_emitBRLEE[];
        static UINT8            m_BRLEEsymbolEncode[];

    private:
        UINT8 getQiSymbol( const UINT64* pQ, const INT32 i );
        UINT8 getQiSymbolPair( const UINT64* pQ, const INT32 i );
        UINT32 emitBRLEE2bps( char* pbuf, INT16 N, UINT64* pQ );
        UINT32 emitBRLEE3bps( char* pbuf, INT16 N, UINT64* pQ );
        void emitBRLEErunLength( char*& p, BRLEAbyteType bbType, INT32 runLength );
        void emitBRLEEsymbols( char*& p, BRLEAbyteType bbType, INT32 runLength, const UINT64* pQ, INT32 i );
        void emitBRLEEmatch( char*& p, INT32 runLength, const UINT64* pQ, INT32& i );
        void emitBRLEEmismatch( char*& p, INT32 runLength, const UINT64* pQ, INT32& i );
        void emitBRLEEgapR( char*& p, INT32 runLength, const UINT64* pQ, INT32& i );
        void emitBRLEEgapQ( char*& p, INT32 runLength, const UINT64* pQ, INT32& i );
        
    protected:
        virtual UINT32 emitFLAG( char* pbuf, INT64 sqId, QAI* pQAI ) = 0;
        UINT32 emitMAPQ( char* pbuf, INT64 sqId, QAI* pQAI );
        UINT32 emitSpanR( char* pbuf, INT64 sqId, QAI* pQAI );              // subId, startAt, SpanR
        UINT32 emitV( char* pbuf, INT64 sqId, QAI* pQAI );                  // alignment score (V)
        UINT32 emitNM( char* pbuf, INT64 sqId, QAI* pQAI );                 // edit distance
        UINT32 emitNA( char* pbuf, INT64 sqId, QAI* pQAI );                 // number of valid alignments
        UINT32 emitBRLEE( char* pbuf, INT64 sqId, QAI* pQAI );              // binary RLE edit string

        UINT32 emitBRLEEfr( char* pbuf, QAI* pQAI );
        UINT32 emitBRLEEu( char* pbuf, QAI* pQAI );

    public:
        TSEBuilderBase( QBatch* pqb );
        virtual ~TSEBuilderBase( void );
};
