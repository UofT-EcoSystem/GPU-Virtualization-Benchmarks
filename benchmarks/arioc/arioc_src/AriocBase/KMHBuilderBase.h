/*
  KMHBuilderBase.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __KMHBuilderBase__

/// <summary>
/// Class <c>KMHBuilderBase</c> is the base implementation for reporting kmer-hashed read sequences
///  for the Terabase Search Engine in SQL Server binary data import/export format
/// </summary>
class KMHBuilderBase : public TSEFormatBase
{
    protected:
        typedef UINT32( KMHBuilderBase::*mfnEmitField )(char* pbuf, INT64 sqId, QAI* pQAI);

#pragma pack(push, 1)
        struct KMH_FIXED_FIELDS      // fixed-size fields
        {
            UINT64  s64;        // bigint      8 bytes
            UINT64  sqId;       // bigint      8 bytes
//          UINT8   mapCat;     // tinyint     1 byte
//          UINT8   rgId;       // tinyint     1 byte
//          INT16   sqLen;      // smallint    2 bytes
//          UINT32  hash32;     // int         4 bytes
//          UINT8   minBQS;     // tinyint     1 byte
//          UINT8   maxBQS;     // tinyint     1 byte
//          INT16   BRLEQ;      // varbinary   2 bytes (plus variable-length RLE quality scores)
//          INT16   a21;        // varbinary   2 bytes (plus variable-length A21-encoded symbols)
//          INT16   SEQ;        // varchar     2 bytes (plus variable-length character string)
        };
#pragma pack(pop)

    private:
        Hash                    m_Hash;
        const INT32             m_kmerSize;
        UINT64                  m_maskNonN;
        UINT64                  m_maskK;
        UINT32                  m_nSketchBits;  // number of 1-bits in a 64-bit "sketch bitmap"

    protected:
        static mfnEmitField     m_emitField[];

    private:
        static int Xcomparer( const void* a, const void* b );
        UINT32 computeX( UINT32* X, UINT32 qid );

    protected:
        UINT32 emitS64( char* pbuf, INT64 sqId, QAI* pQAI );                // kmer "sketch"
        UINT32 emitA21( char* pbuf, INT64 sqId, QAI* pQAI );                // A21-encoded read sequence

    public:
        KMHBuilderBase( QBatch* pqb );
        virtual ~KMHBuilderBase( void );
};
