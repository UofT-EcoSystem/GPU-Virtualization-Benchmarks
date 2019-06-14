/*
  TSEFormatBase.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __TSEFormatBase__

/// <summary>
/// Class <c>TSEFormatBase</c> implements functionality common to TSE SBF format and KMH format.
/// </summary>
class TSEFormatBase : public baseARowBuilder
{
    private:
        static const INT32      m_bitsPerBQS = 3;
        static const INT32      m_nBQSbins = 1 << m_bitsPerBQS;
        static const UINT8      m_2bpq = 0x80;                  // BRLEQ bits 6..7: 10
        static const UINT8      m_3bpq = 0xC0;                  // BRLEQ bits 6..7: 11

    protected:
        static const INT16      m_maxNforTSE = 320;             // maximum Q sequence length for TSE database
        UINT8                   m_mapCat;

    private:
        UINT8 getFirstBQS( UINT64** ppBQS, UINT8* pnConsumed, char* pMbuf );
        UINT8 getNextBQS( UINT64** ppBQS, UINT64* pBQS8, UINT8* pnConsumed );
        INT32 appendBRLEQRunLength( UINT8* pBRLEQ, INT16& cbBRLEQ, INT32 runLength );
        INT32 appendBRLEQBinValues( UINT8* pBRLEQ, INT16& cbBRLEQ, const UINT8* pbv, INT32 i, INT32 j );

    protected:
        UINT32 emitSqId( char* pbuf, INT64 sqId, QAI* pQAI );
        UINT32 emitMapCat( char* pbuf, INT64 sqId, QAI* pQAI );             // mapping type (see enum MappingCategory above)
        UINT32 emitHash32( char* pbuf, INT64 sqId, QAI* pQAI );             // 32-bit hash of the read sequence
        UINT32 emitHash64( char* pbuf, INT64 sqId, QAI* pQAI );             // 64-bit hash of the read sequence
        virtual UINT32 emitBRLEQ( char* pbuf, INT64 sqId, QAI* pQAI ) = 0;  // binary RLE BQSs
        UINT32 emitSqLen( char* pbuf, INT64 sqId, QAI* pQAI );              // sequence length
        UINT32 emitSEQ( char* pbuf, INT64 sqId, QAI* pQAI );                // sequence
        virtual UINT32 emitRGID( char* pbuf, INT64 sqId, QAI* pQAI ) = 0;   // index into read group ID lookup table

        UINT32 emitBRLEQfr( char* pbuf, QAI* pQAI, UINT32 iMate, UINT32 iMetadata );
        UINT8 getRGIDfromReadMetadata( QAI* pQAI, UINT32 iMate, UINT32 iMetadata );

    public:
        TSEFormatBase( QBatch* pqb, UINT32 cbFixedFields, INT16 nEmitFields );
        virtual ~TSEFormatBase( void );
};
