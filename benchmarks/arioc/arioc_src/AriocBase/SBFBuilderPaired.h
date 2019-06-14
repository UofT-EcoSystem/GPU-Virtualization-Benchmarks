/*
  SBFBuilderPaired.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __SBFBuilderPaired__

/// <summary>
/// Class <c>SBFBuilderPaired</c> reports paired-end alignments in SQL Server binary format
/// </summary>
class SBFBuilderPaired : public SBFBuilderBase
{
    private:
        QAI*        m_pQAIx;            // alignment info for opposite mate
        INT32       m_fragmentLength;   // signed fragment length (TLEN)

#pragma pack(push, 1)
        struct SBFIF    // invariant fields for unmapped SBF records
        {
            INT16   rname;      // -1       smallint    2 bytes
            INT32   pos;        // 0        int         4 bytes
            UINT8   mapq;       // 255      tinyint     1 byte
            struct
            {
                INT16   cch;
                char    value;
            }       cigar;      // *        varchar     3 bytes (2-byte length followed by '*')
            INT16   rnext;      // -1       smallint    2 bytes
            INT32   pnext;      // 0        int         4 bytes
            INT32   tlen;       // 0        int         4 bytes
        }           m_invariantFields;
#pragma pack(pop)

    private:
        UINT32 emitYS( char* pbuf, INT64 sqId, QAI* pQAI );
        char* writeMate1Unpaired( char* p, INT64 sqId, PAI* pPAI );
        char* writeMate2Unpaired( char* p, INT64 sqId, PAI* pPAI );
        INT64 writeMatesUnpaired( baseARowWriter* pw, INT64 sqId, PAI* pPAI );

    protected:
        // required fields
        virtual UINT32 emitQNAME( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitFLAG( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitRNEXT( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitPNEXT( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitTLEN( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitQUAL( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitRG( char* pbuf, INT64 sqId, QAI* pQAI );

    public:
        SBFBuilderPaired( QBatch* pqb = NULL );
        virtual ~SBFBuilderPaired( void );
        virtual INT64 WriteRowPc( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
        virtual INT64 WriteRowPd( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
        virtual INT64 WriteRowPr( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
        virtual INT64 WriteRowPu( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
 };
