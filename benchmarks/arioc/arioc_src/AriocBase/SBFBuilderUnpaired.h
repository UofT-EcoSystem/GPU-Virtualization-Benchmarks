/*
  SBFBuilderUnpaired.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __SBFBuilderUnpaired__

/// <summary>
/// Class <c>SBFBuilderUnpaired</c> reports unpaired alignments in SQL Server binary format
/// </summary>
class SBFBuilderUnpaired : public SBFBuilderBase
{
    private:
#pragma pack(push, 1)
        struct      // invariant fields for unmapped SBF records
        {
            UINT16  flag;       // 0        smallint    2 bytes
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
        SBFBuilderUnpaired( QBatch* pqb = NULL );
        virtual ~SBFBuilderUnpaired( void );
        virtual INT64 WriteRowUu( baseARowWriter* pw, INT64 sqId, QAI* pQAI );
        virtual INT64 WriteRowUm( baseARowWriter* pw, INT64 sqId, QAI* pQAI );
};
