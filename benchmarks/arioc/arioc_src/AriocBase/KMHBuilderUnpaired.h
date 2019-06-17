/*
  KMHBuilderUnpaired.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __KMHBuilderUnpaired__

/// <summary>
/// Class <c>KMHBuilderUnpaired</c> reports kmer-hashed unpaired read sequences in SQL Server binary format.
/// </summary>
class KMHBuilderUnpaired : public KMHBuilderBase
{
    protected:
        virtual UINT32 emitBRLEQ( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitRGID( char* pbuf, INT64 sqId, QAI* pQAI );

    public:
        KMHBuilderUnpaired( QBatch* pqb = NULL );
        virtual ~KMHBuilderUnpaired( void );
        virtual INT64 WriteRowUu( baseARowWriter* pw, INT64 sqId, QAI* pQAI );
        virtual INT64 WriteRowUm( baseARowWriter* pw, INT64 sqId, QAI* pQAI );
};
