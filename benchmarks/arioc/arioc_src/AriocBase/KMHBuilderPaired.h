/*
  KMHBuilderPaired.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __KMHBuilderPaired__

/// <summary>
/// Class <c>KMHBuilderPaired</c> reports kmer-hashed paired-end read sequences in SQL Server binary format.
/// </summary>
class KMHBuilderPaired : public KMHBuilderBase
{
    private:
        PAI*    m_pPAI;             // pair alignment info
        QAI*    m_pQAIx;            // alignment info for opposite mate

    private:
        char* writeMate1Unpaired( char* p, INT64 sqId, PAI* pPAI );
        char* writeMate2Unpaired( char* p, INT64 sqId, PAI* pPAI );
        INT64 writeMatesUnpaired( baseARowWriter* pw, INT64 sqId, PAI* pPAI );

    protected:
        virtual UINT32 emitBRLEQ( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitRGID( char* pbuf, INT64 sqId, QAI* pQAI );

    public:
        KMHBuilderPaired( QBatch* pqb = NULL );
        virtual ~KMHBuilderPaired( void );
        virtual INT64 WriteRowPc( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
        virtual INT64 WriteRowPd( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
        virtual INT64 WriteRowPr( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
        virtual INT64 WriteRowPu( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
 };
