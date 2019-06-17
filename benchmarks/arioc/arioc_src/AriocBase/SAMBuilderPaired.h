/*
  SAMBuilderPaired.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __SAMBuilderPaired__

/// <summary>
/// Class <c>SAMBuilderPaired</c> builds SAM alignment records for paired-end reads.
/// </summary>
class SAMBuilderPaired : public SAMBuilderBase
{
    private:
        QAI*        m_pQAIx;            // alignment info for opposite mate
        INT32       m_fragmentLength;   // signed fragment length (TLEN)

    private:
        UINT32 emitYS( char* pbuf, INT64 sqId, QAI* pQAI );
        char* writeMate1Unpaired( char* p, INT64 sqId, PAI* pPAI );
        char* writeMate2Unpaired( char* p, INT64 sqId, PAI* pPAI );
        INT64 writeMatesUnpaired( baseARowWriter* pw, INT64 sqId, PAI* pPAI, const char* pYT );

    protected:
        // required fields
        virtual UINT32 emitQNAME( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitFLAG( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitRNEXT( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitPNEXT( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitTLEN( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitQUAL( char* pbuf, INT64 sqId, QAI* pQAI );

        // optional fields
        virtual UINT32 emitMQ( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitRG( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitXM( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitXR( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitXG( char* pbuf, INT64 sqId, QAI* pQAI );

    public:
        SAMBuilderPaired( QBatch* pqb = NULL );
        virtual ~SAMBuilderPaired( void );
        virtual INT64 WriteRowPc( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
        virtual INT64 WriteRowPd( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
        virtual INT64 WriteRowPr( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
        virtual INT64 WriteRowPu( baseARowWriter* pw, INT64 sqId, PAI* pPAI );
 };
