/*
  SAMBuilderUnpaired.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __SAMBuilderUnpaired__

/// <summary>
/// Class <c>SAMBuilderUnpaired</c> builds SAM alignment records for unpaired reads
/// </summary>
class SAMBuilderUnpaired : public SAMBuilderBase
{
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
        SAMBuilderUnpaired( QBatch* pqb = NULL );
        virtual ~SAMBuilderUnpaired( void );
        virtual INT64 WriteRowUu( baseARowWriter* pw, INT64 sqId, QAI* pQAI );
        virtual INT64 WriteRowUm( baseARowWriter* pw, INT64 sqId, QAI* pQAI );
};
