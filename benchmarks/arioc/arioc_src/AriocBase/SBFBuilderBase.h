/*
  SBFBuilderBase.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It  subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __SBFBuilderBase__

/// <summary>
/// Class <c>SBFBuilderBase</c> is the base implementation for reporting alignments with SAM-like fields in SQL Server binary data import/export format
/// </summary>
class SBFBuilderBase : public SAMFormatBase
{
    protected:
        typedef UINT32( SBFBuilderBase::*mfnEmitField )(char* pbuf, INT64 sqId, QAI* pQAI);

    protected:
        static mfnEmitField     m_emitField[];
        AlignmentResultFlags    m_arf;              // alignment result flags for the current SAM row
        const char*             m_YT;               // YT value for current SAM row
        INT32                   m_cbInvariantFields;

    protected:

        // required fields
        virtual UINT32 emitQNAME( char* pbuf, INT64 sqId, QAI* pQAI ) = 0;
        virtual UINT32 emitFLAG( char* pbuf, INT64 sqId, QAI* pQAI ) = 0;
        UINT32 emitRNAME( char* pbuf, INT64 sqId, QAI* pQAI );
        UINT32 emitPOS( char* pbuf, INT64 sqId, QAI* pQAI );
        UINT32 emitMAPQ( char* pbuf, INT64 sqId, QAI* pQAI );
        UINT32 emitCIGAR( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitRNEXT( char* pbuf, INT64 sqId, QAI* pQAI ) = 0;
        virtual UINT32 emitPNEXT( char* pbuf, INT64 sqId, QAI* pQAI ) = 0;
        virtual UINT32 emitTLEN( char* pbuf, INT64 sqId, QAI* pQAI ) = 0;
        UINT32 emitSEQ( char* pbuf, INT64 sqId, QAI* pQAI );
        virtual UINT32 emitQUAL( char* pbuf, INT64 sqId, QAI* pQAI ) = 0;

        // optional fields
        UINT32 emitid( char* pbuf, INT64 sqId, QAI* pQAI );             // sqId
        UINT32 emitaf( char* pbuf, INT64 sqId, QAI* pQAI );             // alignment result flags
        UINT32 emitqf( char* pbuf, INT64 sqId, QAI* pQAI );             // Q-sequence alignment information flags
        UINT32 emitAS( char* pbuf, INT64 sqId, QAI* pQAI );             // alignment score (V)
        UINT32 emitXS( char* pbuf, INT64 sqId, QAI* pQAI );             // second-best alignment score (V)
        UINT32 emitNM( char* pbuf, INT64 sqId, QAI* pQAI );             // edit distance
        UINT32 emitMD( char* pbuf, INT64 sqId, QAI* pQAI );             // Q-to-R edit transcript
        UINT32 emitNA( char* pbuf, INT64 sqId, QAI* pQAI );             // number of valid alignments
        UINT32 emitNB( char* pbuf, INT64 sqId, QAI* pQAI );             // number of traceback origins
        UINT32 emitNullOptionalFields( char* pbuf );                    // null optional fields
        virtual UINT32 emitRG( char* pbuf, INT64 sqId, QAI* pQAI ) = 0; // read group ID

    public:
        SBFBuilderBase( QBatch* pqb );
        virtual ~SBFBuilderBase( void );
};
