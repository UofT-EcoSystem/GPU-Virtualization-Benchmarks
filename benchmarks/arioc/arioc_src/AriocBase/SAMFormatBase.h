/*
  SAMFormatBase.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __SAMFormatBase__

#pragma region enums
enum samFLAG : UINT16
{                               // Description per SAM v1.4-r985:
    sfNone =            0x0000,
    sfPaired =          0x0001, // template having multiple segments in sequencing
    sfPairMapped =      0x0002, // each segment properly aligned according to the aligner
    sfMateUnmapped =    0x0004, // segment unmapped
    sfReadUnmapped =    sfMateUnmapped,
    sfOppMateUnmapped = 0x0008, // next segment in the template unmapped
    sfRC =              0x0010, // SEQ being reverse complemented
    sfOppRC =           0x0020, // SEQ of the next segment in the template being reversed
    sfPair0 =           0x0040, // the first segment in the template
    sfPair1 =           0x0080, // the last segment in the template
    sfSecondary =       0x0100, // secondary alignment
    sfQualityFail =     0x0200, // not passing quality controls
    sfDuplicate =       0x0400  // PCR or optical duplicate
};
#pragma endregion

/// <summary>
/// Class <c>SAMFormatBase</c> implements SAM formatting functionality common to derived classes (SAMBuilder*, SBFBuilder*).
/// </summary>
class SAMFormatBase : public baseARowBuilder
{
    private:
        typedef UINT8 (SAMFormatBase::*mfnGetNextQi)( UINT64& Qi, const UINT64*& pQi );
        typedef void (SAMFormatBase::*mfnEmitXM)( char*& p, INT32 runLength, UINT64& R, const UINT64*& pR, UINT64& Qi, const UINT64*& pQi, INT32& i, mfnGetNextQi getNextQi );
        typedef void (SAMFormatBase::*mfnEmitMDsymbols)( char*& p, UINT32 j, INT16 cchRun, const UINT64* pR0 );

        static mfnEmitXM    m_emitXM[];
        static char         m_ctxXM[];
        const char*         m_CIGARsymbolMID;
        char                m_CIGARsymbolS;
        mfnEmitMDsymbols    m_emitMDsymbols;
        bool                m_emitMDstandard;

    private:
        static INT32 initRval;
        static INT32 initializer( void );
        INT16 getRunLengthBRLEA( BRLEAbyte*& pB, const BRLEAbyte* pBlimit );
        void emitMDsymbolsStandard( char*& p, UINT32 j, INT16 cchRun, const UINT64* pR0 );
        void emitMDsymbolsCompact( char*& p, UINT32 j, INT16 cchRun, const UINT64* pR0 );
        void emitCIGARsymbols( char*& p, BRLEAbyte* pB, INT16 cbBRLEA );
        UINT8 getNextR( UINT64& R, const UINT64*& pR );
        UINT8 getNextQiF( UINT64& Qi, const UINT64*& pQi );
        UINT8 getNextQiRC( UINT64& Qi, const UINT64*& pQi );
        void emitXMmatch( char*& p, INT32 runLength, UINT64& R, const UINT64*& pR, UINT64& Qi, const UINT64*& pQi, INT32& i, mfnGetNextQi getNextQi );
        void emitXMmismatch( char*& p, INT32 runLength, UINT64& R, const UINT64*& pR, UINT64& Qi, const UINT64*& pQi, INT32& i, mfnGetNextQi getNextQi );
        void emitXMgapR( char*& p, INT32 runLength, UINT64& R, const UINT64*& pR, UINT64& Qi, const UINT64*& pQi, INT32& i, mfnGetNextQi getNextQi );
        void emitXMgapQ( char*& p, INT32 runLength, UINT64& R, const UINT64*& pR, UINT64& Qi, const UINT64*& pQi, INT32& i, mfnGetNextQi getNextQi );
        void convertXM( char* p, UINT32 cb, QAI* pQAI );

    protected:
        UINT32 emitCIGARf( char* pbuf, QAI* pQAI );
        UINT32 emitCIGARr( char* pbuf, QAI* pQAI );
        void emitQUALf( char* pbuf, QAI* pQAI, char* pMbuf );
        void emitQUALr( char* pbuf, QAI* pQAI, char* pMbuf );
        UINT32 emitQUALfr( char* pbuf, QAI* pQAI, UINT32 iMate, UINT32 iMetadata );
        UINT32 emitSubIdAsRNAME( char* pbuf, INT16 subId );
        UINT32 emitMDfr( char* p, QAI* pQAI );
        INT16 emitQNAMEpu( char* p, QAI* pQAI, UINT32 iMate, UINT32 iMetadata );
        INT16 emitRGID( char* p, QAI* pQAI, UINT32 iMate, UINT32 iMetadata );
        UINT32 emitXMfr( char* pbuf, QAI* pQAI );

    public:
        SAMFormatBase( QBatch* pqb, INT16 nEmitFields, bool emitA21TraceFields );
        virtual ~SAMFormatBase( void );
};
