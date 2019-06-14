/*
  baseEncode.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseEncode__

/// <summary>
/// Class <c>baseEncode</c> is the base implementation for the FASTA and FASTQ encoders.
/// </summary>
class baseEncode : public tuBaseA
{
    typedef void (baseEncode::*mfnWrite)( INT64 readId, char* pRaw, INT64 cb );
    typedef void (baseEncode::*mfnEndRowRaw)( void );
    typedef bool (baseEncode::*mfnComputeKmerHash)( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl );

    private:
        static const INT32 QMDBUFFERSIZE = 512;

        INT64                   m_cbUnread;
        RaiiFile                m_outFileSqm;
        WinGlobalPtr<char>      m_outBufSqm;
        RaiiFile                m_outFileRaw;
        WinGlobalPtr<char>      m_outBufRaw;
        INT64                   m_cbRawCurrent;
        RaiiFile                m_outFileSqq;
        WinGlobalPtr<char>      m_outBufSqq;
        INT64                   m_cbSqqCurrent;
        RaiiFile                m_outFileA21;
        WinGlobalPtr<char>      m_outBufA21;
        RaiiFile                m_outFileKmers;
        WinGlobalPtr<char>      m_outBufKmers;
        INT32                   m_maxEncodeHash;
        INT32                   m_subId;
        bool                    m_isRminus;
        bool                    m_pairFlag;
        bool                    m_deleteSCW;        // true: destruct the SAMConfigWriter instance in this baseEncode instance's dtor
        WinGlobalPtr<char>      m_qmdRG;            // read-group strings captured from Q-sequence metadata
        WinGlobalPtr<UINT32>    m_ofsqmdRG;         // offsets of captured read-group strings
        WinGlobalPtr<UINT32>    m_hashqmdRG;        // 32-bit hash of captured read-group strings
        mfnComputeKmerHash      m_computeKmerHash;

    protected:
        static const INT32 INPUTBUFFERSIZE = (10 * 1024*1024);      // 10Mb
        static const INT32 OUTPUTBUFFERSIZE = (10 * 1024*1024);

        AriocEncoderParams*     m_psip;
        SqCategory              m_sqCategory;   // Q, R+, R-
        INT16                   m_iInputFile;   // index of input file specification
        SAMConfigWriter*        m_pscw;         // pointer to SAMConfigWriter instance
        RaiiFile                m_inFile;
        WinGlobalPtr<char>      m_inBuf;
        INT16                   m_cbHash;       // number of bytes required for a hash value of size SeedBaseParameters::hashBitsWidth
        mfnWrite                m_writeSqm;
        mfnWrite                m_writeRaw;
        mfnWrite                m_writeSqq;
        mfnEndRowRaw            m_endRowRaw;
        RaiiSemaphore*          m_psemComplete;
        char                    m_baseName[_MAX_FNAME];
        HiResTimer              m_hrt;

    private:
        void reverseComplement( char* pSqRaw, INT64 cchRaw );
        bool computeKmerHash21R( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl );
        bool computeKmerHash42R( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl );
        bool computeKmerHash84R( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl );
        bool computeKmerHash21Q( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl );
        bool computeKmerHash42Q( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl );
        void writeSqmR( INT64 readId, char* pRaw, INT64 cb );
        void writeRawR( INT64 readId, char* pRaw, INT64 cb );
        void writeSqqR( INT64 readId, char* pRaw, INT64 cb );
        void writeAriocR( INT64 readId, char* pRaw, INT64 nSymbols );
        void writeKmersR( INT64 sqId, INT64* pB2164, INT64 nSymbols );
        void endRowRawR( void );
        void writeSqmQ( INT64 readId, char* pRaw, INT64 cb );
        void writeRawQ( INT64 readId, char* pRaw, INT64 cb );
        void writeSqqQ( INT64 readId, char* pRaw, INT64 cb );
        void writeAriocQ( INT64 readId, char* pRaw, INT64 nSymbols );
        void writeKmersQ( INT64 sqId, INT64* pB2164, INT64 nSymbols );
        void endRowRawQ( void );
        void getNextCaptureInPattern( char** ppcstart, char** ppcend, char* ppat );
        bool getNextOperatorInPattern( char** pppat, char* pcstart, char* pcend );
        INT32 parseQmetadata( char* capture, char* md, INT32 cb, char* pattern );
        INT32 getQmetadataReadGroupIndex( char* pqrg, INT32 cb );

    protected:
        baseEncode( void );
        INT64 readInputFile( char* pRead, INT64 cbRead );
        char* findSOL( char* p, char** ppLimit );
        char* findEOL( char** pp, char** ppLimit );
        void endRowSqqQ( void );
        void flushSqm( void );
        void flushRaw( bool flushCurrent );
        void flushSqq( bool flushCurrent );
        void flushA21( void );
        void flushKmers( void );
        void writeConfigFile( void );

    public:
        baseEncode( AriocEncoderParams* psip, SqCategory sqCat, INT16 iInputFile, RaiiSemaphore* psem, SAMConfigWriter* pscw );
        virtual ~baseEncode( void );
};
