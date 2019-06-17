/*
  AriocE.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocE__

#pragma region structs
struct JBcount
{
    UINT32  nJ;     // number of values in J-list bucket
    UINT32  nB;     // number of buckets with nJ values
};

struct HJpair
{
    UINT32  h;      // hash key
    UINT32  ij;     // index of J value in the J list ("bucket") for the hash key
    Jvalue8 j;      // J value
};
#pragma endregion

class AriocE
{
    private:
        static const DWORD ENCODERWORKERTHREAD_TIMEOUT =    (120 * 60000);      // worker thread timeout
        static const INT64 INPUTFILECUTSIZE =               (1024 * 1024);      // minimum file size for multithreaded parsing
        static const DWORD SNIFFBUFFERSIZE =                (32 * 1024);
        static const INT64 MAXM =                           0x10000000;         // maximum M (reference sequence length)
        static const DWORD FILE_PREALLOCATE_TIMEOUT =       (5*60*1000);        // file preallocation timeout for C, H, and J files

        struct BBregionInfo
        {
            INT64   ofs;    // offset of the region relative to the start of the big-bucket list
            UINT32  n;      // number of adjacent big-bucket kmers in the region
        };

        double                      m_samplingRatio;
        WinGlobalPtr<char>          m_buf;                  // buffer that contains raw sequence data
        WinGlobalPtr<INT64>         m_packed;               // buffer that contains packed sequence data
        WinGlobalPtr<INT64>         m_SqFileCutpoints;      // offsets of cutpoints for concurrent input-file reads
        RaiiFile                    m_outFileSqm;
        RaiiFile                    m_outFileRaw;
        RaiiFile                    m_outFileA21;
        WinGlobalPtr<Hvalue8>       m_H;                    // H table
        WinGlobalPtr<UINT8>         m_xH;                   // bit flags for H table (see shuffleJ())
        WinGlobalPtr<Jvalue8>       m_J;                    // J table
        INT64                       m_celJ;                 // number of elements in J table (including header element and list counts)
        INT64                       m_nJ;                   // total number of J values
        WinGlobalPtr<Cvalue>        m_C;                    // J list (collision list) sizes
        WinGlobalPtr<UINT32>        m_CH;                   // CH table (offsets into C list, ordered by hash value)
        WinGlobalPtr<HJpair>        m_BBHJ;                 // "big bucket" (large collision list) H and J values
        WinGlobalPtr<BBregionInfo>  m_BBRI;                 // big-bucket region info
        INT32                       m_maxJ;                 // maximum J list (collision list) size
        size_t                      m_iCshort;              // index of first "short" J-list reference in sorted C list
        size_t                      m_iCzero;               // index of first zero-length J-list reference in sorted C list
        INT64                       m_nBBHJ;                // number of big-bucket HJ values
        INT64                       m_nBBRI;                // number of big-bucket regions
        DWORD                       m_encoderWorkerThreadTimeout;
        AppMain*                    m_pam;

    public:
        AriocEncoderParams      Params;
        static AppPerfMetrics   PerfMetrics;

    private:
        char* findSOL( char* p, char* pLimit );
        char* findEOL( char* p, char* pLimit );
        void preallocateCHJ( RaiiFile& rfFile, RaiiSyncEventObject& rseoComplete, INT64 cb, const char cLUTtype[2] );
        INT64 writeCHJ( RaiiFile& rfFile, RaiiSyncEventObject* prseoComplete, const void* buf, INT64 cel, INT32* pmsElapsed );
        void accumulateJlistSize( size_t* _piC, UINT32* _pnCard );
        void computeJlistSizes( void );
        void validateHJ( void );
        void validateSubIds( size_t _cbJ );
        INT64 countBBJ( WinGlobalPtr<INT64>* pnBBJ );
        void buildBBJlists( WinGlobalPtr<INT64>* pnBBJ );
        void identifyBBJ( void );
        void setEOLflags( void );
        void encodeR( SAMConfigWriter* pscw );
        void encodeQ();
        void validateQ();
        void buildJlists( void );
        void setBBJlistFlags( void );
        size_t compactHtable( void );
        size_t compactJtable( void );
        void sortBBHJ( WinGlobalPtr<INT64>* pnBBJ );
        void sortJcpu( void );
        void sortJgpu( void );

        static int CvalueComparer_nJ( const void*, const void* );

    public:
        AriocE( INT32 srcId, InputFileGroup& ifgRaw, const Nencoding enc, A21SeedBase* pa21sb, char* outDir, const INT32 maxThreads, double samplingRatio, char qualityScoreBias, bool emitKmers, UINT32 gpuMask, char* qmdQNAME, INT32 maxJ, AppMain* pam );
        ~AriocE( void );
        SqFileFormat SniffSqFile( char* fileSpec );
        void ImportR( void );
        void ImportQ( void );
        static UINT32 Hash6432( UINT64 x );
};
