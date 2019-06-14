/*
  baseQReader.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseQReader__

// forward declaration
class QBatch;

/// <summary>
/// Describes a Q-sequence file or a partition thereof.
/// </summary>
struct QfileInfo
{
    InputFileGroup::FileInfo*   pfi;
    INT64                       fp0;        // initial file position
    INT64                       fpLimit;    // final file position
    INT64                       fp;         // current file position
    INT64                       cbPerRead;  // estimated number of bytes in one encoded read
    UINT32                      cbInBuf;    // number of usable bytes in the input buffer
    bool                        eod;        // flag set when the end of the input data is reached

    QfileInfo() : pfi(NULL), fp0(0), fpLimit(0), fp(0), cbPerRead(0), cbInBuf(0), eod(false)
    {
    }
};

/// <summary>
/// Describes a metadata (.sqm or .sqq) file.
/// </summary>
struct MfileInfo
{
    INT64       fp;         // file pointer
    INT64       fpLimit;    // file pointer limit
    const char* ext;        // filename extension ("sqm", "sqq")
    INT64       cb;         // running total number of metadata bytes read
    INT64       nRows;      // running total number of metadata rows read

    MfileInfo() : fp(0), fpLimit(0), ext(NULL), cb(0), nRows(0)
    {
    }
};

/// <summary>
/// Loads encoded short-read data from disk files
/// </summary>
class baseQReader
{
    private:
        static const INT64  SQIDHUNTBUFSIZE = 16*1024;

    protected:
        INT64               m_cbRead;   // total bytes read
        INT64               m_nQ;       // total number of Q sequences (reads) read
        INT16               m_iPart;    // 0-based index of the data partition in the input files
        WinGlobalPtr<UINT8> m_QFB[2];   // Q file buffers
        HiResTimer          m_hrt;

    public:
        QfileInfo           QFI[2];     // Q file info
        MfileInfo           MFIm[2];    // metadata file info
        MfileInfo           MFIq[2];

    protected:
        INT64 openQfile( RaiiFile* pFileQ, QfileInfo* pqfi );
        void openMfile( RaiiFile* pFileM, MfileInfo* pmfi, QfileInfo* pqfi );
        void initQFB( WinGlobalPtr<UINT8>* pQFB, QfileInfo* pqfi, UINT32 nQwarpsPerBatch );
        UINT8* loadFromQfile( RaiiFile* pFileQ, QfileInfo* pqfi, WinGlobalPtr<UINT8>* pQFB, INT32 cb );
        void copyToQwarp( Qwarp* pQw, INT16 iq, UINT64* const pQiBuffer, InputFileGroup::DataRowHeader* pdrh, UINT64* pA21 );
        static INT32 scanBufferForSqId( UINT64 _sqId, WinGlobalPtr<UINT8>* _pbuf, INT64* _pofsFrom, INT64* _pofsTo );

    public:
        baseQReader( INT16 partitionId = -1 );
        virtual ~baseQReader( void );
        virtual bool LoadQ( QBatch* pqb ) = 0;

        static INT64 sqIdHunt( RaiiFile* pFileM, INT64 initialFp, UINT64 sqId );
};
