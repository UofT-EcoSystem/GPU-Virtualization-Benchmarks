/*
  baseARowWriter.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
   The idea here is that each calling thread (i.e., one per GPU) fills its own output buffer; the ARowWriter implementation
    flushes that buffer to disk in a thread-safe manner.
*/
#pragma once
#define __baseARowWriter__

// forward declaration
class AriocBase;

#pragma region structs
struct RowWriteCounts
{
    INT64   nMapped1;
    INT64   nMapped2;
    INT64   nUnmapped;
};

struct PairWriteCounts
{
    INT64   nConcordantRows;
    INT64   nDiscordantRows;
    INT64   nRejectedRows;
    INT64   nUnmappedRows;
};
#pragma endregion

/// <summary>
/// Class <c>baseARowWriter</c> is the base implementation for class ARowWriter, which writes alignment data to a file.
/// </summary>
class baseARowWriter
{
    static const INT32 BITBUCKET_SIZE = 10240;  // size of bit bucket referenced by LockOutputBuffer

    protected:
        AriocBase*              m_pab;
        OutputFileInfo          m_ofi;
        char                    m_filespecStub[FILENAME_MAX];
        char                    m_filespecExt[4];
        INT16                   m_outFileSeqNo;
        RaiiFile                m_outFile;
        INT16                   m_nBufs;        // number of output buffers for this baseARowWriter instance
        INT64                   m_nA;           // current number of reads in the current output file
        WinGlobalPtr<char>*     m_outBuf;       // output buffers (one per GPU)
        RaiiMutex               m_mtx;          // mutex

    public:
        bool                    IsActive;
        INT64                   TotalA;         // total number of alignment results written to this ARowWriter object

    public:
        baseARowWriter( AriocBase* pabase );
        virtual ~baseARowWriter( void );
        virtual char* Lock( INT16 iBuf, UINT32 cb );
        virtual void Release( INT16 iBuf, UINT32 cb );
        virtual void Flush( INT16 iBuf );
        virtual void Close( void );
};
