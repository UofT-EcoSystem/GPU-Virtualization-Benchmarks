/*
  RaiiFile.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiFile__

#ifndef __RaiiSyncEventObject__
#include "RaiiSyncEventObject.h"
#endif


/* File allocation granularity (AKA "sector size" or "cluster size"):
    - 1MB is a multiple of all known disk file-allocation sizes (see "Default cluster size for NTFS, FAT, and exFAT" at http://support.microsoft.com/kb/140365)
*/
#define FILE_ALLOCATION_GRANULARITY     (1024 * 1024)

/* Maximum size of a file's "base name" (i.e., the first dot-separated element in the filename), including a terminal null.
*/
#define BASENAME_MAX    32

/// <summary>
/// Class <c>RaiiFile</c> provides a "resource acquisition is initialization" wrapper for a read/write, sequential-access Win32 file.
/// </summary>
class RaiiFile
{
    private:
        static const DWORD PREALLOCATE_WORKER_THREAD_TIMEOUT = 5000;
        static const INT32 WRITECHUNKSIZE = 10 * 1024*1024;                 // 10Mb

    public:
        INT32               Handle;
        WinGlobalPtr<char>  FileSpec;

    private:
        void internalOpen( char* fileSpec, int _openFlag, int _shareFlag = 0, int _permissionMode = 0 );

    public:
        RaiiFile( void );
        RaiiFile( char* fileSpec );
        RaiiFile( char* fileSpec, bool readOnly );
        ~RaiiFile( void );
        void Close( void );
        void Open( char* fileSpec );
        void OpenNoTruncate( char* fileSpec );
        void OpenReadOnly( char* fileSpec );
        INT64 Read(  void* buf, INT64 cb );
        INT64 Write( const void* buf, INT64 cb );
        INT64 Seek( INT64 ofs, int origin );
        bool EndOfFile( void );
        INT64 FileSize( void );
        void Preallocate( INT64 cb, RaiiSyncEventObject* prseoComplete );
        INT64 ConcurrentFill( const INT64 pos, const void* buf, INT64 cb, INT32 nThreads );
        static bool Exists( const char* fileSpec );
};

