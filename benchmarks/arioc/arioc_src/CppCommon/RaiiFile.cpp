/*
  RaiiFile.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#ifdef _WIN32
#pragma warning ( push )
#pragma warning ( disable : 4996 )		// don't nag us about strerror being "deprecated"
#endif

#pragma region constructors and destructor
// default constructor
RaiiFile::RaiiFile() : Handle(-1)
{
}

// constructor (char*)
RaiiFile::RaiiFile( char* fileSpec ) : Handle(-1)
{
    this->Open( fileSpec );
}

// constructor (char*, bool)
RaiiFile::RaiiFile( char* fileSpec, bool readOnly ) : Handle(-1)
{
    if( readOnly )
        this->OpenReadOnly( fileSpec );
    else
        this->Open( fileSpec );
}

// destructor
RaiiFile::~RaiiFile()
{
    this->Close();
}
#pragma endregion

#pragma region private methods
/// [private] method internalOpen
void RaiiFile::internalOpen( char* fileSpec, int _openFlag, int _shareFlag, int _permissionMode )
{
    // sanity check
    if( this->Handle >= 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: file %s is already open", __FUNCTION__, fileSpec );

    // save a copy of the file specification
    INT32 cb = static_cast<INT32>(strlen( fileSpec )) + 1;
    this->FileSpec.Realloc( cb, false );
    memcpy_s( this->FileSpec.p, cb, fileSpec, cb );

    // open the file using the specified flags
#ifdef _WIN32
    errno_t errNo =_sopen_s( &this->Handle, fileSpec, _openFlag, _shareFlag, _permissionMode );
    if( errNo )
        throw new ApplicationException( __FILE__, __LINE__, "_sopen_s failed for %s (error %d: %s)", this->FileSpec.p, errNo, strerror(errNo) );
#endif
#ifdef __GNUC__
    this->Handle = open( fileSpec, _openFlag, _permissionMode );
    if( this->Handle < 0 )
        throw new ApplicationException( __FILE__, __LINE__, "open failed for %s (error %d: %s)", this->FileSpec.p, errno, strerror(errno) );
#endif

    CDPrint( cdpCDc, "%s: this=0x%016llx fileSpec=%s", __FUNCTION__, this, this->FileSpec.p );
}
#pragma endregion

#pragma region public methods
/// [public] method Close
void RaiiFile::Close()
{
    if( this->Handle >= 0 )
    {
        _close( this->Handle );
        this->Handle = -1;

        CDPrint( cdpCDc, "%s: this=0x%016llx fileSpec=%s", __FUNCTION__, this, this->FileSpec.p );
    }
}

/// [public] method Open
void RaiiFile::Open( char* fileSpec )
{
    /* Open the file for sequential reading and writing:
        - if the file does not exist, it is created
        - if the file does exist, it is truncated
    */
    internalOpen( fileSpec, 
                  _O_BINARY | _O_CREAT | _O_TRUNC | _O_SEQUENTIAL | _O_RDWR,
                  _SH_DENYNO,
                  _S_IREAD | _S_IWRITE );
}

/// [public] method OpenNoTruncate
void RaiiFile::OpenNoTruncate( char* fileSpec )
{
    // open the file for sequential reading and writing
    internalOpen( fileSpec,
                  _O_BINARY | _O_CREAT | _O_SEQUENTIAL |_O_RDWR,
                  _SH_DENYNO,
                  _S_IREAD | _S_IWRITE );
}

/// [public] method OpenReadOnly
void RaiiFile::OpenReadOnly( char* fileSpec )
{
    // open the file for reading only
    internalOpen( fileSpec,
                  _O_BINARY | _O_RDONLY,
                  _SH_DENYWR,
                  _S_IREAD );     // (ignored because _O_CREAT is not specified)
}

/// [public] method Read
INT64 RaiiFile::Read( void* buf, INT64 cb )
{
    // if there are fewer than 2**30 bytes to read, do it in one chunk
    if( cb <= 0x3FFFFFFF )
    {
        INT32 rval = _read( this->Handle, buf, static_cast<INT32>(cb) );
        if( rval < 0 )
        {
#ifdef _WIN32
            DWORD errVal = GetLastError();
            throw new ApplicationException( __FILE__, __LINE__, "error reading %s (errno %u: %s, Windows error %u)", this->FileSpec.p, errno, strerror(errno), GetLastError() );
#endif
#ifdef __GNUC__
            throw new ApplicationException( __FILE__, __LINE__, "error reading %s (errno %u: %s)", this->FileSpec.p, errno, strerror(errno) );
#endif
        }

        return rval;
    }

    // read chunks of 2**30 bytes per chunk
    char* p = static_cast<char*>(buf);
    INT64 cbRemaining = cb;
    while( cbRemaining && !this->EndOfFile() )
    {
        // read one maximum-sized chunk
        INT32 cbRead = static_cast<INT32>(min2(cbRemaining, static_cast<INT64>(0x40000000)));
        INT32 cbChunk = _read( this->Handle, p, cbRead );
        if( cbChunk < 0 )
        {
#ifdef _WIN32
            DWORD errVal = GetLastError();
            throw new ApplicationException( __FILE__, __LINE__,
										    "error reading %s (errno %u: %s, Windows error %u): cbRead=%d cbRemaining=%lld buf=0x%016llx",
				                            this->FileSpec.p, errno, strerror(errno), GetLastError(), cbRead, cbRemaining, buf );
#endif
#ifdef __GNUC__
            throw new ApplicationException( __FILE__, __LINE__,
											"error reading %s (errno %u: %s): cbRead=%d cbRemaining=%lld buf=0x%016llx",
											this->FileSpec.p, errno, strerror(errno), cbRead, cbRemaining, buf);
#endif
        }

        // track the number of bytes remaining to be read
        cbRemaining -= cbChunk;

        // point to the next free byte in the buffer
        p += cbChunk;
    }

    return cb - cbRemaining;
}

/// [public] method Write
INT64 RaiiFile::Write( const void* buf, INT64 cb )
{
    INT64 cbWritten = 0;

    do
    {
        // write the data in reasonably-sized chunks
        INT32 cbToWrite = static_cast<INT32>(min2( (cb-cbWritten), WRITECHUNKSIZE ));
        INT32 rval = _write( this->Handle, static_cast<const char*>(buf)+cbWritten, cbToWrite );
        if( rval == -1 )
            throw new ApplicationException( __FILE__, __LINE__, "_write for %s failed (error %u: %s)", this->FileSpec.p, errno, strerror(errno) );

        cbWritten += rval;
    }
    while( cbWritten < cb );

    return cbWritten;
}

/// [public] method Seek
INT64 RaiiFile::Seek( INT64 ofs, int origin )
{
    return _lseeki64( this->Handle, ofs, origin );
}

/// [public] method EOF
bool RaiiFile::EndOfFile()
{
    INT64 posCurrent = _lseeki64( this->Handle, 0, SEEK_CUR );
    INT64 posEof = _lseeki64( this->Handle, 0, SEEK_END );
    if( posCurrent == posEof )
        return true;                                    // the current file position is at the end of the file

    _lseeki64( this->Handle, posCurrent, SEEK_SET );    // restore the file position (not at the end of the file)
    return false;
}

/// [public] method FileSize
INT64 RaiiFile::FileSize()
{
    INT64 posCurrent = _lseeki64( this->Handle, 0, SEEK_CUR );
    INT64 posEof = _lseeki64( this->Handle, 0, SEEK_END );
    _lseeki64( this->Handle, posCurrent, SEEK_SET );
    return posEof;
}

/// [public] method Preallocate
void RaiiFile::Preallocate( INT64 cb, RaiiSyncEventObject* prseoComplete )
{
    /* This method preallocates space in a physical file.  In Windows, writing at a file position beyond the current end of file extends the
        file; all space between the current end of file and the new end of file that is not explicitly written is zeroed by the OS.  This can
        take a significant amount of time for a multi-gigabyte file -- hence this little function, which does this operation on a one-shot
        worker thread.
    */
    RaiiFileWorkerP::LaunchWorkerThread( prseoComplete, this, cb );
}

/// [public] method ConcurrentFill
INT64 RaiiFile::ConcurrentFill( INT64 pos, const void* buf, INT64 cb, INT32 nThreads )
{
    /* This method uses worker threads to write the contents of a "huge" (i.e. gigabyte-scale) buffer:
        - a call to RaiiFile::Preallocate must be used to ensure that the file contents are preallocated and zeroed prior to calling this method
        - each thread writes a number of bytes that is a multiple of the file-allocation granularity, so there should be no file-allocation units
            that are written by more than one thread
        - an additional thread writes any leftover bytes
        - if necessary, the file size is shrunk to the specified value
    */

    RaiiInverseSemaphore isemWorkers;
    const char* pFrom = reinterpret_cast<const char*>(buf);
    INT64 cbRemaining = cb;
    INT64 cbWritten = 0;

    // launch all but one of the specified number of worker threads
    for( INT32 n=0; (n<(nThreads-1)) && (cbRemaining > FILE_ALLOCATION_GRANULARITY); ++n )
    {
        // compute the size of the n'th thread's data chunk
        INT64 cbChunk = blockdiv( cbRemaining, nThreads-n );

        // compute the number of file allocation units in the chunk by rounding upward to the nearest file-allocation boundary
        cbChunk = blockdiv( cbChunk, FILE_ALLOCATION_GRANULARITY );

        // compute the number of bytes to be written by the n'th thread 
        cbChunk *= FILE_ALLOCATION_GRANULARITY;

        // launch the n'th thread
        RaiiFileWorkerF::LaunchWorkerThread( &isemWorkers, this, pFrom, pos, cbChunk, &cbWritten );

        // point to the start of the next thread's data
        pFrom += cbChunk;

        // point to the next thread's file position
        pos += cbChunk;

        // update the total number of bytes to be written by the remaining threads
        cbRemaining -= cbChunk;
    }

    // launch one more worker thread to write the remaining bytes
    RaiiFileWorkerF::LaunchWorkerThread( &isemWorkers, this, pFrom, pos, cbRemaining, &cbWritten );

    // wait for the worker threads to terminate
    DWORD msTimeout = static_cast<DWORD>(1000 * cb / (1*1024*1024));    // assume a worst-case minimum disk write speed of 1Mb/s
    isemWorkers.Wait( msTimeout );

    // set the file size
    INT64 posEof = _lseeki64( this->Handle, 0, SEEK_END );
    if( posEof > cb )
    {
        errno_t errNo = _chsize_s( this->Handle, cb );
        if( errNo )
            throw new ApplicationException( __FILE__, __LINE__, "%s: unable to resize file %s (error %d: %s)", __FUNCTION__, this->FileSpec.p, errNo, strerror(errNo) );
    }

    return cbWritten;
}
#pragma endregion

#pragma region static member functions
/// <summary>
/// Tests whether the specified file exists
/// </summary>
/// <remarks>This is a C++ static method.</remarks>
bool RaiiFile::Exists( const char* fileSpec )
{
    struct _stat64 statBuf;

    // we use a 64-bit version of the stat function so as to accommodate files whose size is greater than 4GB.
    int rval = _stat64( fileSpec, &statBuf );
    if( rval )
    {
        int errVal = errno;
        switch( errVal )
        {
            case ENOENT:        // file not found
                return false;

            case EINVAL:
                throw new ApplicationException( __FILE__, __LINE__, "_stat64 failed for '%s' (invalid parameter)", fileSpec );

            default:
                throw new ApplicationException( __FILE__, __LINE__, "_stat64 failed unexpectedly (errno=%d) for '%s'", errVal, fileSpec );
        }
    }

    return true;
}
#pragma endregion

#ifdef _WIN32
#pragma warning ( pop )     // disable : 4996
#endif
