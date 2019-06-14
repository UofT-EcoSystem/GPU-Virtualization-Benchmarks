/*
  RaiiFileWorkerP.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// [public] constructor (workerThreadParams*)
RaiiFileWorkerP::RaiiFileWorkerP( workerThreadParams* pwtp ) : m_wtp(*pwtp)
{
}

/// [public] destructor
RaiiFileWorkerP::~RaiiFileWorkerP()
{
}
#pragma endregion

#pragma region static methods
/// [public static] method LaunchWorkerThread
void RaiiFileWorkerP::LaunchWorkerThread( RaiiSyncEventObject* prseoWorkerTerminated, RaiiFile* prfInstance, INT64 cb )
{
    RaiiSyncEventObject rseoWorkerStarted( true, false );

    // build a parameter list for the worker thread
    workerThreadParams wtp = { &rseoWorkerStarted, prseoWorkerTerminated, prfInstance, cb};

    // launch a CPU worker thread
    RaiiWorkerThread rwt( RaiiFileWorkerP::workerThreadProc, &wtp );

    // wait for the worker thread to signal that it is executing
    rseoWorkerStarted.Wait( WORKER_THREAD_TIMEOUT );
}

/// [private static] method workerThreadProc
THREADPROC_RVAL __stdcall RaiiFileWorkerP::workerThreadProc( LPVOID pp )
{
    THREADPROC_RVAL rval = 0;

    try
    {
        // call the worker thread implementation
        RaiiFileWorkerP Ariocwl( static_cast<workerThreadParams*>(pp) );
        rval = Ariocwl.workerThreadImpl();
    }
    catch( ApplicationException* pex )
    {
        exit( pex->Dump() );
    }
    catch( ... )
    {
        // at this point the best we can do is indicate that an unhandled exception occurred
        CDPrint( cdpCD0, "%s: unhandled exception in worker thread", __FUNCTION__ );
        exit( EXIT_FAILURE );
    }

    return rval;
}
#pragma endregion

#pragma region private methods
/// [private] method workerThreadImpl
THREADPROC_RVAL RaiiFileWorkerP::workerThreadImpl()
{
    // set signals
    m_wtp.prseoWorkerStarted->Set();  // signal that this thread is executing

    CDPrint( cdpCDd, "%s starts for %s: %lld bytes", __FUNCTION__, m_wtp.prf->FileSpec.p, m_wtp.cb );

    /* The following (or something similar) ought to work in Windows without the cost of zeroing the allocated space
        in the file:
    
            HANDLE hFile = reinterpret_cast<HANDLE>( _get_osfhandle( m_wtp.prf->Handle ) );

            LARGE_INTEGER pos;
            pos.QuadPart = cb;

            SetFilePointerEx( hFile, pos, NULL, FILE_END );
            SetEndOfFile( hFile );

            pos.QuadPart = 0;
            SetFilePointerEx( hFile, pos, NULL, FILE_BEGIN );
            SetFileValidData( hFile, cb-1 );

       The problem is that SetFileValidData seems to fail with Windows error 1314 even if the process is owned by a login
        that has SE_MANAGE_VOLUME_NAME permission.  But, in any event, it is more secure to zero everything despite the
        performance penalty.
    */

    /* Change the physical size of the file
        - if the file is being extended, the OS appends zeroes to extend the file to the specified size
        - the Windows implementation of _chsize_s seems to be much faster than seeking to the end of the file and writing some data
            (perhaps they're using SetFileValidData as above, but the C++ source code for _chsize_s is hidden so we don't know for sure)
    */
    errno_t errNo = _chsize_s( m_wtp.prf->Handle, m_wtp.cb );

#pragma warning ( push )
#pragma warning ( disable : 4996 )		// don't nag us about strerror being "deprecated"
    if( errNo )
        throw new ApplicationException( __FILE__, __LINE__, "unable to preallocate file %s to %lld bytes (error %d: %s)", m_wtp.prf->FileSpec.p, m_wtp.cb, errNo, strerror(errNo) );
#pragma warning ( pop )

        CDPrint( cdpCDd, "%s completed", __FUNCTION__ );

        // signal that this worker thread is exiting
        m_wtp.prseoWorkerTerminated->Set();

    return 0;
}
#pragma endregion
