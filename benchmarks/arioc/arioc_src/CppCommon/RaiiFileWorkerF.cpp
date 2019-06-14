/*
  RaiiFileWorkerF.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// [public] constructor (workerThreadParams*)
RaiiFileWorkerF::RaiiFileWorkerF( workerThreadParams* pwtp ) : m_wtp(*pwtp)
{
}

/// [public] destructor
RaiiFileWorkerF::~RaiiFileWorkerF()
{
}
#pragma endregion

#pragma region static methods
/// [public static] method LaunchWorkerThread
void RaiiFileWorkerF::LaunchWorkerThread( RaiiInverseSemaphore* pisemWorkers, RaiiFile* prfInstance, const char* buf, INT64 pos, INT64 cb, INT64* pcbWritten )
{
    RaiiSyncEventObject rseoWorkerStarted( true, false );

    // build a parameter list for the worker thread
    workerThreadParams wtp = { &rseoWorkerStarted, pisemWorkers, prfInstance, buf, pos, cb, pcbWritten };

    // launch a CPU worker thread
    RaiiWorkerThread rwt( RaiiFileWorkerF::workerThreadProc, &wtp );

    // wait for the worker thread to signal that it is executing
    rseoWorkerStarted.Wait( WORKER_THREAD_TIMEOUT );
}

/// [private static] method workerThreadProc
THREADPROC_RVAL __stdcall RaiiFileWorkerF::workerThreadProc( LPVOID pp )
{
    THREADPROC_RVAL rval = 0;

    try
    {
        // call the worker thread implementation
        RaiiFileWorkerF Ariocwl( static_cast<workerThreadParams*>(pp) );
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
THREADPROC_RVAL RaiiFileWorkerF::workerThreadImpl()
{
    // set signals
    m_wtp.pisemWorkers->Bump( 1 );    // increment the worker-thread count
    m_wtp.prseoWorkerStarted->Set();  // signal that this thread is executing

    CDPrint( cdpCDd, "%s: starts (buf=0x%016llX pos=%lld cb=%lld)...", __FUNCTION__, m_wtp.buf, m_wtp.pos, m_wtp.cb );

    RaiiFile f;
    f.OpenNoTruncate( m_wtp.prf->FileSpec.p );

    // seek to the specified position
    f.Seek( m_wtp.pos, SEEK_SET );

    // write the specified number of bytes
    INT64 cbWritten = f.Write( m_wtp.buf, m_wtp.cb );

    // update the total number of bytes written
    InterlockedExchangeAdd( reinterpret_cast<volatile UINT64*>(m_wtp.pcbWritten), static_cast<UINT64>(cbWritten) );

    CDPrint( cdpCDd, "%s completed (%lld bytes written)", __FUNCTION__, cbWritten );

    // signal that this worker thread is exiting
    m_wtp.pisemWorkers->Set();

    return 0;
}
#pragma endregion
