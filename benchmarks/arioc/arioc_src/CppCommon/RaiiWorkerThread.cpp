/*
  RaiiWorkerThread.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#ifdef _WIN32
// constructor
RaiiWorkerThread::RaiiWorkerThread( LPTHREAD_START_ROUTINE _threadProc, LPVOID _param ) : m_handle(NULL)
{
    m_handle = ::CreateThread( NULL,          // default security attributes
                               0,             // default stack size
                               _threadProc,   // thread proc entry point
                               _param,        // thread proc parameter
                               0,             // flags (thread runs immediately)
                               NULL );        // pointer to thread ID
    if( m_handle == NULL )
    {
        DWORD dwErr = GetLastError();
        throw new ApplicationException( __FILE__, __LINE__, "%s: CreateThread failed: error %d (%x)", __FUNCTION__, dwErr, dwErr );
    }
}

// destructor
RaiiWorkerThread::~RaiiWorkerThread()
{
    if( m_handle )
    {
        ::CloseHandle( m_handle );
        m_handle = INVALID_HANDLE_VALUE;
    }
}

/// [public] method Wait
bool RaiiWorkerThread::Wait( DWORD _msTimeout )
{
    // wait for the specified timeout interval
    DWORD dwRval = ::WaitForSingleObject( m_handle, _msTimeout );

    if( dwRval == WAIT_OBJECT_0 )
        return true;                // the object is signalled (i.e., the thread has terminated)

    if( (dwRval == WAIT_TIMEOUT) && (_msTimeout == 0) )
        return false;               // the object is not signalled but the caller did not wait

    // at this point we either timed out or there is a synchronization error
    DWORD dwErr = GetLastError();
    throw new ApplicationException( __FILE__, __LINE__, "%s: WaitForSingleObject (timeout=%u) returned %x (GetLastError returned %d (%x))",
        __FUNCTION__, _msTimeout, dwRval, dwErr, dwErr );
}
#endif

#ifdef __GNUC__
// constructor
RaiiWorkerThread::RaiiWorkerThread( LPTHREAD_START_ROUTINE _threadProc, void* _param )
{
    /* We use a "detached" thread because the following scenario leads to a memory leak.  (No doubt this is a either a "feature" of pthreads or else
         we're using the worker thread "incorrectly", but no matter.)

         - The worker thread starts.
         - The worker thread terminates.
         - The caller (i.e. the thread on which this RaiiWorkerThread object is constructed) subsequently calls Wait() which calls pthread_timedjoin_np().

       In this case, there is a memory leak, apparently of thread-local storage allocated within pthread_create().  Using a "detached" thread seems to
       prevent this from occurring.
    */
    pthread_attr_t threadAttrs;
    int rval = pthread_attr_init( &threadAttrs );
    if( rval )
    	throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_attr_init returned %d", __FUNCTION__, rval );

    rval = pthread_attr_setdetachstate( &threadAttrs, PTHREAD_CREATE_DETACHED );
    if( rval )
    	throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_attr_setdetachstate returned %d", __FUNCTION__, rval );

    rval = pthread_create( &m_tid,          // thread ID
                           &threadAttrs,    // default thread attributes
                           _threadProc,     // thread proc entry point
                           _param );        // thread proc parameter
    if( rval )
        throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_create returned %d", __FUNCTION__, rval );
}

// destructor
RaiiWorkerThread::~RaiiWorkerThread()
{
    // (nothing to do)
}

/// [public] method Wait
bool RaiiWorkerThread::Wait( DWORD _msTimeout )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}
#endif
