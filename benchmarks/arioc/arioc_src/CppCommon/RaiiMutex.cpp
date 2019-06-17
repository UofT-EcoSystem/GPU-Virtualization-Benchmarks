/*
  RaiiMutex.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#ifdef _WIN32
/// constructor (bool)
RaiiMutex::RaiiMutex( bool _initialOwner ) : m_handle(INVALID_HANDLE_VALUE)
{
    // create an unnamed mutex with default security attributes
    m_handle = CreateMutex( NULL, _initialOwner, NULL );
}

// destructor
RaiiMutex::~RaiiMutex()
{
    if( m_handle != INVALID_HANDLE_VALUE )
    {
        CloseHandle( m_handle );
        m_handle = INVALID_HANDLE_VALUE;
    }
}

/// [public] method Wait
void RaiiMutex::Wait( DWORD msTimeout )
{
    if( msTimeout == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: timeout value may not be zero", __FUNCTION__ );

    DWORD dwRval = WaitForSingleObject( m_handle, msTimeout );
    if( dwRval )
    {
        DWORD dwErr = ::GetLastError();
        throw new ApplicationException( __FILE__, __LINE__, "%s: WaitForSingleObject (timeout=%u) returned %x (GetLastError returned %d (%x))",
                                                            __FUNCTION__, msTimeout, dwRval, dwErr, dwErr );
    }

    /* At this point we own the mutex. */
}

/// [public] method Release
void RaiiMutex::Release()
{
    ReleaseMutex( m_handle );
}
#endif

#ifdef __GNUC__
/// constructor (bool)
RaiiMutex::RaiiMutex( bool _initialOwner )
{
    int rval = pthread_mutex_init( &this->Mtx, NULL );
    if( rval )
        throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_mutex_init returned %d", __FUNCTION__, rval );

    if( _initialOwner )
        this->Wait( 500 );
}

// destructor
RaiiMutex::~RaiiMutex()
{
    int rval = pthread_mutex_destroy( &this->Mtx );
    if( rval )
        CDPrint( cdpCD0, "%s: pthread_mutex_destroy returned %d", __FUNCTION__, rval );
}

/// [public] method Wait
void RaiiMutex::Wait( DWORD msTimeout )
{
    if( msTimeout == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: timeout value may not be zero", __FUNCTION__ );

    if( msTimeout == INFINITE )
    {
        int rval = pthread_mutex_lock( &this->Mtx );
        if( rval )
            throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_mutex_lock returned %d", __FUNCTION__, rval );
    }
    else
    {
        // compute the absolute time at which the specified timeout would occur
        timespec ts;
        HiResTimer::IntervalToAbsoluteTime( ts, msTimeout );

        // wait for ownership of the mutex
        int rval = pthread_mutex_timedlock( &this->Mtx, &ts );
        if( rval )
            throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_mutex_timedlock (timeout=%u) returned %d", __FUNCTION__, msTimeout, rval );
    }

    /* At this point we own the mutex */
}

/// [public] method Release
void RaiiMutex::Release()
{
    int rval = pthread_mutex_unlock( &this->Mtx );
    if( rval )
        throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_mutex_unlock returned %d", __FUNCTION__, rval );
}
#endif
