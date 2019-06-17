/*
  RaiiSemaphore.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#ifdef _WIN32
/// constructor (INT32, INT32)
RaiiSemaphore::RaiiSemaphore( INT32 _initialCount, INT32 _maximumCount ) : Handle(INVALID_HANDLE_VALUE), MaxCount(_maximumCount)
{
    // create an unnamed semaphore object
    this->Handle = CreateSemaphoreA( NULL, _initialCount, _maximumCount, NULL );
}

// destructor
RaiiSemaphore::~RaiiSemaphore()
{
    if( this->Handle != INVALID_HANDLE_VALUE )
    {
        CloseHandle( this->Handle );
        this->Handle = INVALID_HANDLE_VALUE;
    }
}

/// [public] method Wait
void RaiiSemaphore::Wait( DWORD msTimeout )
{
    DWORD dwRval = WaitForSingleObject( this->Handle, msTimeout );
    if( dwRval )
    {
        DWORD dwErr = GetLastError();
        throw new ApplicationException( __FILE__, __LINE__, "%s: WaitForSingleObject (timeout=%u) returned %x (GetLastError returned %d (%x))",
                                                            __FUNCTION__, msTimeout, dwRval, dwErr, dwErr );
    }
}

/// [public] method Release
void RaiiSemaphore::Release( INT32 _releaseCount )
{
    ReleaseSemaphore( this->Handle, _releaseCount, NULL );
}
#endif

#ifdef __GNUC__
/// constructor (INT32, INT32)
RaiiSemaphore::RaiiSemaphore( INT32 _initialCount, INT32 _maximumCount ) : MaxCount(_maximumCount)
{
    // create an unnamed intraprocess semaphore object
    int rval = sem_init( &m_sem, 0, _initialCount );
    if( rval )
        throw new ApplicationException( __FILE__, __LINE__, "%s: sem_init returned %d", __FUNCTION__, rval );
}

// destructor
RaiiSemaphore::~RaiiSemaphore()
{
    int rval = sem_destroy( &m_sem );
    if( rval )
        CDPrint( cdpCD0, "%s: sem_destroy returned %d", __FUNCTION__, rval );
}

/// [public] method Wait
void RaiiSemaphore::Wait( DWORD msTimeout )
{
    if( msTimeout == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: timeout value may not be zero", __FUNCTION__ );

    if( msTimeout == INFINITE )
    {
        int rval = sem_wait( &m_sem );
        if( rval )
            throw new ApplicationException( __FILE__, __LINE__, "%s: sem_wait returned %d", __FUNCTION__, rval );
    }
    else
    {
        timespec ts;
        HiResTimer::IntervalToAbsoluteTime( ts, msTimeout );
        int rval = sem_timedwait( &m_sem, &ts );
        if( rval )
            throw new ApplicationException( __FILE__, __LINE__, "%s: sem_timedwait (timeout=%u) returned %d", __FUNCTION__, msTimeout, rval );
    }
}

/// [public] method Release
void RaiiSemaphore::Release( INT32 releaseCount )
{
    if( releaseCount <= 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: release count must be greater than zero", __FUNCTION__ );

    do
    {
        int rval = sem_post( &m_sem );
        if( rval )
            throw new ApplicationException( __FILE__, __LINE__, "%s: sem_post returned %d", __FUNCTION__, rval );
    }
    while( --releaseCount );

    // validate the semaphore count
    int count = 0;
    sem_getvalue( &m_sem, &count );
    if( count > this->MaxCount )
        throw new ApplicationException( __FILE__, __LINE__, "%s: semaphore count %d is greater than the configured maximum count (%d)", __FUNCTION__, count, this->MaxCount );
}
#endif
