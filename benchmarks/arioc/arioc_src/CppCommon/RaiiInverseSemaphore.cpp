/*
  RaiiInverseSemaphore.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#ifdef _WIN32
/// constructor (INT32)
RaiiInverseSemaphore::RaiiInverseSemaphore( UINT32 initialCount ) : m_ev(true,(initialCount==0)), Count(initialCount)
{
    /* The constructor initializes a Win32 synchronization event object with:
        - default security
        - manual reset
        - initial state = nonsignalled if initialCount > 0
        - no name
    */

    if( initialCount < 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: count must be non-negative", __FUNCTION__ );

    // set up the CriticalSection structure
    InitializeCriticalSection( &m_criticalSection );
}

/// destructor
RaiiInverseSemaphore::~RaiiInverseSemaphore()
{
    /* The following call to EnterCriticalSection prevents a race condition where
        - thread A instantiates an RaiiInverseSemaphore object
        - thread A passes the object to thread B
        - thread A waits on the object
        - thread B calls Set on the object
        - thread A destructs the object (before Set calls LeaveCriticalSection)
       By calling EnterCriticalSection here, we block until Set (in thread B) calls LeaveCriticalSection
    */
    EnterCriticalSection( &m_criticalSection );

    // at this point we can delete the critical section
    DeleteCriticalSection( &m_criticalSection );
}

/// [public] method Set
void RaiiInverseSemaphore::Set()
{
    EnterCriticalSection( &m_criticalSection );

    // decrement the count and conditionally set the synchronization signal
    if( --this->Count <= 0 )
        m_ev.Set();

    LeaveCriticalSection( &m_criticalSection );
}

/// [public] method Reset
void RaiiInverseSemaphore::Reset( INT32 count )
{
    if( count <= 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: count must be greater than 0", __FUNCTION__ );

    EnterCriticalSection( &m_criticalSection );

    // reset the signal and initialize the count
    m_ev.Reset();
    this->Count = count;

    LeaveCriticalSection( &m_criticalSection );
}

/// [public] method Bump
void RaiiInverseSemaphore::Bump( INT32 increment )
{
    if( increment <= 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: increment must be greater than 0", __FUNCTION__ );

    EnterCriticalSection( &m_criticalSection );

    // reset the signal and add the specified increment to the count
    m_ev.Reset();
    this->Count += increment;

    LeaveCriticalSection( &m_criticalSection );
}

/// [public] method Wait
void RaiiInverseSemaphore::Wait( DWORD msTimeout )
{
    m_ev.Wait( msTimeout );
}
#endif

#ifdef __GNUC__
/// default constructor (INT32)
RaiiInverseSemaphore::RaiiInverseSemaphore( UINT32 initialCount ) : m_ev(true,true), m_mtx(false), Count(initialCount)
{
    /* The pthreads synchronization event object is initialized with:
        - manual reset
        - initial state = signalled

       The pthreads mutex object is initialized with:
        - initially owned = false
    */
}

/// destructor
RaiiInverseSemaphore::~RaiiInverseSemaphore()
{
}

/// [public] method Set
void RaiiInverseSemaphore::Set()
{
    m_mtx.Wait( INFINITE );

    // decrement the count and conditionally set the synchronization signal
    if( --this->Count <= 0 )
        m_ev.Set();

    m_mtx.Release();
}

/// [public] method Reset
void RaiiInverseSemaphore::Reset( INT32 count )
{
    if( count <= 0 )
        throw new ApplicationException( __FILE__, __LINE__, "Reset: count must be greater than 0" );

    m_mtx.Wait( INFINITE );

    // reset the signal and initialize the count
    m_ev.Reset();
    this->Count = count;

    m_mtx.Release();
}

/// [public] method Bump
void RaiiInverseSemaphore::Bump( INT32 increment )
{
    if( increment <= 0 )
        throw new ApplicationException( __FILE__, __LINE__, "Bump: increment must be greater than 0" );

    m_mtx.Wait( INFINITE );

    // reset the signal and add the specified increment to the count
    m_ev.Reset();
    this->Count += increment;

    m_mtx.Release();
}

/// [public] method Wait
void RaiiInverseSemaphore::Wait( DWORD msTimeout )
{
    m_ev.Wait( msTimeout );
}
#endif
