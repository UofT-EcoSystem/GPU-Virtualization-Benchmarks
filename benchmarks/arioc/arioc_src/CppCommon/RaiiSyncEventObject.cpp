/*
  RaiiSyncEventObject.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#ifdef _WIN32
/// constructor (bool, bool)
RaiiSyncEventObject::RaiiSyncEventObject( bool _manualReset, bool _initialState ) : m_handle(NULL)
{
    m_handle = ::CreateEvent( NULL, _manualReset, _initialState, NULL );
}

// destructor
RaiiSyncEventObject::~RaiiSyncEventObject()
{
    if( m_handle )
    {
        ::CloseHandle( m_handle );
        m_handle = NULL;
    }
}

/// [public] method Wait
bool RaiiSyncEventObject::Wait( DWORD msTimeout )
{
    // wait for the specified timeout interval
    DWORD dwRval = ::WaitForSingleObject( m_handle, msTimeout );

    if( dwRval == WAIT_OBJECT_0 )
        return true;                // the object is signalled

    if( (dwRval == WAIT_TIMEOUT) && (msTimeout == 0) )
        return false;               // the object is not signalled but the caller did not wait

    // at this point we either timed out or there is a synchronization error
    DWORD dwErr = GetLastError();
    throw new ApplicationException( __FILE__, __LINE__, "%s: WaitForSingleObject (timeout=%u) returned %x (GetLastError returned %d (%x))",
                                                        __FUNCTION__, msTimeout, dwRval, dwErr, dwErr );
}

/// [public] method Set
void RaiiSyncEventObject::Set()
{
    SetEvent( m_handle );
}

/// [public] method Reset
void RaiiSyncEventObject::Reset()
{
    ResetEvent( m_handle );
}
#endif

#ifdef __GNUC__
/* Notes on proper usage of signal and wait operations and their mutex:

    bool initialised = false;
    mutex mt;
    convar cv;

    void *thread_proc(void *)
    {
       ...
       mt.lock();
       initialised = true;
       cv.signal();
       mt.unlock();
    }

    int main()
    {
       ...
       mt.lock();
       while(!initialised) cv.wait(mt);
       mt.unlock();
    }

   From: https://stackoverflow.com/questions/11471930/how-to-wait-for-starting-thread-to-have-executed-init-code/11472124#11472124
*/

/// constructor (bool, bool)
RaiiSyncEventObject::RaiiSyncEventObject( bool _manualReset, bool _initialState ) : m_state(_initialState), m_autoReset(!_manualReset), m_mtx(false)
{
    int rval = pthread_cond_init( &m_cond, NULL );
    if( rval )
        throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_cond_init returned %d", __FUNCTION__, rval );
}

// destructor
RaiiSyncEventObject::~RaiiSyncEventObject()
{
    int rval = pthread_cond_destroy( &m_cond );
    if( rval )
        CDPrint( cdpCD0, "%s: pthread_cond_destroy returned %d", __FUNCTION__, rval );
}

/// [private] method waitInLoop
bool RaiiSyncEventObject::waitInLoop( DWORD msWait )
{
    if( m_state )
    {
        /* At this point we own m_mtx, but m_state is already true:
            - if this is an autoreset event, we need to reset
            - otherwise, all is as it should be after the event has been signalled
        */
        if( m_autoReset )
            m_state = false;

        return true;
    }

    /* At this point m_state is false (i.e., non-signalled) */
	if( msWait == 0 )
        return false;       // the event isn't signalled but the caller doesn't want to wait

    /* At this point we own m_mtx (i.e., a nonzero timeout interval was specified, so pthread_mutex_lock must have succeeded) */
    if( msWait == INFINITE )
    {
        /* A pthreads signal is basically a signal associated with a mutex-protected value in memory (in this case,
            m_state):

            The pthread_cond_signal() function shall unblock at least one of the threads that are blocked on the specified
            condition variable cond (if any threads are blocked on cond).
                
            If more than one thread is blocked on a condition variable, the scheduling policy shall determine the order in
            which threads are unblocked.
                
            When each thread unblocked as a result of a pthread_cond_broadcast() or pthread_cond_signal() returns from its
            call to pthread_cond_wait() or pthread_cond_timedwait(), the thread shall own the mutex with which it called
            pthread_cond_wait() or pthread_cond_timedwait().
                
            The thread(s) that are unblocked shall contend for the mutex according to the scheduling policy (if applicable),
            and as if each had called pthread_mutex_lock().

            The point is that the same mutex must be used around m_state in the thread that sets the signal and in all of
            the threads that are waiting on the signal.  Even if multiple threads are signalled, the mutex ensures that
            their access to m_state is serialized so that the "autoreset" policy can be implemented.
                
            We use a loop on pthread_cond_wait() mainly because pthread_cond_wait() is designed so that it is possible for
            two different threads to be signalled even if pthread_cond_signal() is called only once!  (In the *nix world,
            this is officially documented as a "spurious wakeup" and considered a "feature".)
            
            Since two different threads can return successfully from pthread_cond_wait() after a single call to
            pthread_cond_signal(), each thread must examine the signal state to be sure that the signal wasn't
            already handled in the other thread -- hence the loop on pthread_cond_wait().

            This problem is mitigated, however, by using pthread_cond_signal() only for autoreset event objects.  In this case,
            the first thread to return from pthread_cond_wait() grabs the mutex and resets the state, so that any other thread(s)
            that happened to return from pthread_cond_wait() due to the "spurious wakeup" phenomenon will see that m_state == false
            and keep looping.

            For manual-reset objects, we intend for all waiting threads to be signalled and to see m_state == true, so we use
            pthread_cond_broadcast() instead of pthread_cond_signal() anyway.
        */
        while( !m_state )
        {
            int rval = pthread_cond_wait( &m_cond, &m_mtx.Mtx );
            if( rval )
                throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_cond_wait (tag=\"%s\") returned %d", __FUNCTION__, rval );
        }
    }

    else    // timed wait
    {
        // compute the absolute system time at which the timeout will occur if the condition variable isn't signalled
		timespec ts;
        HiResTimer::IntervalToAbsoluteTime( ts, msWait );

        // (see comments above)
        while( !m_state )
        {
            int rval = pthread_cond_timedwait( &m_cond, &m_mtx.Mtx, &ts );
            if( rval )
                throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_cond_timedwait (timeout=%u) returned %d", __FUNCTION__, msWait, rval );
        }
    }

    /* At this point:
        - the current thread owns m_mtx
        - the wait succeeded (otherwise an exception would have been thrown) and m_state == true (i.e., set by the thread
            that called Set())
    */

    // if the reset policy is "autoreset", we need to reset the state
	if( m_autoReset )
		m_state = false;

    return true;
}

/// [public] method Wait
bool RaiiSyncEventObject::Wait( DWORD msTimeout )
{
    if( msTimeout == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: timeout value may not be zero", __FUNCTION__ );

    // wait for ownership of the mutex
    m_mtx.Wait( msTimeout );

    // wait for a signal
    bool waitResult = waitInLoop( msTimeout );

    // release the mutex
    m_mtx.Release();

    return waitResult;
}

/// [public] method Set
void RaiiSyncEventObject::Set()
{
    m_mtx.Wait( INFINITE );

    // set the state flag
    m_state = true;

    // the reset policy determines whether we want to signal just one thread or all threads (see comments in waitInLoop() above):
    if( m_autoReset )        
    {
        int rval = pthread_cond_signal( &m_cond );
        if( rval )
            throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_cond_signal returned %d", __FUNCTION__, rval );
    }

    else    // (manual reset)
    {
        int rval = pthread_cond_broadcast( &m_cond );
        if( rval )
            throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_cond_broadcast returned %d", __FUNCTION__, rval );
    }

    m_mtx.Release();
}

/// [public] method Reset
void RaiiSyncEventObject::Reset()
{
    m_mtx.Wait( INFINITE );
    m_state = false;            // reset the state flag
    m_mtx.Release();
}
#endif
