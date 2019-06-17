/*
  RaiiCriticalSection.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiCriticalSection__

#ifdef _WIN32
/// <summary>
/// Class <c>RcsFactory</c> creates a per-template instance of a Win32 critical section.
/// </summary>
class RcsFactory
{
    private:
        static const UINT32 MAX_CS_COUNT = 32;          // maximum number of CRITICAL_SECTION instances that can be managed here

        static CRITICAL_SECTION m_CS[MAX_CS_COUNT];     // list of CRITICAL_SECTION structs
        static bool             m_isCS[MAX_CS_COUNT];   // list of flags that indicate whether the corresponding CRITICAL_SECTION instance has been initialized

    public:
        template <class T>
        static CRITICAL_SECTION* GetCriticalSection()
        {
            /* There is one instance of this static variable for each specialization of this templated method.  Because the variable
                is a C++ static, its value is initialized (through a call to getNextCS()) only the first time each specialization is called.
            */
            static CRITICAL_SECTION* pCS( getNextCS() );

            // return a pointer to the iCS'th critical section instance
            return pCS;
        }

    private:
        static CRITICAL_SECTION* getNextCS( void );

    public:
        RcsFactory( void );
        ~RcsFactory( void );
};

/// <summary>
/// Class <c>RaiiCriticalSection&lt;T&gt;</c> provides a "resource acquisition is initialization" wrapper for a Win32 critical section.
/// </summary>
/// <remarks>
/// This class is templated so that critical sections can be "scoped" by class.
/// </remarks>
template <class T> class RaiiCriticalSection
{
    /* Getting this right is nontrivial:
        - We need to initialize one critical section per specialization of this class.  See comments in RcsFactory::getNextCS().
        - If this constructor is called by multiple threads for the same T, we need to deal with race conditions.  (See
            http://blogs.msdn.com/b/oldnewthing/archive/2004/03/08/85901.aspx.)  The rules of C++ guarantee that the constructor executes
            exactly once, but there is no guarantee of thread safety.  In effect, the C++ compiler generates a guard flag in the constructor
            to enforce this behavior:

                ctor()
                {
                    static bool ctorHasBeenInvoked = false;     // guard bit
                    if( !ctorHasBeenInvoked )
                    {
                        ctorHasBeenInvoked = true;              // set the guard bit

                        getNextCS();                            // get a new CRITICAL_SECTION reference
                    }
                }

          If the current thread invokes the constructor after another thread has already set the guard bit but before the
           CRITICAL_SECTION reference has been initialized, the current thread's CRITICAL_SECTION reference will be null.
             
          Our workaround is simple: we wait until the pointer to the CRITICAL_SECTION instance has been initialized before
           we try to use it.

          Since the value of the pointer can be changed by a thread other than the current thread, we designate the pointer value
           as "volatile" to encourage the C++ compiler to reevaluate the conditional on every iteration of the spin loop.
    */
    CRITICAL_SECTION* volatile m_pCS;

    public:
        RaiiCriticalSection() : m_pCS( RcsFactory::GetCriticalSection<T>() )
        {
            /* if the CRITICAL_SECTION reference is uninitialized, spin until it gets initialized, either by another thread (see above)
                or by the current thread */
            while( m_pCS == NULL )
            {
                /* give up the remainder of this thread's current time slice and let all other same-priority threads execute;
                    hopefully this will let another thread initialize the static critical-section pointer */
                Sleep( 0 );

                if( m_pCS == NULL )
                {
                    // no other thread did the work, so this thread will do it 
                    m_pCS = RcsFactory::GetCriticalSection<T>();
                }
            }

            EnterCriticalSection( m_pCS );
        }

        ~RaiiCriticalSection()
        {
            LeaveCriticalSection( m_pCS );
        }
};
#endif

#ifdef __GNUC__
/// <summary>
/// Class <c>RcsFactory</c> creates a per-template instance of a Win32 critical section.
/// </summary>
class RcsFactory
{
    private:
        static const UINT32 MAX_CS_COUNT = 32;          // maximum number of pthread_mutex_t instances that can be managed here

        static pthread_mutex_t  m_CS[MAX_CS_COUNT];     // list of pthread_mutex_t structs
        static bool             m_isCS[MAX_CS_COUNT];   // list of flags that indicate whether the corresponding pthread_mutex_t instance has been initialized

    public:
        template <class T>
        static pthread_mutex_t* GetCriticalSection()
        {
            /* There is one instance of this static variable for each specialization of this templated method.  Because the variable
                is a C++ static, its value is initialized (through a call to getNextCS()) only the first time each specialization is called.
            */
            static pthread_mutex_t* pCS( getNextCS() );

            // return a pointer to the iCS'th critical section instance
            return pCS;
        }

    private:
        static pthread_mutex_t* getNextCS( void );

    public:
        RcsFactory( void );
        ~RcsFactory( void );
};

/// <summary>
/// Class <c>RaiiCriticalSection&lt;T&gt;</c> provides a "resource acquisition is initialization" wrapper for a pthreads pthread_mutex_t
/// (a reasonable approximation of a Windows CRITICAL_SECTION).
/// </summary>
/// <remarks>
/// This class is templated so that critical sections can be "scoped" by class.
/// </remarks>
template <class T> class RaiiCriticalSection
{
    /* See notes for the Windows implementation.  Presumably gcc generates similar initializer code so the same implementation
        is reasonable (but that's an untested assumption).
    */
    pthread_mutex_t* volatile m_pCS;

    public:
        RaiiCriticalSection() : m_pCS( RcsFactory::GetCriticalSection<T>() )
        {
            /* if the pthread_mutex_t reference is uninitialized, spin until it gets initialized, either by another thread (see above)
                or by the current thread */
            while( m_pCS == NULL )
            {
                /* give up the remainder of this thread's current quantum and let all other threads execute;
                    hopefully this will let another thread initialize the static critical-section pointer */
                sched_yield();

                if( m_pCS == NULL )
                {
                    // no other thread did the work, so this thread will do it 
                    m_pCS = RcsFactory::GetCriticalSection<T>();
                }
            }

            int rval = pthread_mutex_lock( m_pCS );
            if( rval )
                throw new ApplicationException( __FILE__, __LINE__, "%s: pthread_mutex_lock returned %d", __FUNCTION__, rval );
        }

        ~RaiiCriticalSection()
        {
            pthread_mutex_unlock( m_pCS );
        }
};
#endif
