/*
  RaiiInverseSemaphore.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiInverseSemaphore__

/// <summary>
/// Class <c>RaiiInverseSemaphore</c> implements an "inverse semaphore", i.e. a semaphore that is initially
/// not signaled, and is signaled when its count reaches zero.
/// </summary>
class RaiiInverseSemaphore
{
    private:
        RaiiSyncEventObject m_ev;

#ifdef _WIN32
        CRITICAL_SECTION    m_criticalSection;
#endif

#ifdef __GNUC__
//        pthread_mutex_t     m_mtx;
        RaiiMutex           m_mtx;
#endif

    public:
        volatile int    Count;

    public:
        RaiiInverseSemaphore( UINT32 initialCount = 0 ); 
        virtual ~RaiiInverseSemaphore( void );
        void Reset( INT32 initialCount );
        void Set( void );
        void Bump( INT32 increment );
        void Wait( DWORD msTimeout );
};
