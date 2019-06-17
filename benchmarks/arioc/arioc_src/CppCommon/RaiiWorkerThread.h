/*
  RaiiWorkerThread.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiWorkerThread__

#ifdef _WIN32
typedef DWORD THREADPROC_RVAL;
#endif
#ifdef __GNUC__
typedef void* THREADPROC_RVAL;
#endif

/// <summary>
/// Class <c>RaiiWorkerThread</c> provides a "resource acquisition is initialization" wrapper for a Windows thread.
/// </summary>
class RaiiWorkerThread
{
    private:
#ifdef _WIN32
        HANDLE  m_handle;
#endif

#ifdef __GNUC__
        pthread_t   m_tid;  // pthread thread ID
#endif

    public:
        RaiiWorkerThread( LPTHREAD_START_ROUTINE, LPVOID );
        virtual ~RaiiWorkerThread( void );
        bool Wait( DWORD msTimeout );
};
