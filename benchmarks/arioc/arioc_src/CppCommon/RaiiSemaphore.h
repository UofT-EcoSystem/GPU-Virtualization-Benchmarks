/*
  RaiiSemaphore.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiSemaphore__

/// <summary>
/// Class <c>RaiiSemaphore</c> provides a "resource acquisition is initialization" wrapper for a Windows semaphore.
/// </summary>
class RaiiSemaphore
{
    public:
        const INT32 MaxCount;

    private:
#ifdef _WIN32
        HANDLE      Handle;
#endif

#ifdef __GNUC__
        sem_t       m_sem;
#endif

    public:
        RaiiSemaphore( INT32 _initialCount, INT32 _maximumCount = 0 );
        virtual ~RaiiSemaphore( void );
        void Wait( DWORD msTimeout );
        void Release( INT32 _releaseCount );
};
