/*
  RaiiMutex.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiMutex__

/// <summary>
/// Class <c>RaiiMutex</c> provides a "resource acquisition is initialization" wrapper for a Windows mutex.
/// </summary>
class RaiiMutex
{
    private:
#ifdef _WIN32
        HANDLE  m_handle;
#endif

#ifdef __GNUC__
    public:
        pthread_mutex_t     Mtx;
#endif

    public:
        RaiiMutex( bool _InitialOwner = false );
        virtual ~RaiiMutex( void );
        void Wait( DWORD msTimeout );
        void Release( void );
};
