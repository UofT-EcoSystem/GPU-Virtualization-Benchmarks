/*
  RaiiSyncEventObject.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiSyncEventObject__

/// <summary>
/// Class <c>RaiiSyncEventObject</c> provides a "resource acquisition is initialization" wrapper for a synchronization
/// event object ("condition variable").
/// </summary>
class RaiiSyncEventObject
{
    private:
#ifdef _WIN32
        HANDLE  m_handle;
#endif

#ifdef __GNUC__
    	bool            m_state;
        bool            m_autoReset;
		pthread_cond_t  m_cond;
        RaiiMutex       m_mtx;

        bool waitInLoop( DWORD msTimeout );
#endif

    public:
        RaiiSyncEventObject( bool _manualReset, bool _initialState );
        virtual ~RaiiSyncEventObject( void );
        bool Wait( DWORD msTimeout );
        void Set( void );
        void Reset( void );
};
