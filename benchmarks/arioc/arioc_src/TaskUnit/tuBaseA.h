/*
  tuBaseA.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuBaseA__

/// <summary>
/// Task unit base class implementation (asynchronous)
/// </summary>
class tuBaseA
{
    protected:
        RaiiWorkerThread*       m_pWorker;
        RaiiSyncEventObject*    m_pevWorkerStarted;
        RaiiSyncEventObject     m_evWorkerTerminated;

    public:
        static DWORD WorkerThreadLaunchTimeout;

    private:
        static THREADPROC_RVAL __stdcall  workerThreadEntry( LPVOID p );

    protected:
        virtual void main( void ) = 0;

    public:
        tuBaseA( void );
        virtual ~tuBaseA( void );
        void Start( void );
        void Wait( UINT32 msTimeout = INFINITE );
};
