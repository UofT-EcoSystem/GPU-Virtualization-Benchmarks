/*
  RaiiFileWorkerF.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiFileWorkerF__


// foreward reference
class RaiiFile;

/// <summary>
/// Class <c>RaiiFileWorkerF</c> implements a worker thread for the RaiiFile::ConcurrentFill function.
/// </summary>
class RaiiFileWorkerF
{
    private:
        static const DWORD WORKER_THREAD_TIMEOUT =  5000;

        struct workerThreadParams
        {
            RaiiSyncEventObject*    prseoWorkerStarted;
            RaiiInverseSemaphore*   pisemWorkers;
            RaiiFile*               prf;
            const char*             buf;        // output buffer
            INT64                   pos;        // output file position
            INT64                   cb;         // number of bytes of data to write
            volatile INT64*         pcbWritten; // pointer to the total number of bytes written
        };

    private:
        workerThreadParams      m_wtp;

    private:
        static THREADPROC_RVAL __stdcall workerThreadProc( LPVOID pp );

    protected:
        THREADPROC_RVAL workerThreadImpl( void );

    public:
        RaiiFileWorkerF( workerThreadParams* pwtp );
        virtual ~RaiiFileWorkerF( void );
        static void LaunchWorkerThread( RaiiInverseSemaphore* pisemWorkers, RaiiFile* prfInstance, const char* buf, INT64 pos, INT64 cb, INT64* pcbWritten );
};
