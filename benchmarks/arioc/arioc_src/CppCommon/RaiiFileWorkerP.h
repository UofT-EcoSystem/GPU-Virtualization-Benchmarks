/*
  RaiiFileWorkerP.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiFileWorkerP__


// foreward reference
class RaiiFile;

/// <summary>
/// Class <c>RaiiFileWorkerP</c> implements a worker thread for the RaiiFile::Preallocate function.
/// </summary>
class RaiiFileWorkerP
{
    private:
        static const DWORD WORKER_THREAD_TIMEOUT =  5000;
        static const INT32 EOFBLOCKSIZE = 16384;            // size of a small block of data to write at the end of the file

        struct workerThreadParams
        {
            RaiiSyncEventObject*    prseoWorkerStarted;
            RaiiSyncEventObject*    prseoWorkerTerminated;
            RaiiFile*               prf;
            INT64                   cb;
        };

    private:
        workerThreadParams      m_wtp;

    private:
        static THREADPROC_RVAL __stdcall workerThreadProc( LPVOID pp );

    protected:
        THREADPROC_RVAL workerThreadImpl( void );

    public:
        RaiiFileWorkerP( workerThreadParams* pwtp );
        virtual ~RaiiFileWorkerP( void );
        static void LaunchWorkerThread( RaiiSyncEventObject* prseoWorkerTerminated, RaiiFile* prfInstance, INT64 cb );
};
