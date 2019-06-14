/*
  tuBaseA.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"
#include <stdexcept>

#pragma region static member variables
DWORD tuBaseA::WorkerThreadLaunchTimeout = 30000;   // 30 seconds
#pragma endregion

#pragma region constructor/destructor
/// default constructor
tuBaseA::tuBaseA() : m_pWorker(NULL),
                     m_pevWorkerStarted(NULL),
                     m_evWorkerTerminated(true,false) // manual reset; initially nonsignalled
{
}

/// destructor
tuBaseA::~tuBaseA()
{
    /* Destruct RaiiWorkerThread instance that contains the worker thread.  In Windows, this closes the thread
        handle.  In Linux pthreads, this is a no-op. */
    delete m_pWorker;
}
#pragma endregion

#pragma region private methods
/// <summary>
/// Worker thread entry point (i.e. the first code executed on the worker thread)
/// </summary>
THREADPROC_RVAL __stdcall tuBaseA::workerThreadEntry( LPVOID p )
{
    // address the caller tuBaseA instance    
    tuBaseA* pInstance = reinterpret_cast<tuBaseA*>(p);

    try
    {
        // signal that the worker thread is running (see Start())
        pInstance->m_pevWorkerStarted->Set();

        // run the worker thread implementation
        pInstance->main();
    }
    catch( ApplicationException* pex )
    {
        // dump the exception and terminate as abruptly as possible
        exit( pex->Dump() );
    }
    catch( std::runtime_error ex )
    {
        /* We avoid making this C++ project dependent on the CUDA Thrust source-code distribution by catching the std type
            from which the Thrust exception type (thrust::system::system_error) inherits. */
        UINT32 tid = GetCurrentThreadId();
        ApplicationException* pex = new ApplicationException( __FILE__, __LINE__, "unhandled exception in TaskUnit worker thread %u (0x%08x): %s", tid, tid, ex.what() );
        exit( pex->Dump() );
    }
    catch( ... )
    {
        UINT32 tid = GetCurrentThreadId();
        ApplicationException* pex = new ApplicationException( __FILE__, __LINE__, "unhandled C++ exception in TaskUnit worker thread %u (0x%08x)", tid, tid );
        exit( pex->Dump() );
    }

    // signal that the worker thread has completed
    pInstance->m_evWorkerTerminated.Set();

    return 0;
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Runs the virtual implementation of main() on a worker thread
/// </summary>
void tuBaseA::Start()
{
    // create a thread sync object
    RaiiSyncEventObject evWorkerStarted( true, false );         // manual reset; initially nonsignalled
    m_pevWorkerStarted = &evWorkerStarted;

    // launch a CPU worker thread
    m_pWorker = new RaiiWorkerThread( workerThreadEntry, this );

    // wait for the worker thread to signal that it is executing
    evWorkerStarted.Wait( tuBaseA::WorkerThreadLaunchTimeout );
    
    m_pevWorkerStarted = NULL;
}

/// <summary>
/// Waits for the virtual implementation of main() to terminate
/// </summary>
void tuBaseA::Wait( UINT32 msTimeout )
{
    // wait for the worker thread to complete
    m_evWorkerTerminated.Wait( msTimeout );
}
#pragma endregion
