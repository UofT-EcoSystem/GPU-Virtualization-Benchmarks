/*
  tuGpuP.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] default constructor
tuGpuP::tuGpuP()
{
}

/// <param name="iPart">a 0-based ordinal (assigned by the caller) (one per GPU)</param>
/// <param name="pab">a reference to the application's <c>AriocP</c> instance</param>
tuGpuP::tuGpuP( INT16 _gpuDeviceOrdinal, AriocBase* _pab ) : tuGpu( _gpuDeviceOrdinal, _pab )
{
#if TODO_CHOP
    // look for the optional "cgaReserved" Xparam
    INT32 i = pab->paamb->Xparam.IndexOf( "cgaReserved" );
    if( i >= 0 )
        m_cbCgaReserved = max2( CUDAMINRESERVEDGLOBALMEMORY, pab->paamb->Xparam.Value(i) );
#endif
}

/// destructor
tuGpuP::~tuGpuP()
{
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Defines a pipelined workflow for doing paired-end alignments
/// </summary>
/// <remarks>There is one <c>tuGpuP</c> instance per GPU.</remarks>
void tuGpuP::main()
{
    CRVALIDATOR;

    // set up a CudaDeviceBinding instance on the current thread
    CudaDeviceBinding cdb( CUDADEVICEBINDING_TIMEOUT );

    CDPrint( cdpCDe, "[%d] %s...", cdb.GetDeviceProperties()->cudaDeviceId, __FUNCTION__ );

#if TODO_CHOP_WHEN_DEBUGGED
    // increase the CUDA printf FIFO buffer size from its default (1MB)
    CRVALIDATE = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 250*1024*1024);
#endif


    try
    {
        // create a per-GPU info container
        GpuInfo gi( &cdb, m_gpuDeviceOrdinal );

        // load R sequence data and H lookup tables
        loadR( &gi );
        loadH( &gi );

        /* Here we reserve the largest remaining chunk of contiguous device memory so that we can manage
            our own buffer allocations and avoid fragmentation (as well as any bugs remaining in the CUDA
            runtime memory-management APIs).

           We need to cooperate with the Thrust global-memory allocator as well.  See notes in Windux.h.            
        */
        CREXEC( CudaGlobalAllocator cga(cdb, m_cbCgaReserved) );
        gi.pCGA = &cga;
        CDPrint( cdpCDb, "[%d] tuGpuP::main: CGA = %lld bytes", gi.deviceId, cga.cbFree );

        // create a pool of QBatch instances to use in the pipeline
        QBatchPool qbp( 2, m_pab, &gi );

        // get an empty QBatch instance
        INT32 iPart = -1;                   // set up to start with the first input-file partition
        QBatch* pqb = qbp.Depool( m_pab->pifgQ->GetEstimatedNmax(), m_pab->pifgQ->GetNextPartition( iPart ) );

        // initialize a reference to the final task unit in the pipeline
        tuTailP* tailP = NULL;

        m_pab->Watchdog.Watch( gi.deviceId );

        // move each pair of input files (and their associated metadata files) through the pipeline
        while( pqb->pfiQ )
        {
            /* Process the reads from the current file partition.  The current thread is associated with one GPU
                and the file handles and the data are associated with this thread, so we can read the data without
                doing any thread synchronization. */

            QReaderP qrp( pqb, iPart );
            while( qrp.LoadQ( pqb ) )
            {
                if( m_pab->doMainLoopJoin )
                {
                    // join all tuGpuP threads here
                    m_pab->semMainLoop.Set();                         // decrement the inverse-semaphore count
                    m_pab->semMainLoop.Wait( MAINLOOPJOIN_TIMEOUT );  // wait for all tuGpuP threads to reach this point
                    m_pab->semMainLoop.Bump( 1 );                     // increment the inverse-semaphore count
                }

                HiResTimer hrt(ms);
                CDPrint( cdpCDe, "[%d] %s: top of loop (batch 0x%016llx)", gi.deviceId, __FUNCTION__, pqb );

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 0 );

#if TODO_CHOP_WHEN_DEBUGGED
                CDPrint( cdpCD0, "%s: first sqId=0x%016llx QwBuffer.n=%d DB.nQ=%u", __FUNCTION__, pqb->QwBuffer.p->sqId[0], pqb->QwBuffer.n, pqb->DB.nQ );
#endif

                // copy the Qwarp and interleaved Q buffers for the current batch into GPU global memory
                loadQ( pqb );

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 10 );

                // load Q-sequence metadata
                tuLoadMP loadMm( pqb, qrp.QFI, qrp.fileMm, qrp.MFIm, pqb->MFBm );    // row metadata
                tuLoadMP loadMq( pqb, qrp.QFI, qrp.fileMq, qrp.MFIq, pqb->MFBq );    // quality scores
                loadMm.Start();
                loadMq.Start();

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 20 );

                // setup for nongapped alignment
                tuSetupN setupN( pqb );
                setupN.Start();
                setupN.Wait();

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 30 );

                // do nongapped alignment
                tuAlignN alignN( pqb );
                alignN.Start();
                alignN.Wait();

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 40 );

                // build nongapped BRLEAs
                tuFinalizeN finalizeN( pqb );
                finalizeN.Start();

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 50 );

                // do seed-and-extend gapped alignment for unmapped mates
                tuAlignGwn alignGwn( pqb );
                alignGwn.Start();
                alignGwn.Wait();

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 60 );

                // finalize BRLEAs
                tuFinalizeGwn finalizeGwn( pqb );
                finalizeGwn.Start();

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 70 );

                // do seed-and-extend gapped alignment
                tuAlignGs alignGs( pqb );
                alignGs.Start();
                alignGs.Wait();

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 80 );

                // compute KMH for unmapped reads
                tuComputeKMH computeKMH( pqb );
                computeKMH.Start();

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 90 );

                // finalize BRLEAs
                tuFinalizeGs finalizeGs( pqb );
                finalizeGs.Start();

                // finalize BRLEAs
                tuFinalizeGc finalizeGc( pqb );
                finalizeGc.Start();

                // finalize BRLEAs
                tuFinalizeGwc finalizeGwc( pqb );
                finalizeGwc.Start();

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 100 );

                // wait for metadata to load
                loadMm.Wait();
                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 110 );

                loadMq.Wait();
                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 120 );

                finalizeN.Wait();
                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 130 );
                finalizeGwn.Wait();
                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 140 );
                finalizeGs.Wait();
                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 150 );
                finalizeGc.Wait();
                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 160 );
                finalizeGwc.Wait();
                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 170 );

                // wait for KMH computation to complete
                computeKMH.Wait();
                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 180 );

                // free the GPU buffers that contain the Qwarp and Qi data
                unloadQ( pqb );
                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 190 );

                /* The final task unit does the following:
                    - classify alignment results
                    - write alignment results to disk
                    - release the current QBatch instance

                   Unlike the other task units in this loop, which go out of scope on each iteration, this final task unit executes
                   concurrently with the subsequent iteration, so we keep a reference to it outside the loop.

                   At this point, tailP contains a reference to the tuTailP instance from the previous loop iteration.
                */
                if( tailP )
                {
                    INT32 msElapsed = hrt.GetElapsed( false );
                    m_pab->Watchdog.SetWatchdogState( gi.deviceId, 200 );

                    tailP->Wait();
                    m_pab->Watchdog.SetWatchdogState( gi.deviceId, 210 );

                    delete tailP;
                    m_pab->Watchdog.SetWatchdogState( gi.deviceId, 220 );

                    msElapsed = hrt.GetElapsed( false ) - msElapsed;
                    if( msElapsed )
                        CDPrint( cdpCDe, "[%d] %s: waited for tailP (batch 0x%016llx): %dms", gi.deviceId, __FUNCTION__, pqb, msElapsed );
                }

                // create and start a new "tail" instance
                tailP = new tuTailP( pqb );
                tailP->Start();

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 300 );

                // update progress
                m_pab->updateProgress( pqb->DB.nQ );

                CDPrint( cdpCDe, "[%d] %s: bottom of loop (batch 0x%016llx): %dms", gi.deviceId, __FUNCTION__, pqb, hrt.GetElapsed( true ) );

                // get an empty QBatch instance
                pqb = qbp.Depool( pqb->Nmax, pqb->pfiQ );

                m_pab->Watchdog.SetWatchdogState( gi.deviceId, 310 );
            }

            // get another partition of file data
            pqb->pfiQ = m_pab->pifgQ->GetNextPartition( iPart );
        }

        // remove this thread's most recent contribution to the inverse semaphore
        m_pab->semMainLoop.Set();

        // wait for the last task unit in the pipeline to complete
        if( tailP )
        {
            INT32 msElapsed = m_hrt.GetElapsed( false );
            m_pab->Watchdog.SetWatchdogState( gi.deviceId, 400 );

            tailP->Wait();
            m_pab->Watchdog.SetWatchdogState( gi.deviceId, 410 );

            delete tailP;
            m_pab->Watchdog.SetWatchdogState( gi.deviceId, 420 );

            msElapsed = m_hrt.GetElapsed(false) - msElapsed;
            CDPrint( cdpCDe, "[%d] %s: waited for final tailP (batch 0x%016llx): %dms", gi.deviceId, __FUNCTION__, pqb, msElapsed );
        }

        // discard the current contents of the Qwarp and Qi buffers
        pqb->DB.Qi.Free();
        pqb->DB.Qw.Free();

        // return the last QBatch instance (which is empty) to the pool
        pqb->Release();
        if( !qbp.Wait( 1000 ) )
            throw new ApplicationException( __FILE__, __LINE__, "workflow complete but all QBatch instances have not been processed" );

        m_pab->Watchdog.Halt( gi.deviceId );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }

    CDPrint( cdpCDe, "[%d] %s completed", cdb.GetDeviceProperties()->cudaDeviceId, __FUNCTION__ );
}
#pragma endregion
