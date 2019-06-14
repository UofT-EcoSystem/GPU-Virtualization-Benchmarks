/*
  tuGpuU.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#if TODO_CHOP_WHEN_DEBUGGED
#include "nvToolsExtCuda.h"
#endif

#pragma region constructor/destructor
/// [private] default constructor
tuGpuU::tuGpuU()
{

}

/// <param name="iPart">a 0-based ordinal (assigned by the caller) (one per GPU)</param>
/// <param name="pab">a reference to the application's <c>AriocU</c> instance</param>
tuGpuU::tuGpuU( INT16 _gpuDeviceOrdinal, AriocBase* _pab ) : tuGpu(_gpuDeviceOrdinal,_pab)
{
#if TODO_CHOP
    // look for the optional "cgaReserved" Xparam
    INT32 i = pab->paamb->Xparam.IndexOf( "cgaReserved" );
    if( i >= 0 )
        m_cbCgaReserved = max2( CUDAMINRESERVEDGLOBALMEMORY, pab->paamb->Xparam.Value(i) );
#endif
}

/// destructor
tuGpuU::~tuGpuU()
{
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Defines a pipelined workflow for doing paired-end alignments
/// </summary>
/// <remarks>There is one <c>tuGpuU</c> instance per GPU.</remarks>
void tuGpuU::main()
{
    CRVALIDATOR;

    // set up a CudaDeviceBinding instance on the current thread
    CudaDeviceBinding cdb( CUDADEVICEBINDING_TIMEOUT );

    CDPrint( cdpCD3, "[%d] %s...", cdb.GetDeviceProperties()->cudaDeviceId, __FUNCTION__ );


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

        /* Here we reserve the largest available chunk of contiguous device memory so that we can manage
            our own buffer allocations and avoid fragmentation (as well as any bugs remaining in the CUDA
            runtime memory-management APIs).

           We need to cooperate with the Thrust global-memory allocator as well.  See notes in Windux.h.            
        */
        CREXEC( CudaGlobalAllocator cga(cdb, m_cbCgaReserved) );
        gi.pCGA = &cga;
        CDPrint( cdpCDb, "[%d] tuGpuU::main: CGA = %lld bytes", gi.deviceId, cga.cbFree );

        // create a pool of QBatch instances to use in the pipeline
        QBatchPool qbp( 2, m_pab, &gi );

        // get an empty QBatch instance
        INT32 iPart = -1;                   // set up to start with the first input-file partition
        QBatch* pqb = qbp.Depool( m_pab->pifgQ->GetEstimatedNmax(), m_pab->pifgQ->GetNextPartition(iPart) );

        // initialize a reference to the final task unit in the pipeline
        tuTailU* tailU = NULL;

        // move each input file (and its associated metadata) through the pipeline
        while( pqb->pfiQ )
        {
            /* Process the reads from the current file partition.  The current thread is associated with one GPU
                and the file handles and the data are associated with this thread, so we can read the data without
                doing any thread synchronization. */
            QReaderU qru( pqb, iPart );
            while( qru.LoadQ( pqb ) )
            {
                if( m_pab->doMainLoopJoin )
                {
                    // join all tuGpuU threads here
                    m_pab->semMainLoop.Set();                         // decrement the inverse-semaphore count
                    m_pab->semMainLoop.Wait( MAINLOOPJOIN_TIMEOUT );  // wait for all tuGpuU threads to reach this point
                    m_pab->semMainLoop.Bump( 1 );                     // increment the inverse-semaphore count
                }

                HiResTimer hrt(ms);
                CDPrint( cdpCDe, "[%d] %s: top of loop", gi.deviceId, __FUNCTION__ );

#if TRACE_SQID
    Qwarp* pQwFirst = pqb->QwBuffer.p;
    Qwarp* pQwLast = pqb->QwBuffer.p + pqb->QwBuffer.n - 1;
    CDPrint( cdpCD0, "%s: sqId 0x%016llx - 0x%016llx", __FUNCTION__, pQwFirst->sqId[0], pQwLast->sqId[pQwLast->nQ-1] );
#endif
    
                // copy the Qwarp and interleaved Q buffers for the current batch into GPU global memory
                loadQ( pqb );

                // load Q-sequence metadata
                tuLoadMU loadMm( pqb, qru.QFI, &qru.fileMm, qru.MFIm, pqb->MFBm );   // row metadata
                tuLoadMU loadMq( pqb, qru.QFI, &qru.fileMq, qru.MFIq, pqb->MFBq );   // quality scores
                loadMm.Start();
                loadMq.Start();

                // setup for nongapped alignment
                tuSetupN setupN( pqb );
                setupN.Start();
                setupN.Wait();

                // do nongapped alignment
                tuAlignN alignN( pqb );
                alignN.Start();
                alignN.Wait();

                // build nongapped BRLEAs
                tuFinalizeN finalizeN( pqb );
                finalizeN.Start();

                // do seed-and-extend gapped alignment
                tuAlignGs alignGs( pqb );
                alignGs.Start();
                alignGs.Wait();

                // finalize BRLEAs
                tuFinalizeGc finalizeGc( pqb );
                finalizeGc.Start();

                // wait for metadata to load
                loadMm.Wait();
                loadMq.Wait();

                // wait for all BRLEAs to finalize
                finalizeN.Wait();
                finalizeGc.Wait();

                // free the GPU buffers that contain the Qwarp and Qi data
                unloadQ( pqb );

                /* The final task unit does the following:
                    - classify alignment results
                    - write alignment results to disk
                    - release the current QBatch instance

                   Unlike the other task units in this loop, which go out of scope on each iteration, this final task unit executes
                   concurrently with the subsequent iteration, so we keep a reference to it outside the loop.

                   At this point, tailU contains a reference to the tuTailU instance from the previous loop iteration.
                */
                if( tailU )
                {
                    INT32 msElapsed = hrt.GetElapsed( false );

                    tailU->Wait();
                    delete tailU;

                    msElapsed = hrt.GetElapsed( false ) - msElapsed;
                    if( msElapsed )
                        CDPrint( cdpCDe, "[%d] %s: waited for tailU (batch 0x%016llx): %dms", gi.deviceId, __FUNCTION__, pqb, msElapsed );
                }

                // create and start a new "tail" instance
                tailU = new tuTailU( pqb );
                tailU->Start();

                // update progress
                m_pab->updateProgress( pqb->DB.nQ );

                // get an empty QBatch instance
                pqb = qbp.Depool( pqb->Nmax, pqb->pfiQ );

                CDPrint( cdpCDe, "[%d] %s: bottom of loop (batch 0x%016llx): %dms", gi.deviceId, __FUNCTION__, pqb, hrt.GetElapsed( true ) );
            }

            // get another partition of file data
            pqb->pfiQ = m_pab->pifgQ->GetNextPartition( iPart );    // (iPart is updated by GetNextPartition())
        }

        // remove this thread's most recent contribution to the inverse semaphore
        m_pab->semMainLoop.Set();

        // wait for the last task unit in the pipeline to complete
        if( tailU )
        {
            INT32 msElapsed = m_hrt.GetElapsed( false );

            tailU->Wait();
            delete tailU;

            msElapsed = m_hrt.GetElapsed( false ) - msElapsed;
            CDPrint( cdpCDe, "[%d] %s: waited for final tailU (batch 0x%016llx): %dms", gi.deviceId, __FUNCTION__, pqb, msElapsed );
        }

        // discard the current contents of the Qwarp and Qi buffers
        pqb->DB.Qi.Free();
        pqb->DB.Qw.Free();

        // return the current QBatch instance (which is empty) to the pool
        pqb->Release();
        if( !qbp.Wait( 1000 ) )
            throw new ApplicationException( __FILE__, __LINE__, "workflow complete but all QBatch instances have not been processed" );


#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( m_gpuDeviceOrdinal, cdpCD0, "[%d] %s: CudaGlobalAllocator::msMalloc=%llu nMalloc=%llu", gi.deviceId, __FUNCTION__, CudaGlobalAllocator::msMalloc, CudaGlobalAllocator::nMalloc );
#endif
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }

    CDPrint( cdpCD4, "[%d] %s completed", cdb.GetDeviceProperties()->cudaDeviceId, __FUNCTION__ );
}
#pragma endregion
