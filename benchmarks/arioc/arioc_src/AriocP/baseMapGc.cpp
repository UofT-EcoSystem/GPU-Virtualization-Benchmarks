/*
  baseMapGc.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseMapGc::baseMapGc()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbg">a reference to a <c>DeviceBuffersG</c> instance</param>
/// <param name="phb">a reference to a <c>HostBuffers</c> instance</param>
baseMapGc::baseMapGc( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, HostBuffers* phb ) : baseMapCommon( pqb, pdbg, phb )
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseMapGc", m_cudaThreadsPerBlock );
    m_ptum = AriocBase::GetTaskUnitMetrics( ptumKey, "baseMapGc" );

    // set up the seed-iteration loop limit
    m_isiLimit = pqb->pab->aas.ACP.seedDepth - 1;

    /* Set up the constant parameter values for CUDA kernels; the DeviceBuffersG pointer references an object defined in
        the current QBatch instance so these values will remain available for subsequent CUDA kernels to use ... */
    pdbg->InitKernelParameters( pqb );
}

/// destructor
baseMapGc::~baseMapGc()
{
}
#pragma endregion

#pragma region protected methods
/// [private] method initGlobalMemory
void baseMapGc::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

           low:     Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      QIDs of unmapped mates
                    Ru      subId bits for QIDs of unmapped mates
                    Dc      mapped candidates with unmapped opposite mates
                   
            high:   Di      isolated D values (initially empty)
                    Du      unmapped D values (initially empty)
        */

        // allocate an empty buffer to contain previously-processed D values
        CREXEC( m_pdbg->Du.Alloc( cgaHigh, 1, false ) );
        m_pdbg->Du.n = 0;
        SET_CUDAGLOBALPTR_TAG(m_pdbg->Du,"Du");

        /* Allocate a buffer to contain isolated D values (i.e., "isolated" as in "in the seed iterations so far,
            no nearby seed location has been found for these D values, but nearby seeds may turn up in subsequent
            seed iterations").

           This buffer is freed either in the final seed iteration (see filterJforSeedInterval()) or in the
            resetGlobalMemory() implementation (see below).
        */
        CREXEC( m_pdbg->Di.Alloc( cgaHigh, 1, false ) );
        m_pdbg->Di.n = 0;
        SET_CUDAGLOBALPTR_TAG(m_pdbg->Di,"Di");
    }
    catch( ApplicationException* pex )
    {
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
        CRTHROW;
    }
}

/// [private] method initGlobalMemory_LJSI
void baseMapGc::initGlobalMemory_LJSI()
{
    CRVALIDATOR;

    /* CUDA global memory layout after initialization:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs of unmapped mates
                Ru      subId bits for QIDs of unmapped mates
                Dc      mapped candidates with unmapped opposite mates
                DBj.D   D values for candidates for seed-and-extend alignment

       high:    DBj.cnJ cumulative per-Q J-list sizes
                DBj.nJ  per-Q J-list sizes
                DBj.oJ  per-seed J-list offsets
                Di      isolated D values
                Du      unmapped D values
    */
    try
    {
        /* The memory-management strategy is:

            - At the start of each outer-loop iteration (in baseMapGc::main):
                - An empty consolidated Dj list (in the low memory-allocation region) is allocated.
            - The Dj list is filled in inner-loop iterations (in baseMapGc::loadJforSeedInterval)
            - At the end of each outer-loop iteration:
                - The consolidated Dj list is updated.
                - The temporary buffers are freed.
        */

        // initially, the D list is empty
        CREXEC( m_pqb->DBj.D.Alloc( cgaLow, 1, false ) );
        m_pqb->DBj.D.n = 0;
    }
    catch( ApplicationException* pex )
    {
        CDPrint( cdpCD0, "%s: exception!", __FUNCTION__ );
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
        CRTHROW;
    }
}

/// [private] method initGlobalMemory_LJSIIteration
void baseMapGc::initGlobalMemory_LJSIIteration( UINT32 nJ )
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      QIDs of unmapped mates
                    Ru      subId bits for QIDs of unmapped mates
                    Dc      mapped candidates with unmapped opposite mates
                    DBj.D   D values for candidates for seed-and-extend alignment

            high:   Diter   J list (D values) for current iteration
                    DBj.cnJ cumulative per-Q J-list sizes
                    DBj.nJ  per-Q J-list sizes
                    DBj.oJ  per-seed J-list offsets
                    Di      isolated D values
                    Du      unmapped D values
        */

        /* The memory-management strategy is:
            - At the start of each inner-loop iteration (in baseMapGc::loadJforSeedInterval):
                - A temporary buffer is allocated in the high memory-allocation region to handle the D values
                    for the iteration.
            - At the end of each inner-loop iteration:
                - The Dj list is consolidated.
                - The temporary buffer is freed.
        */

        CREXEC( m_pdbg->Diter.Alloc(cgaHigh, nJ, true ) );
        m_pdbg->Diter.n = nJ;
        SET_CUDAGLOBALPTR_TAG(m_pdbg->Diter, "Diter");
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory_LJSIIteration
void baseMapGc::resetGlobalMemory_LJSIIteration()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after reset:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      QIDs of unmapped mates
                    Ru      subId bits for QIDs of unmapped mates (null)
                    Dm      QIDs of mapped mates
                    Dj      consolidated list of D values for candidates for seed-and-extend alignment
                   
            high:   cnJ     cumulative per-Q J-list sizes
                    nJ      per-Q J-list sizes
                    oJ      per-seed J-list offsets
                    Du      unmapped D values
        */

        // expand the consolidated Dj buffer
        UINT32 cel = m_pqb->DBj.D.n + m_pdbg->Diter.n;
        m_pqb->DBj.D.Resize( cel );

        // copy the current iteration's Dj values into the consolidated buffer
        CREXEC( m_pdbg->Diter.CopyInDevice( m_pqb->DBj.D.p+m_pqb->DBj.D.n, m_pdbg->Diter.n ) );
        m_pqb->DBj.D.n = cel;

        // discard the current iteration's Dj values
        CREXEC( m_pdbg->Diter.Free() );
    }
    catch( ApplicationException* pex )
    {
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory_LJSI
void baseMapGc::resetGlobalMemory_LJSI()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after reset:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      QIDs of unmapped mates
                    Ru      subId bits for QIDs of unmapped mates (null)
                    Dc      QIDs of mapped mates
                    DBj.D   consolidated Dj list

            high:   Di
                    Du      unmapped D values
        */

        // discard the J-list sizes
        m_pqb->DBj.cnJ.Free();
        m_pqb->DBj.nJ.Free();
        m_pqb->DBj.oJ.Free();
    }
    catch( ApplicationException* pex )
    {
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseMapGc::resetGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after reset:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      QIDs of unmapped mates
                    Ru      subId bits for QIDs of unmapped mates
                    Dc      mapped D values without opposite-mate mappings

            high:   (unallocated)
        */

        /* If the seed-iteration loop terminates early (i.e., because all mappings have been found), then
            the Di buffer will not yet have been freed. (If the Di buffer has already been freed, the call to
            Free() does nothing.) */
        m_pdbg->Di.Free();

        // discard the Du list
        m_pdbg->Du.Free();
    }
    catch( ApplicationException* pex )
    {
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
        CRTHROW;
    }
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Performs seed-and-extend gapped alignment at reference-sequence locations prioritized by seed coverage.
/// </summary>
void baseMapGc::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    // set up CUDA memory
    initGlobalMemory();
    const UINT32 cbSharedPerBlock = initSharedMemory();

    // loop from maximum to minimum seed interval
    for( m_isi=0; (m_isi<=m_isiLimit)&&(m_pdbg->Qu.n); ++m_isi )
    {
        // count J values for seeds for all Q sequences
        baseCountJs countJs( m_ptum->Key, m_pqb, m_pdbg, m_isi );
        countJs.Start();
        countJs.Wait();

        // identify candidate locations for seed-and-extend alignment (DBj.D)
        initGlobalMemory_LJSI();
        loadJforSeedInterval();
        resetGlobalMemory_LJSI();

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: m_isi=%u: DBj.D.n=%u (new candidates)  Du.n=%u (previously processed candidates)",
            m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
            m_isi, m_pqb->DBj.D.n, m_pdbg->Du.n );
#endif

        // build and filter a list of D values that represent potential mappings (Dx)
        filterJforSeedInterval( cbSharedPerBlock );

        // do gapped alignment on the D values in the Dx list
        mapJforSeedInterval();

        // remove mapped QIDs from the list of unmapped QIDs
        pruneQu();
    }

    resetGlobalMemory();

    CDPrint( cdpCD3, "[%d] %s::%s: completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
#pragma endregion
