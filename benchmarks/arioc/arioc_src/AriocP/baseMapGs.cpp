/*
  baseMapGs.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseMapGs::baseMapGs()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbg">a reference to a <c>DeviceBuffersG</c> instance</param>
/// <param name="phb">a reference to a <c>HostBuffers</c> instance</param>
baseMapGs::baseMapGs( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, HostBuffers* phb ) : baseMapCommon( pqb, pdbg, phb )                                                                                 
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseMapGs", m_cudaThreadsPerBlock );
    m_ptum = AriocBase::GetTaskUnitMetrics( ptumKey, "baseMapGs" );

    // set up the seed-iteration loop limit
    m_isiLimit = pqb->pab->aas.ACP.seedDepth - 1;

    /* Set up the constant parameter values for CUDA kernels; the DeviceBuffersG pointer references an object defined in
        the current QBatch instance so these values will remain available for subsequent CUDA kernels to use ... */
    pdbg->InitKernelParameters( pqb );
}

/// destructor
baseMapGs::~baseMapGs()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void baseMapGs::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBgs.Dc mapped D values without opposite-mate mappings

            high:   Du      unmapped D values
        */

        // allocate an empty buffer to contain previously-processed D values
        CREXEC( m_pdbg->Du.Alloc( cgaHigh, 1, false ) );
        m_pdbg->Du.n = 0;
        SET_CUDAGLOBALPTR_TAG(m_pdbg->Du,"Du");

        // initially, the Dc list is empty
        CREXEC( m_pdbg->Dc.Alloc( cgaLow, 1, false ) );
        m_pdbg->Dc.n = 0;
        SET_CUDAGLOBALPTR_TAG(m_pdbg->Dc,"Dc");
    }
    catch( ApplicationException* pex )
    {
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
        CRTHROW;
    }
}

/// [private] method initGlobalMemory_LJSI
void baseMapGs::initGlobalMemory_LJSI()
{
    CRVALIDATOR;

    /* CUDA global memory layout after initialization:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs of unmapped mates
                Ru      subId bits for QIDs of unmapped mates
                DBgs.Dc mapped D values without opposite-mate mappings
                Dj      Dj values for candidates for seed-and-extend alignment

       high:    cnJ     cumulative per-Q J-list sizes
                nJ      per-Q J-list sizes
                oJ      per-seed J-list offsets
    */
    try
    {
        /* The memory-management strategy is:

            - At the start of each outer-loop iteration (in baseMapGs::main):
                - An empty consolidated Dj list (in the low memory-allocation region) is allocated.
            - The Dj list is filled in inner-loop iterations (in baseMapGs::loadJforSeedInterval)
            - At the end of each outer-loop iteration:
                - The consolidated Dj list is updated.
                - The temporary buffers are freed.
        */


#if TODO_CHOP_WHEN_DEBUGGED
        if( m_pqb->DBj.D.p ) DebugBreak();
#endif


        // initially, the D list is empty
        CREXEC( m_pqb->DBj.D.Alloc( cgaLow, 1, false ) );
        m_pqb->DBj.D.n = 0;


#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: DBj.D at 0x%016llx", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_pqb->DBj.D.p );
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
#endif
    }
    catch( ApplicationException* pex )
    {
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
        CRTHROW;
    }
}

/// [private] method initGlobalMemory_LJSIIteration
void baseMapGs::initGlobalMemory_LJSIIteration( UINT32 nJ )
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      QIDs of unmapped mates
                    Ru      subId bits for QIDs of unmapped mates
                    DBgs.Dc mapped D values without opposite-mate mappings
                    D       D values for candidates for seed-and-extend alignment
                   
            high:   Diter   J list (D values) for current iteration
                    cnJ     cumulative per-Q J-list sizes
                    nJ      per-Q J-list sizes
                    oJ      per-seed J-list offsets
        */

        /* The memory-management strategy is:
            - At the start of each inner-loop iteration (in baseMapGs::loadJforSeedInterval):
                - A temporary buffer is allocated in the high memory-allocation region to handle the D values
                    for the iteration.
            - At the end of each inner-loop iteration:
                - The Dj list is consolidated.
                - The temporary buffer is freed.
        */

        CREXEC( m_pdbg->Diter.Alloc(cgaHigh, nJ+1, true ) );
        m_pdbg->Diter.n = nJ;
        SET_CUDAGLOBALPTR_TAG(m_pdbg->Diter, "Diter");
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory_LJSIIteration
void baseMapGs::resetGlobalMemory_LJSIIteration()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      QIDs of unmapped mates
                    Ru      subId bits for QIDs of unmapped mates
                    DBgs.Dc mapped D values without opposite-mate mappings
                    D       consolidated list of D values for candidates for seed-and-extend alignment
                   
            high:   cnJ     cumulative per-Q J-list sizes
                    nJ      per-Q J-list sizes
                    oJ      per-seed J-list offsets
        */

        // expand the consolidated Dj buffer
        UINT32 cel = m_pqb->DBj.D.n + m_pdbg->Diter.n;
        m_pqb->DBj.D.Resize( cel );

        // copy the current iteration's Dj values into the consolidated buffer
        CREXEC( m_pdbg->Diter.CopyInDevice( m_pqb->DBj.D.p+m_pqb->DBj.D.n, m_pdbg->Diter.n ) );
        m_pqb->DBj.D.n = cel;
        
        // discard the current iteration's D-list buffer
        CREXEC( m_pdbg->Diter.Free() );


#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: after freeing Diter", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
#endif


    }
    catch( ApplicationException* pex )
    {
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory_LJSI
void baseMapGs::resetGlobalMemory_LJSI()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after reset:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      QIDs for D values
                    Ru      subId bits for QIDs
                    DBgs.Dc mapped D values without opposite-mate mappings
                    D       consolidated Dj list

            high:   (unallocated)
        */

        // discard the J-list sizes
        m_pqb->DBj.cnJ.Free();
        m_pqb->DBj.nJ.Free();
        m_pqb->DBj.oJ.Free();
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseMapGs::resetGlobalMemory()
{
    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs for D values
                Ru      subId bits for QIDs
                DBgs.Dc mapped D values without opposite-mate mappings

        high:   (unallocated)
    */

    // discard the Du list
    m_pdbg->Du.Free();
}

/// <summary>
/// Performs seed-and-extend alignment on a list of Q sequences.
/// </summary>
void baseMapGs::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    // set up CUDA memory
    initGlobalMemory();
    const UINT32 cbSharedPerBlock = initSharedMemory();



#if TODO_CHOP_WHEN_DEBUGGED
    // zap all concordant-pair counts
    m_pqb->DB.Qw.CopyToHost( m_pqb->QwBuffer.p, m_pqb->QwBuffer.n );
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; ++iq )
            pQw->nAc[iq] = 0;
    }
    m_pqb->DB.Qw.CopyToDevice( m_pqb->QwBuffer.p, m_pqb->QwBuffer.n );

    // use this to count concordant mappings
    m_pdbg->nDm1 = 0;

#endif


    // loop from maximum to minimum seed interval
    for( m_isi=0; (m_isi<=m_isiLimit)&&(m_pdbg->Qu.n); ++m_isi )
    {
#if TODO_CHOP_WHEN_DEBUGGED
        // look for QIDs in the Dc list that are not in the Qu list
        WinGlobalPtr<UINT64> Dcxxx( m_pdbg->Dc.n, false );
        m_pdbg->Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );
        WinGlobalPtr<UINT32> Quxxx( m_pdbg->Qu.n ,false );
        m_pdbg->Qu.CopyToHost( Quxxx.p, Quxxx.Count );

        UINT32 nAnomalies = 0;
        for( UINT32 iDc=0; iDc<m_pdbg->Dc.n; ++iDc )
        {
            UINT64 D = Dcxxx.p[iDc];
            UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
            qid ^= 1;

            bool notFound = true;
            for( UINT32 iQu=0; notFound && (iQu<m_pdbg->Qu.n); ++iQu )
                notFound = (qid != Quxxx.p[iQu]);

            if( notFound )
            {
                ++nAnomalies;
                CDPrint( cdpCD0, "%s: %u: 0x%08x in Dc (D=0x%016llx) but not in Qu!", __FUNCTION__, iDc, qid, D );
            }
        }

        CDPrint( cdpCD0, "[%d] %s::%s: %u D values with QIDs not in Qu", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, nAnomalies );
#endif

        // count J values for seeds for all Q sequences
        baseCountJs countJs( m_ptum->Key, m_pqb, m_pdbg, m_isi );
        countJs.Start();
        countJs.Wait();

        // combine the subId bits for adjacent Q sequences (opposite mates)
        combineRuForSeedInterval();

        // identify candidate locations for seed-and-extend alignment (DBj.D)
        initGlobalMemory_LJSI();
        loadJforSeedInterval();
        resetGlobalMemory_LJSI();

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: m_isi=%u: DBj.D.n = %u (new candidates)",
                            m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                            m_isi, m_pqb->DBj.D.n );
#endif

        // build and filter a list of D values that represent potential mappings (Dx)
        filterJforSeedInterval( cbSharedPerBlock );

        // do gapped alignment on the D values in the Dx list
        mapJforSeedInterval();

        // remove mapped QIDs from the list of unmapped QIDs
        pruneQu();
    }

    // clean up CUDA global memory
    resetGlobalMemory();

    CDPrint( cdpCD3, "[%d] %s::%s: completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
#pragma endregion
