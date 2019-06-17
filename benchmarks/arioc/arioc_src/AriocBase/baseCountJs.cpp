/*
  baseCountJs.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseCountJs::baseCountJs() : m_isi(0)
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
/// <param name="iter">0-based seed iteration number</param>
baseCountJs::baseCountJs( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, UINT32 isi ) : m_pqb(pqb),
                                                                                                    m_pab(pqb->pab),
                                                                                                    m_kc(pqb->pab),
                                                                                                    m_pdbb(pdbb),
                                                                                                    m_isi(isi),
                                                                                                    m_nSeedPos(0),
                                                                                                    m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseCountJs", m_cudaThreadsPerBlock );
}

/// destructor
baseCountJs::~baseCountJs()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeGridDimensions
void baseCountJs::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pdbb->Qu.n );
}

/// [private] method initGlobalMemory
void baseCountJs::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    Qu      QIDs
                    ...

           high:    DBj.cnJ cumulative per-Q J-list sizes
                    DBj.nJ  per-Q J-list sizes
                    DBj.oJ  per-seed J-list offsets
                    ...
        */

        /* compute the maximum number of seed positions for the current iteration (assuming that all Q sequences in the
            current batch have the maximum length)
        */

        // scan the precomputed list of all possible seed positions for the current iteration
        const UINT32 iLimit = m_pqb->Nmax - m_pab->a21hs.seedWidth;  // maximum seed position for any read in the current batch
        UINT16 isp0 = m_pab->a21hs.ofsSPSI.p[m_isi];
        UINT16 ispLimit = m_pab->a21hs.ofsSPSI.p[m_isi+1];
        m_pdbb->AKP.seedsPerQ = 0;
        for( UINT16 isp=isp0; isp<ispLimit; isp++ )
        {
            // if the isp'th seed position is beyond the maximum seed position for any read in the current batch...
            if( m_pab->a21hs.SPSI.p[isp] > iLimit )
            {
                m_pdbb->AKP.seedsPerQ = isp - isp0;
                break;
            }
        }

        if( m_pdbb->AKP.seedsPerQ == 0 )
            throw new ApplicationException( __FILE__, __LINE__, "maximum seed position exceeded for Nmax=%d", m_pqb->Nmax );


#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: m_isi=%u iStart=%u iLimit=%u seedsPerQ=%u",
                         m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_isi, iStart, iLimit, m_pdbb->AKP.seedsPerQ );
#endif

        /* Allocate and zero three parallel lists:
            - J-list offset
            - J-list size
            - cumulative J-list offsets
           For each list:
            - the list contains one element for each seed position in each Q sequence
            - the buffer size is doubled for bsDNA alignments where both strands of each Q sequence are seeded
            - the values corresponding to a Q sequence are stored in consecutive elements in each list (i.e. they are NOT interleaved)
        */
        m_pqb->DBj.celJ = m_pdbb->Qu.n * m_pdbb->AKP.seedsPerQ * m_kc.sps;  // (see also tuSetupN10::initGlobalMemory())

        // J-list offsets
        CREXEC( m_pqb->DBj.oJ.Alloc( cgaHigh, m_pqb->DBj.celJ, true ) );
        SET_CUDAGLOBALPTR_TAG(m_pqb->DBj.oJ, "DBj.oJ");

        /* J-list cardinalities
        
           We use a trailing zero-valued element in the buffer so that thrust::exclusive_scan() will
            compute the sum of the elements.
        */
        CREXEC( m_pqb->DBj.nJ.Alloc( cgaHigh, m_pqb->DBj.celJ+1, true ) );
        SET_CUDAGLOBALPTR_TAG(m_pqb->DBj.nJ, "DBj.nJ");

        /* cumulative J-list cardinalities
        */
        CREXEC( m_pqb->DBj.cnJ.Alloc( cgaHigh, m_pqb->DBj.celJ+1, true ) );
        SET_CUDAGLOBALPTR_TAG(m_pqb->DBj.cnJ, "DBj.cnJ");
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method copyKernelResults
void baseCountJs::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // if the kernel is initializing the subId bits, save the subId bits list count
        if( m_pdbb->Ru.Count && (m_pdbb->Ru.n == 0) )
            m_pdbb->Ru.n = m_pdbb->Qu.n;

        // block until the kernel completes
        CREXEC( waitForKernel() );


#if TODO_CHOP_WHEN_DEBUGGED
            // verify data returned by the kernel
            WinGlobalPtr<UINT32> Quxxx( m_pdbb->Qu.n, false );
            m_pdbb->Qu.CopyToHost( Quxxx.p, Quxxx.Count );
            WinGlobalPtr<UINT64> oJxxx( m_pqb->DBj.oJ.Count, true );
            m_pqb->DBj.oJ.CopyToHost( oJxxx.p, oJxxx.Count );
            WinGlobalPtr<UINT32> nJxxx( m_pqb->DBj.nJ.Count, true );
            m_pqb->DBj.nJ.CopyToHost( nJxxx.p, nJxxx.Count );

            CDPrint( cdpCD0, "%s", __FUNCTION__ );
#endif


#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: looking for sqId 0x%016llx in D list...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    
    UINT32 qidxxx = _UI32_MAX;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            if( pQw->sqId[iq] == TRACE_SQID )
            {
                qidxxx = PACK_QID(iw,iq);

                // dump the Q sequence
                CDPrint( cdpCD0, "%s: sqId 0x%016llx <--> qid 0x%08x", __FUNCTION__, TRACE_SQID, qidxxx );
                UINT64* pQi = m_pqb->QiBuffer.p + pQw->ofsQi + iq;
                AriocCommon::DumpA21( pQi, pQw->N[iq], CUDATHREADSPERWARP );
                break;
            }
        }
    }

    if( qidxxx == _UI32_MAX )
        CDPrint( cdpCD0, "[%d] %s::%s: sqId 0x%016llx not in current batch", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    else
    {
        // verify data returned by the kernel
        WinGlobalPtr<UINT32> Quxxx( m_pdbb->Qu.n, false );
        m_pdbb->Qu.CopyToHost( Quxxx.p, Quxxx.Count );
        WinGlobalPtr<UINT32> Ruxxx( m_pdbb->Ru.n, false );
        m_pdbb->Ru.CopyToHost( Ruxxx.p, Ruxxx.Count );
        WinGlobalPtr<UINT64> oJxxx( m_pqb->DBj.oJ.Count, false );
        m_pqb->DBj.oJ.CopyToHost( oJxxx.p, oJxxx.Count );
        WinGlobalPtr<UINT32> nJxxx( m_pqb->DBj.nJ.Count, false );
        m_pqb->DBj.nJ.CopyToHost( nJxxx.p, nJxxx.Count );

        for( UINT32 iqid=0; iqid<m_pdbb->Qu.n; iqid++ )
        {
            UINT32 qid = Quxxx.p[iqid];
            if( (qid ^ qidxxx) <= 1 )
            {
                // seed iteration info
                UINT32 nSeedsPerQ = m_pdbb->AKP.seedsPerQ * m_pab->StrandsPerSeed;
                UINT32 iJ = iqid * nSeedsPerQ;
                CDPrint( cdpCD0, "%s: iqid=%u qid=0x%08x m_isi=%u AKP.seedsPerQ=%u iJ=%u",
                                    __FUNCTION__, iqid, qid, m_isi, m_pdbb->AKP.seedsPerQ, iJ );

                UINT32 iSeedPos = m_pab->a21hs.ofsSPSI.p[m_isi];
                UINT16* pSeedPos = m_pab->a21hs.SPSI.p + iSeedPos;

                // dump J list offsets and counts
                UINT64* poJ = oJxxx.p + iJ;
                UINT32* pnJ = nJxxx.p + iJ;
                for( UINT32 s=0; s<nSeedsPerQ; ++s )
                    CDPrint( cdpCD0, "%s: qid=0x%08x pos=%d%c oJ=%u nJ=%u",
                                        __FUNCTION__, qid, pSeedPos[s/m_pab->StrandsPerSeed], "fr"[s&1], poJ[s], pnJ[s] );

                // dump rbits
                UINT32 rbits = (Ruxxx.Count ? Ruxxx.p[iqid] : _UI32_MAX);
                bool rbitsContain = ((rbits & (1 << TRACE_SUBID)) != 0);
                CDPrint( cdpCD0, "[%d] %s::%s: iqid=%u qid=0x%08X rbits=0x%08x (subId %d %sin rbits)",
                                    m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                    iqid, qid, rbits, TRACE_SUBID, (rbitsContain ? "" : "not ") );
            }
        }
        CDPrint( cdpCD0, __FUNCTION__ );
    }
#endif

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseCountJs::resetGlobalMemory()
{
    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs
                ...

       high:    DBj.cnJ cumulative per-Q J-list sizes
                DBj.nJ  per-Q J-list sizes
                DBj.oJ  per-seed J-list offsets
                ...
    */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Hashes Q sequences to determine the number of reference locations associated with seed-and-extend seeds.
/// </summary>
void baseCountJs::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    // set up and run a CUDA kernel to do the work
    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    launchKernel( d3g, d3b, initSharedMemory() );
    copyKernelResults();
    
    resetGlobalMemory();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s::%s: completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
#pragma endregion
