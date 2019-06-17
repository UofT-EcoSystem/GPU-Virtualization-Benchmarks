/*
  baseMaxVw.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseMaxVw::baseMaxVw()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbg">a reference to an initialized <c>DeviceBuffersG</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
/// <param name="ofsD">offset into D buffer</param>
/// <param name="iD">number of values in D buffer</param>
baseMaxVw::baseMaxVw( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, RiFlags flagRi, UINT32 ofsD, UINT32 nD ) :
                        m_pqb( pqb ),
                        m_pab(pqb->pab),
                        m_FV(pqb->pgi->pCGA),
                        m_pdbg(pdbg),         
                        m_pD(NULL),
                        m_nD(0),
                        m_pVmax(NULL),
                        m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseMaxVw"))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseMaxVw", m_cudaThreadsPerBlock );

    // determine which buffer to use
    m_pdbg->flagRi = flagRi;
    switch( flagRi )
    {
        case riDc:
            m_pD = m_pdbg->Dc.p + ofsD;
            m_pVmax = m_pdbg->VmaxDc.p + ofsD;
            m_nD = (nD == _UI32_MAX) ? m_pdbg->Dc.n : nD;
            break;

        case riDm:
            m_pD = m_pdbg->Dm.p + ofsD;
            m_pVmax = m_pdbg->VmaxDm.p + ofsD;
            m_nD = (nD == _UI32_MAX) ? m_pdbg->Dm.n : nD;
            break;

        case riDx:
            m_pD = m_pdbg->Dx.p + ofsD;
            m_pVmax = m_pdbg->VmaxDx.p + ofsD;
            m_nD = (nD == _UI32_MAX) ? m_pdbg->Dx.n : nD;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for flagRi: %d", flagRi );
            break;
    }
}

/// destructor
baseMaxVw::~baseMaxVw()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void baseMaxVw::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

           low:     Qw      Qwarps
                    Qi      interleaved Q sequence data
                    D       D list (candidates for windowed gapped alignment)
                    Ri      interleaved R sequence data
                    FV      scoring-matrix buffer

            high:   Vmax    Vmax for all Q sequences (mapped or unmapped)
        */


#if TRACE_SQID
        // look for a specific sqId
        WinGlobalPtr<UINT64> Dcxxx( m_nD, false );
        cudaMemcpy( Dcxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

        for( UINT32 n=0; n<m_nD; ++n )
        {
            UINT64 Dc = Dcxxx.p[n];
            UINT32 qid = static_cast<UINT32>(Dc >> 39) & AriocDS::QID::maskQID;
            UINT32 iw = QID_IW(qid);
            INT16 iq = QID_IQ(qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];

            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
                CDPrint( cdpCD0, "[%d] %s::%s: n=%u qid=%08x sqId=0x%016llx Dc=0x%016llx",
                                    m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                    n, qid, pQw->sqId[iq], Dc );

        }
#endif

#if TODO_CHOP_IF_UNUSED
        /* Allocate the Vmax buffer:
            - there is one element in this buffer for each candidate D value
        */
        switch( m_pdbg->flagRi )
        {
            case riDc:
                CREXEC( m_pdbg->VmaxDc.Alloc( cgaHigh, m_nD, true ) );
                m_pdbg->VmaxDc.n = m_nD;
                m_pVmax = m_pdbg->VmaxDc.p;
                break;

            case riDm:
                CREXEC( m_pdbg->VmaxDm.Alloc( cgaHigh, m_nD, true ) );
                m_pdbg->VmaxDm.n = m_nD;
                m_pVmax = m_pdbg->VmaxDm.p;
                break;

            case riDx:
                CREXEC( m_pdbg->VmaxDx.Alloc( cgaHigh, m_nD, true ) );
                m_pdbg->VmaxDx.n = m_nD;
                m_pVmax = m_pdbg->VmaxDx.p;
                break;

            default:
                throw new ApplicationException( __FILE__, __LINE__, "unexpected value for flagRi" );
        }
#endif

        // allocate the FV buffer
        CREXEC( m_FV.Alloc( cgaLow, m_nD*m_pqb->Mrw, false ) );

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s:  %lld bytes used for FV buffer (%lld bytes remaining) m_nD=%u", __FUNCTION__, m_FV.cb, m_pqb->pgi->pCGA->GetAvailableByteCount(), m_nD );
#endif

    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}


#if TODO_CHOP_WHEN_DEBUGGED
static UINT32 totalVmax = 0;
static UINT32 totalHits = 0;
#endif

/// [private] method copyKernelResults
void baseMaxVw::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->n.CandidateD, m_nD );

        // block until the CUDA kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );




#if TODO_CHOP_WHEN_DEBUGGED

        WinGlobalPtr<UINT64> Rixxx( m_pqb->DB.Ri.Count, false );
        m_pqb->DB.Ri.CopyToHost( Rixxx.p, Rixxx.Count );
        for( size_t n=0; n<Rixxx.Count; ++n )
        {
            if( Rixxx.p[n] & 0x8000000000000000 )
                CDPrint( cdpCD0, "%s: Ri is corrupted", __FUNCTION__ );

        }
#endif


#if defined(TRACE_SQID)
        // look at D and Vmax for a specific sqId
        WinGlobalPtr<UINT64> Dcxxx( m_nD, false );
        cudaMemcpy( Dcxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

        WinGlobalPtr<INT16> Vmaxxxx( m_nD, false );
        cudaMemcpy( Vmaxxxx.p, m_pVmax, m_nD*sizeof(INT16), cudaMemcpyDeviceToHost );

        for( UINT32 n=0; n<m_nD; ++n )
        {
            UINT64 D = Dcxxx.p[n];
            UINT32 qid = AriocDS::D::GetQid( D );
            UINT32 iw = QID_IW(qid);
            INT16 iq = QID_IQ(qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];

            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {

                CDPrint( cdpCD0, "%s::%s: n=%u D=0x%016llx Vmax=%d", m_ptum->Key, __FUNCTION__, n, D, Vmaxxxx.p[n] );


            }

        }
        CDPrint( cdpCD0, __FUNCTION__ );

#endif



#if TODO_CHOP_WHEN_DEBUGGED
        WinGlobalPtr<UINT64> Dcxxx( m_nD, false );
        cudaMemcpy( Dcxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
        
        WinGlobalPtr<INT16> Vmaxxxx( m_nD, false );
        cudaMemcpy( Vmaxxxx.p, m_pVmax, m_nD*sizeof(INT16), cudaMemcpyDeviceToHost );

        // count the number of QIDs mapped and the number of mappings   WE'RE USING the Qwarps here!
        UINT32 nQmapped1 = 0;
        UINT32 nQmapped2 = 0;
        UINT32 nMapped = 0;
        for( UINT32 n=0; n<Vmaxxxx.Count; ++n )
        {
            if( Vmaxxxx.p[n] == 0 )
                continue;

            ++nMapped;

            UINT64 Dc = Dcxxx.p[n];
            UINT32 qid = static_cast<UINT32>(Dc >> 39) & AriocDS::QID::maskQID;
            Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
            pQw->nAg[QID_IQ(qid)]++ ;
        }

        for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
        {
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            for( INT16 iq=0; iq<pQw->nQ; ++iq )
            {
                if( pQw->nAg[iq] )
                {
                    ++nQmapped1;
                    if( pQw->nAg[iq] >= 2 )
                        ++nQmapped2;
                }
            }
        }



        //CDPrint( cdpCD0, "baseMaxVw::copyKernelResults: nQ=%u nMapped=%u nQmapped1=%u nQmapped2=%u", nQ, nMapped, nQmapped1, nQmapped2 );
        CDPrint( cdpCD0, "%s: nMapped=%u (%3.2f%%) nQmapped1=%u nQmapped2=%u", __FUNCTION__, nMapped, (100.0*nMapped/m_nD), nQmapped1, nQmapped2 );

#endif

#if TODO_MAYBE
        // allocate host buffers for the results
        pii->Vmax.Realloc( pii->nQ, false );
        pii->tbo.Realloc( pii->nQ, false );

        // performance metrics
        InterlockedExchangeAdd( &AriocBase::PerfMetrics.nGwD, pii->nQ );

        if( pii->iQ == 0 )      // record GPU memory utilization on the first iteration only
        {
            RaiiCriticalSection<tuAlignGw> rcs;
            AriocBase::PerfMetrics.cbMaxUnusedGPUMemoryGw = max2( AriocBase::PerfMetrics.cbMaxUnusedGPUMemoryGw, static_cast<UINT64>(m_pqb->pgi->pCGA->GetAvailableByteCount()) );
            CDPrint( cdpCDb, "baseMaxVw::copyKernelResults:  %lld bytes unused for nQ=%u", m_pqb->pgi->pCGA->GetAvailableByteCount(), pii->nQ );
        }

        // block until the CUDA kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &AriocBase::PerfMetrics.msAlignGw1_Launch, m_hrt.GetElapsed(true) );

        // copy the Vmax values for the successful gapped alignments
        CRVALIDATE = cudaMemcpy( pii->Vmax.p, m_pbufW->Vmax.p+pii->iQ, pii->Vmax.cb, cudaMemcpyDeviceToHost );

        // copy the traceback origins for the successful gapped alignments
        CRVALIDATE = cudaMemcpy( pii->tbo.p, m_pbufW->tbo.p+pii->iQ, pii->tbo.cb, cudaMemcpyDeviceToHost );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
        // TODO: DEBUG: look at the range of traceback origins
        INT32 mintbo = INT_MAX;
        INT32 maxtbo = INT_MIN;
        for( UINT32 n=0; n<pii->nQ; ++n )
        {
            mintbo = min2(mintbo, pii->tbo.p[n]);
            maxtbo = max2(maxtbo, pii->tbo.p[n]);
        }
        CDPrint( cdpCD0, "baseMaxVw::copyKernelResults: mintbo=%d maxtbo=%d", mintbo, maxtbo );

        // TODO: DEBUG: COUNT THE NUMBER OF HITS!
        UINT32 nHits = 0;
        UINT32 nHitsPlus = 0;
        UINT32 nHitsMinus = 0;
        INT16 maxVmax = 0;
        INT16 minVmax = 32767;
        for( UINT32 n=0; n<pii->Vmax.Count; ++n )
        {
            ////UINT32 qid = m_pqb->GB.qid.p[pii->iQ+n];
            ////Qwarp* pQw = m_pqb->QwBuffer.p+(qid>>5);
            ////INT64 sqId = pQw->sqId[qid&31];

            ////if( ((sqId & (~SQID_MATEID_MASK))>= 0x000002140073a0f4) && ((sqId & ~SQID_PAIRID_MASK)<= 0x000002140073a1fa) )
            ////    CDPrint( cdpCDb, "qid=0x%08x sqId=0x%016llx: Vmax=%d", qid, sqId, pii->Vmax.p[n] );


            
            ////if( pii->iQ && (n >= 214752) )
            ////    CDPrint( cdpCDb, "qid=0x%08x sqId=0x%016llx: Vmax=%d", qid, sqId, pii->Vmax.p[n] );

            //////if( sqId == 0x0000021400673722 )
            //////{
            //////    for( UINT32 x=n; x<n+100; ++x )
            //////    {
            //////        UINT32 qid = m_pqb->GB.qid.p[pii->iQ+x];
            //////        Qwarp* pQw = m_pqb->QwBuffer.p+(qid>>5);
            //////        INT64 sqId = pQw->sqId[qid&31];
            //////        CDPrint( cdpCDb, "Vmax for 0x%016llx: %d", sqId, pii->Vmax.p[x] );
            //////    }
            //////}

            if( pii->Vmax.p[n] )
            {
                if( pii->Vmax.p[n] < 0 )
                    DebugBreak();
                
                maxVmax = max2(maxVmax, pii->Vmax.p[n]);
                minVmax = min2(minVmax, pii->Vmax.p[n]);
                nHits++ ;


                // how many hits on each strand polarity?
                UINT64 J = m_pqb->GB.J.p[n];
                if( J & 0x80000000 )
                    ++nHitsMinus;
                else
                    ++nHitsPlus;
            }
        }

        totalHits += nHits;
        totalVmax += static_cast<UINT32>(pii->Vmax.Count);
        CDPrint( cdpCDb, "baseMaxVw::copyKernelResults: iQ=%u nQ=%u nHits=%u (%2.1f%%) maxVmax=%d minVmax=%d totalVmax=%u totalHits=%u nHitsPlus=%u nHitsMinus=%u",
                        pii->iQ, pii->nQ, nHits, 100.0*nHits/pii->Vmax.Count, maxVmax, minVmax, totalVmax, totalHits, nHitsPlus, nHitsMinus );

#endif




    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseMaxVw::resetGlobalMemory()
{
    /* CUDA global memory layout after reset:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data
                Dc      Dc list (candidates for windowed gapped alignment)

        high:   Vmax    Vmax for all Q sequences (mapped or unmapped)
    */

    // discard the buffers that are no longer needed
    m_FV.Free();
    m_pqb->DB.Ri.Free();
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Performs windowed gapped alignment on unmapped paired-end reads whose mates have nongapped mappings.
/// </summary>
void baseMaxVw::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s ...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    /* Because there may not be sufficient CUDA global memory to align the entire Dc list in a single kernel invocation,
        we may need to execute the CUDA kernel iteratively. */

    // setup common to all iterated instances of the CUDA kernel
    initConstantMemory();

    // initialize CUDA global memory and compute the number of Dc values that can be processed in one kernel invocation
    initGlobalMemory();

    // set up the CUDA kernel
    dim3 d3b;
    dim3 d3g;
    computeKernelGridDimensions( d3g, d3b, m_nD );

    // launch the CUDA kernel
    launchKernel( d3g, d3b, initSharedMemory() );

    copyKernelResults();
    resetGlobalMemory();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s::%s completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
