/*
  baseCountC.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseCountC::baseCountC()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
/// <param name="doExclude">flag indicating whether to exclude previously-mapped D values</param>
baseCountC::baseCountC( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, RiFlags flagRi, bool doExclude ) : m_pqb(pqb),
                                                                                                                      m_pab(pqb->pab),
                                                                                                                      m_pdbb(pdbb),
                                                                                                                      m_pD(NULL), m_nD(0),
                                                                                                                      m_doExclude(doExclude),
                                                                                                                      m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseCountC")),
                                                                                                                      m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseCountC", m_cudaThreadsPerBlock );

    // use a flag specified by the caller to determine which buffer to use
    pdbb->flagRi = flagRi;
    switch( pdbb->flagRi )
    {
        case riDc:
            m_pD = pdbb->Dc.p;
            m_nD = pdbb->Dc.n;
            break;

        case riDm:
            m_pD = pdbb->Dm.p;
            m_nD = pdbb->Dm.n;
            break;

        case riDx:
            m_pD = pdbb->Dx.p;
            m_nD = pdbb->Dx.n;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for flagRi: %d", flagRi );
            break;
    }
}

/// destructor
baseCountC::~baseCountC()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeGridDimensions
void baseCountC::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nD );
}

/// [private] method initGlobalMemory
void baseCountC::initGlobalMemory()
{
    /* CUDA global memory layout is unchanged. */


#if TODO_CHOP_IF_DEBUGGED
    WinGlobalPtr<UINT64> Dmyyy( m_nD, false );
    cudaMemcpy( Dmyyy.p, m_pD, Dmyyy.cb, cudaMemcpyDeviceToHost );

    UINT32 nmax = 100;
    CDPrint( cdpCD0, "baseCountC::initGlobalMemory: first %u Dm values:", nmax );
    for( UINT32 n=0; n<nmax; ++n )
    {
        UINT64 Dm = Dmyyy.p[n];
        INT16 subId = static_cast<INT16>((Dm & AriocDS::D::maskSubId) >> 32);
        INT32 pos = Dm & 0x7FFFFFFF;
        if( Dm & AriocDS::D::maskStrand )
            pos = (m_pab->M.p[subId] - 1) - pos;

        CDPrint( cdpCD0, "baseCountC::initGlobalMemory: %4u 0x%016llx qid=0x%08llx subId=%d J=%d",
                            n, Dm, (Dm>>39)&AriocDS::QID::maskQID, subId, pos );
    }
#endif

#if TRACE_SQID
    WinGlobalPtr<UINT64> Dxxx( m_nD, false );
    cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

    UINT64 minSqId = MASKA21;
    UINT64 maxSqId = 0;
    for( UINT32 n=0; n<m_nD; ++n )
    {
        UINT64 D = Dxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( D );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;

        UINT64 sqId = pQw->sqId[iq];
        minSqId = min2(minSqId, sqId);
        maxSqId = max2(maxSqId, sqId);

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 J = D & 0x7FFFFFFF;
            INT32 Jf = (D & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

            CDPrint( cdpCD0, "[%d] %s::%s: %5u: sqId=0x%016llx D=0x%016llx qid=0x%08x subId=%d J=%d Jf=%d",
                                m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                n, sqId, D, qid, subId, J, Jf );
        }
    }

    CDPrint( cdpCD0, "[%d] %s::%s: minSqId=0x%016llx maxSqId=0x%016llx", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, minSqId, maxSqId );
#endif
}

/// [private] method copyKernelResults
void baseCountC::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // block until the kernel completes
        CREXEC( waitForKernel() );



#if TRACE_SQID
    WinGlobalPtr<UINT64> Dmxxx( m_nD, false );
    cudaMemcpy( Dmxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
    WinGlobalPtr<Qwarp> Qwxxx( m_pqb->QwBuffer.n, false );
    m_pqb->DB.Qw.CopyToHost( Qwxxx.p, Qwxxx.Count );


    // look for a specific QID
    bool sqIdFound = false;
    for( UINT32 n=0; n<static_cast<UINT32>(Dmxxx.Count); ++n )
    {
        UINT64 D = Dmxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( D );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
            
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>(D >> 32) & 0x007F;
            INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

            CDPrint( cdpCD0, "[%d] %s::%s: %5u: sqId=0x%016llx D=0x%016llx qid=0x%08x subId=%d J=%d Jf=%d",
                                m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                n, sqId, D, qid, subId, J, Jf );
            sqIdFound = true;
        }
    }

    if( sqIdFound )
        CDPrint( cdpCD0, "%s::%s", m_ptum->Key, __FUNCTION__ );

#if TODO_COUNT_CANDIDATES
    // count candidates for windowed gapped alignment
    UINT32 nCandidates = 0;
    for( UINT32 n=0; n<m_nD; ++n )
    {
        UINT64 Dm = Dmxxx.p[n];

        if( Dm & AriocDS::D::flagCandidate )
            ++nCandidates;
    }

    CDPrint( cdpCD0, "[%d] %s::%s: %u candidates/%d mappings", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, nCandidates, m_nD );

    // let's count the mappings in concordant pairs 
    UINT32 nConcordantPairs[4] = { 0 };
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        Qwarp* pQw = Qwxxx.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; ++iq )
            nConcordantPairs[ min2(pQw->nAc[iq],3) ]++ ;
    }

    for( UINT32 n=0; n<4; ++n )
        CDPrint( cdpCD0, "[%d] %s::%s: %u concordant mapping(s): %u Dm", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, n, nConcordantPairs[n] );
#endif
#endif




        // performance metrics
        InterlockedExchangeAdd( &m_ptum->us.Launch, m_hrt.GetElapsed(true) );

#if TODO_CHOP_WHEN_DEBUGGED
        WinGlobalPtr<UINT64> Dmxxx(m_pqb->DBn.Dm.Count, false);
        m_pqb->DBn.Dm.CopyToHost( Dmxxx.p, Dmxxx.Count );

        for( UINT32 n=0; n<100; ++n )
        {
            UINT32 qid = AriocDS::D::GetQid( Dm );
            UINT32 iw = QID_IW(qid);
            INT16 iq = QID_IQ(qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            CDPrint( cdpCD0, "%s::baseCountC::copyKernelResults: %3d: 0x%016llx sqId=0x%016llx", m_ptum->Key, n, Dmxxx.p[n], pQw->sqId[iq] );
        }



        CDPrint( cdpCD0, "%s::baseCountC::copyKernelResults", m_ptum->Key );
#endif

        
#if TODO_CHOP_WHEN_DEBUGGED
    // count concordant pairs
    UINT32 nPairs = 0;
    WinGlobalPtr<Qwarp> Qwyyy( m_pqb->DB.Qw.n, false );
    m_pqb->DB.Qw.CopyToHost( Qwyyy.p, Qwyyy.Count );
    for( UINT32 iw=0; iw<m_pqb->DB.Qw.n; ++iw )
    {
        Qwarp* pQw = Qwyyy.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; iq+=2 )
        {
            INT16 iqo = iq + 1;
            if( pQw->nAc[iq] && pQw->nAc[iqo] ) nPairs++ ;
        }
    }

    CDPrint( cdpCD0, "%s::%s: nPairs=%u/%u", m_ptum->Key, __FUNCTION__, nPairs, m_pqb->DB.nQ/2 );

    UINT32 nQconcordant = 0;
    for( UINT32 iw=0; iw<m_pqb->DB.Qw.n; ++iw )
    {
        Qwarp* pQw = Qwyyy.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; iq++ )
        {
            if( pQw->nAc[iq] ) nQconcordant++ ;
        }
    }
    
    CDPrint( cdpCD0, "%s::%s: nQconcordant=%u/%u", m_ptum->Key, __FUNCTION__, nQconcordant, m_pqb->DB.nQ );
#endif



    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseCountC::resetGlobalMemory()
{
    /* CUDA global memory layout is unchanged. */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Uses a CUDA kernel to identify nongapped concordantly-mapped pairs
/// </summary>
void baseCountC::main()
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
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s::%s: completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
#pragma endregion
