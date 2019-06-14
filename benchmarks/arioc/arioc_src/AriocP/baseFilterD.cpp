/*
  baseFilterD.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"


#pragma region constructor/destructor
/// [private] constructor
baseFilterD::baseFilterD()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
/// <param name="At">configured minimum number of concordant mappings per pair</param>
baseFilterD::baseFilterD( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, RiFlags flagRi, INT16 At ) : m_pqb(pqb),
                                                                                                                  m_pab(pqb->pab),
                                                                                                                  m_pdbb(pdbb),
                                                                                                                  m_pD(NULL), m_nD(0),
                                                                                                                  m_At(At),
                                                                                                                  m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseFilterD")),
                                                                                                                  m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseFilterD", m_cudaThreadsPerBlock );


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: m_At=%d", pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_At );
#endif


    // determine which buffer to use
    m_pdbb->flagRi = flagRi;
    switch( flagRi )
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
baseFilterD::~baseFilterD()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeGridDimensions
void baseFilterD::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nD );
}

/// [private] method initGlobalMemory
void baseFilterD::initGlobalMemory()
{
    /* CUDA global memory layout unchanged */
}

/// [private] method copyKernelResults
void baseFilterD::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // block until the kernel completes
        CREXEC( waitForKernel() );



#if TRACE_SQID

    CDPrint( cdpCD0, "[%d] %s::%s: D list is: D%c (At=%d)",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        "?cxm"[m_pdbb->flagRi], m_At );
    WinGlobalPtr<UINT64> Dxxx( m_nD, false );
    cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

    // look at the D data
    bool sqIdFound = false;
    for( UINT32 n=0; n<static_cast<UINT32>(Dxxx.Count); ++n )
    {
        UINT64 D = Dxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( D );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
            
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>(D >> 32) & 0x007F;
            INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jo = static_cast<UINT32>(m_pab->M.p[subId] - 1) - J;

#if TRACE_SUBID
        if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "[%d] %s::%s: %3d: D=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d Jo=%d",
                                m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                n, D, sqId, qid, subId, J, Jo );
            sqIdFound = true;
        }
    }

    if( sqIdFound )
        CDPrint( cdpCD0, "[%d] %s::%s", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    // count candidates for windowed gapped alignment
    UINT32 nCandidates = 0;
    for( UINT32 n=0; n<m_nD; ++n )
    {
        UINT64 D = Dxxx.p[n];

        if( D & AriocDS::D::flagCandidate )
            ++nCandidates;
    }

    CDPrint( cdpCD0, "[%d] %s::%s: D%c: %u candidates/%d mappings",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        "?cxm"[m_pdbb->flagRi], nCandidates, m_nD );
#endif



        // performance metrics
        InterlockedExchangeAdd( &m_ptum->us.Launch, m_hrt.GetElapsed(true) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseFilterD::resetGlobalMemory()
{
    /* CUDA global memory layout unchanged */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Flags concordantly-mapped pairs as non-candidates for subsequent windowed gapped alignment
/// </summary>
void baseFilterD::main()
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
