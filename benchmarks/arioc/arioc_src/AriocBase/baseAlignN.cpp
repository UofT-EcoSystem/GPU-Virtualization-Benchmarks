/*
  baseAlignN.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"


#pragma region constructor/destructor
/// [private] constructor
baseAlignN::baseAlignN()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbn">a reference to an initialized <c>DeviceBuffersN</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
/// <param name="freeRi">flag set to cause the Ri (interleaved R-sequence) buffer to be freed after alignments are complete
baseAlignN::baseAlignN( const char* ptumKey, QBatch* pqb, DeviceBuffersN* pdbn, RiFlags flagRi, bool freeRi ) : m_pqb(pqb),
                                                                                                   m_pab(pqb->pab),
                                                                                                   m_pdbn(pdbn), m_pD(NULL), m_nD(0),
                                                                                                   m_freeRi(freeRi),
                                                                                                   m_baseConvertCT(pqb->pab->a21ss.baseConvert == A21SpacedSeed::bcCT),
                                                                                                   m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseAlignN")),
                                                                                                   m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseAlignN", m_cudaThreadsPerBlock );

    // use a flag specified by the caller to determine which buffer to use
    pdbn->flagRi = flagRi;
    switch( flagRi )
    {
        case riDc:
            m_pD = pdbn->Dc.p;
            m_nD = pdbn->Dc.n;
            break;

        case riDm:
            m_pD = pdbn->Dm.p;
            m_nD = pdbn->Dm.n;
            break;

        case riDx:
            m_pD = pdbn->Dx.p;
            m_nD = pdbn->Dx.n;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for flagRi: %d", flagRi );
            break;
    }
}

/// destructor
baseAlignN::~baseAlignN()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeGridDimensions
void baseAlignN::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nD );
}

/// [private] method initGlobalMemory
void baseAlignN::initGlobalMemory()
{
    /* CUDA global memory layout after initialization:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                ...
                Ri      interleaved R sequence data

        high:   (unallocated)
    */
}

/// [private] method copyKernelResults
void baseAlignN::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->n.CandidateD, m_nD );
        AriocTaskUnitMetrics* ptumN = AriocBase::GetTaskUnitMetrics( "tuFinalizeN" );
        InterlockedExchangeAdd( &ptumN->n.CandidateD, m_nD );

        // block until the kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->us.Launch, m_hrt.GetElapsed(true) );


#if TRACE_SQID
    // examine the Dc list (unpaired) or Dx list (paired)
    WinGlobalPtr<UINT64> Dxxx( m_nD, false );

    if( m_pD == m_pqb->DBn.Dc.p )
    {
        if( m_nD != m_pqb->DBn.Dc.n )
            DebugBreak();
        CDPrint( cdpCD0, "[%d] %s: DBn.Dc.n=%u", m_pqb->pgi->deviceId, __FUNCTION__, m_pqb->DBn.Dc.n );
        m_pqb->DBn.Dc.CopyToHost( Dxxx.p, Dxxx.Count );
    }
    else
    if( m_pD == m_pqb->DBn.Dx.p )
    {
        if( m_nD != m_pqb->DBn.Dx.n )
            DebugBreak();
        CDPrint( cdpCD0, "[%d] %s: DBn.Dx.n=%u", m_pqb->pgi->deviceId, __FUNCTION__, m_pqb->DBn.Dx.n );
        m_pqb->DBn.Dx.CopyToHost( Dxxx.p, Dxxx.Count );
    }
    else
        DebugBreak();

    for( UINT32 n=0; n<m_nD; ++n )
    {
        UINT64 D = Dxxx.p[n];
        UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 pos = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - pos) : pos;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "%s: DBn.Dcx[%u]: D=0x%016llx qid=0x%08x subId=%d pos=%d Jf=%d",
                            __FUNCTION__, n,
                            D, qid, subId, pos, Jf );
        }
    }

    CDPrint( cdpCD0, __FUNCTION__ );
#endif

    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseAlignN::resetGlobalMemory()
{
    /* CUDA global memory layout after memory is reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                ...
                Ri      [optional] interleaved R sequence data

        high:   (unallocated)
    */

    // selectively release the CUDA global memory buffer in which the interleaved R sequence data resides
    if( m_freeRi )
        m_pqb->DB.Ri.Free();

#if TRACE_SQID
    WinGlobalPtr<UINT64> Dxxx( m_nD, false );
    cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

    char* sDtype;
    switch( m_pdbn->flagRi )
    {
        case riDc:
            sDtype = "Dc";
            break;

        case riDm:
            sDtype = "Dm";
            break;

        case riDx:
            sDtype = "Dx";
            break;

        default:
            break;
    }
    
    for( UINT32 n=0; n<m_nD; ++n )
    {
        UINT64 D = Dxxx.p[n];
        UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 Jf = static_cast<INT32>(D & 0x7FFFFFFF);
            if( D & AriocDS::D::maskStrand )
                Jf = (m_pab->M.p[subId] - 1) - Jf;

#if TRACE_SUBID
        if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "%s: n=%4u %s=0x%016llx qid=0x%08x subId=%d Jf=%d",
                        __FUNCTION__, n, sDtype, D, qid, subId, Jf );
        }
    }
    CDPrint( cdpCD0, __FUNCTION__ );
#endif

}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Uses a CUDA kernel to do nongapped short-read alignments
/// </summary>
void baseAlignN::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    /* TODO: MAYBE:
        Sanity check:  Ensure that the worst-case score for the specified number of mismatches is no lower
            than the configured threshold score
            sanity check
        INT16 Vmin = (m_pqb->Nmax*m_pab->aas.ASP.Wm) - m_pab->a21ss.maxMismatches
    */

    // set up and run a CUDA kernel to do nongapped alignments on the Q sequences in the current QBatch instance
    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    launchKernel( initSharedMemory() );

    copyKernelResults();
    resetGlobalMemory();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s::%s: completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
