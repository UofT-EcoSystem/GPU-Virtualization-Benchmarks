/*
  baseAlignG.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseAlignG::baseAlignG()
{

}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbg">a reference to an initialized <c>DeviceBuffersG</c> instance</param>
/// <param name="phb">a reference to a <c>HostBuffers</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
baseAlignG::baseAlignG( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, HostBuffers* phb, RiFlags flagRi ) : m_pqb(pqb),
                                                                                                                     m_pab(pqb->pab),
                                                                                                                     m_SM(pqb->pgi->pCGA),
                                                                                                                     m_pdbg(pdbg), m_pD(NULL), m_nD(0),
                                                                                                                     m_pVmax(NULL),
                                                                                                                     m_phb(phb),
                                                                                                                     m_nDperIteration(0),
                                                                                                                     m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseAlignG"))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseAlignG", m_cudaThreadsPerBlock );

    // determine which buffer to use
    m_pdbg->flagRi = flagRi;
    switch( flagRi )
    {
        case riDc:
            m_pD = m_pdbg->Dc.p;
            m_pVmax = m_pdbg->VmaxDc.p;
            m_nD = m_pdbg->Dc.n;
            break;

        case riDm:
            m_pD = m_pdbg->Dm.p;
            m_pVmax = m_pdbg->VmaxDm.p;
            m_nD = m_pdbg->Dm.n;
            break;

        case riDx:
            m_pD = m_pdbg->Dx.p;
            m_pVmax = m_pdbg->VmaxDx.p;
            m_nD = m_pdbg->Dx.n;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for flagRi: %d", flagRi );
            break;
    }

}

/// destructor
baseAlignG::~baseAlignG()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeGridDimensions
void baseAlignG::computeGridDimensions( dim3& d3g, dim3& d3b, UINT32 nDm )
{
    computeKernelGridDimensions( d3g, d3b, nDm );
}
#pragma endregion

#pragma region protected methods
/// [private] method initGlobalMemory
void baseAlignG::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

           low:     Qw      Qwarps
                    Qi      interleaved Q sequence data
                    ...     ...
                    SM      scoring-matrix buffer

            high:   BRLEA   BRLEA buffer
                    ...     ...

           The following global-memory buffers must be initialized by the caller:
                    D       D values for mapped Q sequences (Dx, Dm, Dc)
                    Vmax    Vmax values corresponding to mapped D values (VmaxDx, VmaxDm, VmaxDc)
        */

        /* Allocate the BRLEA buffer:
            - there is space in the buffer for one maximum-length BRLEA for each candidate D value
        */
        UINT32 celBRLEAbuffer = m_nD * m_pqb->celBRLEAperQ;
        CREXEC(m_pdbg->BRLEA.Alloc( cgaHigh, celBRLEAbuffer, true ));
        m_pdbg->BRLEA.n = celBRLEAbuffer;

        /* Compute the maximum number of reads that can be aligned concurrently in a single CUDA kernel invocation:
            - The upper bound is the amount of available CUDA global memory, which is the amount left over after all the other
                buffers have been allocated.
            - The scoring-matrix buffer granularity is one warp's worth of cells, i.e. CUDATHREADSPERWARP * celSMperQ, so we
                need to round the number of Q sequences per iteration to an even multiple of CUDATHREADSPERWARP.
        */
        UINT32 cbPerQ = sizeof(UINT32) * m_pdbg->AKP.celSMperQ;
        m_nDperIteration = static_cast<UINT32>(m_pqb->pgi->pCGA->GetAvailableByteCount() / cbPerQ);
        m_nDperIteration &= (-CUDATHREADSPERWARP);          // round down to the nearest multiple of CUDATHREADSPERWARP

        // allocate the SM buffer
        size_t celSMbuffer = static_cast<size_t>(m_nDperIteration) * m_pdbg->AKP.celSMperQ;
        CREXEC( m_SM.Alloc( cgaLow, celSMbuffer, false ) );

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s: %lld bytes used for SM buffer (%lld bytes remaining) nDperIteration=%u", __FUNCTION__, m_SM.cb, m_pqb->pgi->pCGA->GetAvailableByteCount(), m_nDperIteration );
#endif
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [protected] method copyKernelResults
void baseAlignG::copyKernelResults( WinGlobalPtr<UINT32>* phbBRLEA )
{
    CRVALIDATOR;    
    
    try
    {
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->n.CandidateD, m_nD );
            
        // (re)allocate and zero the host buffer
        UINT32 cel = phbBRLEA->n + m_pdbg->BRLEA.n;
        phbBRLEA->Reuse( cel, false );
        UINT32* pBRLEAcurrent = phbBRLEA->p + phbBRLEA->n;
        memset( pBRLEAcurrent, 0, (phbBRLEA->Count-phbBRLEA->n)*sizeof(UINT32) );

        // block until the last kernel completes
        CREXEC( waitForKernel() );

#if TODO_CHOP_WHEN_DEBUGGED
CDPrint( cdpCD0, "[%d] %s: back from kernels in %dms", m_pqb->pgi->deviceId, __FUNCTION__, m_hrt.GetElapsed(false) );
#endif

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );

        // copy (append) the BRLEAs to the host buffer
        CREXEC( m_pdbg->BRLEA.CopyToHost( pBRLEAcurrent, m_pdbg->BRLEA.n ) );
        phbBRLEA->n += m_pdbg->BRLEA.n;

        // performance metrics
        InterlockedExchangeAdd( &AriocBase::aam.ms.XferMappings, m_hrt.GetElapsed(false) );

#if TRACE_SQID
    // copy the D values to a host buffer
    WinGlobalPtr<UINT64> Dxxx( m_nD, false );
    cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
    WinGlobalPtr<INT16> Vmaxxx( m_nD, false );
    cudaMemcpy( Vmaxxx.p, m_pVmax, m_nD*sizeof(INT16), cudaMemcpyDeviceToHost );

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

            CDPrint( cdpCD0, "[%d] %s::%s: %3d: D=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d Jo=%d Vmax=%d",
                                m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                n, D, sqId, qid, subId, J, Jo, Vmaxxx.p[n] );
            sqIdFound = true;
        }
    }

    // look at the BRLEA data
    if( sqIdFound )
    {
        for( UINT32 n=0; n<static_cast<UINT32>(Dxxx.Count); ++n )
        {
            BRLEAheader* pBH = reinterpret_cast<BRLEAheader*>(phbBRLEA->p + (n * m_pqb->celBRLEAperQ));
            UINT32 iw = QID_IW(pBH->qid);
            INT16 iq = QID_IQ(pBH->qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];
            
            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                INT32 J = (pBH->J << 1) >> 1;
                INT32 Jo = static_cast<UINT32>(m_pab->M.p[pBH->subId] - 1) - J;

#if TRACE_SUBID
            if( pBH->subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "[%d] %s::%s: %3d: BRLEA: sqId=0x%016llx qid=0x%08x subId=%d J=%d Jo=%d V=%d",
                                    m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                    n, sqId, pBH->qid, pBH->subId, J, Jo, pBH->V );
            }
        }

        CDPrint( cdpCD0, "[%d] %s::%s", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
    }
#endif

#if TODO_CHOP_WHEN_DEBUGGED
        // DEBUG: look for bad BRLEAs
        for( UINT32 n=0; n<(m_pdbg->BRLEA.n/m_pqb->celBRLEAperQ); ++n )
        {
            BRLEAheader* pBH = reinterpret_cast<BRLEAheader*>(pBRLEAcurrent+n*m_pqb->celBRLEAperQ);
            if( pBH->cb == 0 )
                continue;

            // point to the first BRLEA byte
            BRLEAbyte* p = reinterpret_cast<BRLEAbyte*>(pBRLEAcurrent+((n+1)*m_pqb->celBRLEAperQ)) - pBH->cb;

            UINT32 iw = QID_IW(pBH->qid);
            INT16 iq = QID_IQ(pBH->qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];

#if defined(TRACE_SQID)
            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
                CDPrint( cdpCD0, "[%d] %s::%s: %u: sqId=0x%016llx subId=%d J=0x%08x (%u) V=%d cb=%d",
                                    m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                    n, sqId, pBH->subId, pBH->J, pBH->J&0x7fffffff, pBH->V, pBH->cb );
#endif

            if( p->bbType != bbMatch )      // always start with a match
            {
                CDPrint( cdpCD0, "[%d] %s::%s: %u: BRLEA does not start with a match! (qid=0x%08x sqId=0x%016llx)",
                                    m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                    n, pBH->qid, sqId );
                DebugBreak();
            }
        }
#endif



#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s returned %u BRLEAs", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_nD );
#endif

    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseAlignG::resetGlobalMemory()
{
    /* CUDA global memory layout after reset:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data
                ...     ...

        high:   ...     ...
    */

    // discard the scoring-matrix and BRLEA buffers
    m_pdbg->BRLEA.Free();
    m_SM.Free();
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Performs mapping and traceback for reads mapped by the seed-and-extend gapped aligner
/// </summary>
void baseAlignG::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    initConstantMemory();
    initGlobalMemory();

    launchKernel( initSharedMemory() );

    copyKernelResults( &m_phb->BRLEA );
    resetGlobalMemory();
    
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s::%s: completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
#pragma endregion
