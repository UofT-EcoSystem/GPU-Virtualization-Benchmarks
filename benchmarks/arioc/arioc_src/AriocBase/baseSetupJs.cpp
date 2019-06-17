/*
  baseSetupJs.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseSetupJs::baseSetupJs()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
/// <param name="isi">index of the current seed iteration</param>
/// <param name=iQ>index of the first QID for the current seed iteration </param>
/// <param name=nJ>number of QIDs for the current seed iteration</param>
baseSetupJs::baseSetupJs( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, UINT32 isi, UINT32 iQ, UINT32 nQ ) :
                                    m_pqb(pqb),
                                    m_pab(pqb->pab),
                                    m_pdbb(pdbb),
                                    m_isi(isi),
                                    m_nSeedPos(0),
                                    m_iQ(iQ),
                                    m_nJlists(0),
                                    m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseSetupJs")),
                                    m_baseConvertCT(pqb->pab->a21hs.baseConvert == A21HashedSeed::bcCT),
                                    m_hrt(us)
{
    CRVALIDATOR;

    pqb->GetGpuConfig( "ThreadsPerBlock_baseSetupJs", m_cudaThreadsPerBlock );
    
    /* There is a new baseSetupJs instance for each "seed iteration" in the caller.
    */

    // compute the length of the seed-position list for the specified seed iteration
    UINT32 maxnSeedPos = m_pab->a21hs.ofsSPSI.p[m_isi+1] - m_pab->a21hs.ofsSPSI.p[m_isi];
    m_nSeedPos = min2(static_cast<UINT32>(m_pdbb->AKP.seedsPerQ),maxnSeedPos);

    // compute the total number of J lists to be set up
    m_nJlists = nQ * m_nSeedPos * m_pab->StrandsPerSeed;    // (number of Q sequences) * (number of seeds per Q sequence) * (number of strands per seed)
}

/// destructor
baseSetupJs::~baseSetupJs()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void baseSetupJs::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBgw.Dc
                    DBn.Qu  QIDs for Df values
                    DBn.Ru  rbits for Df values
                    DBj.D   D values
                    
            high:   Diter   D values for the current iteration
                    DBj.cnJ cumulative J-list sizes
                    DBj.nJ  per-seed J-list sizes
                    DBj.oJ  per-seed J-list offsets
                    Du      unmapped D values
        */
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method computeGridDimensions
void baseSetupJs::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nJlists );
}

/// [private] method copyKernelResults
void baseSetupJs::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // wait for the kernel to complete
        CREXEC( waitForKernel() );


#if TRACE_SQIDxxx

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
        WinGlobalPtr<UINT64> Diterxxx( m_pdbb->Diter.Count, false );
        m_pdbb->Diter.CopyToHost( Diterxxx.p, Diterxxx.Count );
        WinGlobalPtr<UINT32> Quxxx( m_pdbb->Qu.n, false );
        m_pdbb->Qu.CopyToHost( Quxxx.p, Quxxx.Count );

        for( UINT32 n=0; n<static_cast<UINT32>(Diterxxx.Count); ++n )
        {
            UINT64 Dq = Diterxxx.p[n];
            bool rcBit = AriocDS::Dq::IsRC( Dq );
            UINT32 iqid = AriocDS::Dq::GetQid(Dq);
            UINT32 spos = AriocDS::Dq::GetSpos(Dq);
            UINT32 ij = AriocDS::Dq::GetIj(Dq);
            UINT32 qid = Quxxx.p[iqid];

            if( qid == qidxxx )
                CDPrint( cdpCD0, "%s: n=%u Dq=0x%016llx iqid=0x%08x qid=0x%08x rcBit=%d spos=%u ij=%u",
                                    __FUNCTION__, n, Dq, iqid, qid, rcBit, spos, ij );
        }
        CDPrint( cdpCD0, __FUNCTION__ );
    }
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
void baseSetupJs::resetGlobalMemory()
{
    CRVALIDATOR;

    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBgw.Dc
                DBn.Qu  QIDs for Df values
                DBn.Ru  rbits for Df values
                DBj.D   D values
                    
        high:   Diter   D values for the current iteration
                DBj.cnJ cumulative J-list sizes
                DBj.nJ  per-seed J-list sizes
                DBj.oJ  per-seed J-list offsets
                Du      unmapped D values
    */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Loads J values from the hash table (builds a list of Df values)
/// </summary>
void baseSetupJs::main()
{
    CRVALIDATOR;

    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    launchKernel( d3g, d3b, initSharedMemory() );     // initialize D values

    copyKernelResults();
    resetGlobalMemory();
    
    CDPrint( cdpCD3, "[%d] %s: completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
#pragma endregion
