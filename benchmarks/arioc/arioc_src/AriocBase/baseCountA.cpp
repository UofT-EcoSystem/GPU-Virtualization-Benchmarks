/*
  baseCountA.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseCountA::baseCountA()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
baseCountA::baseCountA( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, RiFlags flagRi ) : m_pqb(pqb),
                                                                                                      m_pab(pqb->pab),
                                                                                                      m_pdbb(pdbb), m_pD(NULL), m_nD(0),
                                                                                                      m_isNongapped(false),
                                                                                                      m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseCountA")),
                                                                                                      m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseCountA", m_cudaThreadsPerBlock );

    // set a flag to indicate whether we're dealing with nongapped or gapped alignments
    m_isNongapped = (NULL != strstr(m_ptum->Key, "tuAlignN") );

    // determine which buffer to use
    m_pdbb->flagRi = flagRi;
    switch( flagRi )
    {
        case riDc:
            m_pD = pdbb->Dc.p;
            m_nD = pdbb->Dc.n;

#if TODO_CHOP_IF_UNUSED
            // if we're counting nongapped mappings, count only the most recently added mappings in the specified buffer (see tuAlignN::launchKernel61)
            if( m_isNongapped )
            {
                m_pD += m_pqb->DBgw.nDc1;
                m_nD -= m_pqb->DBgw.nDc1;


//#if TODO_CHOP_WHEN_DEBUGGED
                CDPrint( cdpCD0, "%s::baseCountA::ctor: m_pqb->DBgw.Dc.n=%u m_pqb->DBn.nDm1=%u m_pqb->DBn.nDm2=%u",
                                    m_ptum->Key, m_pqb->DBgw.Dc.n, m_pqb->DBn.nDm1, m_pqb->DBn.nDm2 );
                DebugBreak();
//#endif
            }
#endif
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
baseCountA::~baseCountA()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeGridDimensions
void baseCountA::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nD );
}

/// [private] method initGlobalMemory
void baseCountA::initGlobalMemory()
{
    /* CUDA global memory layout after initialization:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                ...

        high:   Dm      Dm list
    */
}

/// [private] method copyKernelResults
void baseCountA::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // block until the kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->us.Launch, m_hrt.GetElapsed(true) );


#if TODO_CHOP_WHEN_DEBUGGED
        WinGlobalPtr<Qwarp> Qwxxx( m_pqb->QwBuffer.n, false );
        Qwxxx.n = m_pqb->QwBuffer.n;
        m_pqb->DB.Qw.CopyToHost( Qwxxx.p, Qwxxx.n );


        UINT32 totalAn = 0;
        UINT32 totalAg = 0;
        UINT32 nMappedQ = 0;
        UINT32 nMappedPairs[9] = { 0 };

        Qwarp* pQw = Qwxxx.p;
        for( UINT32 iw=0; iw<Qwxxx.n; ++iw )
        {
            for( INT16 iq=0; iq<pQw->nQ; ++iq )
            {
                totalAn += pQw->nAn[iq];            // nongapped
                totalAg += pQw->nAg[iq];            // gapped
                if( pQw->nAn[iq] + pQw->nAg[iq] )   // total unduplicated Q
                    ++nMappedQ;

                // profile
                if( iq & 1 )
                {
                    INT16 nAq1 = pQw->nAn[iq^1]+pQw->nAg[iq^1];   // number of nongapped+gapped mappings for mate 1
                    INT16 nAq2 = pQw->nAn[iq]+pQw->nAg[iq];       // number of nongapped+gapped mappings for mate 2

                    nAq1 = min(2, nAq1);    // we only care about 0, 1, or 2 ...
                    nAq2 = min(2, nAq2);
                    nMappedPairs[3*nAq1 + nAq2]++ ;
                }
            }

            ++pQw;
        }

        CDPrint( cdpCD0, "%s::%s: totalAn=%u totalAg=%u nMappedQ=%u", m_ptum->Key, __FUNCTION__, totalAn, totalAg, nMappedQ );
        for( INT16 u=0; u<=2; ++u )
        {
            for( INT16 v=0; v<=2; ++v )
            {
                char banner[8];
                sprintf_s( banner, sizeof banner, "%dp%d", u, v );
                CDPrint( cdpCD2, "%s::%s: %s=%u", m_ptum->Key, __FUNCTION__, banner, nMappedPairs[u*3+v] );
            }
        }

        UINT64 nMapped1Pairs = nMappedPairs[1] + nMappedPairs[2] + nMappedPairs[3] + nMappedPairs[6];
        UINT64 nMapped2Pairs = nMappedPairs[4] + nMappedPairs[5] + nMappedPairs[7] + nMappedPairs[8];
        CDPrint( cdpCD2, "%s::%s: pairs with two mapped mates: %u", m_ptum->Key, __FUNCTION__, nMapped2Pairs );
        CDPrint( cdpCD2, "%s::%s: pairs with one mapped mate: %u", m_ptum->Key, __FUNCTION__, nMapped1Pairs );
#endif

    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseCountA::resetGlobalMemory()
{
    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                ...

        high:   Dm      Dm list
    */

#if TODO_CHOP_WHEN_DEBUGGED
    m_pqb->DB.Qw.CopyToHost( m_pqb->QwBuffer.p, m_pqb->QwBuffer.n );

    for( UINT32 iw=0; iw<4; iw++ )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; iq += 2 )
            CDPrint( cdpCD0, "baseCountA::resetGlobalMemory: sqId=0x%016llx nAn=%d,%d", pQw->sqId[iq], pQw->nAn[iq], pQw->nAn[iq+1] );
    }

#endif
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Uses a CUDA kernel to count per-Q mappings
/// </summary>
void baseCountA::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s ...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

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

    CDPrint( cdpCD3, "[%d] %s::%s completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
#pragma endregion
