/*
  baseLoadRw.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseLoadRw::baseLoadRw()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbg">a reference to an initialized <c>DeviceBuffersG</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
/// <param name="ofsD">offset into D buffer</param>
/// <param name="nD">number of values in D buffer</param>
baseLoadRw::baseLoadRw( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, RiFlags flagRi, UINT32 ofsD, UINT32 nD ) :
                            m_pqb(pqb),
                            m_pab(pqb->pab),
                            m_pD(NULL), m_nD(nD),
                            m_hrt(us),
                            m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseLoadRw"))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseLoadRw", m_cudaThreadsPerBlock );

    pdbg->flagRi = flagRi;
    switch( flagRi )
    {
        case riDc:
            m_pD = pdbg->Dc.p + ofsD;
            m_nD = (nD == _UI32_MAX) ? pdbg->Dc.n : nD;
            break;

        case riDm:
            m_pD = pdbg->Dm.p + ofsD;
            m_nD = (nD == _UI32_MAX) ? pdbg->Dm.n : nD;
            break;

        case riDx:
            m_pD = pdbg->Dx.p + ofsD;
            m_nD = (nD == _UI32_MAX) ? pdbg->Dx.n : nD;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for flagRi: %d", flagRi );
            break;
    }
}

/// destructor
baseLoadRw::~baseLoadRw()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void baseLoadRw::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    D       D list (candidates for windowed gapped alignment)
                    Ri      interleaved R sequence data

            high:   ...     ...
        */

        /* Allocate a buffer to contain interleaved R sequence data.  We allocate the 64-bit values in a set of 2-dimensional
            blocks, each of which contains the interleaved 64-bit values for one CUDA warp.  The size of each block is thus
                celMr * CUDATHREADSPERWARP
            and there are
                nWarps * (celMr * CUDATHREADSPERWARP)
            64-bit elements in the buffer.
        */
        INT64 celMr = blockdiv( m_pqb->Mrw, 21 );               // max 64-bit elements needed to represent Mr
        INT64 nWarps = blockdiv( m_nD, CUDATHREADSPERWARP );    // number of warps required to compute the alignments
        INT64 celRi = nWarps * (celMr * CUDATHREADSPERWARP);    // total number of 64-bit values
        CREXEC( m_pqb->DB.Ri.Alloc( cgaLow, celRi, true ) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method computeGridDimensions
void baseLoadRw::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nD );
}

/// [private] method copyKernelResults
void baseLoadRw::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->n.CandidateD, m_nD );

        {
            RaiiCriticalSection<baseLoadRw> rcs;

            if( static_cast<UINT32>(m_pqb->Nmax) > AriocBase::aam.n.Nmax )
            {
                AriocBase::aam.n.Nmax = m_pqb->Nmax;
                AriocBase::aam.n.bwmax = 2 * m_pab->aas.ComputeWorstCaseGapSpaceCount(m_pqb->Nmax) + 1;
            }
        }

        // block until the CUDA kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->us.Launch, m_hrt.GetElapsed(true) );

        



#if TODO_CHOP_WHEN_DEBUGGED

        // look at the interleaved R data
        WinGlobalPtr<UINT64> Rix( m_pqb->DB.Ri.Count, false );
        CREXEC( m_pqb->DB.Ri.CopyToHost( Rix.p, Rix.Count ) );
        CDPrint( cdpCD0, "baseLoadRw::copyKernelResults" );

        // look at the raw R data
        WinGlobalPtr<UINT64> Rx( m_pqb->pgi->pR->Count, false );
        CREXEC( m_pqb->pgi->pR->CopyToHost( Rx.p, Rx.Count ) );
        CDPrint( cdpCD0, "baseLoadRw::copyKernelResults" );

        //////// verify the interleaved R data
        //////UINT32 Mr = (m_pab->aas.ACP.maxFragLen - m_pab->aas.ACP.minFragLen) + 1;    // number of R symbols to be aligned in each scoring matrix
        //////UINT32 celMr = (Mr + 40) / 21;              // max 64-bit elements needed to represent Mr


        //////for( UINT32 tid=0; tid<m_pqb->GB.qid.n; ++tid )
        //////{
        //////    UINT64 Jval = m_pqb->CjBufferGw.p[tid];

        //////    // point to raw R
        //////    INT16 subId = static_cast<INT16>(JVALUE_SUBID(Jval) - m_pab->minSubId);
        //////    INT32 J = Jval & 0x7FFFFFFF;
        //////    UINT64* pR = m_pab->R.p;
        //////    pR += (J & JVALUE_MASK_S) ? m_pab->ofsRminus.p[subId] : m_pab->ofsRplus.p[subId];
        //////    pR += J/21;

        //////    // point to interleaved R
        //////    UINT32 iWarp = tid / CUDATHREADSPERWARP;    // offset of the first Ri for the CUDA warp that corresponds to the current thread ID
        //////    UINT32 celMr = (Mr + 40) / 21;              // max 64-bit elements needed to represent Mr

        //////    UINT32 ofsRi = (iWarp * celMr * CUDATHREADSPERWARP) + (tid & (CUDATHREADSPERWARP-1));
        //////    const UINT64* pRi = Rix.p + ofsRi;

        //////    // for now we'll do it only for nonshifted R ...
        //////    if( (J % 21) == 0 )
        //////    {
        //////        for( UINT32 jj=0; jj<celMr; ++jj )
        //////        {
        //////            if( pR[jj] != pRi[jj*CUDATHREADSPERWARP] )
        //////            {
        //////                UINT32 qid = m_pqb->GB.qid.p[tid];
        //////                Qwarp* pQw = m_pqb->QwBuffer.p + (qid>>5);
        //////                CDPrint( cdpCDb, "Ri error for tid=%u qid=0x%08X sqId=0x%016llx pR[%d]=0x%016llx pRi[%d]=0x%016llx",
        //////                                tid, qid, pQw->sqId[qid&31], jj, pR[jj], jj*CUDATHREADSPERWARP, pRi[jj*CUDATHREADSPERWARP] );
        //////            }
        //////        }
        //////    }
        //////}





#endif


    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseLoadRw::resetGlobalMemory()
{
    /* CUDA global memory layout after memory is reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                D       D list (candidates for windowed gapped alignment)
                Ri      interleaved R sequence data

        high:   ...     ...
    */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Prepares interleaved R sequence data for the windowed gapped aligner
/// </summary>
void baseLoadRw::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    // set up the CUDA kernel
    initConstantMemory();
    initGlobalMemory();
        
    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    // launch the CUDA kernel
    launchKernel( d3g, d3b, initSharedMemory() );

    // block until the kernel completes
    copyKernelResults();
    resetGlobalMemory();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s::%s completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
