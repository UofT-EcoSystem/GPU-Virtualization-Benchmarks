/*
  baseMaxV.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseMaxV::baseMaxV()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbg">a reference to a <c>DeviceBuffersG</c> instance</param>
/// <param name="flagRi">D value buffer identifier</param>
baseMaxV::baseMaxV( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, RiFlags flagRi ) : m_pqb(pqb),
                                                                                               m_pab(pqb->pab),
                                                                                               m_nDperIteration(0),
                                                                                               m_maxDperIteration(_UI32_MAX),
                                                                                               m_pdbg(pdbg),
                                                                                               m_pD(NULL),
                                                                                               m_pVmax(NULL),
                                                                                               m_nD(0),
                                                                                               m_FV(pqb->pgi->pCGA),
                                                                                               m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseMaxV"))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseMaxV", m_cudaThreadsPerBlock );

    // TODO: USE OR LOSE
    // look for the Xparam that specifies the maximum number of D values per kernel iteration
    INT32 i = m_pab->paamb->Xparam.IndexOf( "DperIterationV" );
    if( i >= 0 )
        m_maxDperIteration = static_cast<UINT32>(m_pab->paamb->Xparam.Value(i));

    // determine which buffer to use
    m_pdbg->flagRi = flagRi;
    switch( flagRi )
    {
        case riDc:
            m_pD = m_pdbg->Dc.p;
            m_nD = m_pdbg->Dc.n;
            m_pVmax = &m_pdbg->VmaxDc;
            break;

        case riDx:
            m_pD = m_pdbg->Dx.p;
            m_nD = m_pdbg->Dx.n;
            m_pVmax = &m_pdbg->VmaxDx;
            break;

        case riDm:
            m_pD = m_pdbg->Dm.p;
            m_nD = m_pdbg->Dm.n;
            m_pVmax = &m_pdbg->VmaxDm;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for flagRi: %d", flagRi );
            break;
    }

}

/// destructor
baseMaxV::~baseMaxV()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void baseMaxV::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    ...     ...
                    FV      scoring-matrix buffer

            high:   Vmax    Vmax for all Q sequences (mapped or unmapped)
                    ...     ...
        */

        /* Allocate the Vmax buffer:
            - there is one element in this buffer for each candidate Dc or Dx value
        */
        CREXEC( m_pVmax->Alloc( cgaHigh, m_nD, true ) );
        SET_CUDAGLOBALPTR_TAG((*m_pVmax), "Vmax");
        m_pVmax->n = m_nD;

        /* Compute the maximum number of reads that can be aligned concurrently in a single CUDA kernel invocation:
            - The upper bound is the amount of available CUDA global memory, which is the amount left over after all the other
                buffers have been allocated.
           - To keep addressing simple in the FV buffer, we want the list of reads in each iteration to align at an even multiple
              of CUDATHREADSPERWARP reads; this means that all iterations need to contain an even multiple of CUDATHREADSPERWARP
              elements.
        */
        UINT32 cbPerQ = sizeof(UINT32) * m_pdbg->AKP.Mr;                    // number of bytes per Q sequence in the FV buffer
        m_nDperIteration = static_cast<UINT32>(m_pqb->pgi->pCGA->GetAvailableByteCount() / cbPerQ);
        m_nDperIteration = min2(m_nDperIteration, m_maxDperIteration);
        m_nDperIteration &= (-CUDATHREADSPERWARP);                          // round down to the nearest multiple of CUDATHREADSPERWARP

        // allocate the FV buffer
        CREXEC( m_FV.Alloc( cgaLow, m_nDperIteration*m_pdbg->AKP.Mr, false ) );
        SET_CUDAGLOBALPTR_TAG(m_FV, "FV");

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "baseMaxV::initGlobalMemory:  %lld bytes (%lld elements) in FV buffer (%lld bytes remaining) m_nDperIteration=%u", m_FV.cb, m_FV.Count, m_pqb->pgi->pCGA->GetAvailableByteCount(), m_nDperIteration );
        m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
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
void baseMaxV::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void baseMaxV::resetGlobalMemory()
{
    /* CUDA global memory layout after reset:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data
                ...     ...

        high:   Vmax    Vmax for all Q sequences (mapped or unmapped)
                ...     ...
    */

    // discard the FV buffer
    m_FV.Free();
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Finds the highest dynamic-programming alignment score for a Q sequence.
/// </summary>
void baseMaxV::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    initConstantMemory();
    initGlobalMemory();

    launchKernel( initSharedMemory() );

    copyKernelResults();
    resetGlobalMemory();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PostLaunch, m_hrt.GetElapsed(true) );

    CDPrint( cdpCD3, "[%d] %s completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
#pragma endregion
