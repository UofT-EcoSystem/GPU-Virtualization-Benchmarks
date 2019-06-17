/*
  baseSetupJn.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseSetupJn::baseSetupJn()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="npos">number of seed positions per Q sequence</param>
baseSetupJn::baseSetupJn( const char* ptumKey, QBatch* pqb, UINT32 npos ) : m_pqb(pqb),
                                                                            m_pab(pqb->pab),
                                                                            m_npos(npos),
                                                                            m_baseConvertCT(pqb->pab->a21ss.baseConvert == A21SpacedSeed::bcCT),
                                                                            m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseSetupJn")),
                                                                            m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseSetupJn", m_cudaThreadsPerBlock );
}

/// destructor
baseSetupJn::~baseSetupJn()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void baseSetupJn::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBn.Qu  QIDs for Df values
                    DBj.cnJ cumulative per-seed J-list sizes
                    
            high:   DBj.D   D values
                    DBj.nJ  per-seed J-list sizes
                    DBj.oJ  per-seed J-list offsets
        */

        /* Allocate a buffer for the D list
        */
        CREXEC( m_pqb->DBj.D.Alloc( cgaHigh, m_pqb->DBj.totalD, true ) );
        m_pqb->DBj.D.n = m_pqb->DBj.totalD;
        SET_CUDAGLOBALPTR_TAG( m_pqb->DBj.D, "DBj.D" );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method computeGridDimensions
void baseSetupJn::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pqb->DBj.celJ );
}

/// [private] method copyKernelResults
void baseSetupJn::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // wait for the kernel to complete
        CREXEC( waitForKernel() );



#if TODO_CHOP_WHEN_DEBUGGED
        WinGlobalPtr<UINT32> cnJxxx( m_pqb->DBj.cnJ.Count, false );
        m_pqb->DBj.cnJ.CopyToHost( cnJxxx.p, cnJxxx.Count );

        WinGlobalPtr<UINT64> Dxxx( m_pqb->DBj.totalD, false );
        m_pqb->DBj.D.CopyToHost( Dxxx.p, Dxxx.Count );
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
void baseSetupJn::resetGlobalMemory()
{
    CRVALIDATOR;

    /* CUDA global memory layout after reset:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data
                DBn.Qu  QIDs for Df values

        high:   Df      Df lists (one per seed)
                DBj.nJ  per-seed J-list sizes
                DBj.oJ  per-seed J-list offsets
    */

    CREXEC(m_pqb->DBj.cnJ.Free());

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, m_hrt.GetElapsed(true) );
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Loads J values from the hash table (builds a list of Df values)
/// </summary>
void baseSetupJn::main()
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
