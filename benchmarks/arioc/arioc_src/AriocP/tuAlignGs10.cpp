/*
  tuAlignGs10.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuAlignGs10::tuAlignGs10()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuAlignGs10::tuAlignGs10( QBatch* pqb ) : m_pqb(pqb),
                                          m_pab(pqb->pab),
                                          m_ptum(AriocBase::GetTaskUnitMetrics( "tuAlignGs10" ))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_tuAlignGs10", m_cudaThreadsPerBlock );
}

/// destructor
tuAlignGs10::~tuAlignGs10()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void tuAlignGs10::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

             low:   Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBgs.Qu QIDs of candidates for seed-and-extend alignment
                    DBgs.Ru subId bitmaps for candidate QIDs

            high:   (unallocated)
        */

        /* Allocate two parallel lists:
            - QIDs of mates
            - corresponding subId bits
        */
        CREXEC( m_pqb->DBgs.Qu.Alloc( cgaLow, m_pqb->DB.nQ, false ) );
        SET_CUDAGLOBALPTR_TAG(m_pqb->DBgs.Qu,"Qu");
        CREXEC( m_pqb->DBgs.Ru.Alloc( cgaLow, m_pqb->DB.nQ, false ) );
        SET_CUDAGLOBALPTR_TAG(m_pqb->DBgs.Ru,"Ru");

        // initialize
        CREXEC( cudaMemset( m_pqb->DBgs.Qu.p, 0xFF, m_pqb->DBgs.Qu.cb ) );  // all Qus are initially null (all bits set)
        m_pqb->DBgs.Qu.n = 0;
        m_pqb->DBgs.Ru.n = 0;
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method computeGridDimensions
void tuAlignGs10::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_pqb->DB.nQ );
}

/// [private] method copyKernelResults
void tuAlignGs10::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // block until the kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void tuAlignGs10::resetGlobalMemory()
{
    /* CUDA global memory layout after reset:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data
                DBgs.Qu QIDs of candidates for seed-and-extend alignment

        high:   (unallocated)
    */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Identifies candidates for gapped alignment.
/// </summary>
void tuAlignGs10::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    launchKernel( d3g, d3b, initSharedMemory() );

    copyKernelResults();
    resetGlobalMemory();
    
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s: completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
#pragma endregion
