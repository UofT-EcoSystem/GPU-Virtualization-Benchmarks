/*
  tuXlatToD.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuXlatToD::tuXlatToD()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pD">output buffer</param>
/// <param name="nD">number of values to be translated</param>
/// <param name="pDf">Df values to be translated</param>
/// <param name="maskDflags">indicates which flag bits should be updated in the kernel</param>
/// <param name="newDflags">new values for flag bits</param>
/// <remarks>The output buffer may be the same as the input buffer</remarks>
tuXlatToD::tuXlatToD( QBatch* pqb, UINT64* pD, UINT32 nD, UINT64* pDf, UINT64 maskDflags, UINT64 newDflags ) : CudaLaunchCommon(4*CUDATHREADSPERWARP),
                                                                                                               m_pqb(pqb),
                                                                                                               m_pab(pqb->pab),
                                                                                                               m_pDf(pDf),
                                                                                                               m_nD(nD),
                                                                                                               m_pD(pD),
                                                                                                               m_maskDflags(maskDflags),
                                                                                                               m_newDflags(newDflags),
                                                                                                               m_ptum(AriocBase::GetTaskUnitMetrics("tuXlatToD")),
                                                                                                               m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_tuXlatToD", m_cudaThreadsPerBlock );
}

/// destructor
tuXlatToD::~tuXlatToD()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeGridDimensions
void tuXlatToD::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nD );
}

/// [private] method copyKernelResults
void tuXlatToD::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // block until the kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->us.Launch, m_hrt.GetElapsed(true) );

#if TODO_CHOP_WHEN_DEBUGGED
        WinGlobalPtr<UINT64> Dxxx(m_nD, false );
        cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
        CDPrint( cdpCD0, "tuXlatToD::copyKernelResults!" );
#endif
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Translates Df values to D values
/// </summary>
void tuXlatToD::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    initConstantMemory();
    
    dim3 d3g;
    dim3 d3b;
    computeGridDimensions( d3g, d3b );
    launchKernel( d3g, d3b, initSharedMemory() );
    copyKernelResults();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s: completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
