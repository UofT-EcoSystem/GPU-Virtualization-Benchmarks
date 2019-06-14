/*
  tuXlatToDf.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuXlatToDf::tuXlatToDf()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pDf">output buffer</param>
/// <param name="nD">number of values to be translated</param>
/// <param name="pD">D values to be translated</param>
/// <param name="maskDflags">indicates which flag bits should be updated in the kernel</param>
/// <param name="newDflags">new values for flag bits</param>
/// <remarks>The output buffer may be the same as the input buffer</remarks>
tuXlatToDf::tuXlatToDf( QBatch* pqb, UINT64* pDf, UINT32 nD, UINT64* pD, UINT64 maskDflags, UINT64 newDflags ) : CudaLaunchCommon(4*CUDATHREADSPERWARP),
                                                                                                                 m_pqb(pqb),
                                                                                                                 m_pab(pqb->pab),
                                                                                                                 m_pDf(pDf),
                                                                                                                 m_nD(nD),
                                                                                                                 m_pD(pD),
                                                                                                                 m_maskDflags(maskDflags),
                                                                                                                 m_newDflags(newDflags),
                                                                                                                 m_ptum(AriocBase::GetTaskUnitMetrics("tuXlatToDf")),
                                                                                                                 m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_tuXlatToD", m_cudaThreadsPerBlock );
}

/// destructor
tuXlatToDf::~tuXlatToDf()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeGridDimensions
void tuXlatToDf::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nD );
}

/// [private] method copyKernelResults
void tuXlatToDf::copyKernelResults()
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
        CDPrint( cdpCD0, "tuXlatToDf::copyKernelResults!" );
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
void tuXlatToDf::main()
{
//    CDPrint( cdpCD0, "%s: [%d] assert 0: %d", __FUNCTION__, m_pqb->pgi->deviceId, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    //    CDPrint( cdpCD0, "%s: [%d] assert 1: %d", __FUNCTION__, m_pqb->pgi->deviceId, m_hrt.GetElapsed(false) );

    initConstantMemory();
    
//    CDPrint( cdpCD0, "%s: [%d] assert 2: %d", __FUNCTION__, m_pqb->pgi->deviceId, m_hrt.GetElapsed(false) );

    dim3 d3g;
    dim3 d3b;
    computeGridDimensions( d3g, d3b );
//    CDPrint( cdpCD0, "%s: [%d] assert 3: %d", __FUNCTION__, m_pqb->pgi->deviceId, m_hrt.GetElapsed(false) );

    launchKernel( d3g, d3b, initSharedMemory() );

//    CDPrint( cdpCD0, "%s: [%d] assert 4: %d", __FUNCTION__, m_pqb->pgi->deviceId, m_hrt.GetElapsed(false) );

    copyKernelResults();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s: completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
