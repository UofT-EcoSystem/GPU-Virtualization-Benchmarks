/*
  baseFilterQu.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data
/// [kernel] baseFilterQu_Kernel
static __global__ void baseFilterQu_Kernel(       UINT32* const pQuBuffer,  // in,out: pointer to QIDs
                                            const UINT32        nQ,         // in: number of QIDs
                                            const INT16         At,         // in: threshold concordant-mapping count
                                            const Qwarp*  const pQwBuffer   // in: pointer to Qwarps
                                          )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nQ )
        return;

    // load the QID for the current CUDA thread
    UINT32 qid = pQuBuffer[tid];

    // point to the corresponding Qwarp
    UINT32 iw = QID_IW(qid);
    UINT32 iq = QID_IQ(qid);
    const Qwarp* pQw = pQwBuffer + iw;

    /* Set the QID to null (all bits set) if it references a mate in a pair that already has the required
        minimum number of concordant mappings.
    */
    if( pQw->nAc[iq] >= At )
        pQuBuffer[tid] = _UI32_MAX;
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseFilterQu::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 baseFilterQu::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseFilterQu_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseFilterQu::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );

    // execute the kernel for the specified D list
    baseFilterQu_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pdbb->Qu.p,    // in,out: QIDs
                                                           m_pdbb->Qu.n,    // in: number of QIDs
                                                           m_At,            // in: required minimum number of concordant mappings per pair
                                                           m_pqb->DB.Qw.p   // in: Qwarps
                                                        );
}
#pragma endregion
