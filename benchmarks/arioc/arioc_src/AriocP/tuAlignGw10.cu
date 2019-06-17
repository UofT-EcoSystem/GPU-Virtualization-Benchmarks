/*
  tuAlignGw10.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"


// TODO: CHOP WHEN DEBUGGED
#include <thrust/device_ptr.h>
#include <thrust/sort.h>




/// [kernel] tuAlignGw10_Kernel
static __global__ void tuAlignGw10_Kernel(       UINT64* const pDcBuffer,   // in, out: D values mapped by nongapped aligner
                                           const UINT32        nDc,         // in: number of D values
                                                 UINT32* const pQu,         // out: QIDs of opposite (unmapped) mates
                                                 UINT32* const pRu          // out: 32-bit bitmap representing subId for mapped D values
                                         )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nDc )
        return;

    // load the D value for the current CUDA thread
    UINT64 Dc = pDcBuffer[tid];

    // set the "exclude" bit (see baseCountC)
    pDcBuffer[tid] = Dc | AriocDS::D::flagX;

    // write the QID for the opposite mate
    UINT32 qid = static_cast<UINT32>(Dc >> 39) & AriocDS::QID::maskQID;
    pQu[tid] = qid ^ 1;     // toggle bit 0 to reference the opposite mate

    // extract the subId from the D value
    INT32 subId = static_cast<INT32>(Dc >> 32) & 0x0000007F;

    /* Write the corresponding bitmap for the subId:
        - if possible, set a bit in the rbits value
        - otherwise, return a null value (all bits set)
    */
    UINT32 rbits = (subId < 32) ? (1 << subId) : _UI32_MAX;
    pRu[tid] = rbits;
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void tuAlignGw10::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 tuAlignGw10::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( tuAlignGw10_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void tuAlignGw10::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );

    // execute the kernel
    tuAlignGw10_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pD,             // in: QIDs of seed-and-extend candidate pairs
                                                          m_nD,             // in: number of QIDs
                                                          m_pdbb->Qu.p,     // out: QIDs of opposite mates
                                                          m_pdbb->Ru.p      // out: subId bits for opposite mates
                                                        );
}
#pragma endregion
