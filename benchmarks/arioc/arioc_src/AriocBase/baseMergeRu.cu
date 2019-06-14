/*
  baseMergeRu.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data
/// [device] function binarySearch32
static inline __device__ UINT32 binarySearch32( const UINT32* const __restrict__ buf, UINT32 lo, UINT32 hi, const UINT32 k )
{
    do
    {
        // find the midpoint
        UINT32 mid = (lo + hi) / 2;
        
        // update the range for the next iteration
        if( k < buf[mid] )
            hi = mid;
        else
            lo = mid;
    }
    while( (hi-lo) > 1 );

    return lo; 
}

/// [kernel] baseMergeRu_Kernel
static __global__ void baseMergeRu_Kernel( const UINT32* const __restrict__ pQuBuffer,  // in: pointer to QIDs
                                           const UINT32                     nQ,         // in: number of QIDs
                                           const UINT64* const __restrict__ pDBuffer,   // in: pointer to D values
                                           const UINT32                     nD,         // in: number of D values
                                                 UINT32* const              pRuBuffer   // in,out: pointer to subId bits
                                         )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nD )
        return;

    // load the D value for the current CUDA thread
    const UINT64 D = pDBuffer[tid];

    // extract the QID and subId from the D value
    UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
    UINT32 subId = static_cast<UINT32>(D >> 32) & 0x7F;

    // find the QID in the Qu buffer
    UINT32 ofs = binarySearch32( pQuBuffer, 0, nQ, qid );

    // set the bit in the Ru buffer that corresponds to the subId in the D value
    if( subId < 32 )
        atomicOr( pRuBuffer+ofs, (1 << subId) );    // set the subId'th bit
    else
        atomicOr( pRuBuffer+ofs, _UI32_MAX );       // null the subId bits (all bits set)
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseMergeRu::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 baseMergeRu::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseMergeRu_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseMergeRu::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );

    // execute the kernel for the specified D list
    baseMergeRu_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pdbb->Qu.p,     // in: QIDs
                                                          m_pdbb->Qu.n,     // in: number of QIDs
                                                          m_pD,             // in: D values
                                                          m_nD,             // in: number of D values
                                                          m_pdbb->Ru.p      // in,out: subId bits for QIDs
                                                        );
}
#pragma endregion
