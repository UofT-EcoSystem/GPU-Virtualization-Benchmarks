/*
  tuSetupN24.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

/// [kernel] tuSetupN24_Kernel
static __global__ void tuSetupN24_Kernel(       UINT64* const __restrict__ pDBuffer,            // in,out: D values
                                          const UINT32                     nD,                  // in: number of D values
                                          const INT32                      seedCoverageN,       // in: nongapped seed coverage threshold
                                          const INT32                      seedCoverageLeftover // in: "leftover" seed coverage threshold
                                        )
{
    // compute the 0-based index of the CUDA thread (one per Q sequence in the current batch)
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nD )
        return;

    // load the D value for the current thread
    UINT64 D = pDBuffer[tid];

    // load the subsequent D value (if it exists)
    UINT64 Dnext = (tid < (nD-1)) ? pDBuffer[tid+1] : _UI64_MAX;
            
    // extract MSB (flag) bits from the D value
    const INT32 coverage = static_cast<INT32>(D >> AriocDS::D::shrFlags);

    // reset MSB bits (which will subsequently be treated as flags)
    D &= (~AriocDS::D::maskFlags);

    // compare QID and reference location of the D values
    Dnext &= (~AriocDS::D::maskFlags);
    if( D != Dnext )
    {
        /* The next D value's QID and/or reference location differ from the QID and reference location in the current
            D value, so the current thread's "coverage" value is the maximum for the QID and reference location. */

        // if the coverage meets the threshold requirement, flag the D value as a candidate for nongapped alignment
        if( coverage >= seedCoverageN )
            D |= AriocDS::D::flagCandidate;
        else
        {
            // if the coverage is greater than an arbitrary threshold, flag the D value as a candidate for gapped alignment
            if( coverage >= seedCoverageLeftover )
                D |= AriocDS::D::flagX;
        }
    }

    // update the D value for the current thread
    pDBuffer[tid] = D;
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void tuSetupN24::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 tuSetupN24::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( tuSetupN24_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void tuSetupN24::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );

    // execute the kernel    
    tuSetupN24_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DBj.D.p,                // in,out: D values
                                                         m_pqb->DBj.D.n,                // in: number of D values
                                                         m_pab->aas.ACP.seedCoverageN,  // in: nongapped seed coverage threshold
                                                         m_seedCoverageLeftover         // in: "leftover" seed coverage threshold
                                                       );
}
#pragma endregion
