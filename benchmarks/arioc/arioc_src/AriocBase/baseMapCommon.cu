/*
  baseMapCommon.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"
#include <thrust/device_ptr.h>
#include <thrust/remove.h>

#pragma region CUDA device code and data
/// [device] function binarySearch64
static inline __host__ __device__ UINT32 binarySearch64( const UINT64* const __restrict__ buf, UINT32 lo, UINT32 hi, const UINT64 k )
{
    /* Performs binary search on elements in the interval [lo, hi) (i.e. between lo and hi-1 inclusive) */
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

    /* Returns the largest value of i such that buf[i] <= k, except:
        - returns lo if k < buf[lo]
        - returns hi-1 if k > buf[hi-1]
    */
    return lo; 
}

/// [kernel] baseMapCommon_Kernel6
static __global__ void baseMapCommon_Kernel(       UINT64* const              pDbuffer,     // in,out: pointer to D values with coalesced coverage
                                             const UINT32                     nD,           // in: number of D values with coalesced coverage
                                             const UINT64* const __restrict__ pDuBuffer,    // in: D values that were not mapped on previous iterations
                                             const UINT32                     nDu,          // in: number of unmapped D values
                                             const UINT32                     wcgsc         // in: width of horizontal dynamic-programming band (for gapped aligner)
                                           )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nD )
        return;

    // get the D value for the current CUDA thread
    UINT64 D = pDbuffer[tid] & ~AriocDS::D::maskFlags;

    // in the Du buffer, find the largest value less than or equal to this thread's D value
    UINT32 iDu = binarySearch64( pDuBuffer, 0, nDu, D );

    /* reset the "candidate" flag if the current thread's D value lies sufficiently close to a previously-unmapped D value:
        - we need to check the Du values prior to and after the current thread's D value
        - the last value in the Du list is handled as a special case (all-bits-set null so that the comparison always fails)
    */
    UINT64 Du = pDuBuffer[iDu] & ~AriocDS::D::maskFlags;    // Du <= D
    if( (D - Du) <= static_cast<UINT64>(wcgsc) )
        pDbuffer[tid] = D & ~AriocDS::D::flagCandidate;
    else
    {
        Du = (++iDu < nDu) ? pDuBuffer[iDu] : _UI64_MAX;    // Du > D
        Du &= ~AriocDS::D::maskFlags;

        if( (Du - D) <= static_cast<UINT64>(wcgsc) )
            pDbuffer[tid] = D & ~AriocDS::D::flagCandidate;
    }
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Removes redundant D values from the current (Dx) list where they are close enough to D values in
//  the previously-evaluated (Du) list.
/// </summary>
void baseMapCommon::exciseRedundantDvalues( DeviceBuffersG* pdbg )
{
    // avoid redoing unsuccessful alignments by removing values in the D list that are close enough to previously processed D values
    CRVALIDATOR;
    dim3 d3g;
    dim3 d3b;
        
    // reset the "candidate" flag (which was set prior to alignment and never reset) in redundant D values
    computeKernelGridDimensions( d3g, d3b, pdbg->Dx.n );
    baseMapCommon_Kernel<<< d3g, d3b, 0 >>>( pdbg->Dx.p,        // in,out: pointer to D values with coalesced coverage
                                             pdbg->Dx.n,        // in: number of D values with coalesced coverage
                                             pdbg->Du.p,        // in: D values that were not mapped on previous iterations
                                             pdbg->Du.n,        // in: number of unmapped D values
                                             pdbg->AKP.wcgsc    // in: width of horizontal dynamic-programming band (for gapped aligner)
                                           );

    // wait for the kernel to complete
    CREXEC( waitForKernel() );

#if TODO_CHOP_WHEN_DEBUGGED
    UINT32 nBefore = m_pdbg->Dx.n;
#endif

    // prune the Dx list
    thrust::device_ptr<UINT64> tpDx( pdbg->Dx.p );
    thrust::device_ptr<UINT64> tpEolD = thrust::remove_if( epCGA, tpDx, tpDx+pdbg->Dx.n, TSX::isNotCandidateDvalue() );
    pdbg->Dx.n = static_cast<UINT32>(tpEolD.get() - tpDx.get() );
    pdbg->Dx.Resize( pdbg->Dx.n );
}
#pragma endregion
