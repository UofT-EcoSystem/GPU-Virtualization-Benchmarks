/*
  tuAlignN32.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

/// [kernel] tuAlignN32_Kernel
static __global__ void tuAlignN32_Kernel( const Qwarp* const __restrict__ pQwBuffer,    // in: Qwarps for the current batch
                                          const UINT32                    nD,           // in: number of D values
                                          const INT16                     AtN,          // in: threshold mapping count
                                                UINT64* const             pDBuffer      // in,out: D values
                                        )
{
    // compute the 0-based index of the CUDA thread (one per Q sequence in the current batch)
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nD )
        return;

    // get the D value for the current thread
    UINT64 D = pDBuffer[tid];

    // reset the "candidate" flag
    D &= (~AriocDS::D::flagCandidate);

    // point to the Qwarp for the current thread
    const UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
    UINT32 iw = QID_IW(qid);
    UINT32 iq = QID_IQ(qid);
    const Qwarp* pQw = pQwBuffer + iw;
    if( iq < pQw->nQ )
    {
        // if the current CUDA thread's Q sequence does not yet have the required number of mappings, flag the D value
        if( pQw->nAn[iq] < AtN )
            D |= AriocDS::D::flagCandidate;
    }

    /* At this point, a D value has its "candidate" flag set if the corresponding read does not yet have the required
        number of mappings, regardless of whether the D value's "mapped" flag is set. */

    // update the D value
    pDBuffer[tid] = D;
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void tuAlignN32::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 tuAlignN32::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( tuAlignN32_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void tuAlignN32::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );
    
    // execute the kernel    
    tuAlignN32_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DB.Qw.p,    // in: Qwarps for the current batch
                                                         m_pqb->DBn.Dc.n,   // in: number of D values
                                                         m_AtN,             // in: threshold mapping count
                                                         m_pqb->DBn.Dc.p    // in,out: D values
                                                       );
}
#pragma endregion
