/*
  tuAlignGs10.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"


/// [kernel] tuAlignGs10_Kernel
static __global__ void tuAlignGs10_Kernel( const Qwarp* const __restrict__ pQwBuffer,   // in: Qwarps for the current batch
                                           const UINT32                    nQ,          // in: number of Q sequences in the current batch
                                           const INT16                     AtN,         // in: threshold concordant (nongapped) mapping count
                                                 UINT32* const             pQu          // out: QIDs of unmapped mates
                                         )
{
    // compute the 0-based index of the CUDA thread (one per Q sequence in the current batch)
    const UINT32 qid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( qid >= nQ )
        return;

    // point to the Qwarp for the current thread
    UINT32 iw = QID_IW(qid);
    UINT32 iq = QID_IQ(qid);
    const Qwarp* pQw = pQwBuffer + iw;
    if( iq >= pQw->nQ )
        return;

    /* If the current CUDA thread's Q sequence does not yet have the required number of concordant mappings,
        write the QID to the output buffer.

       The output buffer should already be initialized with null values that can be subsequently removed.  This kernel
        replaces nulls in the buffer with QIDs that are not yet concordantly mapped.
    */
    if( pQw->nAc[iq] < AtN )
        pQu[qid] = qid;
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void tuAlignGs10::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 tuAlignGs10::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( tuAlignGs10_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void tuAlignGs10::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );
    
    // execute the kernel    
    tuAlignGs10_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DB.Qw.p,       // in: Qwarps for the current batch
                                                          m_pqb->DB.nQ,         // in: number of Q sequences in the current batch
                                                          m_pab->aas.ACP.AtN,   // in: required minimum number of concordant mappings per pair
                                                          m_pqb->DBgs.Qu.p      // out: QIDs of unmapped mates
                                                        );
}
#pragma endregion
