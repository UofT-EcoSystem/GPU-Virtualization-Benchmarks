/*
  baseSetupJs.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data
/// [kernel] baseSetupJs_Kernel
static __global__ void baseSetupJs_Kernel( const UINT32* const __restrict__ pcnJBuffer, // in: cumulative J-list sizes (one per mate)
                                           const UINT32                     iQ,         // in: index of first QID for the current seed iteration
                                           const UINT32                     nJlists,    // in: number of J lists for the current seed iteration
                                           const UINT32                     npos,       // in: number of seed positions per Q sequence
                                           const UINT32                     sps,        // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                                 UINT64* const              pDqBuffer   // out: pointer to D-list buffer
                                         )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // each CUDA thread initializes Dq values for the J list associated with one Q sequence and one seed position
    if( tid >= nJlists )
        return;

    /* Get the QID for the current thread's Q sequence.

       - When only Qf is seeded, there are npos lists for each QID; when Qrc is also seeded, there are
       2*npos lists for each QID.
       - When Qrc is seeded, even-numbered Dq values contain J values for Qf seeds and odd-numbered Dq values
       contain J values for Qrc seeds.
       */
    const UINT32 rcBit = (sps == 2) ? (tid & 1) : 0;
    const UINT32 nSeedsPerQ = npos * sps;
    const UINT32 iQID = iQ + (tid / nSeedsPerQ);    // index of the QID for this CUDA thread's J list
    const UINT32 spos = (tid % nSeedsPerQ) / sps;   // 0-based position of the seed within the Q sequence

    // get the cumulative number of J values preceding the current iteration
    UINT32 iJlist0 = iQ * nSeedsPerQ;
    UINT32 cnJ0 = pcnJBuffer[iJlist0];

    // point to the first D value to be initialized in the current thread
    UINT32 iJlist = iJlist0 + tid;
    UINT64* pDq = pDqBuffer + pcnJBuffer[iJlist] - cnJ0;

    // compute the limiting value for the D-list pointer
    const UINT64* pDlimit = pDqBuffer + pcnJBuffer[iJlist+1] - cnJ0;

    /* Initialize the Dq values for the current thread (Q sequence and seed position).  Yes, this is terrible
        CUDA memory management (i.e., not coalesced) but it only runs once on comparatively little data... */
    UINT32 ij = 0;
    while( pDq < pDlimit )
        *(pDq++) = PACK_DQ( rcBit, iQID, spos, ij++ );
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseSetupJs::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 baseSetupJs::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseSetupJs_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseSetupJs::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: m_nJlists=%d m_nSeedPos=%d cnJ.p=0x%016llx cnJ.Count=%lld",
                        __FUNCTION__, m_nJlists, m_nSeedPos, m_pqb->DBj.cnJ.p, m_pqb->DBj.cnJ.Count );
#endif



    // execute the kernel
    baseSetupJs_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DBj.cnJ.p,         // in: cumulative per-Q J-list sizes
                                                          m_iQ,                     // in: index of first QID for the current iteration
                                                          m_nJlists,                // in: number of J lists for the current iteration
                                                          m_nSeedPos,               // in: number of seed positions per Q sequence
                                                          m_pab->StrandsPerSeed,    // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                                          m_pdbb->Diter.p           // out: D list for the current iteration
                                                        );
}
#pragma endregion
