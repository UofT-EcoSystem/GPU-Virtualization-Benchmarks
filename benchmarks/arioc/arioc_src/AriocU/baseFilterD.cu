/*
  baseFilterD.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data
/// [kernel] baseFilterD_Kernel
static __global__ void baseFilterD_Kernel(       UINT64* const  pDbuffer,   // in,out: pointer to D values for mappings
                                           const UINT32         nD,         // in: number of D values for mappings
                                           const INT16          AtG,        // in: threshold mapping count
                                           const Qwarp*  const  pQwBuffer   // in: pointer to Qwarps
                                         )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nD )
        return;

    /* Load the D value for the current CUDA thread; the D value is bitmapped like this:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
            UINT64  flags :  3;     // bits 61..63: flags
                                    //  bit 61: 0
                                    //  bit 62: set if the D value is a candidate for alignment
                                    //  bit 63: set if the D value is successfully mapped

       When this kernel executes, the default state of the flags is expected to be:
        bit 62: 1
        bit 63: (ignored)
    */
    UINT64 D = pDbuffer[tid];

    // do nothing if the current thread's D value indicates that it is already a candidate for removal from the list
    if( D & AriocDS::D::flagCandidate )
        return;

    // extract the QID
    UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;

    // point to the Qwarp
    UINT32 iw = QID_IW(qid);
    INT16 iq = QID_IQ(qid);
    const Qwarp* pQw = pQwBuffer + iw;

    /* Set the "candidate" flag if the D value belongs to a read that already has the required minimum number
        of gapped mappings.
    */
    if( pQw->nAg[iq] >= AtG )
        pDbuffer[tid] = D | AriocDS::D::flagCandidate;
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseFilterD::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 baseFilterD::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseFilterD_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseFilterD::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );

    // execute the kernel for the specified D list
    baseFilterD_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pD,             // in,out: D values for mappings
                                                          m_nD,             // in: number of mappings
                                                          m_AtG,            // in: required minimum number of gapped mappings per read
                                                          m_pqb->DB.Qw.p    // in: Qwarps
                                                        );
}
#pragma endregion
