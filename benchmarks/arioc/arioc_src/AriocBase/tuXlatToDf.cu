/*
  tuXlatToDf.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data
/* CUDA constant memory
*/
static __device__ __constant__ __align__(4) UINT32  ccM[AriocDS::SqId::MaxSubId+1];


/// [kernel] tuXlatToDf_Kernel
static __global__ void tuXlatToDf_Kernel( const UINT64* const pDBuffer,      // in: D values
                                          const UINT32        totalDf,       // in: total number of elements in the lists
                                          const UINT64        maskDflags,    // in: mask (bits set where flags are to be updated)
                                          const UINT64        newDflags,     // in: new flag bits
                                                UINT64* const pDfBuffer      // out: Df values
                                        )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= totalDf )
        return;

    // load the D value
    const UINT64 d = pDBuffer[tid];

    /* A Df value is bitmapped like this:

        UINT64  qlo   :  1;     // bits 00..00: low-order bit of QID (0: mate 1; 1: mate 2)
        UINT64  s     :  1;     // bits 01..01: strand (0: forward; 1: reverse complement)
        UINT64  d     : 31;     // bits 02..32: 0-based position relative to the forward strand
        UINT64  subId :  7;     // bits 33..39: subId
        UINT64  qhi   : 20;     // bits 40..59: high-order bits of QID
        UINT64  rcQ   :  1;     // bits 60..60: seed from reverse complement
        UINT64  flags :  3;     // bits 61..63: flags

      A D value is bitmapped like this:
        UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
        UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
        UINT64  subId :  7;     // bits 32..38: subId
        UINT64  qid   : 21;     // bits 39..59: QID
        UINT64  rcQ   :  1;     // bits 60..60: seed from reverse complement
        UINT64  flags :  3;     // bits 61..63: flags
    */

    // copy and/or update the flags
    UINT64 df = (d & AriocDS::Df::maskFlags & ~maskDflags) | (newDflags & maskDflags);

    // copy the reverse-complement seed (rcQ) flag
    df |= (d & AriocDS::D::flagRCq);

    // copy the QID
    df |= ((d >> 39) & 1);              // bits 00..00
    df |= (d & 0x0FFFFF0000000000);     // bits 40..59

    // copy the subId
    df |= ((d & 0x0000007F00000000) << 1);

    if( d & 0x80000000 )
    {
        // extract the reference position and subId
        INT32 pos = static_cast<INT32>(d & 0x7FFFFFFF);
        INT32 subId = static_cast<INT32>(d >> 32) & 0x7F;

        // transform the position to a reference to the reverse-complement strand
        const INT32 M = static_cast<INT32>(ccM[subId]);
        pos = (M-1) - pos;

        // set the strand bit (bit 1)
        df |= 0x00000002;

        // copy the position
        df |= (static_cast<UINT64>(pos) << 2);
    }
    else
    {
        // copy the position; the strand bit (bit 1) is 0
        df |= ((d & 0x000000007FFFFFFF) << 2);
    }

    // update the strand bit (bit 1) to reflect whether we have a reverse-complement seed
    df ^= (df & AriocDS::Df::flagRCq) >> AriocDS::Df::shrRCq;

    // save the Df value
    pDfBuffer[tid] = df;
}

#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void tuXlatToDf::initConstantMemory()
{
    CRVALIDATOR;

    // load the size of each R sequence into constant memory on the current GPU
    CRVALIDATE = cudaMemcpyToSymbol( ccM, m_pab->M.p, m_pab->M.cb );
}

/// [private] method initSharedMemory
UINT32 tuXlatToDf::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( tuXlatToDf_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void tuXlatToDf::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );

#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dfxxx(m_nD, false );
    cudaMemcpy( Dfxxx.p, m_pDf, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
#endif



    // execute the kernel
    tuXlatToDf_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pD,           // in: D values
                                                         m_nD,           // in: list count
                                                         m_maskDflags,   // in: flags to update
                                                         m_newDflags,    // in: new flag values
                                                         m_pDf           // out: Df values
                                                       );
}
#pragma endregion
