/*
  tuComputeKMH30.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data

/// [device] function computeS64
static inline __device__ UINT64 computeS64( const INT32 K, const UINT64* __restrict__ pKMH, const INT16 N, const UINT32 nSketchBits )
{
    UINT64 sketchBits = 0;

    // build the "sketch" bitmap from the smallest distinct (unduplicated) hash values
    for( UINT32 pos=0; pos<=(N-K); ++pos )
    {
        // halt if a null hash value is encountered
        if( *pKMH & 0x0000000100000000 )
            break;

        // set the bit whose offset is specified by bits 16..21 of the hash value (i.e., a value between 0 and 63)
        const UINT32 bofs = (*pKMH >> 16) & 0x3F;
        sketchBits |= static_cast<UINT64>(1) << bofs;

        // clamp the popcount
        if( __popc64( sketchBits ) == nSketchBits )
            break;

        // iterate
        ++pKMH;
    }

    return sketchBits;
}

/// [kernel] tuComputeKMH30_Kernel
static __global__ void tuComputeKMH30_Kernel( const UINT32                     nQw,         // in: total number of Q warps
                                              const Qwarp*  const              pQwarp,      // in: list of Q-warp structs
                                              const UINT64* const __restrict__ pQiBuffer,   // in: interleaved Q sequence data
                                              const INT32                      K,           // in: kmer size
                                              const UINT32                     stride,      // in: maximum number of kmer hash values per Q sequence
                                              const UINT64* const __restrict__ pKMHBuffer,  // in: pointer to KMH/qid buffer (one per Q sequence)
                                              const UINT32                     nSketchBits, // in: number of sketch bits
                                                    UINT64* const              pS64Buffer   // out: pointer to S64 buffer
                                        )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // compute the offset of the corresponding Qwarp
    const UINT32 iw = QID_IW(tid);
    if( iw >= nQw )
        return;

    // point to the Qwarp struct that contains the Q sequences for this CUDA warp
    const Qwarp* pQw = pQwarp + iw;

    // get the offset of the Q sequence within the current Qwarp
    INT16 iq = QID_IQ(tid);
    if( iq >= pQw->nQ )
        return;

#if TODO_CHOP_IF_UNUSED
    // do nothing if the Q sequence has at least one nongapped or gapped mapping
    if( pQw->nAn[iq] || pQw->nAg[iq] )
        return;
#endif

    // point to the KMH buffer for the current thread (current Q sequence)
    const UINT64* __restrict__ pKMH = pKMHBuffer + (tid * stride);

    // compute S64 ("sketch bits") for the Q sequence
    pS64Buffer[tid] = computeS64( K, pKMH, pQw->N[iq], nSketchBits );
}

#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void tuComputeKMH30::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 tuComputeKMH30::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( tuComputeKMH30_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void tuComputeKMH30::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );

    // execute the kernel
    tuComputeKMH30_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->QwBuffer.n,     // in: number of Qwarps
                                                             m_pqb->DB.Qw.p,        // in: Qwarp buffer
                                                             m_pqb->DB.Qi.p,        // in: interleaved Q sequence data
                                                             m_pab->KmerSize,       // in: K (kmer size)
                                                             m_pqb->DBkmh.stride,   // in: maximum number of kmer hash values per Q sequence
                                                             m_pqb->DBkmh.KMH.p,    // in: KMH/qid pairs (64-bit values)
                                                             m_nSketchBits,         // in: maximum number of 1 bits in an S64 value
                                                             m_pqb->DBkmh.S64.p     // out: S64 ("sketch bits") values
                                                           );
}
#pragma endregion
