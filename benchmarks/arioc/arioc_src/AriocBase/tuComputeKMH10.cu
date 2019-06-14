/*
  tuComputeKMH10.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data

/// [device] function computeH32 (UINT64) (see Hash.cpp for the corresponding CPU implementation)
static inline __device__ UINT32 computeH32( UINT64 u64 )
{
    /* Computes a 32-bit value by hashing the bits of the specified 64-bit pattern.  The hash implementation is the 32-bit
        version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.

       When the reference kmers are encoded, we isolate the bits of interest by applying a mask, e.g.
            u64 &= this->seedMask;
       But we don't bother doing that here because the bits have already been masked in getQik().
    */

    // constants
    UINT32 c1 = 0xcc9e2d51;
    UINT32 c2 = 0x1b873593;
    UINT32 seed = 0xb0f57ee3;

    // u64 bits 0-31
    register UINT32 k1 = static_cast<UINT32>(u64);

    k1 *= c1;
    k1 = (k1 << 15) | (k1 >> (32-15));  // k1 = _rotl( k1, 15 );
    k1 *= c2;

    UINT32 h = seed ^ k1;
    h = (h << 13) | (h >> (32-13));     // h = _rotl( h, 13 );
    h = 5*h + 0xe6546b64;

    // u64 bits 32-63
    k1 = static_cast<UINT32>(u64 >> 32);
    k1 *= c1;
    k1 = (k1 << 15) | (k1 >> (32-15));  // k1 = _rotl( k1, 15 );
    k1 *= c2;

    h ^= k1;
    h = (h << 13) | (h >> (32-13));     // h = _rotl( h, 13 );
    h = 5*h + 0xe6546b64;

    // avalanche
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

/// [device] function computeKMHvalues
static __device__ void computeKMHvalues( UINT64* pKMH, const UINT32 tid, const INT32 K, const UINT64* __restrict__ pQi, const INT16 N, const UINT32 stride )
{
    /* The list of KMH values is "segmented" by prefixing the 32-bit kmer hash with a QID:
    
    
        bits 00..31: hash value
        bits 32..32: 1=null (no hash value)
        bits 33..63: QID

       The idea is to be able to sort the entire list of KMH values in one operation while maintaining the
        per-QID grouping of the hash values in the list.
    */

    /* Compute a bitmask for counting Ns in a given kmer:
        - we need a bitmask that covers the low-order symbols in an A21-formatted 64-bit value
        - MASK100 masks the high-order bit of each 3-bit symbol
    */
    INT32 shr = 63 - (K*3);
    const UINT64 maskNonN = MASK100 >> shr;

    // compute a bitmask for the low-order bits of an A21-formatted 64-bit value
    const UINT64 maskK = MASKA21 >> shr;

    // build an empty KMH/qid pair
    UINT64 QK = static_cast<UINT64>(tid) << 33;

    // get the first set of encoded symbols
    UINT64 i64 = *pQi;
    UINT64 i64n = 0;

    // iterate through the kmers in the current Q sequence
    const UINT64* pLimit = pKMH + stride;
    for( INT32 pos=0; pos<=(N-K); ++pos )
    {
        UINT64 k64 = i64 & maskK;

        // hash the kmer only if it contains no Ns
        if( (k64 & maskNonN) == maskNonN )
            *(pKMH++) = QK | computeH32( k64 );

        // get the next set of encoded symbols
        if( (i64n == 0) && (pos < (N-21)) )
            i64n = *(pQi += CUDATHREADSPERWARP);

        // shift to the next symbol position
        i64 = (i64 >> 3) | ((i64n & 7) << 60);
        i64n >>= 3;
    }

    // fill any remaining KMH values with null values
    while( pKMH < pLimit )
        *(pKMH++) = QK | 0x00000001FFFFFFFF;
}

#if TODO_CHOP_IF_UNUSED
/// [device] function computeKMHnulls
static __device__ void computeKMHnulls( UINT64* pKMH, const UINT32 tid, const UINT32 stride )
{
    // build an empty KMH/qid pair using a null (all bits set) hash value
    const UINT64 QKnull = (static_cast<UINT64>(tid) << 33) | 0x00000001FFFFFFFF;

    // null the KMH list for the current Q sequence
    for( UINT32 pos=0; pos<stride; ++pos )
        *(pKMH++) = QKnull;
}
#endif

/// [kernel] tuComputeKMH10_Kernel
static __global__ void tuComputeKMH10_Kernel( const UINT32                     nQw,         // in: total number of Q warps
                                              const Qwarp*  const              pQwarp,      // in: list of Q-warp structs
                                              const UINT64* const __restrict__ pQiBuffer,   // in: interleaved Q sequence data
                                              const INT32                      K,           // in: kmer size
                                              const UINT32                     stride,      // in: maximum number of kmer hash values per Q sequence
                                                    UINT64* const              pKMHBuffer   // out: pointer to KMH/qid buffer (one per Q sequence)
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

    // point to the first set of A21-encoded symbols for the current Q sequence
    const UINT64* __restrict__ pQi = pQiBuffer + pQw->ofsQi + iq;

    // point to the KMH buffer for the current thread (current Q sequence)
    UINT64* pKMH = pKMHBuffer + (tid * stride);

#if TODO_CHOP_IF_UNUSED
    // if the Q sequence has at least one nongapped or gapped mapping...
    if( pQw->nAn[iq] || pQw->nAg[iq] )
        computeKMHnulls( pKMH, tid, stride );                       // use null kmer hash "values"
    else
        computeKMHvalues( pKMH, tid, K, pQi, pQw->N[iq], stride );  // compute kmer hash values
#endif
    
    // compute kmer hash values
    computeKMHvalues( pKMH, tid, K, pQi, pQw->N[iq], stride );
}

#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void tuComputeKMH10::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 tuComputeKMH10::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( tuComputeKMH10_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void tuComputeKMH10::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );

    // execute the kernel
    tuComputeKMH10_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->QwBuffer.n,     // in: number of Qwarps
                                                             m_pqb->DB.Qw.p,        // in: Qwarp buffer
                                                             m_pqb->DB.Qi.p,        // in: interleaved Q sequence data
                                                             m_pab->KmerSize,       // in: K (kmer size)
                                                             m_pqb->DBkmh.stride,   // in: maximum number of kmer hash values per Q sequence
                                                             m_pqb->DBkmh.KMH.p     // out: KMH/qid pairs (64-bit values)
                                                           );
}
#pragma endregion
