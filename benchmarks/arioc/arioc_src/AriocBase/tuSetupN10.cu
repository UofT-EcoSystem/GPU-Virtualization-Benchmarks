/*
  tuSetupN10.cu

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
static __device__ __constant__ __align__(8) AlignmentControlParameters  ccACP;
static __device__ __constant__ __align__(8) tuSetupN10::KernelConstants cckc;

/* CUDA shared memory
*/

/// [device] function computeH32 (UINT64, UINT64, UINT64, UINT64, INT16) (see A21SpacedSeed.cpp for the corresponding CPU implementation)
static inline __device__ UINT32 computeH32( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4, const INT16 seedWidth )
{
    /* Computes a hash value for the four specified values (i.e., up to 84 symbols) by applying the optimal "periodic" (i.e. repeating)
        seed mask 0010111 to the specified 2164-encoded values.

        We first pack the 1 bits from each of the four specified 64-bit values into two 64-bit values.
    */
                                                    // ········LL····KK·JJ·II·······HH····GG·FF·EE·······DD····CC·BB·AA     (k1)
    UINT64 u1 = (k1 & 0x00036C061B6030DB) |         // ··············KK·JJ·II·············GG·FF·EE·············CC·BB·AA     (k1 & 0x00036C001B6000DB)
                ((k1 & 0x00C0000600003000) >> 4);   // ············LL···················HH···················DD········     (k1 & 0x00C0000600003000) >> 4
                                                    // ············LLKK·JJ·II···········HHGG·FF·EE···········DDCC·BB·AA     (u1)

                                                    // ········ll····kk·jj·ii·······hh····gg·ff·ee·······dd····cc·bb·aa     (k1)
    UINT64 u2 = (k2 & 0x00036C061B6030DB) |         // ··············kk·jj·ii·············gg·ff·ee·············cc·bb·aa     (k2 & 0x00036C001B6000DB)
                ((k2 & 0x00C0000600003000) >> 4);   // ············ll···················hh···················dd········     (k2 & 0x00C0000600003000) >> 4
                                                    // ············llkk·jj·ii···········hhgg·ff·ee···········ddcc·bb·aa     (u2)

                                                    // ··llkk·jj·ii···········hhgg·ff·ee···········ddcc·bb·aa··········     (u2 << 10)
    u1 |= (u2 << 10);                               // ··llkk·jj·iiLLKK·JJ·II·hhgg·ff·eeHHGG·FF·EE·ddcc·bb·aaDDCC·BB·AA     (u1)

    UINT64 u3 = (k3 & 0x00036C061B6030DB) | ((k3 & 0x00C0000600003000) >> 4);
    UINT64 u4 = (k4 & 0x00036C061B6030DB) | ((k4 & 0x00C0000600003000) >> 4);
    u3 |= (u4 << 10);

    /* Now we hash u1 and u3, which contain the packed bits, to the configured number of hash bits.
        
       This is the 64-bit version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.  Since we start with
        two 64-bit values, there is no "tail" computation.
    */
    const UINT64 c1 = 0x87c37b91114253d5;
    const UINT64 c2 = 0x4cf5ad432745937f;
    const UINT64 seed = 0x5e4c39a7b31672d1;     // (16/32 bits are 1)
    UINT64 h1 = seed;
    UINT64 h2 = seed;

    // body
    u1 *= c1;
    u1 = (u1 << 31) | (u1 >> 33);   // _rotl64( ul, 31 )
    u1 *= c2;
    h1 ^= u1;

    h1 = (h1 << 27) | (h1 >> 37);   // _rotl64( hl, 27 )
    h1 += h2;
    h1 = h1*5 + 0x52dce729;

    u3 *= c2;
    u3 = (u3 << 33) | (u3 >> 31);   // _rotl64( u3, 33 )
    u3 *= c1;
    h2 ^= u3;

    h2 = (h2 << 31) | (h2 >> 33);   // _rotl64( h2, 31 )
    h2 += h1;
    h2 = h2*5 + 0x38495ab5;

    // (no "tail")

    // finalization
    h1 ^= seedWidth;
    h2 ^= seedWidth;

    h1 += h2;
    h2 += h1;

    // finalization mix - force all bits of a hash block to avalanche
    h1 ^= h1 >> 33;                 // h1 = fmix(h1);
    h1 *= 0xff51afd7ed558ccd;
    h1 ^= h1 >> 33;
    h1 *= 0xc4ceb9fe1a85ec53;
    h1 ^= h1 >> 33;

    h2 ^= h2 >> 33;                 // h2 = fmix(h2);
    h2 *= 0xff51afd7ed558ccd;
    h2 ^= h2 >> 33;
    h2 *= 0xc4ceb9fe1a85ec53;
    h2 ^= h2 >> 33;

    h1 += h2;
    h2 += h1;

    // mask the 64-bit result to the required number of bits
    return static_cast<UINT32>(h2 & cckc.hashBitsMask);
}

/// [device] function getV5
static inline __device__ UINT64 getV5( const UINT32 * const __restrict__ p, const UINT64 ofs )
{
    UINT64 ofs32 = (ofs * 5) >> 2;          // compute the offset of the UINT32 in which the specified Hvalue5 begins
    UINT64 u64 = static_cast<UINT64>(p[ofs32]) | (static_cast<UINT64>(p[ofs32+1]) << 32);
    INT32 shr = (ofs & 3) << 3;             // number of bits to shift
    return (u64 >> shr) & 0x000000FFFFFFFFFF;

#if TODO_CHOP_ASAP
    UINT64 ofs32 = (ofs >> 2) * 5;          // compute the offset of the HJ54-formatted data for the specified Hvalue offset


    // TODO: TEST WHETHER ldg DOES ANYTHING (IT'S DOUBTFUL)

#if __CUDA_ARCH__ < 350
    UINT32 hi4 = p[ofs32];                  // get the high-order bytes for the four subsequent Hvalues
#else
    UINT32 hi4 = __ldg( p+ofs32 );          // get the high-order bytes for the four subsequent Hvalues
#endif

    UINT32 mod4 = ofs & 3;                  // compute ofs MOD 4
    ofs32 += mod4 + 1;                      // compute the offset of the four low-order bytes for the specified offset

#if __CUDA_ARCH__ < 350
    UINT32 lo = p[ofs32];                   // load the four low-order bytes
#else
    UINT32 lo = __ldg( p+ofs32 );            // load the four low-order bytes
#endif


    UINT32 hi = __bfe32( hi4, (mod4 << 3), 8 );     // shift the correct high-order byte into bits 0-7

    // return the 5-byte value as an unsigned 8-byte value; the three high-order bytes are zero because that's how bfe works
    return (static_cast<UINT64>(hi) << 32) | lo;
#endif
}

/// [device] function reverseComplement
static inline __device__ UINT64 reverseComplement( UINT64 v )
{
    /* (Documented in AriocCommon::A21ReverseComplement) */

    UINT64 vx = v & (static_cast<UINT64>(7) << 30);
    v = ((v & MASKA21) >> 33) | (v << 33);
    v = ((v & 0x7FFF00003FFF8000) >> 15) | ((v & 0x0000FFFE00007FFF) << 15);
    vx |= (v & 0x01C0038000E001C0);
    v = ((v & 0x7E00FC003F007E00) >> 9) | ((v & 0x003F007E001F803F) << 9);
    v = ((v & 0x7038E070381C7038) >> 3) | ((v & 0x0E071C0E07038E07) << 3);
    v |= vx;
    const UINT64 v100 = v & MASK100;
    return v ^ (v100 - (v100 >> 2));    
}

/// [device] function loadQiF
static inline __device__ void loadQiF( UINT64& i64lo, UINT64& i64m1, UINT64& i64m2, UINT64& i64hi, UINT64& i64n, const UINT64* const __restrict__ pQi )
{
    i64lo = pQi[0];
    i64m1 = pQi[1*CUDATHREADSPERWARP];
    i64m2 = pQi[2*CUDATHREADSPERWARP];
    i64hi = pQi[3*CUDATHREADSPERWARP];
    i64n  = pQi[4*CUDATHREADSPERWARP];
}

/// [device] function loadQiRC
static inline __device__ void loadQiRC( UINT64& i64lo, UINT64& i64m1, UINT64& i64m2, UINT64& i64hi, UINT64& i64n, const UINT64* const __restrict__ pQi, const INT16 N )
{
    /* (See test code in baseQreader::copyToQwarp) */

    /* Build the A21-encoded reverse complement of the Q sequence.

       The encoding is described in AriocCommon.h:
        000     0        (null)
        001     1        (null)
        010     2        Nq (Q sequence N)
        011     3        Nr (R sequence N)
        100     4        A
        101     5        C
        110     6        G
        111     7        T
    */

    INT16 i21 = blockdiv( N, 21 ) - 1;

    // compute the number of bits to shift
    INT32 shr = 3 * (N % 21);
    if( shr == 0 )
        shr = 63;
    INT32 shl = 63 - shr;

    /* Symbols 0-20 */
    UINT64 a21 = pQi[CUDATHREADSPERWARP*i21--];
    UINT64 a21prev = pQi[CUDATHREADSPERWARP*i21--];

    // combine the adjacent A21-encoded values; bit 63 is also affected but it will get zeroed soon anyway
    UINT64 v = (a21 << shl) | (a21prev >> shr);

    // save the reverse-complement of the A21-encoded values
    i64lo = reverseComplement( v );

    /* Symbols 21-41 */
    a21 = a21prev;
    a21prev = pQi[CUDATHREADSPERWARP*i21--];
    v = (a21 << shl) | (a21prev >> shr);
    i64m1 = reverseComplement( v );

    /* Symbols 42-62 */
    a21 = a21prev;
    a21prev = pQi[CUDATHREADSPERWARP*i21--];
    v = (a21 << shl) | (a21prev >> shr);
    i64m2 = reverseComplement( v );

    /* Symbols 63-83 */
    a21 = a21prev;
    a21prev = pQi[0];
    v = (a21 << shl) | (a21prev >> shr);
    i64hi = reverseComplement( v );

    /* Symbols 84- */
    v = a21prev << shl;
    i64n = reverseComplement( v );
}

/// [device] function convertCT
static inline __device__ void convertCT( UINT64& v )
{
    /* Convert all instances of C (101) to T (111) by ORing with a bitmask wherein...
        - bit 1 of each 3-bit field is set only when that subfield's value is 101
        - bits 0 and 2 of each 3-bit field are zero
    */
    v |= (((v >> 1) & (~v) & (v << 1)) & MASK010);
}

/// [device] function convertQiCT
static inline __device__ void convertQiCT( UINT64& i64lo, UINT64& i64m1, UINT64& i64m2, UINT64& i64hi, UINT64& i64n )
{
    convertCT( i64lo );
    convertCT( i64m1 );
    convertCT( i64m2 );
    convertCT( i64hi );
    convertCT( i64n );
}

/// [device] function loadJinfo
static inline __device__ void loadJinfo( UINT64* poJ, UINT32* pnJ,
                                         UINT64 i64lo, UINT64 i64m1, UINT64 i64m2, UINT64 i64hi, UINT64 i64n,
                                         const UINT32* const __restrict__ pH, const UINT32* const __restrict__ pJ, const UINT32 sps )
{
    // load the seed width constant
    const INT16 seedWidth = cckc.seedWidth;

    // compute seeds and find the corresponding J-list sizes
    for( INT16 i=0; i<cckc.npos; ++i )
    {
        // compute the spaced-seed hash for the i'th position in the Q sequence
        UINT32 hashKey = computeH32( i64lo, i64m1, i64m2, i64hi, seedWidth );

        // get the 5-byte Hvalue5 for the hash key
        UINT64 hval = getV5( pH, hashKey );

        // get the offset of the J list for the current hash key
        UINT64 ofsJ = HVALUE_OFSJ(hval);

        // if the J-list offset is nonzero ...
        UINT32 nJ = 0;
        if( ofsJ )
        {
            // get the number of J values in the J list
            nJ = (hval & HVALUE_MASK_NJ) ?      // if the J-list count is in the H value ...
                 HVALUE_NJ(hval) :              // ... extract the count from the H value
                 getV5( pJ, ofsJ++ );           // ... otherwise, get the count from the first item in the J list and increment the J-list offset

#if TODO_CHOP_ASAP
            nJ = (hval & HVALUE_MASK_NJ) ?      // if the J-list count is in the H value ...
                 HVALUE_NJ(hval) :              // ... extract the count from the H value
                 pJ[ofsJ++];                    // ... otherwise, get the count from the first item in the J list and increment the J-list offset
#endif

            // clamp the number of J values to be processed from the list
            nJ = min(nJ, ccACP.maxJn);
        }

        // copy the J-list offset (which references the first J value in the list, even if preceded by the list count) to its output buffer
        *poJ = ofsJ;
        
        // copy the J-list size to its output buffer
        *pnJ = nJ;

        // shift the Q symbols into position for the (i+1)th iteration
        i64lo = ((i64lo >> 3) | (i64m1 << 60)) & MASKA21;
        i64m1 = ((i64m1 >> 3) | (i64m2 << 60)) & MASKA21;
        i64m2 = ((i64m2 >> 3) | (i64hi << 60)) & MASKA21;
        i64hi = ((i64hi >> 3) | (i64n << 60))  & MASKA21;
        i64n >>= 3;

        // update the buffer pointers
        poJ += sps;
        pnJ += sps;
    }
}

/// [kernel] tuSetupN10_Kernel
static __global__ void tuSetupN10_Kernel( const UINT32* const __restrict__ pH,          // in: pointer to the H lookup table
                                          const UINT32* const __restrict__ pJ,          // in: pointer to the J lookup table
                                          const UINT32                     nQw,         // in: total number of Q warps
                                          const Qwarp*  const              pQwarp,      // in: list of Q-warp structs
                                          const UINT64* const __restrict__ pQiBuffer,   // in: interleaved Q sequence data
                                          const UINT32                     celJ,        // in: total number of elements in J-list buffers
                                          const UINT32                     sps,         // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                                UINT64* const              poJBuffer,   // out: pointer to J-list offsets (one per seed)
                                                UINT32* const              pnJBuffer    // out: pointer to J-list sizes (one per seed)
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

    /* do nothing if the Q sequence is shorter than the minimum number of symbols spanned by the overlapping spaced seeds
        (the J-list offset and list-size buffers contain zeros for this Q sequence) */
    const INT16 N = pQw->N[iq];
    if( N < cckc.minSeedN )
        return;

    // point to the Q sequence for the current thread
    const UINT64* const __restrict__ pQi = pQiBuffer + pQw->ofsQi + iq;

    // get the encoded symbols for the Q sequence
    UINT64 i64lo;
    UINT64 i64m1;
    UINT64 i64m2;
    UINT64 i64hi;
    UINT64 i64n;

    if( cckc.doCTconversion )
    {
        // compute the offset of the J-list info
        UINT32 tofs = tid * cckc.npos * sps;

        // point to the output buffers
        UINT64* poJ = poJBuffer + tofs;
        UINT32* pnJ = pnJBuffer + tofs;
    
        // get the A21-encoded symbols for the Q sequence
        loadQiF( i64lo, i64m1, i64m2, i64hi, i64n, pQi );
        convertQiCT( i64lo, i64m1, i64m2, i64hi, i64n );

        // copy J-list counts and offsets for all seed positions
        loadJinfo( poJ, pnJ, i64lo, i64m1, i64m2, i64hi, i64n, pH, pJ, sps );

        if( cckc.sps == 2 )
        {
            // do the same for the reverse complement of the Q sequence
            loadQiRC( i64lo, i64m1, i64m2, i64hi, i64n, pQi, N );
            convertQiCT( i64lo, i64m1, i64m2, i64hi, i64n );

            /* copy J-list counts and offsets for all seed positions; the J-list info for the reverse-complemented,
                CT-converted Q sequence is at an odd-numbered offset */
            loadJinfo( poJ+1, pnJ+1, i64lo, i64m1, i64m2, i64hi, i64n, pH, pJ, sps );
        }
    }

    else
    {
        // get the A21-encoded symbols for the Q sequence
        loadQiF( i64lo, i64m1, i64m2, i64hi, i64n, pQi );

        // point to the output buffers
        UINT32 tofs = tid * cckc.npos;
        UINT64* poJ = poJBuffer + tofs;
        UINT32* pnJ = pnJBuffer + tofs;

        // copy J-list counts and offsets for all seed positions
        loadJinfo( poJ, pnJ, i64lo, i64m1, i64m2, i64hi, i64n, pH, pJ, sps );
    }
}

#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void tuSetupN10::initConstantMemory()
{
    CRVALIDATOR;

    // copy parameters into CUDA constant memory
    CRVALIDATE = cudaMemcpyToSymbol( ccACP, &m_pab->aas.ACP, sizeof ccACP, 0, cudaMemcpyHostToDevice );
    CRVALIDATE = cudaMemcpyToSymbol( cckc, &m_kc, sizeof m_kc, 0, cudaMemcpyHostToDevice );
}

/// [private] method initSharedMemory
UINT32 tuSetupN10::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( tuSetupN10_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void tuSetupN10::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );

    // execute the kernel
    tuSetupN10_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->pgi->pHn,           // in: H lookup table
                                                         m_pab->Jn.p,               // in: J lookup table
                                                         m_pqb->QwBuffer.n,         // in: number of Qwarps
                                                         m_pqb->DB.Qw.p,            // in: Qwarp buffer
                                                         m_pqb->DB.Qi.p,            // in: interleaved Q sequence data
                                                         m_pqb->DBj.celJ,           // in: total number of elements in J-list buffers
                                                         m_pab->StrandsPerSeed,     // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                                         m_pqb->DBj.oJ.p,           // out: per-seed J-list offsets
                                                         m_pqb->DBj.nJ.p            // out: per-Q J-list sizes
                                                       );
}
#pragma endregion
