/*
  baseCountJs.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"



// TODO: CHOP WHEN DEBUGGED
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#pragma region CUDA device code and data
/* CUDA constant memory

    To avoid a compile-time "file uses too much global constant data" error, we limit the maximum number of seed
     positions to what can be represented in a fixed-size vector of unsigned 16-bit values.  This implies a limit
     on the number of query-sequence positions that can be seeded, but that limit is at least 15K or so, so it's
     not a problem for the current "short read" Arioc implementation.

    We also copy only the seed positions associated with the current seed iteration, which should never exceed
     half of the number of precomputed seed positions.  (See A21HashedSeed::initHSP32().)
*/
static __device__ __constant__ UINT16                                       ccSPSI[A21HashedSeed::celSPSI/2];
static __device__ __constant__ __align__(8) baseCountJs::KernelConstants    cckc;


/* CUDA shared memory
*/
/// [device] method initializeSharedMemory
static inline __device__ void initializeSharedMemory()
{
}

/// [device] function computeH32 (UINT64) (see computeH32(UINT64) in CppCommon Hash.cpp for the corresponding CPU implementation)
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

    // mask the 32-bit result to the required number of bits
    return h & static_cast<UINT32>(cckc.hashBitsMask);
}

/// [private] method computeH64(UINT64,UINT64) (see computeH64(UINT64,UINT64) in CppCommon Hash.cpp for the corresponding CPU implementation)
static inline __device__ UINT32 computeH32( UINT64 k1, UINT64 k2 )
{
    /* Computes a 64-bit value by hashing the two specified 64-bit values.

       The hash implementation is the x64 version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.
        Since we start with two 64-bit values, there is no "tail" computation.

       When the reference kmers are encoded, we isolate the bits of interest by applying a mask, e.g.
            k2 &= this->seedMask;
       But we don't bother doing that here because the bits have already been masked in getQik().

       We return only the low-order bits of the result.
    */
    const UINT64 c1 = 0x87c37b91114253d5;
    const UINT64 c2 = 0x4cf5ad432745937f;
    const UINT64 seed = 0x5e4c39a7b31672d1;     // (16/32 bits are 1)
    UINT64 h1 = seed;
    UINT64 h2 = seed;

    // body
    k1 *= c1;
    k1 = (k1 << 31) | (k1 >> (64-31));  // k1 = _rotl64( k1, 31 );
    k1 *= c2;
    h1 ^= k1;

    h1 = (h1 << 27) | (h1 >> (64-27));  // h1 = _rotl64( h1, 27 );
    h1 += h2;
    h1 = h1*5 + 0x52dce729;

    k2 *= c2;
    k2 = (k2 << 33) | (k2 >> (64-33));  // k2 = _rotl64( k2, 33 );
    k2 *= c1;
    h2 ^= k2;

    h2 = (h2 << 31) | (h2 >> (64-31));  // h2 = _rotl64( h2, 31 );
    h2 += h1;
    h2 = h2*5 + 0x38495ab5;

    // (no "tail")

    // finalization
    h1 ^= cckc.sand;   // yes, this is weird but it's how MurmurHash3 is implemented
    h2 ^= cckc.sand;

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

    // return a 32-bit result, masked to the required number of bits 
    return static_cast<UINT32>(h2 & cckc.hashBitsMask);
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

/// [device] method getQikF
static inline __device__ UINT64 getQikF( const UINT64* const __restrict__ pQi, const INT16 i )
{
    /* seedWidth <= 21 */

    // compute the index of the 64-bit A21-formatted value in Qi that contains the i'th symbol
    const INT16 iQi = i / 21;

    // read the 64-bit value that contains the ith symbol
    INT32 ofs = iQi * CUDATHREADSPERWARP;
    UINT64 Qik = pQi[ofs];

    // compute the number of bits to shift
    const INT16 iMod21 = i - (21 * iQi);        // should be faster than i % 21
    INT16 shr = 3 * iMod21;
    if( shr <= 3 )
        Qik >>= shr;
    else
    {
        // the 21-mer spans two consecutive 64-bit values
        UINT64 Qn = pQi[ofs+CUDATHREADSPERWARP];
        Qik = ((Qn << (63-shr)) | (Qik >> shr));
    }

    // mask the low-order bits so as to return the correct number of symbols (seedWidth) for the seed
    return Qik & cckc.seedMask;
}

/// [device] method getQikF2
static inline __device__ ulonglong2 getQikF2( const UINT64* const __restrict__ pQi, const INT16 i )
{
    /* 22 <= seedWidth <= 42 */

    // compute the index of the 64-bit A21-formatted value in Qi that contains the i'th symbol
    const INT16 iQi = i / 21;

    // read the 64-bit value that contains the ith symbol
    INT32 ofs = iQi * CUDATHREADSPERWARP;
    UINT64 Qik1 = pQi[ofs];

    // read the subsequent 64-bit value
    ofs += CUDATHREADSPERWARP;
    UINT64 Qik2 = pQi[ofs];

    // if the seed kmer extends beyond the subsequent 64-bit value, read an additional 64-bit value
    INT32 i0 = iQi * 21;    // bit index of first Qi value
    UINT64 Qik3 = ((i+cckc.seedWidth) > (i0+42)) ? pQi[ofs+CUDATHREADSPERWARP] : 0;    

    // compute the number of bits to shift
    const INT16 iMod21 = i - (21 * iQi);        // should be faster than i % 21
    INT16 shr = 3 * iMod21;
    INT16 shl = 63 - shr;

    // shift the symbols into position in two 64-bit A21-formatted values
    ulonglong2 Qik;
    Qik.x = ((Qik1 >> shr) | (Qik2 << shl)) & MASKA21;
    Qik.y = ((Qik2 >> shr) | (Qik3 << shl)) & cckc.seedMask;
    return Qik;
}

/// [device] method getQikRC
static inline __device__ UINT64 getQikRC( const UINT64* const __restrict__ pQi, const INT16 i, const INT16 N )
{
    /* seedWidth <= 21 */

    // compute the position of the first symbol in the reverse complement
    const INT16 irc = N - (i + cckc.seedWidth);

    // compute the index of the 64-bit A21-formatted value in Qi that contains the irc'th symbol
    const INT16 iQi = irc / 21;

    // read the 64-bit value that contains the irc'th symbol
    const INT16 ofs = iQi * CUDATHREADSPERWARP;
    UINT64 Qik = pQi[ofs];

    // compute the number of bits to shift
    const INT16 iMod21 = irc - (21 * iQi);      // should be faster than i % 21
    INT16 shr = 3 * iMod21;
    if( shr <= 3 )
        Qik >>= shr;
    else
    {
        // the 21-mer spans two consecutive 64-bit values
        UINT64 Qn = pQi[ofs+CUDATHREADSPERWARP];
        Qik = ((Qn << (63-shr)) | (Qik >> shr));
    }

    // compute the reverse complement
    Qik = reverseComplement( Qik );

    // shift the low-order bits so as to return the correct number of symbols (seedWidth) for the seed
    return Qik >> cckc.seedMaskShift;
}

/// [device] method getQikRC2
static inline __device__ ulonglong2 getQikRC2( const UINT64* const __restrict__ pQi, const INT16 i, const INT16 N )
{
    /* 22 <= seedWidth <= 42 */

    // compute the position of the first symbol in the reverse complement
    const INT16 irc = N - (i + cckc.seedWidth);

    // compute the index of the 64-bit A21-formatted value in Qi that contains the irc'th symbol
    const INT16 iQi = irc / 21;

    // read the 64-bit value that contains the irc'th symbol
    INT32 ofs = iQi * CUDATHREADSPERWARP;
    UINT64 Qik1 = pQi[ofs];

    // read the subsequent 64-bit value
    ofs += CUDATHREADSPERWARP;
    UINT64 Qik2 = pQi[ofs];

    // what we do next depends on whether the seed spans two or three 64-bit A21-formatted values in Qi
    ulonglong2 Qik;
    INT32 i0 = iQi * 21;                    // bit index of first Qi value
    if( (irc+cckc.seedWidth) > (i0+42) )
    {
        /* The seed spans 3 64-bit values in Qi.  Let's designate the symbols in the seed with "A"
            representing the symbols in the first Qi value, "B" representing the symbols in the
            second Qi value, and "C" representing the symbols in the third Qi value.  So the seed
            would represented as A0 A1 A2 ... Ax B0 B1 B2 ... B20 C0 C1 C2 ... Cx, and the layout
            in Qi would look like this (LSB=least significant bit, MSB=most significant bit):

                LSB  |---- A0 A1 A2 ... Ax | B0 B1 B2 ... B20 | C0 C1 C2 ... Cx ----|  MSB
        */
        // compute the reverse complement of both 64-bit (21-symbol) Qi values
        UINT64 Qik3 = pQi[ofs+CUDATHREADSPERWARP];
        Qik1 = reverseComplement( Qik1 );       // Qik1: Ax ... A2 A1 A0 ----
        Qik2 = reverseComplement( Qik2 );       // Qik2: B20 B19 ... B2 B1 B0
        Qik3 = reverseComplement( Qik3 );       // Qik3: ---- Cx ... C2 C1 C0

        // compute the number of bits to shift
        INT16 shr = 3 * ((i0+63) - (irc+cckc.seedWidth));
        INT16 shl = 63 - shr;

#if TODO_CHOP_WHEN_DEBUGGED
        if( (shr < 0) || (shr > 63) )
            asm( "brkpt;");
        if( (shl < 0) || (shl > 63) )
            asm( "brkpt;");
#endif

        // shift bits into position in the first 64-bit result value
        Qik.x = Qik3 >> shr;                    // Qik.x: Cx ... C2 C1 C0 000000
        Qik.x |= (Qik2 << shl) & MASKA21;       // Qik.x: Cx ... C2 C1 C0 B20 ...

        // shift bits into position in the second 64-bit result value
        Qik.y = ((Qik2 >> shr) | (Qik3 << shl)) & cckc.seedMask;  // Qik.y: Bx ... B0 Ax ... A2 A1 A0
    }
    else
    {
        /* The seed spans 2 64-bit values in Qi.  Let's designate the symbols in the seed with "A"
            representing the symbols in the first Qi value and "B" representing the symbols in the
            second Qi value.  So the seed would represented as A0 A1 A2 ... Ax B0 B1 B2 ... Bx,
            and the layout in Qi would look like this (LSB=least significant bit, MSB=most significant bit):

                LSB  |---- A0 A1 A2 ... Ax | B0 B1 B2 ... Bx ----|  MSB
        */

        // compute the reverse complement of both 64-bit (21-symbol) Qi values
        Qik1 = reverseComplement( Qik1 );       // Qik1: Ax ... A2 A1 A0 ----
        Qik2 = reverseComplement( Qik2 );       // Qik2: ---- Bx ... B2 B1 B0

        // compute the number of bits to shift
        INT16 shr = 3 * ((i0+42) - (irc+cckc.seedWidth));
        INT16 shl = 63 - shr;


#if TODO_CHOP_WHEN_DEBUGGED
        if( (shr < 0) || (shr > 63) )
            asm( "brkpt;");
        if( (shl < 0) || (shl > 63) )
            asm( "brkpt;");
#endif



        /*** TODO: THINK ABOUT REFACTORING BUT MAKE SURE IT WORKS FIRST! ***/


        // shift bits into position in the first 64-bit result value
        Qik.x = Qik2 >> shr;                    // Qik.x: Bx ... B2 B1 B0 000000
        Qik.x |= (Qik1 << shl) & MASKA21;       // Qik.x: Bx ... B2 B1 B0 Ax ...

        // shift bits into position in the second 64-bit result value
        Qik.y = (Qik1 >> shr) & cckc.seedMask;
    }

    return Qik;
}

// typedefs for methods that are called through a function pointer:
typedef UINT32 (*pfnHashQiKmer)( const UINT64* const __restrict__ pQi, const INT16 i, const INT16 N );
typedef void (*pfnHashProbe)( const INT16 i, const UINT64* const __restrict__ pQi, const INT16 N, UINT64*& poJ, UINT32*& pnJ, UINT32& rbits, const UINT32* const __restrict__ pH, const UINT32* const __restrict__ pJ, const UINT32 tid, pfnHashQiKmer hashKmerF, pfnHashQiKmer hashKmerRC );

/// [device] method hashQiKmerF
static inline __device__ UINT32 hashQiKmerF( const UINT64* const __restrict__ pQi, const INT16 i, const INT16 )
{
    /* Forward kmer, k <= 21 */
    UINT64 Qik = getQikF( pQi, i );     // extract the i'th kmer from the Q sequence
    return computeH32( Qik );           // compute a 32-bit hash for the i'th kmer
}

/// [device] method hashQiKmerFCT
static inline __device__ UINT32 hashQiKmerFCT( const UINT64* const __restrict__ pQi, const INT16 i, const INT16 )
{
    /* Forward kmer, k <= 21 */
    UINT64 Qik = getQikF( pQi, i );     // extract the i'th kmer from the Q sequence
    convertCT( Qik );                   // convert C to T for bsDNA alignments
    return computeH32( Qik );           // compute a 32-bit hash for the i'th kmer
}

/// [device] method hashQiKmerRCCT
static inline __device__ UINT32 hashQiKmerRCCT( const UINT64* const __restrict__ pQi, const INT16 i, const INT16 N )
{
    /* Reverse complement kmer, k <= 21 */
    UINT64 Qik = getQikRC( pQi, i, N ); // extract the i'th kmer from the Q sequence
    convertCT( Qik );                   // convert C to T for bsDNA alignments
    return computeH32( Qik );           // compute a 32-bit hash for the i'th kmer
}

/// [device] method hashQiKmerF2
static inline __device__ UINT32 hashQiKmerF2( const UINT64* const __restrict__ pQi, const INT16 i, const INT16 )
{
    /* Forward kmer, 22 <= k <= 42 */
    ulonglong2 Qik = getQikF2( pQi, i );    // extract the i'th kmer from the Q sequence
    return computeH32( Qik.x, Qik.y );      // compute a 32-bit hash for the i'th kmer
}

/// [device] method hashQiKmerFCT2
static inline __device__ UINT32 hashQiKmerFCT2( const UINT64* const __restrict__ pQi, const INT16 i, const INT16 N )
{
    /* Forward kmer, 22 <= k <= 42 */
    ulonglong2 Qik = getQikF2( pQi, i );    // extract the i'th kmer from the Q sequence
    convertCT( Qik.x );                     // convert C to T for bsDNA alignments
    convertCT( Qik.y );
    return computeH32( Qik.x, Qik.y );      // compute a 32-bit hash for the i'th kmer
}

/// [device] method hashQiKmerRCCT2
static inline __device__ UINT32 hashQiKmerRCCT2( const UINT64* const __restrict__ pQi, const INT16 i, const INT16 N )
{
    /* Reverse complement kmer, 22 <= k <= 42 */
    ulonglong2 Qik = getQikRC2( pQi, i, N );    // extract the i'th kmer from the Q sequence
    convertCT( Qik.x );                         // convert C to T for bsDNA alignments
    convertCT( Qik.y );
    return computeH32( Qik.x, Qik.y );          // compute a 32-bit hash for the i'th kmer
}

/// [device] function getV5
static inline __device__ UINT64 getV5( const UINT32 * const __restrict__ p, const UINT64 ofs )
{
    UINT64 ofs32 = (ofs * 5) >> 2;          // compute the offset of the UINT32 in which the specified Hvalue5 begins
    UINT64 u64 = static_cast<UINT64>(p[ofs32]) | (static_cast<UINT64>(p[ofs32+1]) << 32);
    INT32 shr = (ofs & 3) << 3;             // number of bits to shift
    return (u64 >> shr) & 0x000000FFFFFFFFFF;

#if TODO_CHOP_ASAP
    UINT64 ofs32 = (ofs >> 2) * 5;              // compute the offset of the HJ54-formatted data for the specified Hvalue offset
    UINT32 hi4 = p[ofs32];                      // get the high-order bytes for the four subsequent Hvalues
    UINT32 mod4 = ofs & 3;                      // compute ofs MOD 4
    ofs32 += mod4 + 1;                          // compute the offset of the four low-order bytes for the specified offset
    UINT32 lo = p[ofs32];                       // load the four low-order bytes
    UINT32 hi = __bfe32( hi4, (mod4 << 3), 8 ); // shift the correct high-order byte into bits 0-7

    // return the 5-byte value as an unsigned 8-byte value; the three high-order bytes are zero because that's how bfe works
    return (static_cast<UINT64>(hi) << 32) | lo;
#endif
}

/// [device] method probeLUT
static inline __device__ void probeLUT( UINT64* poJ, UINT32* pnJ, UINT32& rbits, const UINT32 hashKey, const UINT32* const __restrict__ pH, const UINT32* const __restrict__ pJ, const UINT32 tid )
{
    // get the 5-byte Hvalue for the hash key
    UINT64 hval = getV5( pH, hashKey );

    // get the offset of the J list for the current hash key
    UINT64 ofsJ = HVALUE_OFSJ(hval);

    // if the J-list offset is nonzero ...
    UINT32 nJ = 0;
    if( ofsJ )
    {
        // get the number of J values in the J list and post-increment the J-list offset
        nJ = (hval & HVALUE_MASK_NJ) ?  // if the J-list count is in the H value ...
                HVALUE_NJ(hval) :       // ... extract the count from the H value
                getV5( pJ, ofsJ++ );    // ... otherwise, get the count from the first item in the J list and increment the J-list offset

#if TODO_CHOP_ASAP
        nJ = (hval & HVALUE_MASK_NJ) ?  // if the J-list count is in the H value ...
                HVALUE_NJ(hval) :       // ... extract the count from the H value
                pJ[ofsJ++];             // ... otherwise, get the count from the first item in the J list and increment the J-list offset
#endif

        // clamp the number of J values to be processed from the list
        nJ = min(nJ, cckc.maxJg);
    }

    // copy the J-list offset (which references the first J value in the list, even if preceded by the list count) to its output buffer
    *poJ = ofsJ;
        
    // copy the J-list size to its output buffer
    *pnJ = nJ;

    /* Loop across every 5th 4-byte value in the J list to build a bitmap corresponding to subId values in the J list:
        - a null rbits value (all bits set) indicates that the corresponding J list should not be filtered by subId
        - a zero value for rbits should not happen
    */
    while( (rbits != 0xFFFFFFFF) && nJ )
    {
        UINT32 hi4 = pJ[ofsJ];

        UINT32 nJcurrent = min(4, nJ);
        for( UINT32 pos=0; pos<(nJcurrent*8); pos+=8 )
        {
            INT32 subId = __bfe32( hi4, pos, 7 );                   // bits 0-6, 8-14, 16-22, or 24-30
            if( subId < 32 )
                rbits = __bfi32( 0xFFFFFFFF, rbits, subId, 1 );     // set the bit corresponding to the subId value
            else
                rbits = 0xFFFFFFFF;                                 // we can't represent subId values >= 32, so return null (all bits set)
        }

        // iterate
        ofsJ += 5;
        nJ -= nJcurrent;
    }
}

/// [device] method hashProbeF
static inline __device__ void hashProbeF( const INT16 i, const UINT64* const __restrict__ pQi, const INT16 N, UINT64*& poJ, UINT32*& pnJ, UINT32& rbits, const UINT32* const __restrict__ pH, const UINT32* const __restrict__ pJ, const UINT32 tid, pfnHashQiKmer hashKmerF, pfnHashQiKmer )
{
    // hash the forward kmer sequence
    UINT32 hash32 = hashQiKmerF( pQi, i, N );               // compute a 32-bit hash for the i'th kmer
    probeLUT( poJ++, pnJ++, rbits, hash32, pH, pJ, tid );   // hash and count J values; accumulate rbits
}

/// [device] method hashProbeFRC
static inline __device__ void hashProbeFRC( const INT16 i, const UINT64* const __restrict__ pQi, const INT16 N, UINT64*& poJ, UINT32*& pnJ, UINT32& rbits, const UINT32* const __restrict__ pH, const UINT32* const __restrict__ pJ, const UINT32 tid, pfnHashQiKmer hashKmerF, pfnHashQiKmer hashKmerRC )
{
    // hash the forward kmer sequence
    UINT32 hash32 = hashKmerF( pQi, i, N );                 // compute a 32-bit hash for the i'th kmer
    probeLUT( poJ++, pnJ++, rbits, hash32, pH, pJ, tid );   // hash and count J values; accumulate rbits

    /* Hash the reverse-complement kmer sequence.  The J-list info goes into
        the adjacent odd-numbered element in each output buffer. */
    hash32 = hashKmerRC( pQi, i, N );
    probeLUT( poJ++, pnJ++, rbits, hash32, pH, pJ, tid );
}

/// [kernel] baseCountJs_Kernel
static __global__ void baseCountJs_Kernel( const UINT32* const __restrict__ pQuBuffer,  // in: QIDs
                                           const UINT32                     nQu,        // in: number of QIDs
                                           const UINT32* const __restrict__ pH,         // in: H lookup table
                                           const UINT32* const __restrict__ pJ,         // in: J lookup table
                                           const Qwarp*  const              pQwarp,     // in: list of Qwarp structs
                                           const UINT64* const __restrict__ pQiBuffer,  // in: interleaved Q sequence data
                                           const UINT32                     nSeedPos,   // in: number of seed positions for the current seed iteration
                                           const UINT32                     seedsPerQ,  // in: maximum number of seeds for any Q sequence in the current batch
                                           const UINT32                     celJ,       // in: total number of elements in J-list buffers
                                                 UINT64* const              poJBuffer,  // out: pointer to J-list offsets (one per seed)
                                                 UINT32* const              pnJBuffer,  // out: pointer to J-list sizes (one per seed)
                                                 UINT32* const __restrict__ pRuBuffer   // out: subId bits [may be null]
                                         )
{
    initializeSharedMemory();

    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nQu )
        return;

    // load the QID of the Q sequence for the current CUDA thread
    UINT32 qid = pQuBuffer[tid];

    // compute the offset of the corresponding Qwarp
    const INT32 iw = QID_IW(qid);

    // point to the Qwarp struct that contains the Q sequences for this CUDA warp
    const Qwarp* pQw = pQwarp + iw;
    const INT16 iq = QID_IQ(qid);

    // point to the first 21 64-bit encoded symbols for the Q sequence
    const UINT64* const __restrict__ pQi = pQiBuffer + pQw->ofsQi + iq;

    // point to the output buffers
    UINT64* poJ = poJBuffer + (tid*seedsPerQ*cckc.sps);
    UINT32* pnJ = pnJBuffer + (tid*seedsPerQ*cckc.sps);

    // start with the first seed position for the specified seed iteration
    UINT32 iSeedPos = 0;

    // loop setup
    pfnHashProbe hashProbe = (cckc.sps == 2) ? hashProbeFRC : hashProbeF;
    pfnHashQiKmer hashKmerF;
    pfnHashQiKmer hashKmerRC;
    if( cckc.doCTconversion )
    {
        hashKmerF = (cckc.seedWidth <= 21) ? hashQiKmerFCT : hashQiKmerFCT2;
        hashKmerRC = (cckc.seedWidth <= 21) ? hashQiKmerRCCT : hashQiKmerRCCT2;
    }
    else
        hashKmerF = (cckc.seedWidth <= 21) ? hashQiKmerF : hashQiKmerF2;

    /* Loop across seed positions and accumulate rbits (one bit per subId):
        - if no rbits are required, the initial value is null (all bits set)
        - otherwise, the initial value is zero so that probeLUT can set bits for the subId values in
            the J lists
    */
    UINT32 rbits = (pRuBuffer ? 0 : _UI32_MAX);
    INT32 i = ccSPSI[iSeedPos];
    const INT32 iLimit = pQw->N[iq] - cckc.seedWidth;   // maximum possible seed position
    while( i <= iLimit )
    {
        // hash the i'th kmer; count J values; accumulate Rbits
        hashProbe( i, pQi, pQw->N[iq], poJ, pnJ, rbits, pH, pJ, tid, hashKmerF, hashKmerRC );

        // limit the number of seeds examined
        if( ++iSeedPos >= nSeedPos )
            break;

        // get the 0-based position of the next seed
        i = ccSPSI[iSeedPos];
    }
    
    // optionally copy the subId bitmap to its output buffer
    if( pRuBuffer )
        pRuBuffer[tid] = rbits;
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseCountJs::initConstantMemory()
{
    CRVALIDATOR;

    // copy parameters into CUDA constant memory
    CRVALIDATE = cudaMemcpyToSymbol( cckc, &m_kc, sizeof m_kc, 0, cudaMemcpyHostToDevice );

    // copy the seed positions for the current seed iteration
    UINT32 iSeedPos = m_pab->a21hs.ofsSPSI.p[m_isi];
    m_nSeedPos = m_pab->a21hs.ofsSPSI.p[m_isi+1] - iSeedPos;

    CRVALIDATE = cudaMemcpyToSymbol( ccSPSI,
                                     m_pab->a21hs.SPSI.p+iSeedPos,
                                     m_nSeedPos*sizeof(UINT16),
                                     0,
                                     cudaMemcpyHostToDevice );
}

/// [private] method initSharedMemory
UINT32 baseCountJs::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseCountJs_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseCountJs::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{

#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT32> Quxxx( m_pdbb->Qu.n, false );
    m_pdbb->Qu.CopyToHost( Quxxx.p, Quxxx.Count );
    CDPrint( cdpCD0, __FUNCTION__ );
#endif




#if TODO_CHOP_WHEN_DEBUGGED
    // dump the last few qids
    WinGlobalPtr<UINT32> Quxxx( m_pdbb->Qu.Count, false );
    m_pdbb->Qu.CopyToHost( Quxxx.p, Quxxx.Count );
    Quxxx.n = m_pdbb->Qu.n;

    CDPrint( cdpCD0, "baseCountJs::launchKernel [%d]: DBj.Qu.p at 0x%016llx", m_pqb->pgi->deviceId, m_pdbb->Qu.p );
    for( UINT32 n=Quxxx.n-32; n<Quxxx.n; ++n )
        CDPrint( cdpCD0, "baseCountJs::launchKernel [%d]: 0x%08x: qid=0x%08x", m_pqb->pgi->deviceId, n, Quxxx.p[n] );
#endif



    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );

    // if the Ru buffer is empty, fill it up; if it has already been initialized, pass a null Ru buffer pointer to the CUDA kernel
    UINT32* pRuBuffer = (m_pdbb->Ru.n ? NULL : m_pdbb->Ru.p);

    // execute the kernel
    baseCountJs_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pdbb->Qu.p,             // in: QIDs of seed-and-extend candidate pairs
                                                          m_pdbb->Qu.n,             // in: number of QIDs
                                                          m_pqb->pgi->pHg,          // in: H lookup table
                                                          m_pab->Jg.p,              // in: J lookup table
                                                          m_pqb->DB.Qw.p,           // in: Qwarp buffer
                                                          m_pqb->DB.Qi.p,           // in: interleaved Q sequence data
                                                          m_nSeedPos,               // in: number of seed positions for the current seed iteration
                                                          m_pdbb->AKP.seedsPerQ,    // in: maximum seeds computed for each Q sequence
                                                          m_pqb->DBj.celJ,          // in: total number of elements in J-list buffers
                                                          m_pqb->DBj.oJ.p,          // out: per-seed J-list offsets
                                                          m_pqb->DBj.nJ.p,          // out: per-Q J-list sizes
                                                          pRuBuffer                 // out: per-Q subId bits [pointer may be null]
                                                        );
}
#pragma endregion
