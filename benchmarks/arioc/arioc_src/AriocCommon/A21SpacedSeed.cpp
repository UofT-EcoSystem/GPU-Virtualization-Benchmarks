/*
  A21SpacedSeed.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
   For two mismatches, the optimal periodic spaced seed is based on the well-known repeating bit pattern 0010111.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// [public] constructor (INT32, INT32)
A21SpacedSeed::A21SpacedSeed( const char* _si, INT32 maxMismatches ) : spacedSeed(0),
                                                                       spacedSeedWeight(0),
                                                                       spacedSeed1Bits(0),
                                                                       npos(0)
{
    initSI();
    baseInit( _si );

    // get the specified seed info
    SeedInfo si = (m_seedIndex > 0) ? this->SI.Value( m_seedIndex ) : this->SI.Value(A21SpacedSeed::iNone);

    // initialize
    if( (si.k > 0) && (si.k != 84) )
        throw new ApplicationException( __FILE__, __LINE__, "spaced seed width %d is not supported", seedWidth );
    INT16 spacedSeedWeight = 48;    // (this would be different for a different-sized seed)
    initSSP32( si, spacedSeedWeight, maxMismatches, 7, si.nHashBits );
}

/// [public] destructor
A21SpacedSeed::~A21SpacedSeed()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initSI()
void A21SpacedSeed::initSI()
{
    this->SI["error"] = SeedInfo( -1, -1 , -1, 0 );        // the 0th item in the list is used for error info
    this->SI["none"] = SeedInfo( 0, 0, 0, 0 );
    this->SI["ssi00_0_00"] = SeedInfo( 0, 0, 0, 0 );
    this->SI["ssi00_0_00_CT"] = SeedInfo( 0, 0, 0, 1 );

    this->SI["ssi84_2_29"] = SeedInfo( 84, 2, 29, 0 );
    this->SI["ssi84_2_29_CT"] = SeedInfo( 84, 2, 29, 1 );
    this->SI["ssi84_2_30"] = SeedInfo( 84, 2, 30, 0 );
    this->SI["ssi84_2_30_CT"] = SeedInfo( 84, 2, 30, 1 );
}

/// [private] method initSSP32
void A21SpacedSeed::initSSP32( SeedInfo _si, INT16 _spacedSeedWeight, INT16 _maxMismatches, INT16 _npos, INT16 _hashKeyWidth )
{
    this->seedWidth = _si.k;                                                // number of adjacent symbols covered by the seed bit pattern
    this->spacedSeedWeight = _spacedSeedWeight;                             // number of "care" symbols in the seed
    this->maxMismatches = max2(_maxMismatches,_si.maxMismatches);           // maximum number of mismatches accepted in a nongapped spaced-seed mapping
    this->hashKeyWidth = _hashKeyWidth;
    this->npos = _npos;                                                     // number of adjacent spaced seeds required to cover the specified number of mismatches
    this->seedInterval = 0;

    this->spacedSeed1Bits = 2 * _spacedSeedWeight;                          // number of "care" bits in the seed
    this->seedMask = MASKA21;
    if( this->seedWidth%21 )                                                // mask for high-order seed bits in the high-order 64-bit value
        this->seedMask >>= (63 - (3*(this->seedWidth%21)));
    this->maskNonN = MASK100 & this->seedMask;
    this->hashBitsMask = 0xFFFFFFFFFFFFFFFF >> (64 - _hashKeyWidth);
    this->minNonN = this->seedWidth - _maxMismatches;

    this->spacedSeed = SPACED_SEED_BIT_PATTERN & this->seedMask;            // seed pattern for the high-order 64-bit value
}
#pragma endregion

#pragma region public methods
/// [public, virtual method implementation] method computeH32 (UINT64)
UINT32 A21SpacedSeed::ComputeH32( UINT64 b2164 )
{
    // (21 symbols is too few to be practical)
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public, virtual method implementation] method computeH32 (UINT64, UINT64)
UINT32 A21SpacedSeed::ComputeH32( UINT64 k1, UINT64 k2 )
{
    /* Computes a hash value for the two specified values (i.e., up to 42 symbols) by applying the optimal "periodic" (i.e. repeating)
        seed mask 0010111 to the specified 2164-encoded values.

       We first pack the 1 bits from each of the two specified 64-bit values into a single 64-bit value (of which 48 bits are "care" bits).
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

    /* Now we hash to the configured number of hash bits.

       This is the 32-bit version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.
    */

    // constants
    UINT32 c1 = 0xcc9e2d51;
    UINT32 c2 = 0x1b873593;
    UINT32 seed = 0xb0f57ee3;

    // bits 0-31
    register UINT32 v1 = static_cast<UINT32>(u1);

    v1 *= c1;
    v1 = _rotl( v1, 15 );
    v1 *= c2;

    UINT32 h = seed ^ v1;
    h = _rotl( h, 13 );
    h = 5*h + 0xe6546b64;

    // bits 32-63
    v1 = static_cast<UINT32>(u1 >> 32);
    v1 *= c1;
    v1 = _rotl( v1, 15 );
    v1 *= c2;

    h ^= v1;
    h = _rotl( h, 13 );
    h = 5*h + 0xe6546b64;

    // avalanche
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    // mask the 32-bit result to the required number of bits
    return h & this->hashBitsMask;
}

/// [public, virtual method implementation] method ComputeH32 (UINT64, UINT64, UINT64, UINT64)
UINT32 A21SpacedSeed::ComputeH32( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4 )
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
    u1 = _rotl64( u1, 31);
    u1 *= c2;
    h1 ^= u1;

    h1 = _rotl64( h1, 27 );
    h1 += h2;
    h1 = h1*5 + 0x52dce729;

    u3 *= c2;
    u3 = _rotl64( u3, 33 );
    u3 *= c1;
    h2 ^= u3;

    h2 = _rotl64( h2, 31 );
    h2 += h1;
    h2 = h2*5 + 0x38495ab5;

    // (no "tail")

    // finalization
    h1 ^= this->seedWidth;
    h2 ^= this->seedWidth;

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
    return static_cast<UINT32>(h2 & this->hashBitsMask);
}

/// [public, virtual method implementation] method ComputeH64 (UINT64)
UINT64 A21SpacedSeed::ComputeH64( UINT64 b2164 )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public, virtual method implementation] method computeH64
UINT64 A21SpacedSeed::ComputeH64( UINT64 k1, UINT64 k2 )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public, virtual method implementation] method ComputeH64 (UINT64, UINT64, UINT64, UINT64)
UINT64 A21SpacedSeed::ComputeH64( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4 )
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
    u1 = _rotl64( u1, 31);
    u1 *= c2;
    h1 ^= u1;

    h1 = _rotl64( h1, 27 );
    h1 += h2;
    h1 = h1*5 + 0x52dce729;

    u3 *= c2;
    u3 = _rotl64( u3, 33 );
    u3 *= c1;
    h2 ^= u3;

    h2 = _rotl64( h2, 31 );
    h2 += h1;
    h2 = h2*5 + 0x38495ab5;

    // (no "tail")

    // finalization
    h1 ^= this->seedWidth;
    h2 ^= this->seedWidth;

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
    return h2 & this->hashBitsMask;
}
#pragma endregion
