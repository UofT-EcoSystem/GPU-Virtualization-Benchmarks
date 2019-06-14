/*
  Hash.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

// default constructor
Hash::Hash( UINT64 _sand ) : Sand(_sand)
{
    // the default "sand" value is just a very large 64-bit integer
}

// destructor
Hash::~Hash()
{
}

/// [public] method ComputeH32(UINT64)
UINT32 Hash::ComputeH32( UINT64 u64 )
{
    /* Computes a 32-bit value by hashing the specified 64-bit value.

       The hash implementation is the 32-bit version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.
    */
    
    // constants
    UINT32 c1 = 0xcc9e2d51;
    UINT32 c2 = 0x1b873593;
    UINT32 seed = 0xb0f57ee3;

    // u64 bits 0-31
    register UINT32 k1 = static_cast<UINT32>(u64);

    k1 *= c1;
    k1 = _rotl( k1, 15 );
    k1 *= c2;

    UINT32 h = seed ^ k1;
    h = _rotl( h, 13 );
    h = 5*h + 0xe6546b64;

    // u64 bits 32-63
    k1 = static_cast<UINT32>(u64 >> 32);
    k1 *= c1;
    k1 = _rotl( k1, 15 );
    k1 *= c2;

    h ^= k1;
    h = _rotl( h, 13 );
    h = 5*h + 0xe6546b64;

    // (no "sand" here)

    // avalanche
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}

/// [public] method ComputeH64(UINT64)
UINT64 Hash::ComputeH64( UINT64 u64 )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [private] method computeH64(UINT64,UINT64)
UINT64 Hash::ComputeH64( UINT64 k1, UINT64 k2 )
{
    /* Computes a 64-bit value by hashing the two specified 64-bit values.

        The hash implementation is the x64 version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.
        Since we start with two 64-bit values, there is no "tail" computation.
    */
    const UINT64 c1 = 0x87c37b91114253d5;
    const UINT64 c2 = 0x4cf5ad432745937f;
    const UINT64 seed = 0x5e4c39a7b31672d1;     // (16/32 bits are 1)
    UINT64 h1 = seed;
    UINT64 h2 = seed;

    // body
    k1 *= c1;
    k1 = _rotl64( k1, 31 );
    k1 *= c2;
    h1 ^= k1;

    h1 = _rotl64( h1, 27 );
    h1 += h2;
    h1 = h1*5 + 0x52dce729;

    k2 *= c2;
    k2 = _rotl64( k2, 33 );
    k2 *= c1;
    h2 ^= k2;

    h2 = _rotl64( h2, 31 );
    h2 += h1;
    h2 = h2*5 + 0x38495ab5;

    // (no "tail")

    // finalization
    h1 ^= this->Sand;   // yes, this is weird but it's how MurmurHash3 is implemented
    h2 ^= this->Sand;

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

    return h2;
}

#pragma region static methods
/// [public, static] ComputeH32( UINT64*, INT32, INT32 )
UINT32 Hash::ComputeH32( const UINT64* pk, INT32 cel, INT32 incr )
{
    /* Computes a 64-bit value by hashing the specified 64-bit pattern.

        The hash implementation is the x64 version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.
    */
    const UINT64 c1 = 0x87c37b91114253d5;
    const UINT64 c2 = 0x4cf5ad432745937f;
    const UINT64 seed = 0x5e4c39a7b31672d1;     // (16/32 bits are 1)
    UINT64 h1 = seed;
    UINT64 h2 = seed;
    UINT64 k1 = 0;

    /* we traverse the list of A21-encoded 64-bit values two at a time; if there is an odd number of values in the list,
        the final "tail" value is handled separately */
    const UINT64* p = pk;
    const UINT64* const pLimit = pk + ((cel&(~1)) * incr);

    while( p < pLimit )
    {
        // hash a pair of 64-bit values
        k1 = *p;
        p += incr;
        k1 *= c1;
        k1 = _rotl64( k1, 31 );
        k1 *= c2;
        h1 ^= k1;

        h1 = _rotl64( h1, 27 );
        h1 += h2;
        h1 = h1*5 + 0x52dce729;

        UINT64 k2 = *p;
        p += incr;
        k2 *= c2;
        k2 = _rotl64( k2, 33 );
        k2 *= c1;
        h2 ^= k2;

        h2 = _rotl64( h2, 31 );
        h2 += h1;
        h2 = h2*5 + 0x38495ab5;
    }

    if( cel & 1 )
    {
        // hash the "tail" value
        k1 ^= *pLimit;
        k1 *= c1;
        k1 = _rotl64( k1, 31 );
        k1 *= c2;
        h1 ^= k1;
    }

    // finalization
    h1 ^= cel;
    h2 ^= cel;

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

    // return the low-order 32 bits
    return static_cast<UINT32>(h2);
}

/// [public, static] ComputeH64( UINT64*, INT32, INT32 )
UINT64 Hash::ComputeH64( const UINT64* pk, INT32 cel, INT32 stride )
{
    /* Computes a 64-bit value by hashing the specified A21-encoded sequence.

        The hash implementation is the x64 version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.
    */
    const UINT64 c1 = 0x87c37b91114253d5;
    const UINT64 c2 = 0x4cf5ad432745937f;
    const UINT64 seed = 0x5e4c39a7b31672d1;     // (16/32 bits are 1)
    UINT64 h1 = seed;
    UINT64 h2 = seed;
    UINT64 k1 = 0;

    /* we traverse the list of A21-encoded 64-bit values two at a time; if there is an odd number of values in the list,
        the final "tail" value is handled separately */
    const UINT64* p = pk;
    const UINT64* const pLimit = pk + ((cel&(~1)) * stride);

    while( p < pLimit )
    {
        // hash a pair of 64-bit values
        k1 = *p;
        p += stride;
        k1 *= c1;
        k1 = _rotl64( k1, 31 );
        k1 *= c2;
        h1 ^= k1;

        h1 = _rotl64( h1, 27 );
        h1 += h2;
        h1 = h1*5 + 0x52dce729;

        UINT64 k2 = *p;
        p += stride;
        k2 *= c2;
        k2 = _rotl64( k2, 33 );
        k2 *= c1;
        h2 ^= k2;

        h2 = _rotl64( h2, 31 );
        h2 += h1;
        h2 = h2*5 + 0x38495ab5;
    }

    if( cel & 1 )
    {
        // hash the "tail" value
        k1 ^= *pLimit;
        k1 *= c1;
        k1 = _rotl64( k1, 31 );
        k1 *= c2;
        h1 ^= k1;
    }

    // finalization
    h1 ^= cel;
    h2 ^= cel;

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

    // return the 64-bit hash value
    return h2;
}
#pragma endregion
