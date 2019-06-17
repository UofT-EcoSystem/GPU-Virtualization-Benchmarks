/*
  A21SpacedSeed.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __A21SpacedSeed__

#if !defined(__A21SeedBase__)
#include "A21SeedBase.h"
#endif

#define SSMMDEFAULT (-1)    // indicates that the user wants the maximum number of mismatches implicit in the specified spaced seed

class A21SpacedSeed : public A21SeedBase
{
    /* The magic bit pattern is 0010111, expanded to cover the two low-order bits of each of the 21 3-bit encoded symbols in a 64-bit value:
    
        - bit pattern: 0010111
        - bit pattern doubled and expanded over 7 3-bit symbols: 000 000 011 000 011 011 011 = 0 0000 0011 0000 1101 1011 = 0x030DB
    */
    private:
        static const INT64 SPACED_SEED_BIT_PATTERN = static_cast<INT64>(0x30DB) | (static_cast<INT64>(0x30DB) << (3*7)) | (static_cast<INT64>(0x30DB) << (3*14));

    public:
        INT64   spacedSeed;         // "spaced seed" bit pattern
        INT16   spacedSeedWeight;   // number of "care" symbols in the hash seed
        INT16   spacedSeed1Bits;    // number of 1-bits ("care" bits) in the spaced seed bitmask (i.e. log2 of H table size)
        INT16   npos;               // total number of adjacent positions at which the hash seed is applied

    private:
        void initSI( void );
        void initSSP32( SeedInfo _si, INT16 _spacedSeedWeight, INT16 _maxMismatches, INT16 _npos, INT16 _hashKeyWidth );

    public:
        A21SpacedSeed( const char* _si, INT32 maxMismatches = 0 );
        ~A21SpacedSeed( void );

        UINT32 ComputeH32( UINT64 k1 );                                     // implements A21SeedBase::ComputeH32( UINT64 )
        UINT32 ComputeH32( UINT64 k1, UINT64 k2 );                          // implements A21SeedBase::ComputeH32( UINT64, UINT64 )
        UINT32 ComputeH32( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4 );    // implements A21SeedBase::ComputeH32( UINT64, UINT64, UINT64, UINT64 )
        UINT64 ComputeH64( UINT64 k1 );                                     // implements A21SeedBase::ComputeH64( UINT64 )
        UINT64 ComputeH64( UINT64 k1, UINT64 k2 );                          // implements A21SeedBase::ComputeH64( UINT64, UINT64 )
        UINT64 ComputeH64( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4 );    // implements A21SeedBase::ComputeH64( UINT64, UINT64, UINT64, UINT64 )
};
