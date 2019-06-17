/*
  A21HashedSeed.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// [public] constructor (const char*)
A21HashedSeed::A21HashedSeed( const char* _si )
{
    initSI();
    baseInit( _si );

    // get the specified seed info
    SeedInfo si = (m_seedIndex > 0) ? this->SI.Value( m_seedIndex ) : this->SI.Value(A21HashedSeed::iNone);

    // initialize
    initHSP32( si.k, si.maxMismatches, si.nHashBits );
    m_Hash.Sand = seedWidth;    // prepare for hashing
}

/// [public] destructor
A21HashedSeed::~A21HashedSeed()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initSI()
void A21HashedSeed::initSI()
{
    this->SI["error"] = SeedInfo( -1, -1 , -1, 0 );        // the 0th item in the list is used for error info
    this->SI["none"] = SeedInfo( 0, 0, 0, 0 );
    this->SI["hsi00_0_00"] = SeedInfo( 0, 0, 0, 0 );

    // build a range of reasonable seed-info structs
    for( INT16 k=18; k<=42; ++k )
    {
        for( INT16 nHashBits=29; nHashBits<=32; ++nHashBits )
        {
            char s[16];

            // seed info (no base conversion)
            sprintf_s( s, sizeof s, "hsi%02d_0_%02d", k, nHashBits );
            this->SI[s] = SeedInfo( k, 0, nHashBits, 0 );

            // seed info (with CT conversion)
            strcat_s( s, sizeof s, "_CT" );
            this->SI[s] = SeedInfo( k, 0, nHashBits, 1 );
        }
    }

    // experimental cases outside the above range
    this->SI["hsi30_0_61"] = SeedInfo( 30, 0, 61, 0 );
}

/// [private] method initHSP32
void A21HashedSeed::initHSP32( INT16 _seedWidth, INT16 _maxMismatches, INT16 _hashKeyWidth )
{
    this->seedWidth = _seedWidth;
    this->seedMask = MASKA21;
    if( _seedWidth % 21 )                               // mask for high-order seed bits in the high-order 64-bit value
        this->seedMask >>= (63 - (3*(_seedWidth%21)));
    this->maskNonN = MASK100 & this->seedMask;
    this->hashKeyWidth = _hashKeyWidth;
    this->hashBitsMask = 0xFFFFFFFFFFFFFFFF >> (64 - _hashKeyWidth);
    this->maxMismatches = _maxMismatches;
    this->minNonN = _seedWidth - _maxMismatches;
    this->seedInterval = 0;

    /* Initialize seed-iteration lookup tables:

       The logic here assumes that the seed width is between 17 and 32, so we need exactly 6 seed iterations
       to subdivide the seed intervals for each seed iteration.  Each iteration examines (roughly) as many
       new seeds as all of the previous iterations combined.  But the actual number of seeds examined does
       not double with each iteration because seed locations that have already been examined are excluded
       from consideration.

       Instead of coding tricky logic involving mod(seedWidth), we precompute the seed positions for
       each seed iteration.  Examples for 100nt reads:
       
        20-bit seeds (81 seed positions):
            loop 0     00                                      20                                      40                                      60                                      80  ...   5 seeds
            loop 1                         10                                      30                                      50                                      70                      ...   4 seeds
            loop 2               05                  15                  25                  35                  45                  55                  65                  75            ...   8 seeds
            loop 3         02        07        12        17        22        27        32        37        42        47        52        57        62        67        72        77        ...  16 seeds
            loop 4       01  03    06  08    11  13    16  18    21  23    26  28    31  33    36  38    41  43    46  48    51  53    56  58    61  63    66  68    71  73    76  78      ...  32 seeds
            loop 5             04        09        14        19        24        29        34        39        44        49        54        59        64        69        74        79    ...  16 seeds

        21-bit seeds (80 seed positions):
            loop 0     00                                        21                                        42                                        63                                    ...   4 seeds
            loop 1                         10                                        31                                        52                                        73                ...   4 seeds
            loop 2               05                  15                    26                  36                    47                  57                    68                  78      ...   8 seeds
            loop 3         02        07        12          18        23        28        33          39        44        49        54          60        65        70        75            ...  15 seeds
            loop 4       01  03   06   08    11  13    16    19    22  24    27  29    32  34    37    40    43  45    48  50    53  55    58    61    64  66    69  71    74  76    79    ...  31 seeds
            loop 5             04        09        14    17    20        25        30        35    38    41        46        51        56    59    62        67        72        77        ...  18 seeds

       The lookup tables are zeroed by the C++ static initializer.
    */

    // allocate and zero the lookup tables
    this->ofsSPSI.Realloc(nSPSI+1, true);   // offsets of lists of per-seed-iteration seed positions
    this->ofsSPSI.n = nSPSI + 1;
    this->SPSI.Realloc(celSPSI, true);      // seed positions for seed iterations

    if( _seedWidth )
    {
        // allocate and zero a temporary table of flags that indicate which seed positions have been used
        WinGlobalPtr<bool> iUsed(celSPSI, true);

        // seed iteration 0
        UINT16 ofs = 0;
        for( UINT16 i=0; i<celSPSI; i+=static_cast<UINT32>(_seedWidth) )
        {
            this->SPSI.p[ofs++] = i;
            iUsed.p[i] = true;
        }

        // all remaining seed iterations
        for( UINT32 isi=1; isi<nSPSI; isi++ )
        {
#if TODO_CHOP_WHEN_DEBUGGED
            char buf[3192];
            buf[0] = 0;
            for( UINT32 n=0; n<256; ++n )
            {
                if( iUsed.p[n] )
                {
                    size_t cch = strlen( buf );
                    sprintf( buf+cch, "%u ", n );
                }
            }
            CDPrint( cdpCD0, "start of isi=%d: %s", isi, buf );
#endif

            // save the offset of the start of the seed positions for the isi'th seed iteration
            this->ofsSPSI.p[isi] = ofs;

            // iterate through the list of already-used seed positions
            UINT16 iuFrom = 0;
            UINT16 iuTo = iuFrom + 1;
            while( iuTo < celSPSI )
            {
                // find the end of the interval that starts at iuFrom
                while( (iuTo < celSPSI) && !iUsed.p[iuTo] )
                    iuTo++ ;

                if( iuTo < celSPSI )
                {
                    // at this point we have a valid interval [iuFrom,iuTo)
                    UINT16 diff = iuTo - iuFrom;
                    if( diff > 1 )
                    {
                        UINT32 i = iuFrom + (diff / 2);
                        this->SPSI.p[ofs++] = i;
                        iUsed.p[i] = true;
                    }
                }

                // set up for the next interval
                iuFrom = iuTo;
                iuTo = iuFrom + 1;
            }
        }

        // save the final value in the list of offsets (that is, the total number of seed positions in the list)
        this->SPSI.n = this->ofsSPSI.p[nSPSI] = ofs;
        this->SPSI.Realloc( ofs, false );

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s: SPSI contains %u elements...", __FUNCTION__, this->SPSI.n );
        for( int isi=0; isi<6; ++isi )
        {
            char buf[3192];
            buf[0] = 0;
            UINT16 o = this->ofsSPSI.p[isi];
            for( int x=0; (x<128) && (o < this->ofsSPSI.p[isi+1]); ++x )
            {
                size_t cch = strlen( buf );
                sprintf( buf+cch, "%u ", this->SPSI.p[o++] );
            }
            CDPrint( cdpCD0, "%s: isi=%d nSeedPos=%u: %s", __FUNCTION__, isi, this->ofsSPSI.p[isi+1]-this->ofsSPSI.p[isi], buf );
        }
#endif


        // sanity check
        for( UINT16 i=0; i<(celSPSI-static_cast<UINT16>(_seedWidth))+1; ++i )
        {
            if( !iUsed.p[i] )
                throw new ApplicationException( __FILE__, __LINE__, "SPSI initialization failed at i=%u", i );
        }
    }
}
#pragma endregion

#pragma region public methods
/// [public, virtual method implementation] method ComputeH32 (UINT64)
UINT32 A21HashedSeed::ComputeH32( UINT64 u64 )
{
    // isolate the bits of interest
    u64 &= this->seedMask;

    return m_Hash.ComputeH32( u64 ) & this->hashBitsMask;

#if TODO_CHOP
    /* Computes a 32-bit value by hashing the specified 64-bit value.
        
       The hash implementation is the 32-bit version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.
    */

    // isolate the bits of interest
    u64 &= this->seedMask;

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

    // avalanche
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    // mask the 32-bit result to the required number of bits
    return h & this->hashBitsMask;
#endif
}

/// [public, virtual method implementation] method computeH32 (UINT64, UINT64)
UINT32 A21HashedSeed::ComputeH32( UINT64 k1, UINT64 k2 )
{
    // isolate the bits of interest
    k2 &= this->seedMask;

    return static_cast<UINT32>(m_Hash.ComputeH64( k1, k2 ) & this->hashBitsMask);

#if TODO_CHOP
    /* Computes a 64-bit value by hashing the specified 64-bit pattern.
        
       The hash implementation is the x64 version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.
       Since we start with two 64-bit values, there is no "tail" computation.

       This is the same implementation as ComputeH64(UINT64,UINT64) except that only the 32 low-order bits are returned.
    */
    const UINT64 c1 = 0x87c37b91114253d5;
    const UINT64 c2 = 0x4cf5ad432745937f;
    const UINT64 seed = 0x5e4c39a7b31672d1;     // (16/32 bits are 1)
    UINT64 h1 = seed;
    UINT64 h2 = seed;

    // body
    k1 *= c1;
    k1 = _rotl64( k1, 31);
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
    h1 ^= this->seedWidth;          // yes, this is weird but it's how MurmurHash3 is implemented
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

    // mask the 64-bit result to the required number of bits (32 or fewer)
    return static_cast<UINT32>(h2 & this->hashBitsMask);
#endif
}

/// [public, virtual method implementation] method ComputeH32 (UINT64, UINT64, UINT64, UINT64)
UINT32 A21HashedSeed::ComputeH32( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4 )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public, virtual method implementation] method computeH64 (UINT64)
UINT64 A21HashedSeed::ComputeH64( UINT64 b2164 )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public, virtual method implementation] method ComputeH64 (UINT64, UINT64)
UINT64 A21HashedSeed::ComputeH64( UINT64 k1, UINT64 k2 )
{
    return m_Hash.ComputeH64( k1, k2 ) & this->hashBitsMask;

#if TODO_CHOP
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
    k1 = _rotl64( k1, 31);
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
    h1 ^= this->seedWidth;     // yes, this is weird but it's how MurmurHash3 is implemented
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
#endif
}

/// [public, virtual method implementation] method ComputeH64 (UINT64, UINT64, UINT64, UINT64)
UINT64 A21HashedSeed::ComputeH64( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4 )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public] method ComputeH64_61 (UINT64, UINT64)
UINT64 A21HashedSeed::ComputeH64_61( UINT64 k1, UINT64 k2 )
{
    /* The following "hash" packs 30 3-bit symbols into a 64-bit value.  This implementation preserves the two low-order bits
        of each packed 3-bit symbol.  The high-order bit of each of the first 18 (2*9) packed symbols (in k1) is replaced with one of
        the low-order bits of the remaining nine 3-bit symbols (in k2).
        
       This packing is ambiguous if any of the 3-bit symbols is N, so we take the additional step of using an additional bit in the
        result to indicate whether the kmer contains any Ns.
    */
    
    // isolate the bits of interest
    k2 &= this->seedMask;

    // set bit 60 of a 64-bit value if the kmer contains an N
    UINT64 Nbit = (((~k1) & MASK100) | ((~k2) & MASK100 & this->seedMask)) ? 0x1000000000000000 : 0x0000000000000000;

    // pack the two low-order bits from each 3-bit symbol
    
    k1 &= (~MASK100);                                   // ··11·11·11·11·11·11·11·11·11·11·11·11·11·11·11·11·11·11·11·11·11  k1 (21 symbols)
                                                        // ······································22·22·22·22·22·22·22·22·22  k2 (9 symbols)
    UINT64 h = k2 & (MASK001 & 0x0000000001FFFFFF);     // ·······································2··2··2··2··2··2··2··2··2  h
                                                        // ·····································2··2··2··2··2··2··2··2··2··  h << 2
    k1 |= (h << 2);                                     // ··11·11·11·11·11·11·11·11·11·11·11·11211211211211211211211211211  k1

    h = k2 & (MASK010 & 0x000000003FFFFFFF);            // ······································2··2··2··2··2··2··2··2··2·  h
                                                        // ··········2··2··2··2··2··2··2··2··2·····························  h << 28
    k1 |= (h << 28);                                    // ··11·11·11211211211211211211211211211211211211211211211211211211  k1
                                                        // ·····11·························································  k1 & 0x0600000000000000
    k1 += (k1 & 0x0600000000000000);                    // ··1111··11211211211211211211211211211211211211211211211211211211  k1
    h = k1 & 0x3C00000000000000;                        // ··1111··························································  h
    h >>= 2;                                            // ····1111························································  h
                                                        // ········11211211211211211211211211211211211211211211211211211211  k1 & 0x00FFFFFFFFFFFFFF
    h |= (k1 & 0x00FFFFFFFFFFFFFF);                     // ····111111211211211211211211211211211211211211211211211211211211  h

    // set a bit if the kmer contains an N
    h |= Nbit;                                          // ···N1111························································  h

    /* mask the 64-bit result to the required number of bits; in hopes of preserving the information in the N bit, we shift the high-order bits around
        and XOR them into the low-order bits instead of simply masking them off */
    return (h ^ (h >> this->hashKeyWidth)) & this->hashBitsMask;
}
#pragma endregion
