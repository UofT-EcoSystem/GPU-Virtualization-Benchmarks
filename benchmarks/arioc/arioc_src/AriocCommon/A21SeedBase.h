/*
  A21SeedBase.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __A21SeedBase__

// bit masks for packed 3-bit symbols in a 64-bit value
#define MASK001 0x1249249249249249  // masks bit 0 of each 3-bit symbol
#define MASK010 0x2492492492492492  // masks bit 1 of each 3-bit symbol
#define MASK100 0x4924924924924924  // masks bit 2 of each 3-bit symbol
#define MASKA21 0x7FFFFFFFFFFFFFFF  // masks all 21 3-bit symbols

#pragma region structs
#pragma pack(push, 1)
/* 64-bit (8-byte) representation of a 40-bit (5-byte) H value */
struct Hvalue8
{
    UINT64  ofsJ : 35;      //  0..34: offset of J list (bucket) in J table
    UINT64  nJ   :  5;      // 35..39: 0: nJ is first element in J list; 1-31: nJ
    UINT64  nJx  : 24;      // 40..63: modified nJ (AriocE only)
};

#define HVALUE_MAX_NJ   ((static_cast<UINT32>(1) << 5)-1)           // maximum value that can be stored in Hvalue8.nJ
#define HVALUE_MASK_NJ  (static_cast<UINT64>(0x000000F800000000))   // mask for nJ (bits 35-39)
#define HVALUE_OFSJ(h)  (h & 0x7FFFFFFFF)
#define HVALUE_NJ(h)    ((h >> 35) & 0x1F)
#define HVALUE_MAX_OFSJ ((static_cast<INT64>(1) << 35)-1)           // maximum value that can be stored in Hvalue8.ofsJ
#define HVALUE_MAX_NJX  ((static_cast<UINT32>(1) << 24)-1)          // maximum value that can be stored in Hvalue8.nJx

/* 40-bit (5-byte) representation of an H value with J count */
struct Hvalue5
{
    enum bfSize
    {
        bfSize_ofsJlo = 32, //  0..31: offset of J list (bucket) in J table (low-order bits)
        bfSize_ofsJhi =  3, // 32..34: offset of J list (bucket) in J table (high-order bits)
        bfSize_nJ =      5  // 35..39: 0: nJ is first element in J list; 1-31: nJ
    };

    UINT32  ofsJlo;
    UINT8   ofsJhi :  bfSize_ofsJhi;
    UINT8   nJ     :  bfSize_nJ;
};
#define HVALUE5_OFS(pH) ((static_cast<INT64>(pH->ofsJhi) << Hvalue5::bfSize_ofsJlo) | pH->ofsJlo)

/* 40-bit (5-byte) representation of a J value */
struct Jvalue5
{
    enum bfSize
    {
        bfSize_J =     31,  //  0..30: J (0-based offset into reference sequence)
        bfSize_s =      1,  // 31..31: strand (0: R+; 1: R-)
        bfSize_subId =  7,  // 32..38: subId (e.g., chromosome number)
        bfSize_x =      1   // 39..39: end-of-list flag
    };

    enum bfMaxVal : UINT64
    {
        bfMaxVal_J =     (static_cast<UINT64>(1) << bfSize_J) - 1,
        bfMaxVal_s =     (static_cast<UINT64>(1) << bfSize_s) - 1,
        bfMaxVal_subId = (static_cast<UINT64>(1) << bfSize_subId) - 1,
        bfMaxVal_x =     (static_cast<UINT64>(1) << bfSize_x) - 1
    };

    UINT32  J     : bfSize_J;
    UINT32  s     : bfSize_s;
    UINT8   subId : bfSize_subId;
    UINT8   x     : bfSize_x;
};

/* 64-bit (8-byte) representation of a 40-bit (5-byte) J value */
struct Jvalue8
{
    enum bfSize
    {
        bfSize_J =     31,     //  0..30: J (0-based offset into reference sequence)
        bfSize_s =      1,     // 31..31: strand (0: R+; 1: R-)
        bfSize_subId =  7,     // 32..38: subId (e.g., chromosome number)
        bfSize_x =      1,     // 39..39: flag (used only for sorting and filtering J lists; zero in final J table)
        bfSize_tag =   24      // 40..63: used for sorting (see tuSortJgpu)
    };

    enum bfMaxVal : UINT64
    {
        bfMaxVal_J =     (static_cast<UINT64>(1) << bfSize_J) - 1,
        bfMaxVal_s =     (static_cast<UINT64>(1) << bfSize_s) - 1,
        bfMaxVal_subId = (static_cast<UINT64>(1) << bfSize_subId) - 1,
        bfMaxVal_x =     (static_cast<UINT64>(1) << bfSize_x) - 1,
        bfMaxVal_tag =   (static_cast<UINT64>(1) << bfSize_tag) - 1
    };

    UINT64  J     : bfSize_J;
    UINT64  s     : bfSize_s;
    UINT64  subId : bfSize_subId;
    UINT64  x     : bfSize_x;
    UINT64  tag   : bfSize_tag;
};

#if TODO_CHOP
#define JVALUE_MASK0_J      ((static_cast<UINT64>(1) << bfs_J) - 1)
#define JVALUE_MASK0_S      ((static_cast<UINT64>(1) << bfs_s) - 1)
#define JVALUE_MASK0_SUBID  ((static_cast<UINT64>(1) << bfs_subId) - 1)
#define JVALUE_MASK_S       (JVALUE_MASK0_S << Jvalue8::bfs_J)
#define JVALUE_MASK_SUBID   (JVALUE_MASK0_SUBID << (Jvalue8::bfs_J+Jvalue8)



#define JVALUE_SUBID(j)     ((j >> 32) & 0x7F)
#define JVALUE_J(j)         (j & static_cast<UINT64>(0x007FFFFFFF))
#define JVALUE_POS(j)       (static_cast<INT32>(j & 0x7FFFFFFF))    // synonym for JVALUE_J but typed as an INT32
#define JVALUE_SJ(j)        (j & static_cast<UINT64>(0x00FFFFFFFF))
#define JVALUE_RS(j)        (j & static_cast<UINT64>(0x7F80000000))
#define JVALUE_RSJ(j)       (j & static_cast<UINT64>(0x7FFFFFFFFF))
#define JVALUE_XRS(j)       (j & static_cast<UINT64>(0xFF80000000))
#define JVALUE_MASK_X       (static_cast<UINT64>(1) << 39)  // mask for x (bit 39)
#endif

/* Map Jvalue5 to UINT64 and to a 4-byte value followed by a 1-byte value. */
union JvalueU
{
    Jvalue5 j5;
    Jvalue8 j8;
    UINT64  u64;
    struct
    {
        UINT32  lo32;
        UINT8   hi8;
    }       s;
};
#define PACK_JVALUE(J, s, tag)  (static_cast<UINT64>(J) | (static_cast<UINT64>(s) << 31) | (static_cast<UINT64>(tag) << 32))

/* J table header */
struct JtableHeader
{
    enum bfSize
    {
        bfSize_maxnJ =      24,     //  0..23: maximum J-list size
        bfSize_minSubId =    7,     // 24..30: minimum subId
        bfSize_unused1 =     1,     // 31..31: (unused)
        bfSize_maxSubId =    7,     // 32..38: maximum subId
        bfSize_unused2 =     1      // 39..39: (unused)
    };

    enum bfMaxVal : UINT32
    {
        bfMaxVal_maxnJ =    (static_cast<UINT32>(1) << bfSize_maxnJ) - 1,
        bfMaxVal_minSubId = (static_cast<UINT32>(1) << bfSize_minSubId) - 1,
        bfMaxVal_maxSubId = (static_cast<UINT32>(1) << bfSize_maxSubId) - 1
    };

    UINT32  maxnJ   : bfSize_maxnJ;
    UINT32  minSubId: bfSize_minSubId;
    UINT32  unused1 : bfSize_unused1;
    UINT8   maxSubId: bfSize_maxSubId;
    UINT8   unused2 : bfSize_unused2;
};
#pragma pack(pop)

/* seed info */
struct SeedInfo
{
    INT16   k;              // number of symbols in seed (seed width)
    INT16   maxMismatches;  // maximum mismatches (for spaced seeds)
    INT16   nHashBits;      // number of bits in hashed seed
    INT16   baseConversion; // 0: none; 1: CT

    SeedInfo() : k(0), maxMismatches(0), nHashBits(0), baseConversion(0)
    {
    }

    SeedInfo( INT16 _k, INT16 _maxMismatches, INT16 _nHashBits, INT16 _baseConversion ) : k(_k), maxMismatches(_maxMismatches), nHashBits(_nHashBits), baseConversion(_baseConversion)
    {
    }
};
#pragma endregion

class A21SeedBase
{
    private:
        typedef UINT64 (A21SeedBase::*mfnBaseConvert)( const UINT64 a21 );

    protected:
        INT32           m_seedIndex;    // seed index enum value

    public:
        static const INT16  bcNone = 0; // no symbol conversion
        static const INT16  bcCT = 1;   // CT conversion
        
        AA<SeedInfo>    SI;             // seed info strings
        INT32           iError;         // index of "error" info
        INT32           iNone;          // index of "none" info

        UINT64          maskNonN;       // bitmask that isolates the bits that are set to 1 for non-N symbols (bits 0, 3, 6, ...) in a "2164"-encoded 64-bit value
        UINT64          seedMask;       // bitmask that isolates the low-order symbols in an A21-encoded 64-bit value
        UINT64          hashBitsMask;   // bitmask that isolates the low-order bits of the hash value
        INT32           maxMismatches;  // maximum number of mismatches for the seed
        INT16           seedWidth;      // number of adjacent symbols covered by the seed
        INT16           hashKeyWidth;   // H (hash) key size in bits
        INT16           seedInterval;   // interval between adjacent seeds (1 for spaced seeds; nonzero for kmers)
        INT16           minNonN;        // minimum number of non-N symbols in a "2164"-encoded 64-bit value
        INT16           baseConvert;    // base conversion (e.g., C to T)
        mfnBaseConvert  fnBaseConvert;  // base-conversion implementation for A21-encoded 64-bit values
        char            IdString[16];   // string representation of seed index enum (e.g. "hsi21_0_29")

    protected:
        A21SeedBase( void );
        virtual ~A21SeedBase( void );
        void baseInit( const char* _si );

    private:
        UINT64 convertNone( const UINT64 a21 );
        UINT64 convertCT( const UINT64 a21 );

    public:
        SeedInfo StringToSeedInfo( const char* s );
        bool IsNull( void );

        virtual UINT32 ComputeH32( UINT64 k ) = 0;
        virtual UINT32 ComputeH32( UINT64 k1, UINT64 k2 ) = 0;
        virtual UINT32 ComputeH32( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4 ) = 0;
        virtual UINT64 ComputeH64( UINT64 k ) = 0;
        virtual UINT64 ComputeH64( UINT64 k1, UINT64 k2 ) = 0;
        virtual UINT64 ComputeH64( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4 ) = 0;
};
