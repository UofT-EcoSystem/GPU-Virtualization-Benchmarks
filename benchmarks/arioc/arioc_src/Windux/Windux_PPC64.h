/*
  Windux_PPC64.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
    This include file defines types and aliases for compatibility between Windows and Linux.

    This file contains Linux implementations specific to PPC 64-bit architecture.
*/
#pragma once
#define __Windux_PPC64__

// some fundamental types
typedef int         LONG;        // we want LONG to mean 4-byte signed integer (in PowerPC C++, sizeof(long) = 8, not 4)


/*  TODO CHOP IF UNUSED
 *
 * int __popcnt(unsigned int x)
{
   int c;
   for (c = 0; x != 0; x >>= 1)
     if (x & 1)
        c++;
   return c;
}

unsigned int __popcnt64(UINT64 v)
{
    unsigned int c;
    v = v - ((v >> 1) & (UINT64)0x5555555555555555ULL);
    v = (v & (UINT64)0x3333333333333333ULL) + ((v >> 2) & (UINT64)0x3333333333333333ULL);
    v = ((v + (v >> 4)) & (UINT64)0x0f0f0f0f0f0f0f0fULL) * (UINT64)0x0101010101010101ULL;
    c = (unsigned int)(v >> 56);
    return c;
}**
*
* */

// interlocked (atomic) operations
#define InterlockedExchangeAdd( pTarget, val )      __sync_fetch_and_add(pTarget, val)
#define InterlockedExchangeAdd64( pTarget, val )    __sync_fetch_and_add(pTarget, val)
#define InterlockedIncrement( p )                   __sync_add_and_fetch(p, 1)

inline void* InterlockedExchangePointer( void** ppOld, void* pNew )
{
    __sync_synchronize();
    return __sync_lock_test_and_set( ppOld, pNew );
}

#define InterlockedBitTestAndSet(m,i)    (static_cast<UINT8>((__sync_fetch_and_or(m,(1<<i)) & (1<<i)) != 0))
#define InterlockedBitTestAndReset(m,i)  (static_cast<UINT8>((__sync_fetch_and_and(m,(~(1<<i))) & (1<<i)) != 0))

inline UINT8 _BitScanForward( DWORD* Index, DWORD Mask )
{
    INT8 rval = static_cast<UINT8>(__builtin_ffs(Mask));      // 1-based position of low-order 1 bit
    *Index = rval - 1;                                        // 0-based position of low-order 1 bit
    return rval;
}

inline UINT8 _BitScanForward64( DWORD* Index, UINT64 Mask )
{
    INT8 rval = static_cast<UINT8>(__builtin_ffsll(Mask));    // 1-based position of low-order 1 bit
    *Index = rval - 1;                                        // 0-based position of low-order 1 bit
    return rval;
}

inline UINT8 _BitScanReverse( UINT32* Index, UINT32 Mask )
{
    UINT8 rval = static_cast<UINT8>(__builtin_clz(Mask));     // number of high-order 0 bits preceding a 1 bit
    *Index = (rval ? 31-rval : 0xFFFFFFFF);                   // 0-based position of high-order 1 bit
    return rval;
}

inline UINT8 _BitScanReverse64( UINT32* Index, UINT64 Mask )
{
    UINT8 rval = static_cast<UINT8>(__builtin_clzll(Mask));   // number of high-order 0 bits preceding a 1 bit
    *Index = (rval ? 63-rval : 0xFFFFFFFF);                   // 0-based position of high-order 1 bit
    return rval;
}

inline UINT8 _bittest( const LONG* Mask, LONG Index )
{
    return (((*Mask) & (1<<Index)) != 0);
}

inline UINT8 _bittestandset( LONG* Mask, LONG Index )
{
    LONG testBit = 1 << Index;
    UINT8 rval = (((*Mask) & testBit) != 0);
    (*Mask) |= testBit;
    return rval;
}

inline UINT8 _bittestandreset( LONG* Mask, LONG Index )
{
    LONG testBit = 1 << Index;
    UINT8 rval = (((*Mask) & testBit) != 0);
    (*Mask) &= ~testBit;
    return rval;
}
