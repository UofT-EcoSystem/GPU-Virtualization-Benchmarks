/*
  Windux_x64.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
    This include file defines types and aliases for compatibility between Windows and Linux.

    This file contains Linux implementations specific to Intel 64-bit architecture.
*/
#pragma once
#define __Windux_x64__

// some fundamental types
typedef long        LONG;

// interlocked (atomic) operations
#define InterlockedExchangeAdd( pTarget, val )      __sync_fetch_and_add(pTarget, val)
#define InterlockedExchangeAdd64( pTarget, val )    __sync_fetch_and_add(pTarget, val)

inline UINT32 InterlockedIncrement( volatile UINT32* pU32 )
{
    /* We increment the 32-bit value at the specified address and return the previous value:
        - We don't need a clobber list here; the compiler knows which registers it's using and
           what memory is being modified.
        - The variable used for the return value should get optimized away when this function is inlined.
    */
    UINT32 rval;
    asm volatile( "movl $1,%%eax\n"                // eax := 1
                  "lock; xadd %%eax,%0\n"          // eax := previous value; *pU32 := incremented value
                  "inc %%eax\n"                    // eax := incremented value
                    : "=m" (*pU32), "=a" (rval)    // output parameters: =m: memory operand; a: eax
                    : "m" (*pU32)                  // input parameters: m: memory operand
                );
    return rval;
}

inline UINT64 InterlockedIncrement( volatile UINT64* pU64 )
{
    UINT64 rval;
    asm volatile( "movq $1,%%rax\n"                // eax := 1
                  "lock; xadd %%rax,%0\n"          // eax := previous value; *pU64 := incremented value
                  "inc %%rax\n"                    // eax := incremented value
                    : "=m" (*pU64), "=a" (rval)    // output parameters: =m: memory operand; a: eax
                    : "m" (*pU64)                  // input parameters: m: memory operand
                );
    return rval;
}

inline void* InterlockedExchangePointer( void** ppOld, void* pNew )
{
    /* We exchange the 64-bit pointer values and return the previous value:
        - We don't need an explicit lock prefix; it's implicit in the xchg instruction.
        - We don't need a clobber list here; the compiler knows which registers it's using and
           what memory is being modified.
        - The variable used for the return value should get optimized away when this function is inlined.
    */
    void* rval;
    asm volatile( "xchg %0,%1"
                    : "=m" (*ppOld), "=q" (rval)    // output parameters: =m: memory operand; =q: register a,b,c, or d
                    : "m" (*ppOld), "1" (pNew)      // input parameters: m: memory operand; 1: same register as used for the 1th operand in the list
                );
    return rval;
}

inline UINT8 InterlockedBitTestAndSet( volatile LONG* Mask, UINT32 Index )
{
    bool rval;
    asm volatile ( "lock; bts %2,%1\n"
                   "setc %0\n"
                     : "=q" (rval), "+m" (*Mask)    // output parameters: +m: in/out memory operand; =q: register a,b,c, or d
                     : "r" (Index)                  // input parameter: r: any general-purpose register
                 );
    return rval;
}

inline UINT8 InterlockedBitTestAndReset( volatile LONG* Mask, UINT32 Index )
{
    bool rval;
    asm volatile ( "lock; btr %2,%1\n"
                   "setc %0\n"
                     : "=q" (rval), "+m" (*Mask)    // output parameters: +m: in/out memory operand; =q: register a,b,c, or d
                     : "r" (Index)                  // input parameter: r: any general-purpose register
                 );
    return rval;
}

inline UINT8 _BitScanForward( DWORD* Index, DWORD Mask )
{
    UINT8 rval;
    asm volatile ( "bsf %%ecx,%%ecx\n"              // bit scan forward on ecx (32-bit operation)
                   "setnz %%al\n"                   // return nonzero if Mask != 0
                     : "=a" (rval), "=c" (*Index)   // output parameters: =a: al; =c: ecx
                     : "c" (Mask)                   // input parameter: c: rcx
                 );
    return rval;
}

inline UINT8 _BitScanForward64( DWORD* Index, UINT64 Mask )
{
    UINT8 rval;
    asm volatile ( "bsf %%rcx,%%rcx\n"              // bit scan forward on rcx (64-bit operation)
                   "setnz %%al\n"                   // return nonzero if Mask != 0
                     : "=a" (rval), "=c" (*Index)   // output parameters: =a: al; =c: ecx
                     : "c" (Mask)                   // input parameter: c: rcx
                 );
    return rval;
}

inline UINT8 _BitScanReverse( UINT32* Index, UINT32 Mask )
{
    UINT8 rval;
    asm volatile ( "bsr %%ecx,%%ecx\n"              // bit scan reverse on ecx (32-bit operation)
                   "setnz %%al\n"                   // return nonzero if Mask != 0
                     : "=a" (rval), "=c" (*Index)   // output parameters: =a: al; =c: ecx
                     : "c" (Mask)                   // input parameter: c: rcx
                 );
    return rval;
}

inline UINT8 _BitScanReverse64( UINT32* Index, UINT64 Mask )
{
    UINT8 rval;
    asm volatile ( "bsr %%rcx,%%rcx\n"              // bit scan reverse on rcx (64-bit operation)
                   "setnz %%al\n"                   // return nonzero if Mask != 0
                     : "=a" (rval), "=c" (*Index)   // output parameters: =a: al; =c: ecx
                     : "c" (Mask)                   // input parameter: c: rcx
                 );
    return rval;
}

inline UINT8 _bittest( const LONG* Mask, LONG Index )
{
    bool rval;
    asm volatile ( "bt %2,%1\n"                     // bit test
                   "setc %%al\n"                    // the carry flag is a copy of the bit being tested
                     : "=a" (rval)                  // output parameter: =a: al
                     : "m" (*Mask), "r" (Index)     // input parameters: m: memory operand; r: any general-purpose register
                 );
    return rval;
}

inline UINT8 _bittestandset( LONG* Mask, LONG Index )
{
    bool rval;
    asm volatile ( "bts %2,%1\n"                    // bit test and set
                   "setc %%al\n"                    // the carry flag is a copy of the bit prior to being set
                     : "=a" (rval), "+m" (*Mask)    // output parameters: =a: al; +m: in/out memory operand
                     : "r" (Index)                  // input parameter: r: any general-purpose register
                 );
    return rval;
}

inline UINT8 _bittestandreset( LONG* Mask, LONG Index )
{
    bool rval;
    asm volatile ( "btr %2,%1\n"                    // bit test and reset
                   "setc %%al\n"                    // the carry flag is a copy of the bit prior to being reset
                     : "=a" (rval), "+m" (*Mask)    // output parameters: =a: al; +m: in/out memory operand
                     : "r" (Index)                  // input parameter: r: any general-purpose register
                 );
    return rval;
}

