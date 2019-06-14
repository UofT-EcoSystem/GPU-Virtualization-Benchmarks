/*
  AriocCommon.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variable initialization
char AriocCommon::m_symbolDecode[8] = { '·', '·', 'N', 'N', 'A', 'C', 'G', 'T' };
#pragma endregion

#pragma region constructor/destructor
/// constructor
AriocCommon::AriocCommon()
{
}

/// destructor
AriocCommon::~AriocCommon()
{
}
#pragma endregion

#pragma region static methods
/// <summary>
/// Computes the reverse complement of the specified 64-bit A21-encoded value.
/// </summary>
UINT64 AriocCommon::A21ReverseComplement( UINT64 v )
{
    /* The idea here is to avoid looping through the encoded value and copying 3-bit fields on each iteration.
    
       We start with 21 3-bit encoded symbols (bits 0-2, 3-5, 6-8, ..., 60-62):        

       MSB                                                         LSB
        20 19 18 17 16 15 14 13 12 11 10 09 08 07 06 05 04 03 02 01 00
    */

    // save a copy of symbol 10 (already in place)
    UINT64 vx = v & (static_cast<UINT64>(7) << 30);

    /* Swap symbols 11-20 <--> 00-09:

        09 08 07 06 05 04 03 02 01 00 ·· 20 19 18 17 16 15 14 13 12 11

       (Symbol 10 (bits 30-32) is a "don't care" value that will soon get zeroed anyway.)
    */
    v = ((v & MASKA21) >> 33) | (v << 33);

    /* Swap symbols
        05-09 <--> 00-04
        16-20 <--> 11-15
    
        04 03 02 01 00 09 08 07 06 05 ·· 15 14 13 12 11 20 19 18 17 16

       The bit masks are:
        0 111 111 111 111 111 000 000 000 000 000 000 111 111 111 111 111 000 000 000 000 000 = 0x7FFF00003FFF8000
        0 000 000 000 000 000 111 111 111 111 111 000 000 000 000 000 000 111 111 111 111 111 = 0x0000FFFE00007FFF
    */
    v = ((v & 0x7FFF00003FFF8000) >> 15) | ((v & 0x0000FFFE00007FFF) << 15);

    /* Save a copy of symbols 02, 07, 13, and 18 (now in place).

       The bit mask is:
        0 000 000 111 000 000 000 000 111 000 000 000 000 000 111 000 000 000 000 111 000 000 = 0x01C0038000E001C0

       Now we are working with:
        04 03 ·· 01 00 09 08 ·· 06 05 ·· 15 14 ·· 12 11 20 19 ·· 17 16
    */
    vx |= (v & 0x01C0038000E001C0);

    /* Swap symbols
        00-01 <--> 03-04
        05-06 <--> 08-09
        14-15 <--> 11-12
        19-20 <--> 16-17

        01 00 ·· 04 03 06 05 ·· 09 08 ·· 12 11 ·· 15 14 17 16 ·· 20 19

       The bit masks are:
        0 111 111 000 000 000 111 111 000 000 000 000 111 111 000 000 000 111 111 000 000 000 = 0x7E00FC003F007E00
        0 000 000 000 111 111 000 000 000 111 111 000 000 000 000 111 111 000 000 000 111 111 = 0x003F007E001F803F
    */
    v = ((v & 0x7E00FC003F007E00) >> 9) | ((v & 0x003F007E001F803F) << 9);

    /* Swap symbols:
        00 <--> 01
        03 <--> 04
        05 <--> 06
        08 <--> 09
        11 <--> 12
        14 <--> 15
        16 <--> 17
        19 <--> 20
    
        00 01 ·· 03 04 05 06 ·· 08 09 ·· 11 12 ·· 14 15 16 17 ·· 19 20

       The bit masks are:
        0 111 000 000 111 000 111 000 000 111 000 000 111 000 000 111 000 111 000 000 111 000 = 0x7038E070381C7038
        0 000 111 000 000 111 000 111 000 000 111 000 000 111 000 000 111 000 111 000 000 111 = 0x0E071C0E07038E07    
    */
    v = ((v & 0x7038E070381C7038) >> 3) | ((v & 0x0E071C0E07038E07) << 3);

    /* Restore the saved symbols:

        00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20
    */
    v |= vx;

    /* Compute the complement for each symbol:
        A -> T
        C -> G
        G -> C
        T -> A
        Nq -> Nq (unchanged)
        Nr -> Nr (unchanged)
    
       We first isolate bit 2 of each 3-bit symbol.  We then compute a mask wherein each 3-bit symbol
        is either 011 (for ACGT) or 000 (for N) and use this mask to toggle bits 0-1 of ACGT while
        leaving N unchanged.
    */
    const UINT64 v100 = v & MASK100;
    return v ^ (v100 - (v100 >> 2));    
}

/// <summary>
/// Dumps the 21 symbols in a 64-bit encoded value.
/// </summary>
void AriocCommon::DumpB2164( UINT64 b2164, INT32 j )
{
    if( b2164 == 0 )
    {
        CDPrint( cdpCD0, "(null)" );
        return;
    }

    char s[22];
    char* p = s;
    INT64 i64 = b2164;
    do
    {
        *(p++) = m_symbolDecode[i64&7];
        i64 >>= 3;
    }
    while( i64 );

    *p = 0;
    if( j )
        CDPrint( cdpCD0, "%d: 0x%016llx %s", j, b2164, s );
    else
        CDPrint( cdpCD0, "0x%016llx %s", b2164, s );
}

/// <summary>
/// Dumps a set of N A21-encoded symbols.
/// </summary>
void AriocCommon::DumpA21( UINT64* pa21, INT16 N, UINT32 incr )
{
    INT32 cel = blockdiv(N,21);

    // output buffer
    WinGlobalPtr<char> buf( cel*17 + N + 8, true );
    char* p = buf.p;

    // hexadecimal representation
    for( INT32 u=0; u<cel; ++u )
    {
        sprintf_s( p, 18, "%016llx ", pa21[u*incr] );
        p += 17;
    }

    // character representation
    INT16 n = N;
    INT64 i64 = 0;
    while( n-- )
    {
        if( i64 > 7 )
            i64 >>= 3;
        else
        {
            i64 = *pa21;
            pa21 += incr;
        }

        *(p++) = m_symbolDecode[i64&7];
    }

    // N
    sprintf_s( p, buf.cb-(p-buf.p), " (%d)", N );

    CDPrint( cdpCD0, buf.p );
}

/// <summary>
/// Copies the reverse complement of an A21-encoded sequence to a specified buffer.
/// </summary>
void AriocCommon::A21ReverseComplement( UINT64* prc, const UINT64* pa21, const INT16 incr, const INT16 N )
{
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

    // compute the number of bits to shift
    INT32 shr = 3 * (N % 21);
    if( shr == 0 )
        shr = 63;
    INT32 shl = 63 - shr;

    // point to the rightmost A21-encoded value for the sequence
    const INT16 cel = blockdiv( N, 21 );
    const UINT64* pFrom = pa21 + ((cel-1) * incr);

    // point to the target buffer
    UINT64* pTo = prc;

    // iterate from the last to the first 64-bit A21-encoded value
    UINT64 a21 = *pFrom;
    pFrom -= incr;
    while( pFrom >= pa21 )
    {
        UINT64 a21prev = *pFrom;
        pFrom -= incr;

        // combine the adjacent A21-encoded values
        UINT64 v = ((a21 << shl) & MASKA21) | (a21prev >> shr);

        // save the reverse-complement of the A21-encoded values
        *(pTo++) = AriocCommon::A21ReverseComplement( v );

        // iterate
        a21 = a21prev;
    }

    // isolate and reverse-complement the remaining symbols (i.e. the first N%21 symbols in the sequence)
    UINT64 v = a21 << shl;
    *pTo = AriocCommon::A21ReverseComplement( v );
}

/// <summary>
/// Dumps the reverse complement of a set of N A21-encoded symbols.
/// </summary>
void AriocCommon::DumpA21RC( UINT64* pa21, INT16 N, UINT32 incr )
{
    WinGlobalPtr<UINT64> buf( blockdiv(N,21), true );
    AriocCommon::A21ReverseComplement( buf.p, pa21, incr, N );
    AriocCommon::DumpA21( buf.p, N );
}
#pragma endregion
