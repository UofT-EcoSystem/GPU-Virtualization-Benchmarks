/*
  AriocCommon.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocCommon__

#define CONFIG_SAM_VERSION "1.6"

#pragma region enums
/* Symbol encoding looks like this:

        binary  decimal  symbol
        000     0        (null)
        001     1        (null)
        010     2        Nq (Q sequence N)
        011     3        Nr (R sequence N)
        100     4        A
        101     5        C
        110     6        G
        111     7        T
*/
enum Nencoding
{
    NencodingQ = 0x02,
    NencodingR = 0x03
};
#pragma endregion

/// <summary>
/// Implements common functionality for Arioc aligner implementations
/// </summary>
class AriocCommon
{
    private:
        static char m_symbolDecode[8];

    public:
        AriocCommon( void );
        ~AriocCommon( void );
        static UINT64 A21ReverseComplement( UINT64 v );
        static void A21ReverseComplement( UINT64* prc, const UINT64* pa21, const INT16 incr, const INT16 N );
        static void DumpB2164( UINT64 b2164, INT32 j = 0 );
        static void DumpA21( UINT64* pa21, INT16 N, UINT32 incr = 1 );
        static void DumpA21RC( UINT64* pa21, INT16 N, UINT32 incr = 1 );
};
