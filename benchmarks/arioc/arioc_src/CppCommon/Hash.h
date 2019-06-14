/*_
  Hash.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __Hash__

/// <summary>
/// Class <c>Hash</c> implements basic hash functions.
/// </summary>
class Hash
{
    public:
        UINT64 Sand;

    public:
        Hash( UINT64 _sand = 0xE8D7C6B5A4938271 );
        virtual ~Hash( void );
        UINT32 ComputeH32( UINT64 u64 );
        UINT64 ComputeH64( UINT64 u64 );
        UINT64 ComputeH64( UINT64 k1, UINT64 k2 );

        static UINT32 ComputeH32( const UINT64* pk, INT32 cel, INT32 incr = 1 );
        static UINT64 ComputeH64( const UINT64* pk, INT32 cel, INT32 incr = 1 );
};
