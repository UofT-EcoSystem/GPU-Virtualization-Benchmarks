/*
  WinGlobalPtr.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variables

/* The following threshold was arrived at "empirically" by running AriocP to align ERP001960.ERR194161
    with GRCh38.p4, and using umdh.exe to observe the growth of large Windows heap allocations.  */
size_t WinGlobalPtrHelper::cbmaxWinGlobalAlloc = 128 * 1024 * 1024;
#pragma endregion

size_t WinGlobalPtrHelper::EstimateAllocMax( size_t _cb )
{
#ifdef _WIN32
    HANDLE  hHeap = ::GetProcessHeap();
#endif

    size_t l = 0;           // low end of range
    size_t r = _cb;         // high end of range
    size_t m = _cb / 2;     // midpoint of the range

    // binary search for the largest successful memory allocation
    while( r > (l+32) )     // loop until the range is 32 bytes or less
    {
        // try to allocate
#ifdef _WIN32
        void* p = HeapAlloc( hHeap, 0, m ); 
#endif

#ifdef __GNUC__
        void* p = malloc( m );
#endif
        // if the allocation succeeded...
        if( p )
        {
            // free the allocation
#ifdef _WIN32
            HeapFree( hHeap, 0, p );
#endif

#ifdef __GNUC__
            free( p );
#endif
            // iterate on the right-hand part of the range
            l = m;
        }

        else    // the allocation failed
        {
            // iterate on the left-hand part of the range
            r = m;
        }

        // recompute the midpoint
        m = (r + l) / 2;
    }

    // at this point l is the largest number of bytes that were successfully allocated
    return l;
}
