/*
  tuSortBB.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Sort big-bucket lists.
/// </summary>
tuSortBB::tuSortBB( WinGlobalPtr<HJpair>* pBB, WinGlobalPtr<INT64>* pnBBJ, WinGlobalPtr<INT64>* pofsBBJ, volatile UINT32* pnSorted ) :
                        m_pBB(pBB),
                        m_pnBBJ(pnBBJ),
                        m_pofsBBJ(pofsBBJ),
                        m_pnSorted(pnSorted)
{
}

/// [public] destructor
tuSortBB::~tuSortBB()
{
}
#pragma endregion

#pragma region static methods
/// [private static] method HJpairComparer
int tuSortBB::HJpairComparer( const void* a, const void* b )
{
    const HJpair* pa = reinterpret_cast<const HJpair*>(a);
    const HJpair* pb = reinterpret_cast<const HJpair*>(b);

    // order by subId, strand, J
    int rval = static_cast<int>(pa->j.subId) - static_cast<int>(pb->j.subId);
    if( rval == 0 )
    {
        rval = static_cast<int>(pa->j.s) - static_cast<int>(pb->j.s);
        if( rval == 0 )
            rval = static_cast<int>(pa->j.J) - static_cast<int>(pb->j.J);
    }
    else
        throw new ApplicationException( __FILE__, __LINE__, "%s: unexpected subIds %d and %d", __FUNCTION__, pa->j.subId, pb->j.subId );

    return rval;
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Sorts the big-bucket lists.
/// </summary>
void tuSortBB::main()
{
    CDPrint( cdpCD4, "%s...", __FUNCTION__ );

    // count the total number of J values that are sorted in this thread
    UINT32 subId = InterlockedExchangeAdd( m_pnSorted, 1 );
    while( subId < static_cast<UINT32>(m_pnBBJ->Count) )
    {
        // sort the subtable
        if( m_pnBBJ->p[subId] )
        {
#if TODO_CHOP_WHEN_DEBUGGED
            CDPrint( cdpCD0, "subId=%d: sorting %u values at %u", subId, m_pnBBJ->p[subId], m_pofsBBJ->p[subId] );
#endif
            qsort( m_pBB->p+m_pofsBBJ->p[subId], m_pnBBJ->p[subId], sizeof(HJpair), HJpairComparer );
        }

        // get the index of the next subtable to be sorted
        subId = InterlockedExchangeAdd( m_pnSorted, 1 );
    }

    CDPrint( cdpCD4, "%s completed", __FUNCTION__ );
}
#pragma endregion
