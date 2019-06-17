/*
  tuSortBB.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuSortBB__

/// <summary>
/// Class <c>tuSortBB</c> sorts big-bucket lists.
/// </summary>
class tuSortBB : public tuBaseA
{
    private:
        WinGlobalPtr<HJpair>*   m_pBB;
        WinGlobalPtr<INT64>*    m_pnBBJ;
        WinGlobalPtr<INT64>*    m_pofsBBJ;
        volatile UINT32*        m_pnSorted;

    private:
        static int HJpairComparer( const void*, const void* );

    protected:
        tuSortBB( void );
        void main( void );

    public:
        tuSortBB( WinGlobalPtr<HJpair>* pBB, WinGlobalPtr<INT64>* pnBBJ, WinGlobalPtr<INT64>* pofsBBJ, volatile UINT32* pnSortedJ );
        virtual ~tuSortBB( void );
};