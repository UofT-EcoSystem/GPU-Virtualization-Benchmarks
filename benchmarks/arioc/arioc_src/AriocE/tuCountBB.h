/*
  tuCountBB.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuCountBB__

/// <summary>
/// Class <c>tuCountBB</c> counts big-bucket lists.
/// </summary>
class tuCountBB : public tuBaseA
{
    private:
        INT32                   m_maxJ;     // maximum J-list size
        WinGlobalPtr<Hvalue8>*  m_pH;       // H table
        INT64                   m_nH;       // number of elements in H table
        WinGlobalPtr<Jvalue8>*  m_pJ;       // J table
        volatile INT64*         m_pHash;    // current hash value (index into H table)
        volatile INT64*         m_pnBB;     // cumulative number of "big buckets"
        WinGlobalPtr<INT64>*    m_pnBBJ;    // cumulative per-subId big-bucket counts

    protected:
        tuCountBB( void );
        void main( void );

    public:
        tuCountBB( INT32 _maxJ, WinGlobalPtr<Hvalue8>* _pH, INT64 _nH, WinGlobalPtr<Jvalue8>* _pJ, volatile INT64* _pHash, volatile INT64* _pnBB, WinGlobalPtr<INT64>* _pnBBJ );
        virtual ~tuCountBB( void );
};