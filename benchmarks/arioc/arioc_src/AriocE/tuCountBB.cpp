/*
  tuCountBB.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Count big-bucket lists
/// </summary>
tuCountBB::tuCountBB( INT32 _maxJ, WinGlobalPtr<Hvalue8>* _pH, INT64 _nH, WinGlobalPtr<Jvalue8>* _pJ, volatile INT64* _pHash, volatile INT64* _pnBB, WinGlobalPtr<INT64>* _pnBBJ ) :
                        m_maxJ(_maxJ),
                        m_pH(_pH),
                        m_nH(_nH),
                        m_pJ(_pJ),
                        m_pHash(_pHash),
                        m_pnBB(_pnBB),
                        m_pnBBJ(_pnBBJ)
{
}

/// [public] destructor
tuCountBB::~tuCountBB()
{
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Counts the big-bucket lists.
/// </summary>
void tuCountBB::main()
{
    CDPrint( cdpCD4, "%s...", __FUNCTION__ );

    // initialize counters for this worker thread
    INT64 nBB = 0;                              // number of big buckets
    WinGlobalPtr<INT64> nBBJ( 256, true );      // per-subId counts

    // get the next hash value (index into the H table)
    INT64 h = InterlockedExchangeAdd64( m_pHash, 1 );
    while( h < m_nH )
    {
        // get a pointer to the H value corresponding to the h'th hash key
        Hvalue8* pH = m_pH->p+h;

        // if the h'th hash key has one or more J values associated with it...
        if( pH->ofsJ )
        {
            // get a pointer to the start of the J list for the current H value
            Jvalue8* pJ = m_pJ->p + pH->ofsJ;

            UINT32 nJ = pH->nJ;
            if( nJ == 0 )
            {
                nJ = pJ->J;     // copy the J list count from the first element in the list
                pJ++;           // point to the first J value in the list
            }

            // if the J-list size exceeds the configured threshold...
            if( nJ > static_cast<UINT32>(m_maxJ) )
            {
                // update the counters
                nBB++;

                for( UINT32 ij=0; ij<nJ; ++ij )
                {
                    nBBJ.p[pJ->subId]++;
                    pJ++;
                }
            }
        }

        // get the index of the next H value to be evaluated
        h = InterlockedExchangeAdd64( m_pHash, 1 );
    }

    // update the global totals
    InterlockedExchangeAdd64( m_pnBB, nBB );
    for( INT16 subId=0; subId<256; ++subId )
    {
        if( nBBJ.p[subId] )
            InterlockedExchangeAdd64( m_pnBBJ->p+subId, nBBJ.p[subId] );
    }

    CDPrint( cdpCD4, "%s completed", __FUNCTION__ );
}
#pragma endregion
