/*
  tuValidateSubIds.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Validate J lists and count subunit IDs.
/// </summary>
tuValidateSubIds::tuValidateSubIds( INT64 _iC0, INT64 _iClimit,
                                    WinGlobalPtr<Cvalue>* _pC, WinGlobalPtr<Hvalue8>* _pH, WinGlobalPtr<Jvalue8>* _pJ,
                                    size_t _cbJ ) :
                                        m_iC0(_iC0), m_iClimit(_iClimit),
                                        m_pC(_pC), m_pH(_pH), m_pJ(_pJ), m_cbJ(_cbJ),
                                        MinSubId(0x7F), MaxSubId(-1), SubIdsPerH(128,true), TotalJ(128,true)
{
}

/// [public] destructor
tuValidateSubIds::~tuValidateSubIds()
{
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Sorts the big-bucket lists.
/// </summary>
void tuValidateSubIds::main()
{
    CDPrint( cdpCD4, "%s (0x%08llx-0x%08llx)...", __FUNCTION__, m_iC0, m_iClimit );

    // prepare to count subIds for each hash value
    WinGlobalPtr<UINT32> nSubIds( Jvalue5::bfMaxVal_subId+1, true );

    // point to the start of 5-byte data in the H and J buffers
    Hvalue5* const pH5base = reinterpret_cast<Hvalue5*>(m_pH->p);
    Jvalue5* const pJ5base = reinterpret_cast<Jvalue5*>(m_pJ->p);
    
    // point to the H and J tables
    INT64 ofsJlimit = m_cbJ / sizeof(UINT32);       // compute a limit for the number of 32-bit values in the J table 

    // traverse the C table so that we validate J-table offsets in order
    INT64 expectedOfsJ = -1;

    for( INT64 iC=m_iC0; iC<m_iClimit; ++iC )
    {
        // get the hash key
        UINT32 h = m_pC->p[iC].hashKey;

        // get the H value
        Hvalue5* pH = pH5base + h;

        // get the J-list offset
        INT64 ofsJ = HVALUE5_OFS(pH);

        // do nothing if the hash key is unused
        if( ofsJ == 0 )
            continue;

        // verify that the J-table offset is as expected
        if( expectedOfsJ != ofsJ )
        {
            if( (expectedOfsJ == -1) && (ofsJ >= 1) && (ofsJ < ofsJlimit) )
                expectedOfsJ = ofsJ;
            else
                CDPrint( cdpCD0, "%s: inconsistent list count near hash key 0x%08x", __FUNCTION__, h );
        }

        /* Accumulate the number of different subIds and the maximum subId */

        // get the J-list count
        UINT32 nJ = pH->nJ;
        Jvalue5* pJ = pJ5base + ofsJ;
        if( nJ == 0 )
            nJ = *reinterpret_cast<UINT32*>(pJ++);

        // reset the counts for the current hash key
        memset( nSubIds.p, 0, nSubIds.cb );

        // traverse the J list for the current hash key
        Jvalue5* pJlimit = pJ + nJ;
        while( pJ < pJlimit )
        {
            // count the number of J values for each subId
            UINT8 subId = pJ->subId;
            nSubIds.p[subId]++;

            // find the minimum and maximum subId values for the current hash key
            this->MaxSubId = max2(this->MaxSubId, subId);
            this->MinSubId = min2(this->MinSubId, subId);

            // iterate
            pJ++;
        }

        // count the number of different subIds
        UINT32 nSubIdCurrent = 0;
        for( UINT32 n=0; n<static_cast<UINT32>(nSubIds.Count); ++n )
        {
            if( nSubIds.p[n] )
                ++nSubIdCurrent;
        }

        this->SubIdsPerH.p[nSubIdCurrent]++ ;
        this->TotalJ.p[nSubIdCurrent] += nJ;

        // update the J-list offset
        expectedOfsJ = pJlimit - pJ5base;
    }

    CDPrint( cdpCD4, "%s (0x%08llx-0x%08llx) completed", __FUNCTION__, m_iC0, m_iClimit );
}
#pragma endregion
