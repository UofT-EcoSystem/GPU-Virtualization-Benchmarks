/*
  tuSortJcpu.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Sort J lists.
/// </summary>
/// <param name="psip">Reference to a common parameter structure</param>
/// <param name="celH">Number of elements in the H table</param>
/// <param name="pnSortedH">Total number of sorted H values</param>
tuSortJcpu::tuSortJcpu( AriocEncoderParams* psip, INT64 celH, volatile INT64* pnSortedH ) : m_psip(psip), m_celH(celH), m_pnSortedH(pnSortedH)
{
}

/// [public] destructor
tuSortJcpu::~tuSortJcpu()
{
}
#pragma endregion

#pragma region static methods
/// [private static] method JvalueComparer
int tuSortJcpu::JvalueComparer( const void* a, const void* b )
{
    /* a and b point to Jvalue8 structs:
    
        struct Jvalue8
        {
            UINT64  J     : 31;     //  0..30: J (0-based offset into reference sequence)
            UINT64  s     :  1;     // 31..31: strand (0: R+; 1: R-)
            UINT64  subId :  7;     // 32..38: subId (e.g., chromosome number)
            UINT64  x     :  1;     // 39..39: flag (used only for filtering J lists; zero in final J table)
            UINT64  tag   : 24;
        };

       Since we want to order by subunit ID, strand, and J, we could compare these bit fields with logic
        like the following:
            
            const Jvalue8* pa = reinterpret_cast<const Jvalue8*>(a);
            const Jvalue8* pb = reinterpret_cast<const Jvalue8*>(b);

            int rval = static_cast<int>(pa->subId) - static_cast<int>(pb->subId);
            if( rval == 0 )
            {
                rval = static_cast<int>(pa->s) - static_cast<int>(pb->s);
                if( rval == 0 )
                    rval = static_cast<int>(pa->J) - static_cast<int>(pb->J);
            }

       But subId, s, and J are mapped so that we can extract them as a single value and compare the
        extracted values:

            const INT64 ja = JVALUE_RSJ(*pa);
            const INT64 jb = JVALUE_RSJ(*pb);
            const INT64 diff = ja - jb;
            return (diff > 0) ? 1 : ((diff < 0) ? -1 : 0);

       Even better, at this point in the pipeline, tag and x are zero, so we can actually just compare
        the two referenced structs as INT64 values:

            const INT64 diff = *pa - *pb;
    */
    const INT64* pa = reinterpret_cast<const INT64*>(a);
    const INT64* pb = reinterpret_cast<const INT64*>(b);
    const INT64 diff = *pa - *pb;
    return (diff > 0) ? 1 : ((diff < 0) ? -1 : 0);
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Sorts the J lists.
/// </summary>
void tuSortJcpu::main()
{
    CDPrint( cdpCD3, "%s...", __FUNCTION__ );

    // get the hash key for the next J list to be sorted
    INT64 h = InterlockedExchangeAdd64( m_pnSortedH, 1 );
    while( h < m_celH )
    {
        Hvalue8* pH = reinterpret_cast<Hvalue8*>(m_psip->H+h);
        if( pH->ofsJ )
        {
            // point to the first element in the J list
            Jvalue8* pJ = m_psip->J + pH->ofsJ;

            // get the J-list count
            INT32 nJ = pH->nJ;
            if( nJ != 1 )
            {
                // if the count is not stored in the H value ...
                if( nJ == 0 )
                {
                    nJ = pJ->J;     // ... get the count from the first element in the J list
                    ++pJ;           // ... point to the first J value
                }

                // sort the h'th J list
                qsort( pJ, nJ, sizeof(Jvalue8), JvalueComparer );
            }
        }

        // get the hash key for the next J list to be sorted
        h = InterlockedExchangeAdd64( m_pnSortedH, 1 );
    }

    CDPrint( cdpCD3, "%s completed", __FUNCTION__ );
}
#pragma endregion
