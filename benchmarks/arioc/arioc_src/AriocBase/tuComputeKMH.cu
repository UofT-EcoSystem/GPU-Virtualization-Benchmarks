/*
  tuComputeKMH.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
   We try to localize the Thrust-dependent code in one compilation unit so as to minimize the overall compile time.
*/
#include "stdafx.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#pragma region computeKMH20
/// [private] method launchKernel20
void tuComputeKMH::launchKernel20()
{
    /* Sort the kmer hash values.
    
       We do a pseudo-segmented sort by prefixing each 32-bit hash value with a 32-bit qid.

       Unused (null) hash values are formatted as all 1 bits so that they sort last (for each qid) and
        can be excluded from the S64 computation in tuComputeKMH30.
    */

    thrust::device_ptr<UINT64> tpKQ( m_pqb->DBkmh.KMH.p );
    thrust::stable_sort( epCGA, tpKQ, tpKQ+m_pqb->DBkmh.KMH.n );
}
#pragma endregion
