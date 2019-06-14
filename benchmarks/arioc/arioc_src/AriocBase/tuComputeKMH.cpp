/*
  tuComputeKMH.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"
#include <thrust/system_error.h>

#pragma region constructor/destructor
/// [private] constructor
tuComputeKMH::tuComputeKMH()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuComputeKMH::tuComputeKMH( QBatch* pqb ) : m_pqb(pqb),
                                            m_pab(pqb->pab),
                                            m_ptum(AriocBase::GetTaskUnitMetrics("tuComputeKMH"))
{
}

/// destructor
tuComputeKMH::~tuComputeKMH()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeKMH10
void tuComputeKMH::computeKMH10()
{
    /* CUDA global memory here:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   (unallocated)
    */

    /* hash the kmers in unmapped Q sequences */
    tuComputeKMH10 k10( m_pqb );
    k10.Start();
    k10.Wait();
}

/// [private] method computeKMH20
void tuComputeKMH::computeKMH20()
{
    /* CUDA global memory here:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data
                KMH     kmer hash values (paired with qids)
                S64     "sketch bits" (one per unmapped Q sequence)

        high:   (unallocated)
    */

    /* sort kmer hash values */
    launchKernel20();
}

/// [private] method computeKMH30
void tuComputeKMH::computeKMH30()
{
    /* CUDA global memory here:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data
                KMH     kmer hash values (paired with qids)
                S64     "sketch bits" (one per unmapped Q sequence)

        high:   (unallocated)
    */

    /* compute S64 ("sketch bits") for unmapped Q sequences */
    tuComputeKMH30 k30( m_pqb );
    k30.Start();
    k30.Wait();
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Computes kmer hashes and "sketch bits" for unmapped reads.
/// </summary>
void tuComputeKMH::main()
{
    CDPrint( cdpCD3, "[%d] %s ...", m_pqb->pgi->deviceId, __FUNCTION__ );

    try
    {
        computeKMH10();
        computeKMH20();
        computeKMH30();
    }
    catch( thrust::system_error& ex )
    {
        int cudaErrno = ex.code().value();
        throw new ApplicationException( __FILE__, __LINE__,
                                        "CUDA error %u (0x%08x): %s\r\nCUDA Thrust says: %s",
                                        cudaErrno, cudaErrno, ex.code().message().c_str(), ex.what() );
    }

    CDPrint( cdpCD3, "[%d] %s completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
#pragma endregion
