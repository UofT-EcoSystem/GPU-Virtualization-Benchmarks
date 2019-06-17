/*
  tuSetupN.cpp

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
tuSetupN::tuSetupN()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuSetupN::tuSetupN( QBatch* pqb ) : m_pqb(pqb),
                                    m_pab(pqb->pab),
                                    m_nCandidates(0),
                                    m_Dx(pqb->pgi->pCGA),
                                    m_ptum(AriocBase::GetTaskUnitMetrics("tuSetupN"))
{
}

/// destructor
tuSetupN::~tuSetupN()
{
}
#pragma endregion

#pragma region private methods
/// [private] method setupN10
void tuSetupN::setupN10()
{
    /* CUDA global memory here:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data

        high:   (unallocated)
    */

    /* N10: hash the Q sequences */
    tuSetupN10 k10( m_pqb );
    k10.Start();
    k10.Wait();

    /* N12: build a list of QIDs for all Q sequences to be aligned */
    initGlobalMemory12();
    launchKernel12();

    /* N14: accumulate the J-list sizes for the hash keys */
    initGlobalMemory14();
    launchKernel14();

#if TODO_CHOP_WHEN_DEBUGGED
    if( m_pqb->DBj.totalD == 0 )
    {
        WinGlobalPtr<UINT64> nJxxx( m_pqb->DBj.celJ, true );
        m_pqb->DBj.nJ.CopyToHost( nJxxx.p, nJxxx.Count );
        WinGlobalPtr<UINT32> cnJxxx( m_pqb->DBj.cnJ.Count, true );
        m_pqb->DBj.cnJ.CopyToHost( cnJxxx.p, cnJxxx.Count );

        CDPrint( cdpCD0, __FUNCTION__ );
    }
#endif

    /* N15: build a list of QID and pos for each J value */
    baseSetupJn k15( "tuSetupN15", m_pqb, m_pab->a21ss.npos );
    k15.Start();
    k15.Wait();
}

/// [private] method setupN20
void tuSetupN::setupN20()
{
    /* N20: load J values from the hash table (build a list of D values) */
    baseLoadJn k20( "tuSetupN20", m_pqb, m_pab->a21ss.npos );
    k20.Start();
    k20.Wait();

    /* N21: sort the list of D values */
    launchKernel21();
    resetGlobalMemory21();

    /* N22: compute seed coverage for D values */
    launchKernel22();

    /* N24: flag candidate D values */
    tuSetupN24 k24( m_pqb );
    k24.Start();
    k24.Wait();

    /* N26: build a list (DBn.Dc) of D values that are candidates for nongapped alignment */
    initGlobalMemory26();
    launchKernel26();

    /* N28: build a list (DBn.Dl) of D values that are "leftovers" (candidates for gapped alignment but not for nongapped alignment) */
    initGlobalMemory28();
    launchKernel28();
    resetGlobalMemory28();
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Obtains J-list offsets and sizes for the seed-and-extend seeds in a given set of Q sequences
/// </summary>
void tuSetupN::main()
{
    CDPrint( cdpCD3, "[%d] %s ...", m_pqb->pgi->deviceId, __FUNCTION__ );

    try
    {
        // if the user has configured the nongapped aligner ...
        if( !m_pab->a21ss.IsNull() )
        {
            setupN10();
            setupN20();
        }
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
