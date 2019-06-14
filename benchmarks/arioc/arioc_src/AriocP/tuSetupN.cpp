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
        WinGlobalPtr<UINT32> nJxxx( m_pqb->DBj.celJ, true );
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


#if TODO_CHOP_WHEN_DEBUGGED
int U64comparer( const void * pa, const void * pb )
{
    const UINT64 a = *reinterpret_cast<const UINT64*>(pa);
    const UINT64 b = *reinterpret_cast<const UINT64*>(pb);
    if( a == b ) return 0;
    if( a < b ) return -1;
    return 1;
}
#endif



/// [private] method setupN20
void tuSetupN::setupN20()
{
    /* N20: load J values from the hash table (build a list of Df values) */
    baseLoadJn k20( "tuSetupN20", m_pqb, m_pab->a21ss.npos );
    k20.Start();
    k20.Wait();



#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dfxxx( m_pqb->DBj.totalD, false );
    m_pqb->DBj.D.CopyToHost( Dfxxx.p, Dfxxx.Count );

    //qsort( Dfxxx.p, Dfxxx.Count, sizeof( UINT64 ), U64comparer );

    for( UINT32 n=0; n<1000; ++n )
        CDPrint( cdpCD0, "%s: %4u 0x%016llx", __FUNCTION__, n, Dfxxx.p[n] );
    CDPrint( cdpCD0, "%s: totalD=%u", __FUNCTION__, m_pqb->DBj.totalD );
    m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
#endif






    /* N21: sort the list of Df values */
    launchKernel21();
    resetGlobalMemory21();

    /* N22: unduplicate the Df values */
    launchKernel22();
}

/// [private] method setupN30
void tuSetupN::setupN30()
{
    /* N30: traverse the sorted list to identify pairs of Df values whose positions meet the configured paired-end criteria (TLEN, orientation) */
    baseJoinDf k30( "tuSetupN30", m_pqb  );
    k30.Start();
    k30.Wait();

    /* N31: count the number of Df values whose positions meet the configured paired-end criteria */
    launchKernel31();

    /* N32: triage the Df values to the Dc buffer if they meet the configured paired-end criteria, or to the Dx buffer otherwise */
    initGlobalMemory32();
    launchKernel32();
    resetGlobalMemory32();
}

/// [private] method setupN40
void tuSetupN::setupN40()
{
    /* N40: translate Df-formatted values to D-formatted values in the Dc buffer; reset the flags */
    tuXlatToD xlatDc( m_pqb, m_pqb->DBn.Dc.p, m_pqb->DBn.Dc.n, m_pqb->DBn.Dc.p, AriocDS::D::maskFlags, 0 );
    xlatDc.Start();
    xlatDc.Wait();


#if TODO_CHOP_WHEN_DEBUGGED
    // do tuXlatToD and tuXlatToDf work?
    WinGlobalPtr<UINT64> Dcf0( m_pqb->DBn.Dc.n, false );
    m_pqb->DBn.Dc.CopyToHost( Dcf0.p, Dcf0.Count );

    CudaGlobalPtr<UINT64> Dffff( m_pqb->pgi->pCGA );
    Dffff.Alloc( cgaLow, m_pqb->DBn.Dc.n, false );

    tuXlatToDf xdf( m_pqb, Dffff.p, m_pqb->DBn.Dc.n, m_pqb->DBn.Dc.p, 0, 0 );
    xdf.Start();
    xdf.Wait();

    CudaGlobalPtr<UINT64> Dcccc( m_pqb->pgi->pCGA );
    Dcccc.Alloc( cgaLow, m_pqb->DBn.Dc.n, false );

    tuXlatToD xd( m_pqb, Dcccc.p, m_pqb->DBn.Dc.n, Dffff.p, 0, 0 );
    xd.Start();
    xd.Wait();

    // they should be the same
    WinGlobalPtr<UINT64> Dcf1(m_pqb->DBn.Dc.n, false );
    Dcccc.CopyToHost( Dcf1.p, Dcf1.Count );

    for( UINT32 n=0; n<m_pqb->DBn.Dc.n; ++n )
    {
        UINT64 Dc0 = Dcf0.p[n];
        UINT64 Dc1 = Dcf1.p[n];
        if( Dc0 != Dc1 )
            CDPrint( cdpCD0, "%s: Dc0=0x%016llx Dc1=0x%016llx", __FUNCTION__, Dc0, Dc1 );
    }
    CDPrint( cdpCD0, __FUNCTION__ );
#endif
    

    /* N41: translate Df-formatted values to D-formatted values in the Dx buffer */
    initGlobalMemory41();
    launchKernel41();
    resetGlobalMemory41();
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
            setupN30();
            setupN40();
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
