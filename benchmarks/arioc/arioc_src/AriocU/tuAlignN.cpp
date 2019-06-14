/*
  tuAlignN.cpp

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
tuAlignN::tuAlignN()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuAlignN::tuAlignN( QBatch* pqb ) : m_pqb(pqb), m_pab(pqb->pab)
{
}

/// destructor
tuAlignN::~tuAlignN()
{
}
#pragma endregion

#pragma region private methods
/// [private] method alignN10
void tuAlignN::alignN10()
{
    /* CUDA global memory here:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBn.Dc  candidates for nongapped alignment
                DBn.Dl  leftover candidates for gapped alignment

        high:   (unallocated)
    */

    /* N10: load interleaved R sequence data */
    RaiiPtr<tuBaseS> k10 = baseLoadRix::GetInstance( "tuAlignN10", m_pqb, &m_pqb->DBn, riDc );
    k10->Start();
    k10->Wait();
}

/// [private] method alignN20
void tuAlignN::alignN20()
{
    /* N20: do nongapped alignment (Dc list) */
    baseAlignN k20( "tuAlignN20", m_pqb, &m_pqb->DBn, riDc, true );
    k20.Start();
    k20.Wait();

    /* N22: copy mapped D values to a host buffer */
    initGlobalMemory22();
    launchKernel22();
    copyKernelResults22();
    resetGlobalMemory22();
}

/// [private] method alignN30
void tuAlignN::alignN30()
{
    /* N30: count per-Q mappings (update the Qwarp buffer in GPU global memory) */
    baseCountA k30( "tuAlignN30", m_pqb, &m_pqb->DBn, riDc );
    k30.Start();
    k30.Wait();

    /* N32: flag mapped D values as candidates for subsequent (gapped) alignment */
    tuAlignN32 k32( m_pqb, m_pab->aas.ACP.AtN );
    k32.Start();
    k32.Wait();


#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<Qwarp> Qwxxx( m_pqb->QwBuffer.n, false );
    Qwxxx.n = m_pqb->QwBuffer.n;
    m_pqb->DB.Qw.CopyToHost( Qwxxx.p, Qwxxx.n );

    UINT32 totalAn = 0;
    UINT32 nMappedQ = 0;
    UINT32 totalAn1 = 0;    // QIDs with exactly 1 mapping
    UINT32 totalAn2 = 0;    // QIDs with 2 or more mappings

    Qwarp* pQw = Qwxxx.p;
    for( UINT32 iw=0; iw<Qwxxx.n; ++iw )
    {
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            totalAn += pQw->nAn[iq];            // nongapped
            if( pQw->nAn[iq] )
                ++nMappedQ;

            // profile
            switch( pQw->nAn[iq] )
            {
                case 0:
                    break;

                case 1:
                    totalAn1++ ;
                    break;

                default:
                    totalAn2++ ;
                    break;
            }
        }

        ++pQw;
    }

    CDPrint( cdpCD0, "%s: nMappedQ=%u totalAn=%u totalAn1=%u totalAn2=%u", __FUNCTION__, nMappedQ, totalAn, totalAn1, totalAn2 );
    CDPrint( cdpCD0, __FUNCTION__ );

    CDPrint( cdpCD0, "%s: candidates for the Du list (exactly 1 mapping)", __FUNCTION__ );
    WinGlobalPtr<UINT64> Dxxx( m_pqb->DBn.Dc.n, false );
    m_pqb->DBn.Dc.CopyToHost( Dxxx.p, Dxxx.Count );

    for( UINT32 n=0; n<m_pqb->DBn.Dc.n; ++n )
    {
        UINT64 D = Dxxx.p[n];

        const UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        const Qwarp* pQw = Qwxxx.p + iw;
        if( (D & AriocDS::D::flagMapped) && (pQw->nAn[iq] < static_cast<UINT32>(m_pab->aas.ACP.AtN)) )
            CDPrint( cdpCD0, "%s: n=%u D=0x%016llx qid=0x%08x", __FUNCTION__, n, D, qid );
    }
    CDPrint( cdpCD0, __FUNCTION__ );
#endif


    /* N34: build "candidate" (Du) and "leftover" (Dl) lists for subsequent (gapped) alignment */
    initGlobalMemory34();
    launchKernel34();
    resetGlobalMemory34();
}
#pragma endregion


#pragma region virtual method implementations
/// <summary>
/// Uses a CUDA kernel to do nongapped short-read alignments
/// </summary>
void tuAlignN::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    try
    {
        // if the user has configured the nongapped aligner ...
        if( !m_pab->a21ss.IsNull() )
        {
            alignN10();
            alignN20();
            alignN30();
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
