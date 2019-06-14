/*
  baseCountA.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data
/// [kernel] baseCountA_Kernel
static __global__ void baseCountA_Kernel( const UINT64* const __restrict__ pDbuffer,    // in: D values (one per mapping)
                                          const UINT32                     nD,          // in: number of mappings
                                          const bool                       isNongapped, // in: true: update nongapped mapping counts; false: update gapped mapping counts
                                                Qwarp*  const              pQwBuffer    // in,out: pointer to Qwarps
                                         )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nD )
        return;

    // load the D value for the current CUDA thread
    UINT64 D = pDbuffer[tid];

    // do nothing if the "mapped" flag is not set
    if( 0 == (D & AriocDS::D::flagMapped ) )
        return;

    // unpack the QID
    UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
    const UINT32 iw = QID_IW(qid);
    const INT16 iq = QID_IQ(qid);

    // point to the Qwarp
    Qwarp* pQw = pQwBuffer + iw;

    // point to the start of the mapping-count array in the Qwarp structure
    UINT32* pnA = isNongapped ? pQw->nAn : pQw->nAg;

    // count the mapping
    atomicAdd( pnA+iq, 1 );
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseCountA::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 baseCountA::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseCountA_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseCountA::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );



#if TODO_CHOP_WHEN_DEBUGGED
    // get counts before kernel
    WinGlobalPtr<Qwarp> QwBefore( m_pqb->DB.Qw.n, false );
    m_pqb->DB.Qw.CopyToHost( QwBefore.p, QwBefore.Count );
#endif


    // execute the kernel
    baseCountA_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pD,          // in: mapped candidate D values
                                                         m_nD,          // in: number of candidate D values
                                                         m_isNongapped, // in: true: update nongapped mapping counts; false: update gapped mapping counts
                                                         m_pqb->DB.Qw.p // in: Qwarp buffer
                                                        );

#if TODO_CHOP_WHEN_DEBUGGED
    waitForKernel();

    // get counts after
    WinGlobalPtr<Qwarp> QwAfter( m_pqb->DB.Qw.n, false );
    m_pqb->DB.Qw.CopyToHost( QwAfter.p, QwAfter.Count );

    UINT32 totalAnBefore = 0;
    UINT32 totalAgBefore = 0;
    UINT32 totalAnAfter = 0;
    UINT32 totalAgAfter = 0;
    for( UINT32 iw=0; iw<m_pqb->DB.Qw.n; ++iw )
    {
        Qwarp* pQwBefore = QwBefore.p+iw;
        Qwarp* pQwAfter = QwAfter.p+iw;

        for( INT16 iq=0; iq<pQwBefore->nQ; ++iq )
        {
            totalAnBefore += pQwBefore->nAn[iq];
            totalAgBefore += pQwBefore->nAg[iq];
            totalAnAfter += pQwAfter->nAn[iq];
            totalAgAfter += pQwAfter->nAg[iq];
        }
    }

    CDPrint( cdpCD0, "%s::baseCountA::launchKernel: counted %u new An, %u new Ag", m_ptum->Key, totalAnAfter-totalAnBefore, totalAgAfter-totalAgBefore );
#endif


}
#pragma endregion
