/*
  baseFilterD.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data
/// [kernel] baseFilterD_Kernel
static __global__ void baseFilterD_Kernel(       UINT64* const  pDbuffer,   // in,out: pointer to D values for mappings
                                           const UINT32         nD,         // in: number of D values for mappings
                                           const INT16          At,         // in: threshold concordant-mapping count
                                           const Qwarp*  const  pQwBuffer   // in: pointer to Qwarps
                                         )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nD )
        return;

    /* Load the D value for the current CUDA thread; the D value is bitmapped like this:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
            UINT64  flags :  3;     // bits 61..63: flags
                                    //  bit 61: 0
                                    //  bit 62: set if the D value is a candidate for alignment
                                    //  bit 63: set if the D value is successfully mapped

       When this kernel executes, the default state of the flags is expected to be:
        bit 62: 1 for mappings that are not in concordantly-mapped pairs (see kernel tuAlignN40)
        bit 63: (ignored)
    */
    UINT64 D = pDbuffer[tid];




    
#if TODO_CHOP_WHEN_DEBUGGED
    if( isDebug )
    {
        UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
        if( qid != 0x181d )
            return;

        asm( "brkpt;" );
    }

#endif













    // do nothing if the current thread's D value indicates that it is not a candidate for alignment
    if( 0 == (D & AriocDS::D::flagCandidate) )
        return;

    // extract the QID
    UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;

    // point to the Qwarp
    UINT32 iw = QID_IW(qid);
    INT16 iq = QID_IQ(qid);
    const Qwarp* pQw = pQwBuffer + iw;

    /* Reset the "candidate" flag if the current mapping belongs to a pair that already has the required minimum number
        of concordant mappings.

       For concordantly-mapped D values, this may affect only a small number of mates, but those mates may be highly
        repetitive so they are worth excluding from subsequent gapped mapping.

       For unpaired candidates, this affects all D values for which the requisite number of concordant mappings has
        already been found (i.e., for which gapped alignment is not needed).
    */
    if( pQw->nAc[iq] >= At )
        pDbuffer[tid] = D & ~AriocDS::D::flagCandidate;
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseFilterD::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 baseFilterD::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseFilterD_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseFilterD::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );

#if TODO_CHOP_WHEN_DEBUGGED
    bool isDebug = (m_pdbb->flagRi == riDx);
    if( isDebug )
        isDebug = (0 == strcmp( "tuAlignGs35", m_ptum->Key ));
    if( isDebug )
    {
        WinGlobalPtr<UINT64> Dxxx( m_nD, false );
        cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
        WinGlobalPtr<Qwarp> Qwxxx(m_pqb->QwBuffer.n, false);
        m_pqb->DB.Qw.CopyToHost( Qwxxx.p, Qwxxx.Count );
        for( UINT32 n=0; n<m_nD; ++n )
        {
            UINT64 D = Dxxx.p[n];
            UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
            UINT32 iw = QID_IW(qid);
            INT16 iq = QID_IQ(qid);

            if( qid == 0x181d )
                CDPrint( cdpCD0, "%s::baseFilterD::launchKernel: %u: qid=0x%08x D=0x%016llx nAc=%d", m_ptum->Key, n, qid, D, Qwxxx.p[iw].nAc[iq] );

        }

        CDPrint( cdpCD0, "baseFilterD::launchKernel: isDebug=true" );


        isDebug = false;

    }

#endif


    // execute the kernel for the specified D list
    baseFilterD_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pD,             // in,out: D values for mappings
                                                          m_nD,             // in: number of mappings
                                                          m_At,             // in: required minimum number of concordant mappings per pair
                                                          m_pqb->DB.Qw.p    // in: Qwarps
                                                        );


#if TODO_CHOP_WHEN_DEBUGGED
    waitForKernel();
    if( isDebug )
    {
        WinGlobalPtr<UINT64> Dxxx( m_nD, false );
        cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
        for( UINT32 n=0; n<m_nD; ++n )
        {
            UINT64 D = Dxxx.p[n];
            UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
            if( qid == 0x181d )
                CDPrint( cdpCD0, "%s::baseFilterD::launchKernel: %u: qid=0x%08x D=0x%016llx", m_ptum->Key, n, qid, D );

        }

        CDPrint( cdpCD0, "baseFilterD::launchKernel: after kernel completes" );

    }

#endif







}
#pragma endregion
