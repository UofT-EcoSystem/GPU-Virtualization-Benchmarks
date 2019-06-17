/*
  baseLoadRi.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data
/* CUDA constant memory
*/
static __device__ __constant__ __align__(8) INT64   ccOfsRplus[AriocDS::SqId::MaxSubId+1];
static __device__ __constant__ __align__(8) INT64   ccOfsRminus[AriocDS::SqId::MaxSubId+1];

/// [device] method loadRi
static __device__ void loadRi( const UINT64*  const __restrict__ pR,    // in: pointer to the R sequence data
                               const UINT64                      Dc,    // in: candidate D value
                               const INT32                       Mr,    // in: maximum span of mapped R sequence symbols
                               const INT32                       wcgsc, // in: worst-case gap space count
                                     UINT64*  const              pRi    // out: pointer to interleaved R sequence data
                             )
{
    /* The candidate D value is bitmapped like this:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
            UINT64  flags :  3;     // bits 61..63: flags
     */

    // extract the the 0-based position on the reference strand
    INT32 d = static_cast<INT32>(Dc & AriocDS::D::maskPos); // isolate bits 0-30
    d = (d << 1) >> 1;                                      // sign-extend

    /* Adjust for the worst-case gap space count:
        - this value ensures that the scoring-matrix diagonal crosses the d'th diagonal at the end of the mapping; this
           also assumes that Mr has already been computed so as to take this into account
        - for gapped alignment, the Mr value should account for the worst-case gap space count, e.g.
                - N = 100
                - wcgsc = 20
                - Mr = N + 2*wcgsc + 1 = 141
        - for nongapped alignment, wcgsc should be specified as 0, e.g.:
                - N = 100
                - wcgsc = 0
                - Mr = 100
    */
    d -= wcgsc;

    // extract the subunit ID
    INT32 subId = static_cast<INT32>(Dc >> 32) & 0x7F;

    // point to the R sequence to be copied
    const UINT64* const __restrict__ pR0 = pR + ((Dc & AriocDS::D::maskStrand) ? ccOfsRminus[subId] : ccOfsRplus[subId]);

    /* Compute the offset of the first 64-bit value to copy; the computation differs depending on whether d (the 0-based
        offset into the R sequence) is negative:
        - if d >= 0, the computation is straightforward
        - if d < 0, then we need to adjust the value to ensure that the offset is computed correctly
    */
    INT32 ofsFrom = ((d >= 0) ? d : (d-20)) / 21;

    // initialize the pointers
    UINT64* pTo = pRi;
    const UINT64* pFrom = pR0 + ofsFrom;

    /* Compute the number of bits to shift each 64-bit value so that the first R symbol is in the low-order position:
        - if d >= 0, the computation is straightforward
        - if d < 0, the number of symbols to shift right is ((d+1) % 21) + 20
    */
    INT16 posToShift = (d >= 0) ? (d%21) : ((d+1)%21)+20;
    const INT16 shr = 3 * posToShift;
    const INT16 shl = 63 - shr;

    // copy the 64-bit R values from their linear layout into an interleaved layout
    UINT64 v = *(pFrom++);
    UINT64 vn = 0;

    // iterate until at least Mr symbols are copied
    INT32 jRemaining = Mr;
    while( jRemaining >= 21 )
    {
        vn = *(pFrom++);

        // shift the bits into position
        if( shr )
            v = ((v >> shr) | (vn << shl)) & MASKA21;

        // save one 64-bit value
        *pTo = v;
        pTo += CUDATHREADSPERWARP;

        // set up the next iteration
        v = vn;
        jRemaining -= 21;
    }

    if( jRemaining )
    {
        // shift the remaining bits into position
        v >>= shr;

        // if necessary, read one more 64-bit value to capture the remaining bits
        if( jRemaining > (21-posToShift) )
        {
            vn = *pFrom;
            v |= (vn << shl);
        }

        // mask the tail with zero
        v &= (MASKA21 >> (63 - (jRemaining*3)));

        // save the final 64-bit value
        *pTo = v;
    }
}

/// baseLoadRi_Kernel
static __global__  void baseLoadRi_Kernel( const UINT64*  const __restrict__ pR,             // in: pointer to the R sequence data
                                           const UINT64*  const __restrict__ pDcBuffer,      // in: list of candidate D values
                                           const UINT32                      nCandidates,    // in: number of candidates
                                           const INT32                       Mr,             // in: maximum number of R symbols spanned by a successful mapping
                                           const INT32                       wcgsc,          // in: worst-case gap space count
                                                 UINT64*  const              pRiBuffer       // out: pointer to interleaved R sequence data
                                         )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nCandidates )
        return;

    // load the candidate D value for this thread
    UINT64 Dc = pDcBuffer[tid];

    // point to the start of the Ri data for this thread
    INT64 celMr = blockdiv(Mr, 21); // number of 64-bit values required to represent Mr
    celMr *= CUDATHREADSPERWARP;    // number of 64-bit values per warp
    UINT32 wiw = tid >> 5;          // "warp index" (within Ri buffer)
    INT16 wiq = tid & 0x1F;         // thread index within warp (within Ri buffer)
    UINT64* pRi = pRiBuffer + (wiw * celMr) + wiq;

    // load interleaved R sequence data
    loadRi( pR, Dc, Mr, wcgsc, pRi );
}
#pragma endregion

#pragma region protected methods
/// [protected] method initConstantMemory
void baseLoadRi::initConstantMemory()
{
    CRVALIDATOR;

    // load the offset of each R sequence into constant memory on the current GPU
    CRVALIDATE = cudaMemcpyToSymbol( ccOfsRplus, m_pab->ofsRplus.p, m_pab->ofsRplus.cb );
    CRVALIDATE = cudaMemcpyToSymbol( ccOfsRminus, m_pab->ofsRminus.p, m_pab->ofsRminus.cb );
}

/// [protected] method initSharedMemory
UINT32 baseLoadRi::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseLoadRi_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [protected] method launchKernel
void baseLoadRi::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: m_pD=0x%016llx m_nD=%u", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_pD, m_nD );


    WinGlobalPtr<UINT64> Dxxx( m_nD, false );
    cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
    for( UINT32 n=0; n<100; ++n )
        CDPrint( cdpCD0, "%s: %3u 0x%016llx", __FUNCTION__, n, Dxxx.p[n] );


    CDPrint( cdpCD0, "%s: calling baseLoadRi_Kernel with DB.Ri.p=0x%016llx", __FUNCTION__, m_pqb->DB.Ri.p );
#endif


    // performance metrics
    m_usXferR = m_hrt.GetElapsed( true );       // (pre-launch time elapsed)


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: m_pab->R.p=0x%016llx", __FUNCTION__, m_pab->R.p );
    for( UINT32 n=0; n<100; n+=8 )
    {
        CDPrint( cdpCD0, "%s: 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx", __FUNCTION__,
            m_pab->R.p[n+0], m_pab->R.p[n+1], m_pab->R.p[n+2], m_pab->R.p[n+3], m_pab->R.p[n+4], m_pab->R.p[n+5], m_pab->R.p[n+6], m_pab->R.p[n+7] );
    }
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dcxxx( m_nD, false );
    cudaMemcpy( Dcxxx.p, m_pD, Dcxxx.cb, cudaMemcpyDeviceToHost );
    if( (Dcxxx.p[0] & 0x7FFFFFFF) == 0 )
    {
        CDPrint( cdpCD0, "%s: m_pD=0x%016llx m_nD=%u", __FUNCTION__, m_pD, m_nD );
    
    }
#endif


    // launch the CUDA kernel
    baseLoadRi_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->pgi->pR,    // in: pointer to the R sequence data
                                                         m_pD,              // in: list of candidate D values
                                                         m_nD,              // in: number of candidates
                                                         m_pdbb->AKP.Mr,    // in: maximum number of R symbols spanned by a successful mapping
                                                         m_pdbb->AKP.wcgsc, // in: worst-case gap space count
                                                         m_pqb->DB.Ri.p );  // out: pointer to interleaved R sequence data
}
#pragma endregion
