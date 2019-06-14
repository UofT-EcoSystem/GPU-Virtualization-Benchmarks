/*
  baseAlignN.cu

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
static __device__ __constant__ __align__(8) AlignmentScoreParameters    ccASP;

/// [device] function countMismatches
static inline __device__ INT32 countMismatches( const UINT64 Rcurrent, const UINT64 Qcurrent, const bool baseConvCT, const UINT64 tailMask = _I64_MAX )
{
    UINT64 x = (Qcurrent ^ Rcurrent) & tailMask;        // each 3-bit symbol is nonzero where the symbols don't match
    UINT64 b = ((x >> 1) | x | (x << 1)) & MASK010;     // bit 1 of each 3-bit symbol is set where the symbols don't match

    /* For bsDNA alignments, C (101) in the reference matches T (111) in the query:
        - in this situation, Q^R = 010, but this is ambiguous:

            R      Q    
            C 101  T 111
            T 111  C 101
            A 100  G 110
            G 110  A 100

        - we disambiguate this from the three other possible ways that 010 can occur in the XOR string
            by zeroing bit 1 of each 3-bit symbol in the Q^R value wherever R contains C (101) and Q
            contains T (111)
    */
    if( baseConvCT )
    {
        UINT64 cr = ((Rcurrent >> 1) & (~Rcurrent) & (Rcurrent << 1));  // bit 1 of each 3-bit symbol is set where the symbol is C (101b)
        UINT64 tq = ((Qcurrent >> 1) & Qcurrent & (Qcurrent << 1));     // bit 1 of each 3-bit symbol is set where the symbol is T (111b)
        b &= ~((cr & tq) & MASK010);
    }

    // return the number of mismatches
    return __popc64( b );
}

/// [device] function reverseComplement
static inline __device__ UINT64 reverseComplement( UINT64 v )
{
    /* (Documented in AriocCommon::A21ReverseComplement) */

    UINT64 vx = v & (static_cast<UINT64>(7) << 30);
    v = ((v & MASKA21) >> 33) | (v << 33);
    v = ((v & 0x7FFF00003FFF8000) >> 15) | ((v & 0x0000FFFE00007FFF) << 15);
    vx |= (v & 0x01C0038000E001C0);
    v = ((v & 0x7E00FC003F007E00) >> 9) | ((v & 0x003F007E001F803F) << 9);
    v = ((v & 0x7038E070381C7038) >> 3) | ((v & 0x0E071C0E07038E07) << 3);
    v |= vx;
    const UINT64 v100 = v & MASK100;
    return v ^ (v100 - (v100 >> 2));    
}

/// [device] function testAlignmentF
static inline __device__ bool testAlignmentF( const UINT64* pRi, const UINT64* pQi, const INT16 N, const INT16 nsx, const INT16 nTail, const INT16 maxMismatches, const bool baseConvCT )
{
    // get the first 64-bit values from the Q and R sequences
    UINT64 Qcurrent = *pQi;
    UINT64 Rcurrent = *pRi;

    /* Score the alignment:
        - loop over the 64-bit (21-symbol) values in the encoded Q sequence
        - process any remaining 64-bit value (containing fewer than 21 symbols) at the end of the Q sequence
        - we assume that there are at least 21 symbols in the Q sequence
    */

    // accumulate the total number of mismatches; if the maximum number of mismatches is exceeded, give up on this alignment
    INT32 nAx = countMismatches( Rcurrent, Qcurrent, baseConvCT );
    if( nAx > maxMismatches )
        return false;       // false: no alignment for the specified i and j

    /* align the remaining 64-bit (21-symbol) values */    
    for( INT16 s=1; s<nsx; ++s )
    {
        // get the next 64-bit values in the Q and R sequences
        Qcurrent = *(pQi += CUDATHREADSPERWARP);
        Rcurrent = *(pRi += CUDATHREADSPERWARP);

        // accumulate the total number of mismatches; if the maximum number of mismatches is exceeded, give up on this alignment
        nAx += countMismatches( Rcurrent, Qcurrent, baseConvCT );
        if( nAx > maxMismatches )
            return false;       // false: no alignment (too many mismatches)
    }

    /* align any remaining symbols in the Q sequence */
    if( nTail )
    {
        // get the next 64-bit values in the Q and R sequences
        Qcurrent = *(pQi += CUDATHREADSPERWARP);
        Rcurrent = *(pRi += CUDATHREADSPERWARP);

        // create a mask to zero the bit positions that do not contain alignable symbols
        UINT64 tailMask = (static_cast<UINT64>(1) << (3*nTail)) - 1;

        // accumulate the total number of mismatches; if the maximum number of mismatches is exceeded, give up on this alignment
        nAx += countMismatches( Rcurrent, Qcurrent, baseConvCT, tailMask );
        if( nAx > maxMismatches )
            return false;       // false: no alignment (too many mismatches)
    }

    /* at this point we have a successful nongapped alignment */
    return true;
}

/// [device] function testAlignmentRC
static inline __device__ bool testAlignmentRC( const UINT64* pRi, const UINT64* pQi, const INT16 N, INT16 nsx, const INT16 nTail, const INT16 maxMismatches, const bool baseConvCT )
{
    // point to the end of the forward Q sequence
    pQi += CUDATHREADSPERWARP * (nsx + (nTail ? 0 : -1));

    const INT16 shr = nTail ? (3 * nTail) : 63;
    const INT16 shl = 63 - shr;

    // get the first 64-bit values from the Q and R sequences
    UINT64 Rcurrent = *pRi;
    UINT64 QiL = *pQi;
    UINT64 QiR = *(pQi -= CUDATHREADSPERWARP);
    UINT64 Qcurrent = (QiL << shl) | (QiR >> shr);
    Qcurrent = reverseComplement( Qcurrent );

    /* Score the alignment:
        - loop over the 64-bit (21-symbol) values in the encoded Q sequence
        - process any remaining 64-bit value (containing fewer than 21 symbols) at the end of the Q sequence
        - we assume that there are at least 21 symbols in the Q sequence
    */

    // accumulate the total number of mismatches; if the maximum number of mismatches is exceeded, give up on this alignment
    INT32 nAx = countMismatches( Rcurrent, Qcurrent, baseConvCT );
    if( nAx > maxMismatches )
        return false;       // false: no alignment for the specified i and j

    /* align the remaining 64-bit (21-symbol) values */
    while( --nsx )
    {
        // get the next 64-bit values in the Q and R sequences
        Rcurrent = *(pRi += CUDATHREADSPERWARP);
        QiL = QiR;
        QiR = *(pQi -= CUDATHREADSPERWARP);
        Qcurrent = (QiL << shl) | (QiR >> shr);
        Qcurrent = reverseComplement( Qcurrent );

        // accumulate the total number of mismatches; if the maximum number of mismatches is exceeded, give up on this alignment
        nAx += countMismatches( Rcurrent, Qcurrent, baseConvCT );
        if( nAx > maxMismatches )
            return false;       // false: no alignment (too many mismatches)
    }

    /* align any remaining symbols in the Q sequence */
    if( nTail )
    {
        // get the next 64-bit values in the Q and R sequences
        Rcurrent = *(pRi += CUDATHREADSPERWARP);
        Qcurrent = (QiR << shl);
        Qcurrent = reverseComplement( Qcurrent );

        // create a mask to zero the bit positions that do not contain alignable symbols
        UINT64 tailMask = (static_cast<UINT64>(1) << (3*nTail)) - 1;

        // accumulate the total number of mismatches; if the maximum number of mismatches is exceeded, give up on this alignment
        nAx += countMismatches( Rcurrent, Qcurrent, baseConvCT, tailMask );
        if( nAx > maxMismatches )
            return false;       // false: no alignment (too many mismatches)
    }

    /* at this point we have a successful nongapped alignment */
    return true;
}

/// [kernel] baseAlignN_Kernel
static __global__ void baseAlignN_Kernel( const UINT64* const __restrict__ pRiBuffer,   // in: pointer to the interleaved R sequence data
                                          const UINT32                     nCandidates, // in: number of candidates
                                                Qwarp*  const              pQwBuffer,   // in,out: pointer to Qwarps
                                          const UINT64* const __restrict__ pQiBuffer,   // in: pointer to the interleaved Q sequence data
                                          const INT32                      Mr,          // in: maximum number of R symbols spanned by a successful mapping
                                          const bool                       baseConvCT,  // in: flag set if converting C to T (for bsDNA alignment)
                                                UINT64* const              pDcBuffer    // in,out: list of candidate D values
                                        )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nCandidates )
        return;

    /* Load the candidate D value for the current CUDA thread:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
            UINT64  flags :  3;     // bits 61..63: flags
                                    //  bit 61: 0
                                    //  bit 62: set if the D value is a candidate for alignment
                                    //  bit 63: set if the D value is successfully mapped
    */
    UINT64 Dc = pDcBuffer[tid];

    // point to the start of the Ri data for this thread
    INT64 celMr = blockdiv(Mr,21);  // number of 64-bit values required to represent Mr
    celMr *= CUDATHREADSPERWARP;    // number of 64-bit values per warp
    UINT32 wiw = tid >> 5;          // "warp index" (within Ri buffer)
    INT16 wiq = tid & 0x1F;         // thread index within warp (within Ri buffer)
    const UINT64* pRi = pRiBuffer + (wiw * celMr) + wiq;

    // compute the offset of the Qwarp
    UINT32 qid = static_cast<UINT32>(Dc >> 39) & AriocDS::QID::maskQID;
    const INT32 iw = QID_IW(qid);
    const INT16 iq = QID_IQ(qid);

    // point to the Qwarp struct that contains the Q sequences for this CUDA warp
    Qwarp* const pQw = pQwBuffer + iw;
    if( iq >= pQw->nQ )
        return;

    // point to the Q sequence for the current thread
    const UINT64* const __restrict__ pQi = pQiBuffer + pQw->ofsQi + iq;

    // compute the number of 64-bit values required to represent all N symbols in the Q sequence
    const INT16 N = pQw->N[iq];
    const INT16 nsx = N / 21;
    const INT16 nTail = N % 21;

    // compute the minimum high-score threshold
    const INT16 Vt = (ccASP.sft == sftG) ? ccASP.sfA*log( static_cast<double>(pQw->N[iq]) ) + ccASP.sfB :
                     (ccASP.sft == sftS) ? ccASP.sfA*sqrt( static_cast<double>(pQw->N[iq]) ) + ccASP.sfB :
                     (ccASP.sft == sftL) ? ccASP.sfA*pQw->N[iq] + ccASP.sfB :
                                           ccASP.sfB;

    // clamp the maximum number of mismatches
    const INT16 Vp = pQw->N[iq] * ccASP.Wm;     // perfect score
    INT16 maxMismatches = min(ccASP.maxMismatches, (Vp-Vt)/(ccASP.Wm+ccASP.Wx));

    // if we have a nongapped mapping, set the "mapped" flag in the D value
    bool isMapped;
    if( (Dc & AriocDS::D::flagRCq) == 0 )
        isMapped = testAlignmentF( pRi, pQi, N, nsx, nTail, maxMismatches, baseConvCT );
    else
        isMapped = testAlignmentRC( pRi, pQi, N, nsx, nTail, maxMismatches, baseConvCT );

    if( isMapped )
        pDcBuffer[tid] = Dc | AriocDS::D::flagMapped;
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseAlignN::initConstantMemory()
{
    CRVALIDATOR;

    // copy parameters into CUDA constant memory
    CRVALIDATE = cudaMemcpyToSymbol( ccASP, &m_pab->aas.ASP, sizeof ccASP, 0, cudaMemcpyHostToDevice );
}

/// [private] method initSharedMemory
UINT32 baseAlignN::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseAlignN_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseAlignN::launchKernel( UINT32 cbSharedPerBlock )
{
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );


#if TRACE_SQID
    // what's the mapping status for the sqId?
    UINT32 qid = _UI32_MAX;
    WinGlobalPtr<Qwarp> Qwxxx( m_pqb->DB.Qw.n, false );
    m_pqb->DB.Qw.CopyToHost( Qwxxx.p, Qwxxx.Count );
    for( UINT32 iw=0; iw<m_pqb->DB.Qw.n; ++iw )
    {
        Qwarp* pQw = Qwxxx.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            UINT64 sqId = pQw->sqId[iq];
            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                qid = PACK_QID(iw,iq);
                CDPrint( cdpCD0, "%s: 0x%016llx iw=%u iq=%u qid=0x%08x", __FUNCTION__, sqId, iw, iq, qid );
                if( (iq<pQw->nQ) && (pQw->sqId[iq+1] == pQw->sqId[iq]|1) )
                    CDPrint( cdpCD0, "%s: 0x%016llx iw=%u iq=%u qid=0x%08x", __FUNCTION__, pQw->sqId[iq+1], iw, iq+1, qid+1 );
                break;
            }
        }
    }

    // is the specified sqId in the D list?
    WinGlobalPtr<UINT64> Dxxx( m_nD, false );
    cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

    WinGlobalPtr<UINT64> Qixxx( m_pqb->DB.Qi.Count, false );
    m_pqb->DB.Qi.CopyToHost( Qixxx.p, Qixxx.Count );

    if( m_pab->pifgQ->HasPairs )
    {
        for( UINT32 n=0; n<m_nD; ++n )
        {
            UINT64 D = Dxxx.p[n];

//            CDPrint( cdpCD0, "%s: n=%u D=0x%016llx qid=0x%08x", __FUNCTION__, n, D, AriocDS::D::GetQid(D) );

            if( (qid | 1) == (AriocDS::D::GetQid(D) | 1) )
            {
                UINT32 iw = QID_IW(AriocDS::D::GetQid(D));
                INT16 iq = QID_IQ(AriocDS::D::GetQid(D));

                // point to the Qwarp struct that contains the Q sequences for this CUDA warp
                Qwarp* pQw = m_pqb->QwBuffer.p + iw;

                // point to the Q sequence for the current thread
                UINT64* pQi = Qixxx.p + pQw->ofsQi + iq;
                AriocCommon::DumpB2164( *pQi, 0 );

                CDPrint( cdpCD0, "%s: n=%u D=0x%016llx pos=%d rcQ=%d", __FUNCTION__, n, D, static_cast<INT32>(D & AriocDS::D::maskPos), ((D & AriocDS::D::flagRCq) != 0) );
            }
        }
    }

    else
    {
        for( UINT32 n=0; n<m_nD; ++n )
        {
            UINT64 D = Dxxx.p[n];
            if( qid == AriocDS::D::GetQid(D) )
            {
                UINT32 iw = QID_IW(qid);
                INT16 iq = QID_IQ(qid);

                // point to the Qwarp struct that contains the Q sequences for this CUDA warp
                Qwarp* pQw = m_pqb->QwBuffer.p + iw;

                // point to the Q sequence for the current thread
                UINT64* pQi = Qixxx.p + pQw->ofsQi + iq;
                AriocCommon::DumpB2164( *pQi, 0 );

                CDPrint( cdpCD0, "%s: n=%u D=0x%016llx pos=%d rc=%d", __FUNCTION__, n, D, static_cast<INT32>(D & AriocDS::D::maskPos), ((D & AriocDS::D::flagRCq) != 0) );
            }
        }
    }
    

    //CDPrint( cdpCD0, "%s: looking for sqId 0x%016llx in the D list...", __FUNCTION__, TRACE_SQID );
    //WinGlobalPtr<UINT32> Quxxx( m_pqb->DBgs.Qu.n, false );
    //m_pqb->DBgs.Qu.CopyToHost( Quxxx.p, Quxxx.Count );
    //UINT64 minSqId = _UI64_MAX;
    //UINT64 maxSqId = 0;
    //for( UINT32 n=0; n<m_pqb->DBgs.Qu.n; ++n )
    //{
    //    UINT32 qid = Quxxx.p[n];
    //    UINT32 iw = QID_IW(qid);
    //    UINT32 iq = QID_IQ(qid);
    //    Qwarp* pQw = m_pqb->QwBuffer.p + iw;
    //    UINT64 sqId = pQw->sqId[iq];
    //    if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
    //    {
    //        CDPrint( cdpCD0, "%s: %u: 0x%016llx 0x%08x", __FUNCTION__, n, sqId, qid );
    //    }

    //    minSqId = min(minSqId, sqId);
    //    maxSqId = max(maxSqId, sqId);
    //}
    //CDPrint( cdpCD0, "%s: minSqId=0x%016llx maxSqId=0x%016llx", __FUNCTION__, minSqId, maxSqId );
#endif
    
    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    // execute the kernel
    baseAlignN_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DB.Ri.p,    // in: interleaved R sequence data
                                                         m_nD,              // in: number of candidates
                                                         m_pqb->DB.Qw.p,    // in: Qwarp buffer
                                                         m_pqb->DB.Qi.p,    // in: interleaved Q sequence data
                                                         m_pqb->DBn.AKP.Mr, // in: maximum R sequence span
                                                         m_baseConvertCT,   // in: flag set if converting C to T (for bsDNA alignment)
                                                         m_pD               // in,out: candidate D values
                                                       );
}
#pragma endregion
