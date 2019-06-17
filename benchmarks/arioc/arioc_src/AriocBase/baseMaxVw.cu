/*
  baseMaxVw.cu

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
static __device__ __constant__ AlignmentScoreParameters    ccASP;

/* CUDA shared memory
*/
static __device__ __shared__ INT32 csW[8*8];


#if TODO_CHOP_WHEN_DEBUGGED
static __device__ __shared__ UINT32 csJ;
static __device__ __shared__ INT16 csI;
static __device__ __shared__ INT16 csii;
#endif



/// [device] method resetRj
static inline __device__ void resetRj( UINT64& Rj, const UINT64* const __restrict__ pRj, UINT32& ofsRj )
{
    Rj = *pRj;
    ofsRj = CUDATHREADSPERWARP;
}

/// [device] method getNextRj
static inline __device__ void getNextRj( UINT64& Rj, const UINT64* const __restrict__ pRj, UINT32& ofsRj )
{
    Rj >>= 3;
    if( Rj == 0 )
    {
        Rj = pRj[ofsRj];
        ofsRj += CUDATHREADSPERWARP;
    }
}

/// [device] method getNextQi
static inline __device__ void getNextQi( UINT64& Qi, UINT64& Qn, const UINT64* const __restrict__ pQi, INT16& ofsQi, const INT16 endQi, const INT16 cchOut, const INT16 cchIn )
{
    /* This looks complicated because it is intended to do all of the following things:
        - load Q symbols only once from global memory and then keep them in registers until they are consumed by the dynamic-programming algorithm
        - load Q symbols in groups (e.g. groups of 8) to conform to the "height" of the horizontal swaths in the dynamic-programming implementation
        - read ahead so that the DP implementation stalls for the minimum amount of time on global memory reads
        - do the most common action (i.e. shift a group of symbols into position) in the shortest time
    */

    // shift the specified number of symbols out of Qi
    INT16 u = 3 * cchOut;
    Qi >>= u;

    // we're done if Qi still contains at least the requested number of symbols
    u = 3 * cchIn;
    INT32 pos = __bfindu64( Qi );   // pos = 0-based position of high-order bit, or -1 if Qi == 0
    if( pos >= (u-3) )
        return;

    /* At this point Qi contains fewer than the requested number of symbols */

    // copy additional symbols from Qn into Qi
    INT16 v = 3 * ((pos+3) / 3);  // 3 * (number of symbols remaining in Qi)
    Qi = (Qi | (Qn << v)) & MASKA21;

    // shift the additional symbols out of Qn
    Qn >>= (63 - v);

    // return if Qi contains at least the requested number of symbols or there are no more symbols to load
    pos = __bfindu64( Qi );
    if( (pos >= (u-3)) || (ofsQi == endQi) )
        return;

    /* At this point:
        - Qi still contains fewer than the requested number of symbols
        - Qn == 0
        - there remain symbols to load
    */

    // load more symbols
    Qn = pQi[ofsQi];
    ofsQi += CUDATHREADSPERWARP;

    // copy additional symbols from Qn into Qi
    v = 3 * ((pos+3) / 3);  // 3 * (number of symbols remaining in Qi)
    Qi = (Qi | (Qn << v)) & MASKA21;

    // shift the additional symbols out of Qn
    Qn >>= (63 - v);
}

/// [device] method computeSMcell
static inline __device__ bool computeSMcell( INT16&             Vmax,   // in, out
                                             INT16&             Vi,     // in, out
                                             INT16&             Ei,     // in, out
                                             INT16&             F,      // in, out
                                             const INT16        Rj,     // in
                                             const INT16        Qi,     // in
                                             INT16&             Vd,     // in, out
                                             const INT16        Vv      // in
                                           )
{
    // get the match/mismatch score
    INT16 Wmx = static_cast<INT16>(csW[((Qi&7)<<3)|Rj]);    // get the W (match/mismatch) score for Qi and Rj from the lookup table

    // compute the Smith-Waterman-Gotoh recurrences
    INT16 G = max( 0, Vd+Wmx );                         // Vd in = Vi out from the previous column
    Vd = Vi;                                            // Vd out = Vi in
    Ei = max( Ei, Vi-ccASP.Wg ) - ccASP.Ws;             // Ei out = new E
    F = max( F, Vv-ccASP.Wg ) - ccASP.Ws;               // F out = new F
    Vi = max( G, max(Ei, F) );                          // Vi out = new V score

    // TODO: CHOP WHEN DEBUGGED:
    //if( csI < 24 )
    //if( csI >= 24 && csI < 48 )
    //if( csI >= 48 && csI < 72 )
    //if( csI >= 72 && csI < 88 )
    //if( csI >= 88 )
    //        printf( "%d %u %c %c %d\n", csI+csii, csJ, "??NNACGT"[Qi&7], "??NNACGT"[Rj], Vi, Vi, Vmax );



    /* Track the maximum V score:
        - We set a flag to indicate whether the current cell's V score is no smaller than the previous maximum, so if there are multiple cells
            with the same maximum V score, we only report the last one.
        - Since we only report one Vmax for the entire scoring matrix, we will miss valid secondary mappings (i.e. discrete mappings with
            a lower Vmax) within the scoring matrix.  But the cost of capturing those mappings (e.g. multiple overlapping scoring matrices)
            would outweigh the value of finding them.
    */
    bool rval = (Vi >= Vmax);
    Vmax = max(Vmax, Vi);
    return rval;
}

/// [device] method computeVband
static inline __device__ UINT32 computeVband( INT16& Vmax, const INT16 nRows,
                                              INT32& tbo, const INT16 i, const INT32 j,
                                              UINT32 Qi8, INT16 Rj, INT16 Vd, UINT32 FVv,
                                              INT16& V0, INT16& V1, INT16& V2, INT16& V3, INT16& V4, INT16& V5, INT16& V6, INT16& V7,
                                              INT16& E0, INT16& E1, INT16& E2, INT16& E3, INT16& E4, INT16& E5, INT16& E6, INT16& E7
                                            )
{
    /* Computes a vertical band of between 1 and 8 cells, ending at the specified 0-based offset from the topmost row in the band.
                                                
                +---+---+---+---+---+---+---+---+
             0  |   |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+
             1  |   |   |   |   |   |   |   |   |             
                +---+---+---+---+---+---+---+---+
             2  |   |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+
             3  |   |   |   |   |   |   |   |   |
          i     +---+---+---+---+---+---+---+---+
             4  |   |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+
             5  |   |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+
             6  |   |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+
             7  |   |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+

       If the caller specifies a constant value for nRows (the number of rows to be computed), NVCC suppresses the corresponding
        conditional clauses (i.e., if( nRows == ... )).
    */
    Rj &= 7;
    INT16 F = static_cast<INT16>(FVv >> 16);
    INT16 Vv = static_cast<INT16>(FVv);

    /* row 0 */
///* TODO: CHOP WHEN DEBUGGED */ csii = 0;
    if( computeSMcell( Vmax, V0, E0, F, Rj, Qi8, Vd, Vv ) )
        tbo = j - i;
    if( nRows == 1 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V0);        //  bits 0-15: V; bits 16-31: F

    /* row 1 */
///* TODO: CHOP WHEN DEBUGGED */ csii = 1;
    if( computeSMcell( Vmax, V1, E1, F, Rj, Qi8>>3, Vd, V0 ) )
        tbo = j - (i+1);
    if( nRows == 2 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V1);

    /* row 2 */
///* TODO: CHOP WHEN DEBUGGED */ csii = 2;
    if( computeSMcell( Vmax, V2, E2, F, Rj, Qi8>>6, Vd, V1 ) )
        tbo = j - (i+2);
    if( nRows == 3 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V2);

    /* row 3 */
///* TODO: CHOP WHEN DEBUGGED */ csii = 3;
    if( computeSMcell( Vmax, V3, E3, F, Rj, Qi8>>9, Vd, V2 ) )
        tbo = j - (i+3);
    if( nRows == 4 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V3);

    /* row 4 */
///* TODO: CHOP WHEN DEBUGGED */ csii = 4;
    if( computeSMcell( Vmax, V4, E4, F, Rj, Qi8>>12, Vd, V3 ) )
        tbo = j - (i+4);
    if( nRows == 5 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V4);

    /* row 5 */
///* TODO: CHOP WHEN DEBUGGED */ csii = 5;
    if( computeSMcell( Vmax, V5, E5, F, Rj, Qi8>>15, Vd, V4 ) )
        tbo = j - (i+5);
    if( nRows == 6 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V5);

    /* row 6 */
///* TODO: CHOP WHEN DEBUGGED */ csii = 6;
    if( computeSMcell( Vmax, V6, E6, F, Rj, Qi8>>18, Vd, V5 ) )
        tbo = j - (i+6);
    if( nRows == 7 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V6);

    /* row 7 */
///* TODO: CHOP WHEN DEBUGGED */ csii = 7;
    if( computeSMcell( Vmax, V7, E7, F, Rj, Qi8>>21, Vd, V6 ) )
        tbo = j - (i+7);

    return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V7);
}

/// [device] method computeV
static inline __device__ INT16 computeV(       INT32&                       tbo,
                                         const INT16                        Vt,
                                         const UINT64* const __restrict__   pQi,
                                         const UINT64* const __restrict__   pRj,
                                         const INT16                        N,
                                               UINT32* const                pFV,
                                               UINT32                       ofsFV,
                                         const INT16                        Mr
                                       )
{
    // get the first 64-bit value from the R sequence
    UINT64 Rj;
    UINT32 ofsRj;
    resetRj( Rj, pRj, ofsRj );

    // get the first two 64-bit values from the Q sequence; this assumes that N > 21 (without error checking)
    UINT64 Qi = pQi[0];     // the low-order bits always contain the Q symbols for the current horizontal swath of the scoring matrix
    UINT64 Qn = pQi[CUDATHREADSPERWARP];
    INT16 ofsQi = 2*CUDATHREADSPERWARP;
    INT16 endQi = blockdiv(N, 21) * CUDATHREADSPERWARP;     // (number of 64-bit elements in Q sequence) * (number of threads per CUDA warp)

    /* We compute the scoring matrix in horizontal swaths that are 8 cells high. */
    INT16 V0=0, V1=0, V2=0, V3=0, V4=0, V5=0, V6=0, V7=0;
    INT16 E0=0, E1=0, E2=0, E3=0, E4=0, E5=0, E6=0, E7=0;

    // track the maximum V score across all the cells in the scoring matrix
    INT16 Vmax = 0;

    /* topmost swath */
///* TODO: CHOP WHEN DEBUGGED */           csI = 0;
    for( INT16 j=0; j<Mr; ++j )
    {
///* TODO: CHOP WHEN DEBUGGED */                csJ = j;

        pFV[ofsFV] = computeVband( Vmax, 8, tbo, 0, j, Qi, Rj, 0, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
        getNextRj( Rj, pRj, ofsRj );
        ofsFV += CUDATHREADSPERWARP;
    }

    // reset to the start of the FV buffer (i.e. the leftmost column in the scoring matrix)
    ofsFV -= Mr*CUDATHREADSPERWARP;

    /* interior swaths */
    INT16 i = 8;
    const INT16 iLimit = N & (-8);  // largest value evenly divisible by 8 and less than N
    while( i < iLimit )
    {
///* TODO: CHOP WHEN DEBUGGED */           csI = i;
        resetRj( Rj, pRj, ofsRj );
        getNextQi( Qi, Qn, pQi, ofsQi, endQi, 8, 8 );

        // reset the registers that contain the E and V values for horizontally-adjacent cells
        V0 = V1 = V2 = V3 = V4 = V5 = V6 = V7 = 0;
        E0 = E1 = E2 = E3 = E4 = E5 = E6 = E7 = 0;
        INT16 Vd = 0;
        for( INT16 j=0; j<Mr; ++j )
        {
///* TODO: CHOP WHEN DEBUGGED */                csJ = j;

            UINT32 FVv = pFV[ofsFV];
            pFV[ofsFV] = computeVband( Vmax, 8, tbo, i, j, Qi, Rj, Vd, FVv, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
            getNextRj( Rj, pRj, ofsRj );
            ofsFV += CUDATHREADSPERWARP;
            Vd = static_cast<INT16>(FVv);
        }

        // reset the FV offset
        ofsFV -= Mr*CUDATHREADSPERWARP;
        i += 8;

        // abort this thread if it is impossible to reach the threshold V score
        if( Vmax < (Vt - (N-i)*ccASP.Wm) )
            return 0;
    }

    const INT16 nRows = N & 7;      // number of rows in the bottommost swath
    if( nRows  )
    {
///* TODO: CHOP WHEN DEBUGGED */           csI = i;

        /* bottommost swath */
        resetRj( Rj, pRj, ofsRj );
        getNextQi( Qi, Qn, pQi, ofsQi, endQi, 8, nRows );

        V0 = V1 = V2 = V3 = V4 = V5 = V6 = V7 = 0;
        E0 = E1 = E2 = E3 = E4 = E5 = E6 = E7 = 0;
        INT16 Vd = 0;

        for( INT16 j=0; j<Mr; ++j )
        {
///* TODO: CHOP WHEN DEBUGGED */                csJ = j;
            UINT32 FVv = pFV[ofsFV];
            computeVband( Vmax, nRows, tbo, i, j, Qi, Rj, Vd, FVv, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
            getNextRj( Rj, pRj, ofsRj );
            ofsFV += CUDATHREADSPERWARP;
            Vd = static_cast<INT16>(FVv);
        }
    }

    // return the maximum V score
    return Vmax;
}

/// [device] method initializeSharedMemory
static inline __device__ void initializeSharedMemory()
{
    // the first warp of threads copies the W lookup table into shared memory
    if( threadIdx.x < 32 )
    {
        // the table dimensions are 8 by 8, so each of the 32 threads copies two values
        INT16 qi = threadIdx.x / 4;
        INT16 rj = 2 * (threadIdx.x % 4);

        csW[(qi<<3)|rj] = ccASP.W[qi][rj];
        ++rj;
        csW[(qi<<3)|rj] = ccASP.W[qi][rj];
    }

    // all threads must wait at this point for shared memory initialization to complete
    __syncthreads();
}

/// baseMaxVw_Kernel
static __global__  void baseMaxVw_Kernel( const UINT64*  const __restrict__ pRi,            // in: the interleaved R sequence data
                                          const Qwarp*   const __restrict__ pQwBuffer,      // in: Qwarps
                                          const UINT64*  const __restrict__ pQiBuffer,      // in: interleaved Q sequence data
                                                UINT64*  const __restrict__ pDcBuffer,      // in, out: candidate D values
                                          const UINT32                      iDcBuffer,      // in: 0-based offset of the first D value
                                          const UINT32                      nDc,            // in: number of candidate D values to be processed in the current kernel invocation
                                          const INT32                       Mr,             // in: number of R symbols per scoring matrix
                                                INT16*   const              pVmaxBuffer,    // out: pointer to V scores for high-scoring alignments
                                                UINT32*  const              pFVBuffer       // pointer to the FV buffer



                                            // TODO: CHOP WHEN DEBUGGED
                                               , const bool isDebug
                                               , const UINT32 tidDebug
                                            )
{
    // initialize shared memory
    initializeSharedMemory();

    // compute the 0-based index of the current CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // abort this thread if there is no work to do
    if( tid >= nDc )
        return;


    //// TODO: CHOP WHEN DEBUGGED
    //if( isDebug )
    //{
    //    if( tid != tidDebug )
    //        return;

    //    asm( "brkpt;" );
    //}





    // compute the 0-based index of the candidate Dvalue
    const UINT32 iDc = iDcBuffer + tid;

    /* Get the D value for the unmapped mate:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 22;     // bits 39..60: QID
            UINT64  flags :  3;     // bits 61..63: flags
    */
    UINT64 Dc = pDcBuffer[iDc];

    // unpack the QID
    UINT32 qid = static_cast<UINT32>(Dc >> 39) & AriocDS::QID::maskQID;

    // point to the Qwarp struct
    const Qwarp* pQw = pQwBuffer + QID_IW(qid);
    INT32 iq = QID_IQ(qid);


#if TODO_CHOP_WHEN_DEBUGGED
    if( pQw->sqId[iq] == 0x00000808973166f0 )
    {
        Dc = pDcBuffer[iDc];
    }
    else
        return;
#endif



  


    // point to the interleaved Q sequence data for the current thread
    const UINT64* const __restrict__ pQ0 = pQiBuffer + pQw->ofsQi + iq;

    // point to the interleaved R sequence data for the current thread
    UINT32 iRw = iDc >> 5;              // offset of the first Ri for the CUDA warp that corresponds to the current thread ID
    UINT32 celMr = blockdiv(Mr, 21);    // number of 64-bit elements needed to represent Mr

    UINT32 ofsRi = (iRw * celMr * CUDATHREADSPERWARP) + (iDc & 0x1F);
    const UINT64* __restrict__ pR0 = pRi + ofsRi;

    // point to the FV buffer for the current thread; the FV buffer maps one element for each scoring-matrix column
    UINT32 ofsFV = (tid >> 5) * (CUDATHREADSPERWARP * Mr) + (tid & 0x1F);

#if TODO_CHOP_WHEN_DEBUGGED
    if( isDebug )
    {
        if( (pQw->sqId[iq] != 0x000002140000f47c) /* && (pQw->sqId[iq] != 0x00000216000008FF) */ )
            return;

        asm( "brkpt;" );
    }
#endif

    // compute the minimum high-score threshold
    const INT16 Vt = (ccASP.sft == sftG) ? ccASP.sfA*log( static_cast<double>(pQw->N[iq]) ) + ccASP.sfB :
                     (ccASP.sft == sftS) ? ccASP.sfA*sqrt( static_cast<double>(pQw->N[iq]) ) + ccASP.sfB :
                     (ccASP.sft == sftL) ? ccASP.sfA*pQw->N[iq] + ccASP.sfB :
                                           ccASP.sfB;

    // compute the maximum V score for the unmapped mate within the window
    INT32 tbo = 0;
    INT16 Vmax = computeV( tbo, Vt, pQ0, pR0, pQw->N[iq], pFVBuffer, ofsFV, Mr );

    /* save the newly computed Vmax if it is no less than the threshold V score (Vt) */
    if( Vmax >= Vt )
    {
        // save Vmax
        pVmaxBuffer[iDc] = Vmax;

        /* Compute a new D value based on the traceback origin:
            - the tbo recorded in the kernel is recorded as the diagonal number (j-i) for the cell that contains Vmax, so ...
            - the new D value is the sum of the original D position and the tbo value
            - since the tbo value may be negative, the new D value may also be negative
        */
        INT32 Dnew = static_cast<INT32>(Dc & AriocDS::D::maskPos);  // isolate bits 00..30
        Dnew = (Dnew << 1) >> 1;                                    // sign-extend

#if TODO_CHOP_WHEN_DEBUGGED
if( Dnew < 0 )
    asm( "brkpt;" );
#endif

        Dnew += tbo;                                // add the traceback origin (diagonal number)

#if TODO_CHOP_WHEN_DEBUGGED
if( Dnew < 0 )
    asm( "brkpt;" );
#endif

        // update the Dc value
        Dc &= (~AriocDS::D::maskPos);       // zero bits 00..30
        Dc |= (static_cast<UINT64>(Dnew) & AriocDS::D::maskPos);
        Dc |= AriocDS::D::flagMapped;
        pDcBuffer[iDc] = Dc;
    }
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseMaxVw::initConstantMemory()
{
    CRVALIDATOR;

    CRVALIDATE = cudaMemcpyToSymbol( ccASP, &m_pab->aas.ASP, sizeof(AlignmentScoreParameters) );
}

/// [private] method initSharedMemory
UINT32 baseMaxVw::initSharedMemory()
{
    CRVALIDATOR;
    
    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseMaxVw_Kernel, cudaFuncCachePreferL1 );


#if __CUDA_ARCH__ >= 350
    CRVALIDATE = cudaFuncSetSharedMemConfig( baseMaxVw_Kernel, cudaSharedMemBankSizeFourByte );
    CDPrint( cdpCDb, "%s: kernel uses %u bytes/block", __FUNCTION__, sizeof(csW) );
#endif

    return 0;
}

/// [private] method launchKernel
void baseMaxVw::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    bool isDebug = false;
    UINT32 tidDebug = 0;


#if TRACE_SQID
    bool isTraceId = false;
    WinGlobalPtr<UINT64> Dcxxx( m_nD, false );
    cudaMemcpy( Dcxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
        
    for( UINT32 n=0; n<m_nD; ++n )
    {
        UINT64 Dc = Dcxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( Dc );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
            
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            isTraceId = true;
            INT16 subId = static_cast<INT16>(Dc >> 32) & 0x007F;
                INT32 J = Dc & 0x7FFFFFFF;
                INT32 Jf = (Dc & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

#if TRACE_SUBID
                if( subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "%s::%s: before alignment: %3d: Dc=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d Jf=%d",
                                    m_ptum->Key, __FUNCTION__, n, Dc, sqId, qid, subId, J, Jf );

            // prepare to debug
            isDebug = true;
            tidDebug = n;
            //break;
        }
    }
#endif


    if( isDebug )
        CDPrint( cdpCD0, "%s::%s!", m_ptum->Key, __FUNCTION__ );


    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );




#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Rixxx( m_pqb->DB.Ri.Count, false );
    m_pqb->DB.Ri.CopyToHost( Rixxx.p, Rixxx.Count );
    for( size_t n=0; n<Rixxx.Count; ++n )
    {
        if( Rixxx.p[n] & 0x8000000000000000 )
            CDPrint( cdpCD0, "%s: Ri is corrupted", __FUNCTION__ );

    }
#endif






    // launch the CUDA kernel
    baseMaxVw_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DB.Ri.p,     // in: interleaved R sequence data
                                                        m_pqb->DB.Qw.p,     // in: Qwarps
                                                        m_pqb->DB.Qi.p,     // in: interleaved Q sequence data
                                                        m_pD,               // in,out: candidate D values
                                                        0,                  // in: 0-based offset of the first D value
                                                        m_nD,               // in: number of D values to be processed in the current kernel invocation                                                              
                                                        m_pqb->Mrw,         // in: number of R symbols per scoring matrix
                                                        m_pVmax,            // out: V scores for mapped mates
                                                        m_FV.p              // FV buffer

                                                        , isDebug
                                                        , tidDebug

                                                        );





#if TRACE_SQID
    if( isTraceId )
    {
        cudaMemcpy( Dcxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
        
        for( UINT32 n=0; n<m_nD; ++n )
        {
            UINT64 Dc = Dcxxx.p[n];
            UINT32 qid = AriocDS::D::GetQid( Dc );
            UINT32 iw = QID_IW(qid);
            INT16 iq = QID_IQ(qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];
            
            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                INT16 subId = static_cast<INT16>(Dc >> 32) & 0x007F;
                INT32 J = Dc & 0x7FFFFFFF;
                INT32 Jf = (Dc & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

#if TRACE_SUBID
        if( subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "%s::%s: after alignment: %3d: Dc=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d Jf=%d",
                                    m_ptum->Key, __FUNCTION__, n, Dc, sqId, qid, subId, J, Jf );

            }
        }

        CDPrint( cdpCD0, __FUNCTION__ );
    }
#endif

}
#pragma endregion
