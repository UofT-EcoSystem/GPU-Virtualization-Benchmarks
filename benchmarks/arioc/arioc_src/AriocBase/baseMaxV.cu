/*
  baseMaxV.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

//#define DEBUG_DUMP_SMCELLS 1

#pragma region CUDA device code and data
/* CUDA constant memory
*/
static __device__ __constant__ AlignmentScoreParameters    ccASP;


/* CUDA shared memory
*/
static __device__ __shared__ INT32 csW[8*8];


#if DEBUG_DUMP_SMCELLS
// For a 100 x 100 scoring matrix, the printf FIFO should be 25-30Mb; use something like cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 30*1024*1024) */
static __device__ __shared__ UINT32 csJ;
static __device__ __shared__ INT16 csI;
static __device__ __shared__ INT16 csii;
#endif

/// [device] method resetRj
static inline __device__ void resetRj( UINT64& Rj, const UINT64* const __restrict__ pRj, UINT32& ofsRj, const UINT32 j )
{
    ofsRj = (j / 21) * CUDATHREADSPERWARP;  // compute the offset of the 64-bit value that contains the j'th symbol in R

    UINT32 jmod = j % 21;                   // compute the symbol position within the 64-bit value at ofsRj
    if( jmod )
    {
#if __CUDA_ARCH__ < 350
        Rj = pRj[ofsRj] >> (3*(jmod-1));    // read the 64-bit value at ofsRj and shift it so that the next call to getNextRj() will be correct
#else
        Rj = __ldg(pRj+ofsRj) >> (3*(jmod-1));    // read the 64-bit value at ofsRj and shift it so that the next call to getNextRj() will be correct
#endif
        ofsRj += CUDATHREADSPERWARP;        // update the offset so that it references the subsequent 64-bit value
    }
    else
        Rj = 0;                             // zero the current value of Rj; the next call to getNextRj() will fetch data from the computed offset
}

/// [device] method getNextRj
static inline __device__ void getNextRj( UINT64& Rj, const UINT64* const __restrict__ pRj, UINT32& ofsRj )
{
    Rj >>= 3;
    if( Rj == 0 )
    {
#if __CUDA_ARCH__ < 350
        Rj = pRj[ofsRj];
#else
        Rj = __ldg(pRj+ofsRj);
#endif
        ofsRj += CUDATHREADSPERWARP;
    }
}

/// [device] method reverseComplement
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

// methods getNextQiF and getNextQiRC are called through a function pointer that is defined with the following typedef:
typedef void (*pfnGetNextQi)( UINT64& Qi, UINT64& Qn, const UINT64* const __restrict__ pQi, INT16& ofsQi, const INT16 endQi, const INT16 cchOut, const INT16 cchIn );

/// [device] method getNextQiF
static inline __device__ void getNextQiF( UINT64& Qi, UINT64& Qn, const UINT64* const __restrict__ pQi, INT16& ofsQi, const INT16 endQi, const INT16 cchOut, const INT16 cchIn )
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

/// [device] method getNextQiRC
static inline __device__ void getNextQiRC( UINT64& Qi, UINT64& Qn, const UINT64* const __restrict__ pQi, INT16& ofsQi, const INT16 /* endQi */, const INT16 cchOut, const INT16 cchIn )
{
    /* This implementation of getNextQi returns symbols from the reverse complement of a Q sequence. */

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
    if( (pos >= (u-3)) || (ofsQi < 0) )
        return;

    /* At this point:
        - Qi still contains fewer than the requested number of symbols
        - Qn == 0
        - there remain symbols to load
    */

    // load more symbols
    Qn = pQi[ofsQi];
    ofsQi -= CUDATHREADSPERWARP;

    // compute the reverse complement
    Qn = reverseComplement( Qn );

    // copy additional symbols from Qn into Qi
    v = 3 * ((pos+3) / 3);  // 3 * (number of symbols remaining in Qi)
    Qi = (Qi | (Qn << v)) & MASKA21;

    // shift the additional symbols out of Qn
    Qn >>= (63 - v);
}

/// [device] method computeSMcell
static inline __device__ void computeSMcell( INT16&         Vmax,   // in, out
                                             INT16&         Vi,     // in, out
                                             INT16&         Ei,     // in, out
                                             INT16&         F,      // in, out
                                             const INT16    Rj,     // in
                                             const INT16    Qi,     // in
                                             INT16&         Vd,     // in, out
                                             const INT16    Vv      // in
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

    // track the maximum V score
    Vmax = max( Vmax, Vi );

#if DEBUG_DUMP_SMCELLS
//    if( (csI+csii) <= 88 )
        printf( "%d %u %c %c %d\n", csI+csii, csJ, "??NNACGT"[Qi&7], "??NNACGT"[Rj], Vi );
#endif
}

/// [device] method computeVband
static inline __device__ UINT32 computeVband( INT16& Vmax, const INT16 imax,
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

       If the caller specifies a constant value for imax (the index of the last row to be computed), NVCC suppresses the corresponding
        conditional clauses (i.e., if( imax == ... )).
    */
    INT16 Vv = static_cast<INT16>(FVv);
    INT16 F = static_cast<INT16>(FVv >> 16);
    Rj &= 7;

    /* i=0 */
#if DEBUG_DUMP_SMCELLS
csii = 0;
#endif
    computeSMcell( Vmax, V0, E0, F, Rj, Qi8, Vd, Vv );
    if( imax == 0 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V0);        //  bits 0-15: V; bits 16-31: F

    /* i=1 */
#if DEBUG_DUMP_SMCELLS
csii = 1;
#endif
    computeSMcell( Vmax, V1, E1, F, Rj, Qi8>>3, Vd, V0 );
    if( imax == 1 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V1);

    /* i=2 */
#if DEBUG_DUMP_SMCELLS
csii = 2;
#endif
    computeSMcell( Vmax, V2, E2, F, Rj, Qi8>>6, Vd, V1 );
    if( imax == 2 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V2);

    /* i=3 */
#if DEBUG_DUMP_SMCELLS
csii = 3;
#endif
    computeSMcell( Vmax, V3, E3, F, Rj, Qi8>>9, Vd, V2 );
    if( imax == 3 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V3);

    /* i=4 */
#if DEBUG_DUMP_SMCELLS
csii = 4;
#endif
    computeSMcell( Vmax, V4, E4, F, Rj, Qi8>>12, Vd, V3 );
    if( imax == 4 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V4);

    /* i=5 */
#if DEBUG_DUMP_SMCELLS
csii = 5;
#endif
    computeSMcell( Vmax, V5, E5, F, Rj, Qi8>>15, Vd, V4 );
    if( imax == 5 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V5);

    /* i=6 */
#if DEBUG_DUMP_SMCELLS
csii = 6;
#endif
    computeSMcell( Vmax, V6, E6, F, Rj, Qi8>>18, Vd, V5 );
    if( imax == 6 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V6);

    /* i=7 */
#if DEBUG_DUMP_SMCELLS
csii = 7;
#endif
    computeSMcell( Vmax, V7, E7, F, Rj, Qi8>>21, Vd, V6 );
    return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V7);
}

/// [device] method computeVbandR
static inline __device__ UINT32 computeVbandR( INT16& Vmax, INT16 imin, INT16 imax,
                                               UINT32 Qi8, INT16 Rj, INT16 Vd,
                                               INT16& V0, INT16& V1, INT16& V2, INT16& V3, INT16& V4, INT16& V5, INT16& V6, INT16& V7,
                                               INT16& E0, INT16& E1, INT16& E2, INT16& E3, INT16& E4, INT16& E5, INT16& E6, INT16& E7
                                             )
{
    /* Computes a vertical band of between 1 and 8 cells, starting at the specified 0-based offset from the topmost row in the band.

                +---+
             1  |   |
                +---+---+
             2  |   |   |                                   ComputeVbandR is used to compute the triangular tiles
                +---+---+---+                               at the right-hand edge of each horizontal swath of SM cells.
             3  |   |   |   |  
          i     +---+---+---+---+                           This function is only called with
             4  |   |   |   |   |                               - imax <= 7
                +---+---+---+---+---+                           - imin <= imax
             5  |   |   |   |   |   |
                +---+---+---+---+---+---+
             6  |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+
             7  |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+
    */

    INT16 F = 0;        // F and Vv are initially zero
    INT16 Vv = 0;
    Rj &= 7;

    // the initial value for Vd depends on imin
    switch( imin )
    {
        case 0: break;
        case 1: Vd = V0; break;
        case 2: Vd = V1; break;
        case 3: Vd = V2; break;
        case 4: Vd = V3; break;
        case 5: Vd = V4; break;
        case 6: Vd = V5; break;
        case 7: Vd = V6; break;
    }

    // start at the specified row; each case falls through to the subsequent case
    switch( imin )
    {
        case 0:
#if DEBUG_DUMP_SMCELLS
csii = 0;
#endif
            computeSMcell( Vmax, V0, E0, F, Rj, Qi8, Vd, Vv );
            if( imax == 0 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V0);        //  bits 0-15: V; bits 16-31: F
            Vv = V0;

        case 1:
#if DEBUG_DUMP_SMCELLS
csii = 1;
#endif
            computeSMcell( Vmax, V1, E1, F, Rj, Qi8>>3, Vd, Vv );
            if( imax == 1 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V1);
            Vv = V1;

        case 2:
#if DEBUG_DUMP_SMCELLS
csii = 2;
#endif
            computeSMcell( Vmax, V2, E2, F, Rj, Qi8>>6, Vd, Vv );
            if( imax == 2 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V2);
            Vv = V2;

        case 3:
#if DEBUG_DUMP_SMCELLS
csii = 3;
#endif
            computeSMcell( Vmax, V3, E3, F, Rj, Qi8>>9, Vd, Vv );
            if( imax == 3 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V3);
            Vv = V3;

        case 4:
#if DEBUG_DUMP_SMCELLS
csii = 4;
#endif
            computeSMcell( Vmax, V4, E4, F, Rj, Qi8>>12, Vd, Vv );
            if( imax == 4 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V4);
            Vv = V4;

        case 5:
#if DEBUG_DUMP_SMCELLS
csii = 5;
#endif
            computeSMcell( Vmax, V5, E5, F, Rj, Qi8>>15, Vd, Vv );
            if( imax == 5 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V5);
            Vv = V5;

        case 6:
#if DEBUG_DUMP_SMCELLS
csii = 6;
#endif
            computeSMcell( Vmax, V6, E6, F, Rj, Qi8>>18, Vd, Vv );
            if( imax == 6 ) return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V6);
            Vv = V6;

        case 7:
#if DEBUG_DUMP_SMCELLS
csii = 7;
#endif
            computeSMcell( Vmax, V7, E7, F, Rj, Qi8>>21, Vd, Vv );
            return (static_cast<UINT32>(F) << 16) | static_cast<UINT32>(V7);

        default:
            break;
    }

    // (control should never reach this point)
    asm( "brkpt;" );
    return 0;
}

/// [device] method computeVnarrow
static inline __device__ INT16 computeVnarrow( const INT16                      Vt,
                                               const UINT64* const __restrict__ pQi,
                                               const UINT64* const __restrict__ pRj,
                                               const INT16                      N,
                                                     UINT32* const              pFV,
                                                     UINT32                     ofsFV,
                                                     UINT64                     Qi,
                                                     UINT64                     Qn,
                                                     INT16                      ofsQi,
                                               const INT16                      endQi,
                                                     UINT64                     Rj,
                                                     UINT32                     ofsRj,
                                               const INT16                      bw,
                                                     pfnGetNextQi               getNextQi
                                       )
{
    // track the maximum V score across all the cells in the scoring matrix
    INT16 Vmax = 0;

    // compute the scoring matrix in vertical bands that are at most 7 cells high
    INT16 V0=0, V1=0, V2=0, V3=0, V4=0, V5=0, V6=0, V7=0;
    INT16 E0=0, E1=0, E2=0, E3=0, E4=0, E5=0, E6=0, E7=0;

    /* upper left triangle */
    INT16 j = 0;
    for( ; j<bw; ++j )
    {
        getNextRj( Rj, pRj, ofsRj );
        computeVband( Vmax, j, Qi, Rj, 0, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
    }

    /* interior vertical bands */
    INT16 Vd;
    for( ; j<N; ++j )
    {
        Vd = V0;
        V0 = V1; E0 = E1;
        V1 = V2; E1 = E2;
        V2 = V3; E2 = E3;
        V3 = V4; E3 = E4;
        V4 = V5; E4 = E5;
        V5 = V6; E5 = E6;
        V6 = V7; E6 = E7;
        getNextRj( Rj, pRj, ofsRj );
        getNextQi( Qi, Qn, pQi, ofsQi, endQi, 1, bw );
        computeVband( Vmax, bw-1, Qi, Rj, Vd, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
    }

    /* lower right triangle */
    const INT16 jLimit = N + bw - 1;
    INT16 imax = bw - 2;
    for( ; j<jLimit; ++j )
    {
        Vd = V0;
        V0 = V1; E0 = E1;
        V1 = V2; E1 = E2;
        V2 = V3; E2 = E3;
        V3 = V4; E3 = E4;
        V4 = V5; E4 = E5;
        V5 = V6; E5 = E6;
        V6 = V7; E6 = E7;
        getNextRj( Rj, pRj, ofsRj );
        getNextQi( Qi, Qn, pQi, ofsQi, endQi, 1, bw );
        computeVband( Vmax, imax, Qi, Rj, Vd, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
        --imax;
    }

    return Vmax;
}

/// [device] method computeVwide
static inline __device__ INT16 computeVwide( const INT16                        Vt,
                                             const UINT64* const __restrict__   pQi,
                                             const UINT64* const __restrict__   pRj,
                                             const INT16                        N,
                                                   UINT32* const                pFV,
                                                   UINT32                       ofsFV,
                                                   UINT64                       Qi,
                                                   UINT64                       Qn,
                                                   INT16                        ofsQi,
                                             const INT16                        endQi,
                                                   UINT64                       Rj,
                                                   UINT32                       ofsRj,
                                                   const INT16                  bw,
                                                   pfnGetNextQi                 getNextQi
                                            )
{
    // track the maximum V score across all the cells in the scoring matrix
    INT16 Vmax = 0;

    // compute the scoring matrix in horizontal swaths that are 8 cells high
    INT16 V0=0, V1=0, V2=0, V3=0, V4=0, V5=0, V6=0, V7=0;
    INT16 E0=0, E1=0, E2=0, E3=0, E4=0, E5=0, E6=0, E7=0;

    /* topmost swath */
#if DEBUG_DUMP_SMCELLS
csI = 0;
#endif

    // left-hand triangular tile in the swath
    INT16 j = 0;    // j is a 0-based offset into the scoring matrix
    for( ; j<7; j++ )
    {
#if DEBUG_DUMP_SMCELLS
csJ = j;
#endif
        getNextRj( Rj, pRj, ofsRj );
        computeVband( Vmax, j, Qi, Rj, 0, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
    }
        
    // rectangular tiles in the swath
    for( ; j<bw; j++ )
    {
#if DEBUG_DUMP_SMCELLS
csJ = j;
#endif
        getNextRj( Rj, pRj, ofsRj );
        pFV[ofsFV] = computeVband( Vmax, 7, Qi, Rj, 0, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
        ofsFV += CUDATHREADSPERWARP;
    }

    // right-side triangular tile in the swath
    for( ; j<(bw+7); j++ )
    {
#if DEBUG_DUMP_SMCELLS
csJ = j;
#endif
        getNextRj( Rj, pRj, ofsRj );
        pFV[ofsFV] = computeVbandR( Vmax, (j-(bw-1)), 7, Qi, Rj, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
        ofsFV += CUDATHREADSPERWARP;
    }

    /* interior swaths */
    INT16 i = 8;                        // i is a 0-based offset into the Q sequence
    while( i < (N-8) )
    {
#if DEBUG_DUMP_SMCELLS
csI = i;
#endif

        j = i;                          // each swath starts at the 0th diagonal of the scoring matrix
        resetRj( Rj, pRj, ofsRj, j );
        getNextQi( Qi, Qn, pQi, ofsQi, endQi, 8, 8 );

//        printf( "getNextQi 1: i=%d j=%d Qi=0x%016llx\n", i, j, Qi );

        // left-side triangular tile in the swath
        V0 = V1 = V2 = V3 = V4 = V5 = V6 = V7 = 0;
        E0 = E1 = E2 = E3 = E4 = E5 = E6 = E7 = 0;

        // reset the FV offset
        ofsFV -= (bw * CUDATHREADSPERWARP);

        // get the V value for the diagonal neighbor
        INT16 Vd = static_cast<INT16>(pFV[ofsFV]);
        ofsFV += CUDATHREADSPERWARP;

        for( ; j<(i+7); ++j )
        {
#if DEBUG_DUMP_SMCELLS
csJ = j;
#endif
            UINT32 FVv = pFV[ofsFV];
            getNextRj( Rj, pRj, ofsRj );
            computeVband( Vmax, j-i, Qi, Rj, Vd, FVv, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
            Vd = static_cast<INT16>(FVv);
            ofsFV += CUDATHREADSPERWARP;
        }

        // rectangular tiles in the swath
        for( ; j<(i+bw-1); j++ )
        {
#if DEBUG_DUMP_SMCELLS
csJ = j;
#endif
            UINT32 FVv = pFV[ofsFV];
            getNextRj( Rj, pRj, ofsRj );
            pFV[ofsFV-8*CUDATHREADSPERWARP] = computeVband( Vmax, 7, Qi, Rj, Vd, FVv, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
            Vd = static_cast<INT16>(FVv);
            ofsFV += CUDATHREADSPERWARP;
        }

        // reset ofsFV
        ofsFV -= 8*CUDATHREADSPERWARP;

        // the diagonal neighbor for the first vertical band in the right-side triangular tile is the last cell in the FV buffer
        Vd = static_cast<INT16>(pFV[ofsFV+7*CUDATHREADSPERWARP]);

        // right-side triangular tile in the swath
        for( ; j<(i+bw+7); j++ )
        {
#if DEBUG_DUMP_SMCELLS
csJ = j;
#endif
            getNextRj( Rj, pRj, ofsRj );
            pFV[ofsFV] = computeVbandR( Vmax, j-(i+bw-1), 7, Qi, Rj, Vd, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
            ofsFV += CUDATHREADSPERWARP;
        }

        // advance to the next swath
        i += 8;

        // abort this thread if it is impossible to reach the threshold V score
        if( Vmax < (Vt - (N-i)*ccASP.Wm) )
            return 0;
    }
    
    /* bottommost swath */
    j = i;                          // the swath starts at the 0th diagonal of the scoring matrix

#if DEBUG_DUMP_SMCELLS
csI = i;
#endif

    resetRj( Rj, pRj, ofsRj, j );
    getNextQi( Qi, Qn, pQi, ofsQi, i, 8, N-i );


//    printf( "getNextQi 2: i=%d j=%d Qi=0x%016llx\n", i, j, Qi );


    V0 = V1 = V2 = V3 = V4 = V5 = V6 = V7 = 0;
    E0 = E1 = E2 = E3 = E4 = E5 = E6 = E7 = 0;

    // reset the FV offset
    ofsFV -= (bw * CUDATHREADSPERWARP);

    // left-side triangular tile in the swath
    INT16 jTail = (N-i) - 1;
    INT16 Vd = static_cast<INT16>(pFV[ofsFV]);
    ofsFV += CUDATHREADSPERWARP;

    for( ; j<(i+jTail); ++j )
    {
#if DEBUG_DUMP_SMCELLS
csJ = j;
#endif
        UINT32 FVv = pFV[ofsFV];
        ofsFV += CUDATHREADSPERWARP;
        getNextRj( Rj, pRj, ofsRj );
        computeVband( Vmax, j-i, Qi, Rj, Vd, FVv, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
        Vd = static_cast<INT16>(FVv);
    }

    // rectangular tiles in the swath
    for( ; j<(i+bw-1); j++ )
    {
#if DEBUG_DUMP_SMCELLS
csJ = j;
#endif
        UINT32 FVv = pFV[ofsFV];
        ofsFV += CUDATHREADSPERWARP;
        getNextRj( Rj, pRj, ofsRj );
        computeVband( Vmax, (N-i)-1, Qi, Rj, Vd, FVv, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
        Vd = static_cast<INT16>(FVv);
    }

    // (the diagonal neighbor for the first vertical band in the right-side triangular tile is the topmost cell in the most recently filled vertical band)

    // right-side triangular tile in the swath
    for( ; j<(i+bw+jTail); j++ )
    {
#if DEBUG_DUMP_SMCELLS
csJ = j;
#endif
        getNextRj( Rj, pRj, ofsRj );
        computeVbandR( Vmax, j-(i+bw-1), jTail, Qi, Rj, Vd, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7 );
    }

    return Vmax;
}

/// [device] method computeV
static inline __device__ INT16 computeV( const INT16                        Vt,
                                         const UINT64* const __restrict__   pQi,
                                         const UINT64* const __restrict__   pRj,
                                         const INT16                        N,
                                               UINT32* const                pFV,
                                               UINT32                       ofsFV,
                                         const INT16                        wcgsc,
                                         const bool                         useQiF
                                       )
{
    // get the first 64-bit value from the R sequence
    UINT64 Rj;
    UINT32 ofsRj;
    resetRj( Rj, pRj, ofsRj, 0 );

    // get the first two 64-bit values from the Q sequence; this assumes that N > 21 (without error checking)
    UINT64 Qi;
    UINT64 Qn;
    INT16 ofsQi;
    INT16 endQi = blockdiv(N,21) * CUDATHREADSPERWARP;  // (number of 64-bit elements in Q sequence) * (number of threads per CUDA warp)
    pfnGetNextQi getNextQi;

    if( useQiF )
    {
        /* Prepare to use the forward Qi sequence */

        // load the first two 64-bit (21-symbol) chunks of the Q sequence
        Qi = pQi[0];
        Qn = pQi[CUDATHREADSPERWARP];

        // compute the offset of the next chunk of the Q sequence to be loaded
        ofsQi = 2*CUDATHREADSPERWARP;

        getNextQi = getNextQiF;
    }
    else
    {
        /* Prepare to use the reverse-complement Qi sequence */

        // get the final two 64-bit chunks of the Q sequence; the final value may contain fewer than 21 symbols
        Qi = pQi[endQi-CUDATHREADSPERWARP];
        Qn = pQi[endQi-2*CUDATHREADSPERWARP];

        // compute the offset of the next chunk of the Q sequence to be loaded
        ofsQi = endQi-3*CUDATHREADSPERWARP;

        // reverse complement
        Qi = reverseComplement( Qi );
        Qn = reverseComplement( Qn );

        // at this point Qi is likely to contain fewer than 21 symbols...
        INT16 mod21 = N % 21;   // number of symbols in Qi, or 0 if there are 21 symbols in Qi
        if( mod21 )
        {
            INT16 shl = 3 * mod21;              // number of bits used by symbols in Qi
            INT16 shr = 63 - shl;

            /* reverseComplement() leaves symbols in the high-order bits, so we...
                - shift the symbols in Qi to the LSB
                - copy symbols from the LSB of Qn to the MSB of Qi
                - shift those symbols out of Qn
            */
            Qi = ((Qi >> shr) | (Qn << shl)) & MASKA21;
            Qn >>= shr;
        }

        getNextQi = getNextQiRC;
        endQi = 0;                  // (unused when loading the reverse complement of the Q sequence)
    }

    // compute the width of the diagonal band of scoring-matrix cells to be computed
    INT16 bw = (2*wcgsc) + 1;

    // we use a different (and somewhat faster) implementation for band width < 8
    return (bw >= 8) ? computeVwide( Vt, pQi, pRj, N, pFV, ofsFV, Qi, Qn, ofsQi, endQi, Rj, ofsRj, bw, getNextQi ) :
                       computeVnarrow( Vt, pQi, pRj, N, pFV, ofsFV, Qi, Qn, ofsQi, endQi, Rj, ofsRj, bw, getNextQi );
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

    // all threads in the CUDA block must wait at this point for shared memory initialization to complete
    __syncthreads();
}

/// baseMaxV_Kernel
static __global__  void baseMaxV_Kernel( const UINT64*  const __restrict__ pRiBuffer,   // in: interleaved R sequence data
                                         const Qwarp*   const __restrict__ pQwBuffer,   // in: Qwarps
                                         const UINT64*  const __restrict__ pQiBuffer,   // in: interleaved Q sequence data
                                               UINT64*  const __restrict__ pDcBuffer,   // in, out: candidate D values
                                         const UINT32                      oDc,         // in: offset into the list of D values for the 0th thread in the current kernel invocation
                                         const UINT32                      nDc,         // in: number of candidate D values to be processed in the current kernel invocation
                                         const INT32                       maxMr,       // in: maximum number of R symbols per scoring matrix
                                               INT16*   const              pVmaxBuffer, // out: pointer to V scores for high-scoring alignments
                                               UINT32*  const              pFVBuffer    // pointer to the FV buffer

 // TODO: CHOP WHEN DEBUGGED:                                              , const bool isTraceId
                                           )
{
    // initialize shared memory
    initializeSharedMemory();

    // compute the 0-based index of the current CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // abort this thread if there is no work to do
    if( tid >= nDc )
        return;

    // compute the Dc-list offset for the current CUDA thread
    const UINT32 iDc = oDc + tid;

    // TODO: CHOP WHEN DEBUGGED
    if( iDc == 0xffffffff )
        asm( "brkpt;" );        // this should never happen!


#if DEBUG_DUMP_SMCELLS
    if( isTraceId )
    {
        if( iDc != 327 ) return;       // look at one thread only

        iDc += oDc;  // which is zero, so this is a noop
    }
#endif


    /* Get the D value for the unmapped mate:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: set if using reverse complement of Q sequence (bsDNA alignments only)
            UINT64  flags :  3;     // bits 61..63: flags
    */
    UINT64 Dc = pDcBuffer[iDc];

    // unpack the QID
    UINT32 qid = static_cast<UINT32>(Dc >> 39) & AriocDS::QID::maskQID;

    // point to the Qwarp struct
    const Qwarp* pQw = pQwBuffer + QID_IW(qid);
    const INT32 iq = QID_IQ(qid);

    // point to the interleaved Q sequence data for the current thread
    const UINT64* const __restrict__ pQ0 = pQiBuffer + pQw->ofsQi + iq;

    // point to the interleaved R sequence data for the current thread
    INT64 celMr = blockdiv(maxMr,21);   // number of 64-bit values required to represent Mr
    celMr *= CUDATHREADSPERWARP;        // number of 64-bit values per warp
    UINT32 ixw = iDc >> 5;
    INT16 ixq = iDc & 0x1F;
    const UINT64* pR0 = pRiBuffer + (ixw * celMr) + ixq;

    // point to the FV buffer for the current thread; the FV buffer maps one element for each scoring-matrix column
    UINT32 ofsFV = (tid >> 5) * (CUDATHREADSPERWARP * maxMr) + (tid & 0x1F);

    // compute the minimum high-score threshold
    const INT16 Vt = (ccASP.sft == sftG) ? ccASP.sfA*log( static_cast<double>(pQw->N[iq]) ) + ccASP.sfB :
                     (ccASP.sft == sftS) ? ccASP.sfA*sqrt( static_cast<double>(pQw->N[iq]) ) + ccASP.sfB :
                     (ccASP.sft == sftL) ? ccASP.sfA*pQw->N[iq] + ccASP.sfB :
                                           ccASP.sfB;

    // compute the maximum V score for the unmapped mate within the window
    bool useQiF = ((Dc & AriocDS::D::flagRCq) == 0);
    INT16 Vmax = computeV( Vt, pQ0, pR0, pQw->N[iq], pFVBuffer, ofsFV, pQw->wcgsc, useQiF );

    /* save the newly computed Vmax if it is no less than the threshold V score (Vt) */
    if( Vmax >= Vt )
    {
        pVmaxBuffer[iDc] = Vmax;
        pDcBuffer[iDc] = Dc | AriocDS::D::flagMapped;     // update the corresponding Dc value
    }
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseMaxV::initConstantMemory()
{
    CRVALIDATOR;

    CRVALIDATE = cudaMemcpyToSymbol( ccASP, &m_pab->aas.ASP, sizeof(AlignmentScoreParameters) );
}

/// [private] method initSharedMemory
UINT32 baseMaxV::initSharedMemory()
{
    CRVALIDATOR;
    
    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseMaxV_Kernel, cudaFuncCachePreferL1 );

#if __CUDA_ARCH__ >= 350
    CRVALIDATE = cudaFuncSetSharedMemConfig( baseMaxV_Kernel, cudaSharedMemBankSizeFourByte );
    CDPrint( cdpCD4, "%s: kernel uses %u bytes/block", __FUNCTION__, sizeof(csW) );
#endif

    return 0;
}

/// [private] method launchKernel
void baseMaxV::launchKernel( UINT32 cbSharedPerBlock )
{
    CRVALIDATOR;

    // performance metrics
    UINT32 nIterations = 0;
    InterlockedIncrement( &m_ptum->n.Instances );
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );




    // TODO: TEST LOOP BY DECREASING m_nDperIteration to some ridiculously low value


    // loop through the list of Dc values
    UINT32 iDc = 0;
    UINT32 nDremaining = m_nD;
    while( nDremaining )
    {
        // compute the actual number of D values to be processed in the current iteration
        UINT32 nDc = min2(nDremaining, m_nDperIteration);

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s::baseMaxV::launchKernel: invoking kernel for iteration %u oDc=%u nOc=%u", m_ptum->Key, nIterations, oDc, nDc );
#endif


        // set up the grid for the CUDA kernel
        dim3 d3b;
        dim3 d3g;
        computeKernelGridDimensions( d3g, d3b, nDc );


#if TRACE_SQID
    bool isTraceId = false;
    WinGlobalPtr<UINT64> Dcxxx( m_nD, false );
    cudaMemcpy( Dcxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
        
    for( UINT32 n=iDc; n<(iDc+nDc); ++n )
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

        }
    }
#endif

        // launch the CUDA kernel
        baseMaxV_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DB.Ri.p,      // in: interleaved R sequence data
                                                           m_pqb->DB.Qw.p,      // in: Qwarps
                                                           m_pqb->DB.Qi.p,      // in: interleaved Q sequence data
                                                           m_pD,                // in,out: candidate D values
                                                           iDc,                 // in: offset into the list of candidate D values
                                                           nDc,                 // in: number of D values (offsets) to be processed in the current kernel invocation                                                              
                                                           m_pdbg->AKP.Mr,      // in: maximum number of R symbols per scoring matrix
                                                           m_pVmax->p,          // out: V scores for mapped mates
                                                           m_FV.p               // FV buffer
 // TODO: CHOP WHEN DEBUGGED:                                                         , isTraceId
                                                         );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->n.CandidateD, nDc );

        // wait for the kernel to complete
        CREXEC( waitForKernel() );


#if TRACE_SQID
    if( isTraceId )
    {
        CDPrint( cdpCD0, "%s::%s: looking for sqId=0x%016llx in alignment results...", m_ptum->Key, __FUNCTION__, TRACE_SQID );
        
        cudaMemcpy( Dcxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

        WinGlobalPtr<INT16> Vmaxxx( m_nD, false );
        cudaMemcpy( Vmaxxx.p, m_pVmax->p, m_nD*sizeof(INT16), cudaMemcpyDeviceToHost );
                
        // copy the encoded interleaved R sequence data to a host buffer
        WinGlobalPtr<UINT64> Rixxx( m_pqb->DB.Ri.Count, false );
        m_pqb->DB.Ri.CopyToHost( Rixxx.p, Rixxx.Count );

        // copy the encoded interleaved Q sequence data to a host buffer
        WinGlobalPtr<UINT64> Qixxx( m_pqb->DB.Qi.Count, false );
        m_pqb->DB.Qi.CopyToHost( Qixxx.p, Qixxx.Count );

        for( UINT32 n=iDc; n<(iDc+nDc); ++n )
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
                {
#endif
                    CDPrint( cdpCD0, "%s::%s: after alignment: %3d: Dc=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d Jf=%d V=%d",
                        m_ptum->Key, __FUNCTION__, n, Dc, sqId, qid, subId, J, Jf, Vmaxxx.p[n] );
                    
                    // dump the encoded R sequence
                    INT64 celMr = blockdiv( m_pdbg->AKP.Mr, 21 );  // number of 64-bit values required to represent Mr
                    celMr *= CUDATHREADSPERWARP;                   // number of 64-bit values per warp
                    UINT32 ixw = n >> 5;
                    INT16 ixq = n & 0x1F;
                    const UINT64* pR = Rixxx.p + (ixw * celMr) + ixq;

                    CDPrint( cdpCD0, "%s: R...", __FUNCTION__ );
                    AriocCommon::DumpA21( const_cast<UINT64*>(pR), 2*pQw->N[ixq], CUDATHREADSPERWARP );

                    // dump the encoded Q sequence
                    UINT64* pQ = Qixxx.p + pQw->ofsQi + iq;

                    CDPrint( cdpCD0, "%s: Q...", __FUNCTION__ );
                    AriocCommon::DumpA21( pQ, pQw->N[ixq], CUDATHREADSPERWARP );



                    // examine the R sequence in memory
                    CDPrint( cdpCD0, "%s: reference R...", __FUNCTION__ );

                    // extract the the 0-based position on the reference strand
                    INT32 d = static_cast<INT32>(Dc & AriocDS::D::maskPos); // isolate bits 0-30
                    d = (d << 1) >> 1;                                      // sign-extend
                    d -= m_pdbg->AKP.wcgsc;

                    // point to the R sequence to be copied
                    const UINT64* pR0 = m_pab->R + ((Dc & AriocDS::D::maskStrand) ? m_pab->ofsRminus.p[subId] : m_pab->ofsRplus.p[subId]);

                    /* Compute the offset of the first 64-bit value to copy; the computation differs depending on whether d (the 0-based
                        offset into the R sequence) is negative:
                        - if d >= 0, the computation is straightforward
                        - if d < 0, then we need to adjust the value to ensure that the offset is computed correctly
                        */
                    INT32 ofsFrom = ((d >= 0) ? d : (d-20)) / 21;

                    const UINT64* pFrom = pR0 + ofsFrom;
                    AriocCommon::DumpA21( const_cast<UINT64*>(pFrom), 160 );
#if TRACE_SUBID
                }
#endif

            }
        }

        CDPrint( cdpCD0, __FUNCTION__ );
    }
#endif
    
        // iterate
        iDc += nDc;
        nDremaining -= nDc;
        nIterations++ ;
    }

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->n.Iterations, nIterations );
}
#pragma endregion
