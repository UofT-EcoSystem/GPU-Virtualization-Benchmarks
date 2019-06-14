/*
  baseAlignG.cu

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
static __device__ __constant__ AlignmentScoreParameters     ccASP;
static __device__ __constant__ AlignmentKernelParameters    ccAKP;

/* CUDA shared memory
*/
static __device__ __shared__ INT32 csW[8*8];

//#define DO_SMTRACE 1
#if DO_SMTRACE
static __device__ __shared__ UINT32 csJ;
static __device__ __shared__ INT16 csI;
static __device__ __shared__ INT16 csii;
#endif


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

/// [device] method resetRj
static inline __device__ void resetRj( UINT64& Rj, const UINT64* const __restrict__ pRj, UINT32& ofsRj, const UINT32 j )
{
    ofsRj = (j / 21) * CUDATHREADSPERWARP;  // compute the offset of the 64-bit value that contains the j'th symbol in R

    UINT32 jmod = j % 21;                   // compute the symbol position within the 64-bit value at ofsRj
    if( jmod )
    {
        Rj = pRj[ofsRj] >> (3*(jmod-1));    // read the 64-bit value at ofsRj and shift it so that the next call to getNextRj() will be correct
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
        Rj = pRj[ofsRj];
        ofsRj += CUDATHREADSPERWARP;
    }
}

// typedefs for methods that are called through a function pointer:
typedef void (*pfnResetQi)( UINT64& Qi, const UINT64* pQ0, INT16 i, INT16 cch, const INT16 N );
typedef void (*pfnGetNextQi)( UINT64& Qi, const UINT64* pQ0, INT16 i, INT16 cchOut, INT16 cchIn, const INT16 N );

/// [device] method resetQiF
static inline __device__ void resetQiF( UINT64& Qi, const UINT64* pQ0, INT16 i, INT16 cch, const INT16 )
{
    const UINT64* pQi = pQ0 + ((i/21) * CUDATHREADSPERWARP);    // point to the 64-bit value that contains the i'th symbol in Q
    Qi  = *pQi;                                                 // copy the Q bits for the 21 symbols that contain the i'th symbol
    INT16 shr = 3 * (i%21);
    Qi >>= shr;                                                 // shift the i'th symbol into bits 0-2

    // if there are fewer than cch characters in Qi, get the subsequent 64-bit value in the interleaved Q sequence data
    if( shr > (3 * (21-cch)) )                  
        Qi = (Qi | (pQi[CUDATHREADSPERWARP] << (63-shr))) & MASKA21;
}            

/// [device] method getNextQiF
static inline __device__ void getNextQiF( UINT64& Qi, const UINT64* pQ0, INT16 i, INT16 cchOut, INT16 cchIn, const INT16 )
{
    // shift the current symbols out of Qi
    Qi >>= (3*cchOut);

    // if fewer than cchIn symbols remain in Qi, get some more from the interleaved Q sequence data
    if( Qi < (1 << (3 * (cchIn-1))) )
        resetQiF( Qi, pQ0, i, cchIn, 0 );
}

/// [device] method resetQiRC
static inline __device__ void resetQiRC( UINT64& Qi, const UINT64* pQ0, INT16 i, INT16 cch, const INT16 N )
{
    /* The goal is to build a 64-bit value that contains the specified number of symbols from the reverse complement
        of Q, where i specifies the 0-based index of the first desired symbol in the reverse complement.

       The procedure is:
        - index the first and last required symbols in the Q sequence
        - build a 64-bit value that contains the required substring of the Q sequence
        - shift the substring into the high-order bits of the 64-bit value
        - compute the reverse complement (which leaves the substring in the low-order bits)
    */

    INT16 iFrom = (N - i) - 1;            // 0-based index of first symbol wrt the start of Q (the interleaved Q sequence)

    const UINT64* pQi = pQ0 + ((iFrom/21) * CUDATHREADSPERWARP);    // point to the 64-bit value that contains the iFrom'th symbol
    Qi  = *pQi;                                                     // copy the Q bits for the 21 symbols that contain the iFrom'th symbol

    // shift the first symbol into the high-order bits of Qi
    INT16 mod21From = iFrom % 21;               // position of first symbol wrt Qi 
    INT16 shl = 3 * (20 - mod21From);           // number of bits to shift first symbol into MSB of Qi
    Qi = (Qi << shl);                           // (bit 63 may get clobbered here but it will be zeroed in reverseComplement())

    // optionally append additional symbols from the Q sequence
    INT16 mod21To = mod21From - (cch-1);        // position of last symbol wrt Qi
    if( mod21To < 0 )
    {
        INT16 iTo = (iFrom - cch) + 1;          // 0-based index of last symbol wrt the start of Q
        if( iTo >= 0 )
            Qi |= (pQi[-CUDATHREADSPERWARP] >> (63-shl));  // append symbols
    }

    // compute the reverse complement (which leaves the desired substring in the low-order bits of Qi)
    Qi = reverseComplement( Qi );
}            

/// [device] method getNextQiRC
static inline __device__ void getNextQiRC( UINT64& Qi, const UINT64* pQ0, INT16 i, INT16 cchOut, INT16 cchIn, const INT16 N )
{
    // shift the current symbols out of Qi
    Qi >>= (3*cchOut);

    // if fewer than cchIn symbols remain in Qi, get some more from the interleaved Q sequence data
    if( Qi < (1 << (3 * (cchIn-1))) )
        resetQiRC( Qi, pQ0, i, cchIn, N );
}

/// [device] method computeSMcell
static inline __device__ void computeSMcell(       INT16&           Vmax,   // in, out
                                                   UINT32&          tbo,    // in, out
                                                   UINT32* const &  pSMq,   // in
                                             const UINT32           ofsSMq, // in
                                                   INT16&           Vi,     // in, out
                                                   INT16&           Ei,     // in, out
                                                   INT16&           F,      // in, out
                                             const INT16            Rj,     // in
                                             const INT16            Qi,     // in
                                                   INT16&           Vd,     // in, out
                                             const INT16            Vv      // in
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

#if DO_SMTRACE
    printf( "%d %u %c %c %d\n", csI+csii, csJ, "??NNACGT"[Qi&7], "??NNACGT"[Rj], Vi );
#endif

    // determine the traceback direction
    TracebackDirection32 td32 = (Vi == G) ? ((Wmx > 0) ? td32Dm : td32Dx) :
                                (Vi == F) ? td32V :
                                            td32H;

    /* The SWG recurrence

         F = max( F, V-Wg ) - Ws

       is insensitive to F values less than V-Wg, i.e. F values smaller than V-Wg are effectively the same as V-Wg.  We therefore save bits by
       recording Fdiff = min( Wg, V-F ); we can reconstitute the original F value later by computing V-Fdiff.

       Thus we save the SM cell V value as an unsigned 32-bit value:
        bits  0-15: V
        bits 16-29: Fdiff
        bits 30-31: traceback direction
    */
    INT32 Fdiff = min( ccASP.Wg, Vi-F );
    pSMq[ofsSMq] = td32 | static_cast<UINT32>((Fdiff << 16) | Vi);


#if TODO_CHOP_WHEN_DEBUGGED
    if( ofsSMq >= ccAKP.celSMperQ*CUDATHREADSPERWARP )
        asm( "brkpt;" );
#endif



#if TODO_CHOP_WHEN_DEBUGGED
    if( (threadIdx.x == 134) &&
        (blockIdx.x == 5) && (blockIdx.y == 0) )
    {
        printf( "Vi=%d\n" );
    }
#endif

    /* If the current cell contains the maximum V score, update the traceback origin:
        - The expected maximum Vmax was initially computed by either the windowed gapped aligner or the seed-and-extend aligner.
        - It is possible for the highest-scoring mapping found by the windowed gapped aligner to be "clipped" at one of the ends of the
            scoring-matrix rectangle.  The banded SWG implementation here examines a diagonal "band" of scoring-matrix cells to
            either side of the diagonal in which the highest-scoring cell was found, so it is possible for the band to include cells that
            were clipped by the windowed gapped aligner.  In this case, the banded SWG implementation may find a higher-scoring alignment
            that extends outside the rectangle examined by the windowed gapped aligner.  Empirically this seems to happen less than 1% of
            the time with human 100bp paired-end reads, but the actual frequency should depend on the position and width of the "window"
            in relation to the expected distribution of paired-end fragment lengths.

            For this reason, we track the maximum Vmax actually computed here in the banded SWG scoring matrix, that is, we use the
            previously-computed value of Vmax as a minimum-value filter, but we accept computed V values that exceed that minimum value.

            Of course, these reads should ultimately fall outside the acceptable fragment-length range for concordant pairs, but it's
            better to report them than to attempt to filter them out here.
    */
    if( Vi >= Vmax )
    {
        /* The tbo value is bitmapped as follows:
            bits  0-23: ofsSM (0-based offset from the start of this Q sequence's scoring matrix, without CUDA buffer interleave)
            bits 24-31: nTBO (number of SM cells that contain the maximum V score)
           We clamp the maximum value for nTBO at 0xFF.
        */
        UINT64 b3 = static_cast<UINT64>(tbo & static_cast<UINT32>(0xFF000000)) + 0x01000000;
        tbo = static_cast<UINT32>(min(b3, static_cast<UINT64>(0xFF000000))) | (ofsSMq >> LBCTW);

        // track the maximum value of V
        Vmax = Vi;
    }
}

/// [device] method computeVband
static inline __device__ void computeVband( INT16& Vmax, UINT32& tbo, INT16 imax,
                                            UINT32 Qi8, INT16 Rj, INT16 Vd, UINT32 FVv,
                                            INT16& V0, INT16& V1, INT16& V2, INT16& V3, INT16& V4, INT16& V5, INT16& V6, INT16& V7,
                                            INT16& E0, INT16& E1, INT16& E2, INT16& E3, INT16& E4, INT16& E5, INT16& E6, INT16& E7,
                                            UINT32* const & pSMq, UINT32 ofsSMq, const UINT32 cySM
                                          )
{
    /* Computes a vertical band of between 1 and 8 cells, ending at the specified 0-based offset from the topmost row in the band.


                +---+---+---+---+---+---+---+---+
             0  |   |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+           ComputeVband is used to compute:
             1      |   |   |   |   |   |   |   |
                    +---+---+---+---+---+---+---+               - the triangular tiles at the left-hand edge of each horizontal swath
             2          |   |   |   |   |   |   |                   of scoring-matrix cells
                        +---+---+---+---+---+---+
             3              |   |   |   |   |   |
          i                 +---+---+---+---+---+
             4                  |   |   |   |   |
                                +---+---+---+---+
             5                      |   |   |   |
                                    +---+---+---+
             6                          |   |   |
                                        +---+---+
             7                              |   |
                                            +---+

                                            
                +---+---+---+---+---+---+---+---+
             0  |   |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+
             1  |   |   |   |   |   |   |   |   |             
                +---+---+---+---+---+---+---+---+               - the rectangular tiles in the middle of each horizontal swath of SM cells
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

       If the caller specifies a constant value for imax (the index of the bottommost row to be computed), NVCC suppresses the corresponding
        conditional clauses (i.e., if( imax == ... )).
    */
    INT16 Vv = static_cast<INT16>(FVv);
    INT16 F = Vv - (static_cast<INT16>(FVv >> 16) & 0x3FFF);
    Rj &= 7;

    /* i=0 */
#if DO_SMTRACE
    csii = 0;
#endif
    computeSMcell( Vmax, tbo, pSMq, ofsSMq, V0, E0, F, Rj, Qi8, Vd, Vv );
    if( imax == 0 ) return;

    /* i=1 */
#if DO_SMTRACE
    csii = 1;
#endif
    ofsSMq += cySM;
    computeSMcell( Vmax, tbo, pSMq, ofsSMq, V1, E1, F, Rj, Qi8>>3, Vd, V0 );
    if( imax == 1 ) return;

    /* i=2 */
#if DO_SMTRACE
    csii = 2;
#endif
    ofsSMq += cySM;
    computeSMcell( Vmax, tbo, pSMq, ofsSMq, V2, E2, F, Rj, Qi8>>6, Vd, V1 );
    if( imax == 2 ) return;

    /* i=3 */
#if DO_SMTRACE
    csii = 3;
#endif
    ofsSMq += cySM;
    computeSMcell( Vmax, tbo, pSMq, ofsSMq, V3, E3, F, Rj, Qi8>>9, Vd, V2 );
    if( imax == 3 ) return;

    /* i=4 */
#if DO_SMTRACE
    csii = 4;
#endif
    ofsSMq += cySM;
    computeSMcell( Vmax, tbo, pSMq, ofsSMq, V4, E4, F, Rj, Qi8>>12, Vd, V3 );
    if( imax == 4 ) return;

    /* i=5 */
#if DO_SMTRACE
    csii = 5;
#endif
    ofsSMq += cySM;
    computeSMcell( Vmax, tbo, pSMq, ofsSMq, V5, E5, F, Rj, Qi8>>15, Vd, V4 );
    if( imax == 5 ) return;

    /* i=6 */
#if DO_SMTRACE
    csii = 6;
#endif
    ofsSMq += cySM;
    computeSMcell( Vmax, tbo, pSMq, ofsSMq, V6, E6, F, Rj, Qi8>>18, Vd, V5 );
    if( imax == 6 ) return;

    /* i=7 */
#if DO_SMTRACE
    csii = 7;
#endif
    ofsSMq += cySM;
    computeSMcell( Vmax, tbo, pSMq, ofsSMq, V7, E7, F, Rj, Qi8>>21, Vd, V6 );
}

/// [device] method computeVbandR
static inline __device__ void computeVbandR( INT16& Vmax, UINT32& tbo, INT16 imin, INT16 imax,
                                             UINT32 Qi8, INT16 Rj, INT16 Vd,
                                             INT16& V0, INT16& V1, INT16& V2, INT16& V3, INT16& V4, INT16& V5, INT16& V6, INT16& V7,
                                             INT16& E0, INT16& E1, INT16& E2, INT16& E3, INT16& E4, INT16& E5, INT16& E6, INT16& E7,
                                             UINT32* const & pSMq, UINT32 ofsSMq, const UINT32 cySM
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
#if DO_SMTRACE
    csii = 0;
#endif
            computeSMcell( Vmax, tbo, pSMq, ofsSMq, V0, E0, F, Rj, Qi8, Vd, Vv );
            if( imax == 0 ) return;
            Vv = V0;
            ofsSMq += cySM;

        case 1:
#if DO_SMTRACE
    csii = 1;
#endif
            computeSMcell( Vmax, tbo, pSMq, ofsSMq, V1, E1, F, Rj, Qi8>>3, Vd, Vv );
            if( imax == 1 ) return;
            Vv = V1;
            ofsSMq += cySM;

        case 2:
#if DO_SMTRACE
    csii = 2;
#endif
            computeSMcell( Vmax, tbo, pSMq, ofsSMq, V2, E2, F, Rj, Qi8>>6, Vd, Vv );
            if( imax == 2 ) return;
            Vv = V2;
            ofsSMq += cySM;

        case 3:
#if DO_SMTRACE
    csii = 3;
#endif
            computeSMcell( Vmax, tbo, pSMq, ofsSMq, V3, E3, F, Rj, Qi8>>9, Vd, Vv );
            if( imax == 3 ) return;
            Vv = V3;
            ofsSMq += cySM;

        case 4:
#if DO_SMTRACE
    csii = 4;
#endif
            computeSMcell( Vmax, tbo, pSMq, ofsSMq, V4, E4, F, Rj, Qi8>>12, Vd, Vv );
            if( imax == 4 ) return;
            Vv = V4;
            ofsSMq += cySM;

        case 5:
#if DO_SMTRACE
    csii = 5;
#endif
            computeSMcell( Vmax, tbo, pSMq, ofsSMq, V5, E5, F, Rj, Qi8>>15, Vd, Vv );
            if( imax == 5 ) return;
            Vv = V5;
            ofsSMq += cySM;

        case 6:
#if DO_SMTRACE
    csii = 6;
#endif
            computeSMcell( Vmax, tbo, pSMq, ofsSMq, V6, E6, F, Rj, Qi8>>18, Vd, Vv );
            if( imax == 6 ) return;
            Vv = V6;
            ofsSMq += cySM;

        case 7:
#if DO_SMTRACE
    csii = 7;
#endif
            computeSMcell( Vmax, tbo, pSMq, ofsSMq, V7, E7, F, Rj, Qi8>>21, Vd, Vv );
            return;

        default:
            break;
    }
    
}

/// [device] method computeSMnarrow
static __device__ UINT32 computeSMnarrow(       INT16                      Vmax,  // (not constant -- may be modified in computeSMcell)
                                          const UINT64* const __restrict__ pQ0,
                                          const UINT64* const __restrict__ pRj,
                                          const INT16                      N,
                                                UINT32*                    pSMq,
                                                pfnResetQi                 resetQi,
                                                pfnGetNextQi               getNextQi )
{
    // compute the limiting offset into the scoring-matrix buffer for the current CUDA thread
    const UINT32 cySM = (ccAKP.bw-1) * CUDATHREADSPERWARP;

    // get the first 64-bit value from the Q sequence
    UINT64 Qi;
    resetQi( Qi, pQ0, 0, ccAKP.bw, N );

    // initialize a value to contain the traceback origin for the highest-scoring cell
    UINT32 tbo = 0;

    // compute the scoring matrix in vertical bands that are at most 7 cells high
    INT16 V0=0, V1=0, V2=0, V3=0, V4=0, V5=0, V6=0, V7=0;
    INT16 E0=0, E1=0, E2=0, E3=0, E4=0, E5=0, E6=0, E7=0;

    /* upper-left triangle */
    UINT64 Rj;
    UINT32 ofsRj;
    resetRj( Rj, pRj, ofsRj, 0 );

    UINT32 j = 0;       // j is the 0-based offset into the scoring matrix (NOT the entire R sequence)
    UINT32 ofsSMq = 0;  // 0-based offset into the interleaved scoring-matrix buffer
    for( ; j<ccAKP.bw; j++ )
    {
        ofsSMq = j * CUDATHREADSPERWARP;
        getNextRj( Rj, pRj, ofsRj );
        computeVband( Vmax, tbo, j, Qi, Rj, 0, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
    }

    /* interior vertical bands */
    INT16 Vd;
    INT16 i = 0;
    for( ; j<N; ++j )
    {
        ++i;
        ofsSMq += (cySM+CUDATHREADSPERWARP);
        Vd = V0;
        V0 = V1; E0 = E1;
        V1 = V2; E1 = E2;
        V2 = V3; E2 = E3;
        V3 = V4; E3 = E4;
        V4 = V5; E4 = E5;
        V5 = V6; E5 = E6;
        V6 = V7; E6 = E7;
        getNextRj( Rj, pRj, ofsRj );
        getNextQi( Qi, pQ0, i, 1, ccAKP.bw, N );
        computeVband( Vmax, tbo, ccAKP.bw-1, Qi, Rj, Vd, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
    }

    /* lower right triangle */
    const INT16 jLimit = N + ccAKP.bw - 1;
    INT16 imax = ccAKP.bw - 2;
    for( ; j<jLimit; ++j )
    {
        ++i;
        ofsSMq += (cySM+CUDATHREADSPERWARP);
        Vd = V0;
        V0 = V1; E0 = E1;
        V1 = V2; E1 = E2;
        V2 = V3; E2 = E3;
        V3 = V4; E3 = E4;
        V4 = V5; E4 = E5;
        V5 = V6; E5 = E6;
        V6 = V7; E6 = E7;
        getNextRj( Rj, pRj, ofsRj );
        getNextQi( Qi, pQ0, i, 1, ccAKP.bw, N );
        computeVband( Vmax, tbo, imax, Qi, Rj, Vd, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
        --imax;
    }

    return tbo;
}

/// [device] method computeSMwide
static __device__ UINT32 computeSMwide(       INT16                      Vmax,  // (not constant -- may be modified in computeSMcell)
                                        const UINT64* const __restrict__ pQ0,
                                        const UINT64* const __restrict__ pRj,
                                        const INT16                      N,
                                              UINT32*                    pSMq,
                                              pfnResetQi                 resetQi,
                                              pfnGetNextQi               getNextQi )
{
    // compute the limiting offset into the scoring-matrix buffer for the current CUDA thread
    const UINT32 cySM = (ccAKP.bw-1) * CUDATHREADSPERWARP;

    // get the first 64-bit value from the Q sequence
    UINT64 Qi;
    resetQi( Qi, pQ0, 0, 8, N );

    // initialize a value to contain the traceback origin for the highest-scoring cell
    UINT32 tbo = 0;

    // compute the scoring matrix in horizontal swaths that are 8 cells high
    INT16 V0=0, V1=0, V2=0, V3=0, V4=0, V5=0, V6=0, V7=0;
    INT16 E0=0, E1=0, E2=0, E3=0, E4=0, E5=0, E6=0, E7=0;

    /* topmost swath */
#if DO_SMTRACE
    csI = 0;
#endif
    UINT64 Rj;
    UINT32 ofsRj;
    resetRj( Rj, pRj, ofsRj, 0 );

    // left-hand triangular tile in the swath
    UINT32 j = 0;       // j is the 0-based offset into the scoring matrix (NOT the entire R sequence)
    UINT32 ofsSMq = 0;  // 0-based offset into the interleaved scoring-matrix buffer
    for( ; j<7; j++ )
    {
#if DO_SMTRACE
    csJ = j;
#endif
        getNextRj( Rj, pRj, ofsRj );
        computeVband( Vmax, tbo, j, Qi, Rj, 0, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
        ofsSMq += CUDATHREADSPERWARP;
    }
        
    // rectangular tiles in the swath
    for( ; j<ccAKP.bw; j++ )
    {
#if DO_SMTRACE
        csJ = j;
#endif
        getNextRj( Rj, pRj, ofsRj );
        computeVband( Vmax, tbo, 7, Qi, Rj, 0, 0, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
        ofsSMq += CUDATHREADSPERWARP;
    }

    // the diagonal neighbor for the topmost cell in the first vertical band in the right-side triangular tile is the topmost cell in the most recently filled vertical band
    INT16 Vd = static_cast<INT16>(pSMq[ofsSMq-CUDATHREADSPERWARP]);

    // adjust the SM offset to reference the topmost cell in the next vertical band
    ofsSMq += cySM;

    // right-side triangular tile in the swath
    for( ; j<(ccAKP.bw+7); j++ )
    {
        ///* TODO: CHOP WHEN DEBUGGED */  csJ = j;
        getNextRj( Rj, pRj, ofsRj );
        computeVbandR( Vmax, tbo, (j-(ccAKP.bw-1)), 7, Qi, Rj, Vd, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
        ofsSMq += (ccAKP.bw * CUDATHREADSPERWARP);
    }

    /* interior swaths */
    INT16 i = 8;                        // i is a 0-based offset into the Q sequence
    while( i < (N-8) )
    {
#if DO_SMTRACE
        csI = i;
#endif
        j = i;                          // each swath starts at the 0th diagonal of the scoring matrix
        resetRj( Rj, pRj, ofsRj, j );
        getNextQi( Qi, pQ0, i, 8, 8, N );

        // left-side triangular tile in the swath
        V0 = V1 = V2 = V3 = V4 = V5 = V6 = V7 = 0;
        E0 = E1 = E2 = E3 = E4 = E5 = E6 = E7 = 0;

        // set the SM offset to the first cell in the i'th row
        ofsSMq = (i * ccAKP.bw * CUDATHREADSPERWARP);

        // get the V value for the diagonal neighbor
        INT16 Vd = static_cast<INT16>(pSMq[ofsSMq-(cySM+CUDATHREADSPERWARP)]);

        for( ; j<(i+7); ++j )
        {
#if DO_SMTRACE
            csJ = j;
#endif
            UINT32 FVv = pSMq[ofsSMq-cySM];
            getNextRj( Rj, pRj, ofsRj );
            computeVband( Vmax, tbo, j-i, Qi, Rj, Vd, FVv, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
            Vd = static_cast<INT16>(FVv);
            ofsSMq += CUDATHREADSPERWARP;
        }

        // rectangular tiles in the swath
        for( ; j<(i+ccAKP.bw-1); j++ )
        {
#if DO_SMTRACE
            csJ = j;
#endif
            UINT32 FVv = pSMq[ofsSMq-cySM];
            getNextRj( Rj, pRj, ofsRj );
            computeVband( Vmax, tbo, 8, Qi, Rj, Vd, FVv, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
            Vd = static_cast<INT16>(FVv);
            ofsSMq += CUDATHREADSPERWARP;
        }

        // the diagonal neighbor for the first vertical band in the right-side triangular tile is the rightmost cell in the row above themost recently filled vertical band
        Vd = static_cast<INT16>(pSMq[ofsSMq-(ccAKP.bw*CUDATHREADSPERWARP)]);

        // right-side triangular tile in the swath
        for( ; j<(i+ccAKP.bw+7); j++ )
        {
#if DO_SMTRACE
            csJ = j;
#endif
            getNextRj( Rj, pRj, ofsRj );
            computeVbandR( Vmax, tbo, j-(i+ccAKP.bw-1), 7, Qi, Rj, Vd, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
            ofsSMq += (ccAKP.bw * CUDATHREADSPERWARP);
        }

        // advance to the next swath
        i += 8;
    }

    /* bottommost swath */
#if DO_SMTRACE
    csI = i;
#endif
    j = i;                          // the swath starts at the 0th diagonal of the scoring matrix
    resetRj( Rj, pRj, ofsRj, j );
    getNextQi( Qi, pQ0, i, 8, N-i, N );

    // left-side triangular tile in the swath
    V0 = V1 = V2 = V3 = V4 = V5 = V6 = V7 = 0;
    E0 = E1 = E2 = E3 = E4 = E5 = E6 = E7 = 0;

    // set the SM offset to the first cell in the i'th row
    ofsSMq = (i * ccAKP.bw * CUDATHREADSPERWARP);

    // get the V value for the diagonal neighbor
    Vd = static_cast<INT16>(pSMq[ofsSMq-(cySM+CUDATHREADSPERWARP)]);

    INT16 jTail = (N-i) - 1;

    for( ; j<(i+jTail); j++ )
    {
#if DO_SMTRACE
        csJ = j;
#endif
        UINT32 FVv = pSMq[ofsSMq-cySM];
        getNextRj( Rj, pRj, ofsRj );
        computeVband( Vmax, tbo, j-i, Qi, Rj, Vd, FVv, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
        Vd = static_cast<INT16>(FVv);
        ofsSMq += CUDATHREADSPERWARP;
    }

    // rectangular tiles in the swath
#if DO_SMTRACE
    csJ = j;
#endif
    for( ; j<(i+ccAKP.bw-1); j++ )
    {
        UINT32 FVv = pSMq[ofsSMq-cySM];
        getNextRj( Rj, pRj, ofsRj );
        computeVband( Vmax, tbo, (N-i)-1, Qi, Rj, Vd, FVv, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
        Vd = static_cast<INT16>(FVv);
        ofsSMq += CUDATHREADSPERWARP;
    }

    // the diagonal neighbor for the first vertical band in the right-side triangular tile is the rightmost cell in the row above themost recently filled vertical band
    Vd = static_cast<INT16>(pSMq[ofsSMq-(ccAKP.bw*CUDATHREADSPERWARP)]);

    // right-side triangular tile in the swath
    for( ; j<(i+ccAKP.bw+jTail); j++ )
    {
///* TODO: CHOP WHEN DEBUGGED */  csJ = j;
        getNextRj( Rj, pRj, ofsRj );
        computeVbandR( Vmax, tbo, j-(i+ccAKP.bw-1), jTail, Qi, Rj, Vd, V0, V1, V2, V3, V4, V5, V6, V7, E0, E1, E2, E3, E4, E5, E6, E7, pSMq, ofsSMq, cySM );
        ofsSMq += (ccAKP.bw * CUDATHREADSPERWARP);
    }



    // TODO: CHOP WHEN DEBUGGED
    //if( tbo == 0 )              // this should never happen
    //    asm( "brkpt;" );





    // return the traceback origin
    return tbo;
}

/// [device] method initializeSharedMemory
static __device__ void initializeSharedMemory()
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

/// baseAlignG_Kernel1
static __global__  void baseAlignG_Kernel1( const UINT64* const __restrict__ pRiBuffer,     // in: interleaved R sequence data
                                            const UINT64* const __restrict__ pQiBuffer,     // in: interleaved Q sequence data
                                            const Qwarp*  const __restrict__ pQwBuffer,     // in: Qwarps
                                            const UINT64* const __restrict__ pDmBuffer,     // in: D values
                                            const UINT32                     iDmBuffer,     // in: index into Dm, Ri, Vmax, and BRLEA buffers
                                            const UINT32                     nDm,           // in: number of candidates
                                            const INT16*  const __restrict__ pVmaxBuffer,   // in: Vmax values
                                                  UINT32* const              pSMBuffer,     // out: scoring-matrix buffer
                                                  UINT32* const              pBRLEABuffer   // out: BRLEA buffer
                                          )
{
    // initialize shared memory
    initializeSharedMemory();

    // compute the 0-based index of the current thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nDm )
        return;

    // compute the 0-based index into the buffers for the current thread
    UINT32 iDm = iDmBuffer + tid;

    /* Load the D value for the current CUDA thread:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
            UINT64  flags :  3;     // bits 61..63: flags
    */
    UINT64 Dm = pDmBuffer[iDm];

    // load the previously-computed Vmax for the current thread
    const INT16 Vmax = pVmaxBuffer[iDm];



#if TODO_CHOP_WHEN_DEBUGGED
    INT16 Vmax = pVmaxBuffer[iDm];

    if( Dm == 0x801ffa10855aec65 )
    {
        asm( "brkpt;" );
        Vmax = 100;
    }
#endif




    
    // unpack the QID
    UINT32 qid = static_cast<UINT32>(Dm >> 39) & AriocDS::QID::maskQID;
    UINT32 iw = QID_IW(qid);
    UINT32 iq = QID_IQ(qid);
    const Qwarp* pQw = pQwBuffer + iw;

    // get the length of the Q sequence
    const INT16 N = pQw->N[iq];

    // point to the interleaved Q sequence data for the current thread
    const UINT64* const __restrict__ pQ0 = pQiBuffer + pQw->ofsQi + iq;

    // point to the interleaved R sequence data for the current thread
    INT64 celMr = blockdiv(ccAKP.Mr,21);    // number of 64-bit values required to represent Mr
    celMr *= CUDATHREADSPERWARP;            // number of 64-bit values per warp
    UINT32 ixw = iDm >> 5;                  // "warp index" (within Ri buffer)
    INT16 ixq = iDm & 0x1F;                 // "thread index" within warp (within Ri buffer)
    const UINT64* pR0 = pRiBuffer + (ixw * celMr) + ixq;

    // point to the SM buffer for the current thread
    size_t isw = tid >> 5;
    UINT32 isq = tid & 0x1F;
    UINT32* pSMq = pSMBuffer + (isw*CUDATHREADSPERWARP*ccAKP.celSMperQ) + isq;

    // initialize the BRLEA buffer for the current Q sequence
    UINT32* pBH = pBRLEABuffer + (iDm*ccAKP.celBRLEAperQ);
    pBH[0] = qid | ((Dm & AriocDS::D::flagRCq) ? AriocDS::QID::maskRC : 0); // bit 31 is set if the reverse complement of the Q sequence is mapped

    // forward or reverse-complement Q sequence
    pfnResetQi resetQi = (Dm & AriocDS::D::flagRCq) ? resetQiRC : resetQiF;
    pfnGetNextQi getNextQi = (Dm & AriocDS::D::flagRCq) ? getNextQiRC : getNextQiF;

    // compute this thread's gapped alignment and save the traceback origin in the second 32 bits of the corresponding BRLEA buffer
    if( ccAKP.bw >= 8 )
        pBH[1] = computeSMwide( Vmax, pQ0, pR0, N, pSMq, resetQi, getNextQi );
    else
        pBH[1] = computeSMnarrow( Vmax, pQ0, pR0, N, pSMq, resetQi, getNextQi );
}

/// [device] method prependRunLength
static inline __device__ void prependRunLength( UINT8*& pB, BRLEAbyteType bbType, INT16 cchRun )
{
    /* BRLEA (binary run length encoded alignment) encoding is as follows:

       Terminology:
                    description                                             direction in SWG scoring matrix
        match       a pair of matching symbols (i.e. the same nucleotide)   diagonal
        mismatch    a pair of mismatched symbols                            diagonal
        gapQ        one or more spaces in the Q sequence                    horizontal (each space in Q aligns with one symbol in R)
        gapR        one or more spaces in the R sequence                    vertical (each space in R aligns with one symbol in Q)

       The BRLEA (binary run-length encoded alignment) string consists of a series of 8-bit unsigned values formatted as follows:

            bit   7   6    5   4   3   2   1   0
                  0   0    <--   run length  -->   matches
                  0   1    <--   run length  -->   gap in Q (deletion from R)
                  1   0    <--   run length  -->   mismatches
                  1   1    <--   run length  -->   gap in R (insert Q into R)

       Run length is encoded in big-endian 6-bit chunks (e.g. a run length of 256 matches (binary 100000000) is encoded
        as 0x04 0x00 (00 000100 followed by 00 000000)).  There is no end-of-run marker; instead, the end of a run is indicated
        by encountering a differently encoded element (i.e., bits 6-7 in byte n+1 differ from bits 6-7 in byte n) or by reaching
        the end of the BRLEA.

       BRLEA strings do not encode soft clipping; see the Il and Ir fields of the BRLEAheader struct.
    */
    while( cchRun )
    {
        // prepend a new BRLEA byte
        --pB;                                       // decrement the pointer to the nascent BRLEA string
        *pB = packBRLEAbyte(bbType, cchRun & 0x3F); // save the BRLEA byte type and the low-order 6 bits of the run length
        cchRun >>= 6;                               // shift the high-order bits of the run length into the 6 low-order bits
    }
}

/// [device] method buildRunDm
static inline __device__ void buildRunDm( UINT32& V, UINT8*& pB, UINT32& ofsSM, INT16& i, INT16& j, const INT16 bw, const UINT32* const __restrict__ pSMq )
{
    // iterate along the diagonal until the start of the run
    INT16 i0 = i;
    do
    {
        // move to the diagonal neighbor
        --j;
        ofsSM = (--i >= 0) ? (ofsSM-bw) : _UI32_MAX;

        // get the next V value
        V = (ofsSM != _UI32_MAX) ? pSMq[ofsSM*CUDATHREADSPERWARP] : 0;
    }
    while( ((V&0xC0000000) == td32Dm) && (static_cast<INT16>(V) > 0) );

    // emit the run length
    prependRunLength( pB, bbMatch, i0-i );
}

/// [device] method buildRunDx
static inline __device__ void buildRunDx( UINT32& V, UINT8*& pB, UINT32& ofsSM, INT16& i, INT16& j, const INT16 bw, const UINT32* const __restrict__ pSMq )
{
    // iterate along the diagonal until the start of the run
    INT16 i0 = i;
    do
    {
        // move to the diagonal neighbor
        --j;
        ofsSM = (--i >= 0) ? (ofsSM-bw) : _UI32_MAX;

        // get the next V value
        V = (ofsSM != _UI32_MAX) ? pSMq[ofsSM*CUDATHREADSPERWARP] : 0;
    }
    while( ((V&0xC0000000) == td32Dx) && (static_cast<INT16>(V) > 0) );

    // emit the run length
    prependRunLength( pB, bbMismatch, i0-i );
}

/// [device] method buildRunV
static inline __device__ void buildRunV( UINT32& V, UINT8*& pB, UINT32& ofsSM, INT16& i, INT16& j, const INT16 bw, const UINT32* const __restrict__ pSMq )
{
    // initialize the V score that is expected at the start of the gap
    INT16 Vg = static_cast<INT16>(V) + ccASP.Wg;

    // iterate vertically until the start of the run (a vertical run corresponds to a gap in R)
    INT16 i0 = i;
    do
    {
        // move to the vertical neighbor
        ofsSM = (--i >= 0) ? (ofsSM-(bw-1)) : _UI32_MAX;

        // get the next V value
        V = (ofsSM != _UI32_MAX) ? pSMq[ofsSM*CUDATHREADSPERWARP] : 0;

        // increment the expected starting V score
        Vg += ccASP.Ws;
    }
    while( (Vg > static_cast<INT16>(V)) && (static_cast<INT16>(V) > 0) );

    // emit the run length
    prependRunLength( pB, bbGapR, i0-i );
}

/// [device] method buildRunH
static inline __device__ void buildRunH( UINT32& V, UINT8*& pB, UINT32& ofsSM, INT16& i, INT16& j, const INT16 bw, const UINT32* const __restrict__ pSMq )
{
    // initialize the V score that is expected at the start of the gap
    INT16 Vg = static_cast<INT16>(V) + ccASP.Wg;

    // iterate horizontally until the start of the run (a horizontal run corresponds to a gap in Q)
    INT16 j0 = j;
    do
    {
        // move to the horizontal neighbor
        ofsSM = (--j >= i) ? (ofsSM-1) : _UI32_MAX;

        // get the next V value
        V = (ofsSM != _UI32_MAX) ? pSMq[ofsSM*CUDATHREADSPERWARP] : 0;

        // increment the expected starting V score
        Vg += ccASP.Ws;
    }
    while( (Vg > static_cast<INT16>(V)) && (static_cast<INT16>(V) > 0) );

    // emit the run length
    prependRunLength( pB, bbGapQ, j0-j );
}

/* kernel function reportAlignment
*/
static inline __device__ void reportAlignment( BRLEAheader* pBH, const INT16 celBRLEAq, const UINT32 tbo, const UINT64 Dm, const INT16 wcgsc, const INT16 N, const UINT32* const __restrict__ pSMq )
{
    // get the SM cell value at the traceback origin
    UINT32 ofsSM = tbo & 0x00FFFFFF;            // get the 0-based noninterleaved offset of the tbo cell
    UINT32 V = pSMq[ofsSM*CUDATHREADSPERWARP];

    // if there is no mapping, there is no BRLEA
    if( static_cast<INT16>(V) == 0 )
        return;

    /* Initialize the BRLEAheader for the current alignment; we extract the needed values from the specified D value which, as always, looks like this:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
            UINT64  flags :  3;     // bits 61..63: flags

       The number of cells in the scoring matrix that contained the same Vmax is stored in the high-order byte of the traceback-origin value.
    */
    pBH->V = static_cast<INT16>(V);
    pBH->subId = static_cast<UINT8>(Dm >> 32) & 0x7F;
    pBH->nTBO = tbo >> 24;

    // get the width of the diagonal band of cells to be computed
    INT16 bw = ccAKP.bw;

    /* Compute the initial i and j (0-based coordinates of the traceback origin relative to the start of the alignment):

        - Each row of the scoring matrix contains only the cells in the diagonal band for the alignment; the number of cells in the row is specified by bw.

                                j 
              0   1   2   3   4   5   6   7   8   9   
            +---+---+---+---+---+---+---+---+---+---+
         0  |   |   |   |   |   |   |   |   |   |   |
            +---+---+---+---+---+---+---+---+---+---+---+
         1      |   |   |   |   |   |   |   |   |   |   |
      i         +---+---+---+---+---+---+---+---+---+---+---+
         2          |   |   |   |   |   |   |   |   |   |   |
                    +---+---+---+---+---+---+---+---+---+---+---+
         3              |   |   |   |   |   |   |   |   |   |   |
                        +---+---+---+---+---+---+---+---+---+---+


        - Cells are CUDA-interleaved, so successive cells are CUDATHREADSPERWARP apart.
        - The first cell in each row of the diagonal band corresponds to the row number (i.e., j=i).

       Disregarding CUDA interleave, the traceback neighbors of the cell (i, j) at 0-based offset ofsSM are:
        tdH (i, j-1):   ((j-1) >= i) ? ofsSM-1 : 0
        tdD (i-1, j-1): ((i-1) >= 0) ? ofsSM-bw : 0
        tdV (i-1, j):   ((i-1) >= 0) ? (ofsSM-(bw-1)) : 0    (assumes that a mapping cannot have a vertical traceback from the rightmost cell in the diagonal band)

       The traceback neighbors are computed in the "buildRun*" functions.
    */
    INT16 i = ofsSM / bw;
    INT16 j = (ofsSM % bw) + i;

    // save the offsets of the rightmost aligned symbol
    pBH->Ir = i;
    INT16 jr = j;

    // initialize the end-of-BRLEA pointer
    UINT8* pB = reinterpret_cast<UINT8*>(reinterpret_cast<UINT32*>(pBH) + celBRLEAq);

    // traverse the scoring matrix until the V score is zero
    do
    {
        switch( static_cast<TracebackDirection32>(V&0xC0000000) )
        {
            case td32Dm:    // 00: diagonal (match)
                buildRunDm( V, pB, ofsSM, i, j, bw, pSMq );
                break;

            case td32Dx:    // 01: diagonal (mismatch)
                buildRunDx( V, pB, ofsSM, i, j, bw, pSMq );
                break;
                
            case td32H:     // 10: horizontal
                buildRunH( V, pB, ofsSM, i, j, bw, pSMq );
                break;
            
            default:        // 11: vertical (td32V)
                buildRunV( V, pB, ofsSM, i, j, bw, pSMq );
                break;
        }

        // get the next V value
        V = (ofsSM != _UI32_MAX) ? pSMq[ofsSM*CUDATHREADSPERWARP] : 0;
    }
    while( static_cast<INT16>(V) > 0 );

    /* at this point i and j reference the diagonal neighbor of the first cell in the alignment */
    
    // get the 0-based offset of the start of the R sequence for the current alignment
    INT32 d = static_cast<INT32>(Dm);
    INT32 J0 = d - wcgsc;       // J0 is the 0-based offset into the R sequence (i.e. the 0-based diagonal number)

    // we need to add 1 to the i and j values to reference the first cell in the alignment
    pBH->Il = i + 1;
    pBH->J = J0 + j + 1;

    // compute the number of R symbols spanned by the alignment
    pBH->Ma = jr - j;

    // do some pointer arithmetic to compute and save the number of bytes in the BRLEA
    pBH->cb = reinterpret_cast<UINT8*>(reinterpret_cast<UINT32*>(pBH) + celBRLEAq) - pB;
}

/* kernel baseAlignG_Kernel2
*/
static __global__ void baseAlignG_Kernel2( const Qwarp*  const __restrict__ pQwBuffer,      // in: Qwarps
                                           const UINT64* const __restrict__ pDmBuffer,      // in: list of candidate Dvalues
                                           const UINT32                     iDmBuffer,      // in: index into Dm and BRLEA buffers
                                           const UINT32                     nDm,            // in: number of candidates
                                           const UINT32* const __restrict__ pSMBuffer,      // in: scoring-matrix buffer
                                                 UINT32* const              pBRLEABuffer    // in, out: BRLEA buffer
                                         ) 
{
    // compute the 0-based index of the current thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nDm )
        return;

    // compute the 0-based index into the buffers for the current thread
    UINT32 iDm = iDmBuffer + tid;

    /* Load the D value for the current CUDA thread:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
            UINT64  flags :  3;     // bits 61..63: flags
    */
    UINT64 Dm = pDmBuffer[iDm];


#if TODO_CHOP_WHEN_DEBUGGED
    if( Dm != 0x8005d40902279f01 )
        return;
#endif


    // unpack the QID
    UINT32 qid = static_cast<UINT32>(Dm >> 39) & AriocDS::QID::maskQID;
    UINT32 iw = QID_IW(qid);
    UINT32 iq = QID_IQ(qid);
    const Qwarp* pQw = pQwBuffer + iw;

    // get the length of the Q sequence
    INT16 N =  pQw->N[iq];

    // point to the SM buffer for the current thread
    size_t isw = tid >> 5;
    UINT32 isq = tid & 0x1F;
    const UINT32* pSMq = pSMBuffer + (isw*CUDATHREADSPERWARP*ccAKP.celSMperQ) + isq;

    // point to the BRLEA buffer for the current thread
    BRLEAheader* pBH = reinterpret_cast<BRLEAheader*>(pBRLEABuffer + (iDm*ccAKP.celBRLEAperQ));

    // get the traceback origin from the second 32-bit value in the BRLEA buffer
    UINT32 tbo = *(reinterpret_cast<UINT32*>(pBH) + 1);

    // report the BRLEA for the current alignment
    reportAlignment( pBH, ccAKP.celBRLEAperQ, tbo, Dm, ccAKP.wcgsc, N, pSMq );
}
#pragma endregion

#pragma region private methods
/// [protected] method initConstantMemory
void baseAlignG::initConstantMemory()
{
    CRVALIDATOR;

    // copy parameters into CUDA constant memory
    CRVALIDATE = cudaMemcpyToSymbol( ccASP, &m_pab->aas.ASP, sizeof(AlignmentScoreParameters), 0, cudaMemcpyHostToDevice );
    CRVALIDATE = cudaMemcpyToSymbol( ccAKP, &m_pdbg->AKP, sizeof(AlignmentKernelParameters), 0, cudaMemcpyHostToDevice );
}

/// [protected] method initSharedMemory
UINT32 baseAlignG::initSharedMemory()
{
    CRVALIDATOR;
    
    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseAlignG_Kernel1, cudaFuncCachePreferL1 );
    CRVALIDATE = cudaFuncSetCacheConfig( baseAlignG_Kernel2, cudaFuncCachePreferL1 );

    return 0;
}

/// [protected] method launchKernel
void baseAlignG::launchKernel( UINT32 cbSharedPerBlock )
{
#if TODO_CHOP_IF_UNUSED
    CRVALIDATOR;
#endif


    // performance metrics
    UINT32 nIterations = 0;
    InterlockedIncrement( &m_ptum->n.Instances );
    InterlockedExchangeAdd( &m_ptum->ms.PreLaunch, m_hrt.GetElapsed(true) );


#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Rix(m_pqb->DB.Ri.Count, false);
    m_pqb->DB.Ri.CopyToHost( Rix.p, Rix.Count );


    WinGlobalPtr<INT16> Vmaxxx( m_nD, false );
    cudaMemcpy( Vmaxxx.p, m_pVmax, m_nD*sizeof(INT16), cudaMemcpyDeviceToHost );

    WinGlobalPtr<UINT32> BRLEAxxx( m_pdbg->BRLEA.Count, false );
    m_pdbg->BRLEA.CopyToHost( BRLEAxxx.p, BRLEAxxx.Count );

    WinGlobalPtr<UINT64> Dmxxx( m_nD, false );
    cudaMemcpy( Dmxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

    for( UINT32 n=0; n<100; ++n )
    {
        UINT64 Dm = Dmxxx.p[n];
        UINT32 qid = static_cast<UINT32>(Dm >> 39) & AriocDS::QID::maskQID;
        CDPrint( cdpCD0, "baseAlignG::launchKernel: %4u: Dm=0x%016llx qid=0x%08x", n, Dm, qid );
    }
#endif




    UINT32 iDm = 0;
    UINT32 nRemaining = m_nD;
    while( nRemaining )
    {
        // compute the number of candidates to align in the current iteration
        UINT32 nDm = min( m_nDperIteration, nRemaining );

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: before Kernel1: iDm=%u nDm=%u nRemaining=%u", m_pqb->pgi->deviceId, __FUNCTION__, m_ptum->Key, iDm, nDm, nRemaining );
#endif


#if TODO_CHOP_WHEN_DEBUGGED
        WinGlobalPtr<INT16> Vmaxxx( m_nD, false );
        cudaMemcpy( Vmaxxx.p, m_pVmax, m_nD*sizeof(INT16), cudaMemcpyDeviceToHost );

        // validate Vmax
        if( true )
        {
            INT16 Vp = m_pqb->Nmax * m_pab->aas.ASP.Wm;
            INT16 Vt = Vp - 100;

            for( UINT32 n=0; n<static_cast<UINT32>(Vmaxxx.Count); ++n )
            {
                INT16 V = Vmaxxx.p[n];
                if( (V > Vp) || (V < Vt) )
                    CDPrint( cdpCD0, "%s before kernel1: Vmax=%d (0x%04x) at n=%u", __FUNCTION__, V, V, n );
            }
        }
#endif



#if TRACE_SQID
    WinGlobalPtr<INT16> Vmaxxx( m_nD, false );
    cudaMemcpy( Vmaxxx.p, m_pVmax, m_nD*sizeof(INT16), cudaMemcpyDeviceToHost );

    // copy the D values to a host buffer
    WinGlobalPtr<UINT64> Dxxx( m_nD, false );
    cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

    // look at the D data
    bool sqIdFound = false;
    for( UINT32 n=iDm; n<(iDm+nDm); ++n )
    {
        UINT64 D = Dxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( D );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
            
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>(D >> 32) & 0x007F;
            INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jo = static_cast<UINT32>(m_pab->M.p[subId] - 1) - J;
#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "[%d] %s::%s: before kernel 1: %3d: D=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d Jo=%d",
                                m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                n, D, sqId, qid, subId, J, Jo );

            CDPrint(cdpCD0, "[%d] %s::%s: before kernel 1: Vmax[%d]=%d",
                                m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                n, Vmaxxx.p[n] );
            sqIdFound = true;
        }
    }
#endif



        dim3 d3b;
        dim3 d3g;
        computeGridDimensions( d3g, d3b, nDm );

        // launch the aligner kernel
        baseAlignG_Kernel1<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DB.Ri.p,   // in: interleaved R sequence data
                                                              m_pqb->DB.Qi.p,   // in: interleaved Q sequence data
                                                              m_pqb->DB.Qw.p,   // in: Qwarps
                                                              m_pD,             // in: Dm list (D values for mapped candidates)
                                                              iDm,              // in: index of the first candidate in the current iteration
                                                              nDm,              // in: number of candidates in the current iteration
                                                              m_pVmax,          // in: list of candidate Vmax values
                                                              m_SM.p,           // out: scoring-matrix buffer
                                                              m_pdbg->BRLEA.p   // out: BRLEA buffer
                                                            );


#if TODO_CHOP_WHEN_DEBUGGED
        // look at the BRLEA data (qid,TBO)
        WinGlobalPtr<UINT32> Bbefore( m_pdbg->BRLEA.n, false );
        m_pdbg->BRLEA.CopyToHost( Bbefore.p, Bbefore.Count );

        for( UINT32 n=iDm; n<(iDm+nDm); ++n )
        {
            BRLEAheader* pBH = reinterpret_cast<BRLEAheader*>(Bbefore.p + (n * m_pqb->celBRLEAperQ));
            UINT32 iw = QID_IW( pBH->qid );
            INT16 iq = QID_IQ( pBH->qid );
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];

            UINT32 tbo = reinterpret_cast<UINT32*>(pBH)[1];
            //if( (tbo == 0) /*|| (n == (iDm+nDm-1)) */ )
            if( sqId == 0x00000808973166f0 )
            {
                CDPrint( cdpCD0, "[%d] %s::%s: after kernel1: %3d: BRLEA: qid=0x%04x sqId=0x%016llx tbo=0x%08x (%u)",
                                    m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                    n, pBH->qid, sqId, tbo, tbo );
            }
        }
#endif




#if TRACE_SQID
    // copy the D values to a host buffer
    WinGlobalPtr<UINT64> Dyyy( m_nD, false );
    cudaMemcpy( Dyyy.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

    // look at the D data
    sqIdFound = false;
    for( UINT32 n=iDm; n<(iDm+nDm); ++n )
    {
        UINT64 D = Dyyy.p[n];
        UINT32 qid = AriocDS::D::GetQid( D );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
            
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>(D >> 32) & 0x007F;
            INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jo = static_cast<UINT32>(m_pab->M.p[subId] - 1) - J;
#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "[%d] %s::%s: after kernel 1: %3d: D=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d Jo=%d",
                                m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                n, D, sqId, qid, subId, J, Jo );
            sqIdFound = true;
        }
    }

    // look at the BRLEA data
    if( sqIdFound )
    {
        WinGlobalPtr<UINT32> Bxxx( m_pdbg->BRLEA.n, false );
        m_pdbg->BRLEA.CopyToHost( Bxxx.p, Bxxx.Count );

        for( UINT32 n=iDm; n<(iDm+nDm); ++n )
        {
            BRLEAheader* pBH = reinterpret_cast<BRLEAheader*>(Bxxx.p + (n * m_pqb->celBRLEAperQ));
            UINT32 iw = QID_IW(pBH->qid);
            INT16 iq = QID_IQ(pBH->qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];
            
            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                UINT32 tbo = reinterpret_cast<UINT32*>(pBH)[1];
                CDPrint( cdpCD0, "[%d] %s::%s: after kernel1: %3d: BRLEA: sqId=0x%016llx tbo=0x%08x (%u)",
                                    m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                    n, sqId, tbo, tbo );
            }
        }

        CDPrint( cdpCD0, "[%d] %s::%s", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
    }
#endif

#if TODO_CHOP_WHEN_DEBUGGED
        CREXEC( waitForKernel() );
        CDPrint( cdpCD0, "[%d] %s::%s: before Kernel2: iDm=%u nDm=%u nRemaining=%u", m_pqb->pgi->deviceId, __FUNCTION__, m_ptum->Key, iDm, nDm, nRemaining );
#endif


        // launch the traceback kernel
        baseAlignG_Kernel2<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DB.Qw.p,   // in: Qwarps
                                                              m_pD,             // in: Dm list (D values for mapped candidates)
                                                              iDm,              // in: index of the first candidate in the current iteration
                                                              nDm,              // in: number of candidates in the current iteration
                                                              m_SM.p,           // in: scoring-matrix buffer
                                                              m_pdbg->BRLEA.p   // in, out: BRLEA buffer
                                                            );

#if TODO_CHOP_WHEN_DEBUGGED
        CREXEC( waitForKernel() );
        CDPrint( cdpCD0, "[%d] %s::%s: after Kernel2: iDm=%u nDm=%u nRemaining=%u", m_pqb->pgi->deviceId, __FUNCTION__, m_ptum->Key, iDm, nDm, nRemaining );
#endif


        
#if TODO_CHOP_WHEN_DEBUGGED
        {
            // find the first and last QIDs
            WinGlobalPtr<UINT64> Dmxxx( m_nD, false );
            cudaMemcpy( Dmxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

            UINT64 Dc0 = Dmxxx.p[iDm];
            UINT64 Dc1 = Dmxxx.p[iDm+nDm-1];

            CDPrint( cdpCD0, "baseAlignG::launchKernel: from Dc0=0x%016llx (qid=0x%08x) through Dc1=0x%016llx (qid=0x%08x)", Dc0, AriocDS::D::GetQid(Dc0), Dc1, AriocDS::D::GetQid(Dc1) );

            //if(  (AriocDS::D::GetQid(Dc0) <= 0x01F212) && (AriocDS::D::GetQid(Dc1) >= 0x01F212) )
            //    break;
        }
#endif

#if TODO_CHOP_WHEN_DEBUGGED
        if( Bbefore.Count > 0 )
        {
            // copy the Ds to a host buffer
            WinGlobalPtr<UINT64> Dxxx( m_nD, false );
            cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );


#if TODO_CHOP_WHEN_DEBUGGED
    thrust::device_ptr<UINT64> tpDx0( m_pD );
    bool isSorted = thrust::is_sorted( epCGA, tpDx0, tpDx0+m_nD, TSX::isLessD() );
    if( !isSorted ) DebugBreak();
#endif


            // copy the Vmaxs to a host buffer
            WinGlobalPtr<INT16> Vmaxxx( m_nD, false );
            cudaMemcpy( Vmaxxx.p, m_pVmax, m_nD*sizeof(INT16), cudaMemcpyDeviceToHost );

            // copy the encoded interleaved R sequence data to a host buffer
            WinGlobalPtr<UINT64> Rixxx( m_pqb->DB.Ri.Count, false );
            m_pqb->DB.Ri.CopyToHost( Rixxx.p, Rixxx.Count );

            // copy the encoded interleaved Q sequence data to a host buffer
            WinGlobalPtr<UINT64> Qixxx( m_pqb->DB.Qi.Count, false );
            m_pqb->DB.Qi.CopyToHost( Qixxx.p, Qixxx.Count );

            // copy the BRLEAs to a host buffer
            WinGlobalPtr<UINT32> Bafter( m_pdbg->BRLEA.n, false );
            m_pdbg->BRLEA.CopyToHost( Bafter.p, m_pdbg->BRLEA.n );
            Bafter.n = m_pdbg->BRLEA.n;

            BRLEAheader* pBH = reinterpret_cast<BRLEAheader*>(Bafter.p + iDm*m_pqb->celBRLEAperQ);
            for( UINT32 n=iDm; n<(iDm+nDm); ++n )
            {
                if( pBH->V != Vmaxxx.p[n] )
                {
                    UINT64 D = Dxxx.p[n];
                    UINT32 iw = QID_IW(pBH->qid);
                    UINT32 iq = QID_IQ(pBH->qid);
                    Qwarp* pQw = m_pqb->QwBuffer.p + iw;
                    UINT64 sqId = pQw->sqId[iq];


#if TRACE_SQID
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {

                    CDPrint( cdpCD0, "%s::%s: %u: D=0x%016llx sqId=0x%016llx qid=0x%08x V=%d subId=%d J=0x%08x (%d) (Vmax=%d)",
                                                                m_ptum->Key, __FUNCTION__,
                                                                n, D, sqId, pBH->qid, pBH->V, pBH->subId, pBH->J, pBH->J, Vmaxxx.p[n] );

                    // dump the encoded R sequence
                    static char symbolDecode[] = { '·', '·', 'N', 'N', 'A', 'C', 'G', 'T' };     // forward
                    char buf[200] = { 0 };

                    INT64 celMr = blockdiv( m_pdbg->AKP.Mr, 21 );    // number of 64-bit values required to represent Mr
                    celMr *= CUDATHREADSPERWARP;                // number of 64-bit values per warp
                    UINT32 ixw = n >> 5;
                    INT16 ixq = n & 0x1F;
                    const UINT64* pR = Rixxx.p + (ixw * celMr) + ixq;
                    UINT64 b21 = 0;
                    for( INT32 j=0; j<m_pdbg->AKP.Mr; ++j )
                    {
                        // if necessary, get the next 21 symbols
                        if( b21 == 0 )
                        {
                            b21 = *pR;
                            pR += CUDATHREADSPERWARP;
                        }

                        // emit the ith symbol
                        buf[j] = symbolDecode[b21 & 7];

                        // shift the next symbol into the low-order bits of b21
                        b21 >>= 3;
                    }

                    CDPrint( cdpCD0, "%s::%s: %u: R: %s", m_ptum->Key, __FUNCTION__, n, buf );

                    // dump the encoded Q sequence
                    memset( buf, 0, sizeof buf );
                    UINT64* pQ = Qixxx.p + pQw->ofsQi + iq;
                    b21 = 0;
                    for( INT16 i=0; i<pQw->N[iq]; ++i )
                    {
                        // if necessary, get the next 21 symbols
                        if( b21 == 0 )
                        {
                            b21 = *pQ;
                            pQ += CUDATHREADSPERWARP;
                        }

                        // emit the ith symbol
                        buf[i] = symbolDecode[b21 & 7];

                        // shift the next symbol into the low-order bits of b21
                        b21 >>= 3;
                    }

                    CDPrint( cdpCD0, "%s::%s: %u: Q: %s", m_ptum->Key, __FUNCTION__, n, buf );
        }
#endif
                }

                pBH = reinterpret_cast<BRLEAheader*>(reinterpret_cast<UINT32*>(pBH) + m_pqb->celBRLEAperQ );
            }
        }
#endif



        // advance to the next set of candidates
        iDm += nDm;
        nRemaining -= nDm;
        nIterations++ ;
    }

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->n.Iterations, nIterations );
}
#pragma endregion
