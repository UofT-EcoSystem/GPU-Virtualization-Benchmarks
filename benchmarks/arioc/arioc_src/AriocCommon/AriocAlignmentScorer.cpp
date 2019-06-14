/*
  AriocAlignmentScorer.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// [private] default constructor
AriocAlignmentScorer::AriocAlignmentScorer()
{
    // (do not use this constructor)
}

/// [public] constructor (AlignmentControlParameters, AlignmentScoreParameters)
AriocAlignmentScorer::AriocAlignmentScorer( const AlignmentControlParameters& _acp, const AlignmentScoreParameters& _asp ) : ASP(_asp),
                                                                                                                             ACP(_acp)
{
    /* Force Wx (the mismatch penalty) to be non-negative.

       Typically, users specify Wx as a negative number, but we keep it around as a positive number so as to keep
        extra sign changes from cluttering the code.
    */
    if( ASP.Wx < 0 )
        ASP.Wx = -ASP.Wx;

    // sanity check
    if( ASP.Wm <= 0 )
        throw new ApplicationException( __FILE__, __LINE__, "Wm (specified as %d) must be greater than zero", _asp.Wm );
    if( ASP.Wx == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "Wx (specified as %d) must be nonzero", _asp.Wx );
    if( ASP.Wg < 0 )
        throw new ApplicationException( __FILE__, __LINE__, "Wg (specified as %d) may not be less than zero", _asp.Wg );
    if( ASP.Ws < 0 )
        throw new ApplicationException( __FILE__, __LINE__, "Ws (specified as %d) may not be less than zero", _asp.Ws );
}

/// [public] destructor
AriocAlignmentScorer::~AriocAlignmentScorer()
{
}
#pragma endregion

#pragma region static methods
/// [static] method AlignmentResultFlagsToString
char* AriocAlignmentScorer::AlignmentResultFlagsToString( const AlignmentResultFlags arfValue, const AlignmentResultFlags arfMask )
{
    // use a static buffer (i.e. this method is NOT thread-safe)
    static const size_t cbBuf = 128;
    static char buf[cbBuf];

    // apply the specified mask
    AlignmentResultFlags arf = static_cast<AlignmentResultFlags>(arfValue & arfMask);

    // map bits to strings
    if( arf == arfNone )
        strcpy_s( buf, cbBuf, "(none) " );

    else
    {
        *buf = 0;
        if( arf & arfReportUnmapped )   strcat_s( buf, cbBuf, "unmapped," );
        if( arf & arfReportMapped )                 // (arfReportMapped == arfReportConcordant)
        {
            if( arfValue & (arfMaskOrientation|arfMaskCollision) )
                strcat_s( buf, cbBuf, "concordant," );
            else
                strcat_s( buf, cbBuf, "mapped," );
        }
        if( arf & arfReportDiscordant ) strcat_s( buf, cbBuf, "discordant," );
        if( arf & arfReportRejected )   strcat_s( buf, cbBuf, "rejected," );
        INT32 cch = static_cast<INT32>(strlen(buf));
        buf[cch-1] = ' ';                           // replace the trailing comma with a space

        if( arf & arfOrientationSame )       strcat_s( buf, cbBuf, "same," );
        if( arf & arfOrientationConvergent ) strcat_s( buf, cbBuf, "convergent," );
        if( arf & arfOrientationDivergent )  strcat_s( buf, cbBuf, "divergent," );
        cch = static_cast<INT32>(strlen(buf));
        buf[cch-1] = ' ';                           // replace the trailing comma with a space

        if( arf & arfCollisionOverlap )  strcat_s( buf, cbBuf, "overlap," );
        if( arf & arfCollisionCover )    strcat_s( buf, cbBuf, "cover," );
        if( arf & arfCollisionDovetail ) strcat_s( buf, cbBuf, "dovetail," );
    }

    // replace the trailing comma or space with a null
    INT32 cch = static_cast<INT32>(strlen( buf ));
    buf[cch-1] = 0;

    // return a reference to the string; this is for convenience only (the value will become invalid the next time this method is called)
    return buf;
}

/// [static] method ParsePairOrientation
AlignmentResultFlags AriocAlignmentScorer::ParsePairOrientation( const char* pAlignmentResultFlags )
{
    // orientation is convergent by default
    if( pAlignmentResultFlags == NULL )
        return arfOrientationConvergent;

    // look for garbage
    if( strlen(pAlignmentResultFlags) > 1 )
    {
        if( _strcmpi( pAlignmentResultFlags, "same" ) &&
            _strcmpi( pAlignmentResultFlags, "convergent" ) &&
            _strcmpi( pAlignmentResultFlags, "divergent" ) )
        {
            throw new ApplicationException( __FILE__, __LINE__, "invalid value for attribute pairOrientation='%s'", pAlignmentResultFlags );
        }
    }

    // look at the first character in the parameter
    switch( tolower(*pAlignmentResultFlags) )
    {
        case 's':
            return arfOrientationSame;

        case 'c':
            return arfOrientationConvergent;

        case 'd':
            return arfOrientationDivergent;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "invalid value for attribute pairOrientation='%s'", pAlignmentResultFlags );
    }
}
#pragma endregion

#pragma region public methods
/// [public] method ComputeThresholdScore
INT16 AriocAlignmentScorer::ComputeThresholdScore( INT32 N )
{
    double Vt = (ASP.sft == sftG) ? ASP.sfA*log(static_cast<double>(N)) + ASP.sfB :
                (ASP.sft == sftS) ? ASP.sfA*sqrt(static_cast<double>(N)) + ASP.sfB :
                (ASP.sft == sftL) ? ASP.sfA*N + ASP.sfB :
                                    ASP.sfB;
    return static_cast<INT16>(Vt);
}

/// [public] method ValidateScoreFunction
INT16 AriocAlignmentScorer::ValidateScoreFunction( INT32 N )
{
    // perfect alignment score for a maximum-length Q sequence
    INT32 Vp = static_cast<INT32>(ASP.Wm) * N;

    // sanity check
    if( Vp > _I16_MAX )
        throw new ApplicationException( __FILE__, __LINE__, "The maximum score for a perfectly-mapped read (Wm*Nmax=%d*%d=%d) exceeds %d", ASP.Wm, N, Vp, _I16_MAX );

    // threshold score for a maximum-length Q sequence
    double Vt = (ASP.sft == sftG) ? ASP.sfA*log(static_cast<double>(N)) + ASP.sfB :
                (ASP.sft == sftS) ? ASP.sfA*sqrt(static_cast<double>(N)) + ASP.sfB :
                (ASP.sft == sftL) ? ASP.sfA*N + ASP.sfB :
                                    ASP.sfB;

    if( (static_cast<INT32>(Vt) > Vp) || (static_cast<INT32>(Vt) < 0) )
        throw new ApplicationException( __FILE__, __LINE__, "The specified score function is invalid for a query sequence of length %d", N );

    return static_cast<INT16>(Vt);
}

/// [public] method ComputeWorstCaseMismatchCount
INT32 AriocAlignmentScorer::ComputeWorstCaseMismatchCount( INT16 Nmax )
{
    // perfect alignment score for a maximum-length Q sequence
    INT32 Vp = static_cast<INT32>(ASP.Wm) * Nmax;

    // threshold score for a maximum-length Q sequence
    double Vt = (ASP.sft == sftG) ? ASP.sfA*log(static_cast<double>(Nmax)) + ASP.sfB :
                (ASP.sft == sftS) ? ASP.sfA*sqrt(static_cast<double>(Nmax)) + ASP.sfB :
                (ASP.sft == sftL) ? ASP.sfA*Nmax + ASP.sfB :
                                    ASP.sfB;

    // return the maximum number of mismatches
    return (Vp - static_cast<INT32>(Vt)) / (ASP.Wm + ASP.Wx);
}

/// [public] ComputeWorstCaseGapSpaceCount
INT32 AriocAlignmentScorer::ComputeWorstCaseGapSpaceCount( INT16 Nmax )
{
    // perfect alignment score for a maximum-length Q sequence
    INT32 Vp = static_cast<INT32>(ASP.Wm) * Nmax;
   
    // threshold score for a maximum-length Q sequence
    double Vt = (ASP.sft == sftG) ? ASP.sfA*log(static_cast<double>(Nmax)) + ASP.sfB :
                (ASP.sft == sftS) ? ASP.sfA*sqrt(static_cast<double>(Nmax)) + ASP.sfB :
                (ASP.sft == sftL) ? ASP.sfA*Nmax + ASP.sfB :
                                    ASP.sfB;

    /* return the worst-case number of gaps that would result in a reportable alignment; the computation assumes that
        the worst case is the result of a single maximum-length gap in an end-to-end alignment */
    return max2(0, (Vp-static_cast<INT32>(Vt))-ASP.Wg) / ASP.Ws;
}

/// <summary>Computes the number of elements ("cells") in the dynamic-programming scoring matrix for the specified query-sequence length
INT32 AriocAlignmentScorer::ComputeWorstCaseDPSize( INT16 Nmax, INT16 Dspread )
{
    /* Compute the number of DP scoring matrix cells required for one Q sequence.  This is the number of cells in a diagonal band
        centered on the main diagonal (upper left to lower right).  The width bw of the band is estimated by the maximum number of spaces
        in a gap.  The total number of cells is then computed using the length of the Q sequence (N), assuming that the alignment is
        "global" (i.e. end-to-end, i.e. all symbols in Q are used in the alignment.

       The number of SM cells required is thus computed as N*bw, where bw = 2*wcgsc + Dspread + 1.  (There are N rows of bw cells each.)

       For example: given N=9, wgcsc=3, spread=1: 72 cells are needed

                                          R 
                +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                | 1 | 1 | 1 | · | · | 1 | 1 | 1 |   |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                |   | 2 | 2 | 2 | · | · | 2 | 2 | 2 |   |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                |   |   | 3 | 3 | 3 | · | · | 3 | 3 | 3 |   |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                |   |   |   | 4 | 4 | 4 | · | · | 4 | 4 | 4 |   |   |   |   |   |
                +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
            Q   |   |   |   |   | 5 | 5 | 5 | · | · | 5 | 5 | 5 |   |   |   |   |
                +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                |   |   |   |   |   | 6 | 6 | 6 | · | · | 6 | 6 | 6 |   |   |   |
                +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                |   |   |   |   |   |   | 7 | 7 | 7 | · | · | 7 | 7 | 7 |   |   |
                +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                |   |   |   |   |   |   |   | 8 | 8 | 8 | · | · | 8 | 8 | 8 |   |
                +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                |   |   |   |   |   |   |   |   | 9 | 9 | 9 | · | · | 9 | 9 | 9 |
                +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
    */

    // compute the width of each horizontal row in the diagonal band
    INT32 wcgsc = this->ComputeWorstCaseGapSpaceCount( Nmax );
    INT32 bw = (2*wcgsc) + Dspread + 1;
    return Nmax * bw;
}

/// [public] ComputeWorstCaseBRLEAsize
INT32 AriocAlignmentScorer::ComputeWorstCaseBRLEAsize( INT16 Nmax )
{
    /* The worst-case (maximum-length) BRLEA contains a run between each gap and/or mismatch. */

    // perfect alignment score for a maximum-length Q sequence
    INT32 Vp = static_cast<INT32>(ASP.Wm) * Nmax;
    
    // threshold score for a maximum-length Q sequence
    INT32 Vt = static_cast<INT32>( (ASP.sft == sftG) ? ASP.sfA*log(static_cast<double>(Nmax)) + ASP.sfB :
                                   (ASP.sft == sftS) ? ASP.sfA*sqrt(static_cast<double>(Nmax)) + ASP.sfB :
                                   (ASP.sft == sftL) ? ASP.sfA*Nmax + ASP.sfB :
                                                       ASP.sfB );
    
    // compute the maximum number of gaps and/or runs of mismatches
    INT32 maxAgx = (Vp-Vt) / min2((ASP.Wg+ASP.Ws), (ASP.Wm+ASP.Wx));

    // sanity check
    if( (Vt <= 0) || (maxAgx < 0) )
        throw new ApplicationException( __FILE__, __LINE__, "invalid Vt=%d for Wm=%d, Nmax=%d", Vt, ASP.Wm, Nmax );

    /* Compute the size of the runs that are separated by the gaps and/or mismatches:
        - In the worst case, the maximum number of runs in the BRLEA is one greater than the number of gaps and/or runs of mismatches.
        - In the worst case, those runs are sized such that each is encoded with the minimum run length (1, 64, or 4096) for the
            corresponding number of encoded bytes (1, 2, or 3 respectively).
    */
    INT32 maxAgm = Nmax;                                        // total number of symbols in the runs of matches
    INT32 nRuns = maxAgx + 1;                                   // maximum number of runs of matches

    // compute the maximum number of 3-byte runs of matches
    INT32 nRun3 = min2(nRuns, maxAgm / 0x1000);

    maxAgm -= nRun3 * 0x1000;                                   // symbols remaining in 2-byte and 1-byte runs
    nRuns -= nRun3;                                             // number of remaining runs

    // compute the maximum number of 2-byte runs of matches
    INT32 nRun2 = min2(nRuns, maxAgm / 0x0040);
    maxAgm -= nRun2 * 0x0040;                                   // symbols remaining in 1-byte runs
    nRuns -= nRun2;                                             // number of remaining (1-byte) runs

    // compute the total number of bytes in runs of matches
    INT32 cbRun = 3*nRun3 + 2*nRun2 + nRuns;

    // compute the worst-case number of bytes in the BRLEA encoding for a gap or mismatch
    INT32 cbGapOrMismatch = (maxAgx == 0)     ? 0 :
                            (maxAgx < 0x0040) ? 1 :
                            (maxAgx < 0x1000) ? 2 :
                                                3;

    // compute and return the number of bytes in a worst-case (maximum length) BRLEA for the specified Nmax
    return cbRun + cbGapOrMismatch*maxAgx;
}

/// [public] method ComputeMinimumMappingSize
INT32 AriocAlignmentScorer::ComputeMinimumMappingSize( INT16 Nmax )
{
    // threshold score for a maximum-length Q sequence
    double Vt = (ASP.sft == sftG) ? ASP.sfA*log(static_cast<double>(Nmax)) + ASP.sfB :
                (ASP.sft == sftS) ? ASP.sfA*sqrt(static_cast<double>(Nmax)) + ASP.sfB :
                (ASP.sft == sftL) ? ASP.sfA*Nmax + ASP.sfB :
                                    ASP.sfB;

    // the shortest valid mapping occurs with a single run of matching symbols (and all other symbols "soft-clipped")
    return static_cast<INT32>(Vt / this->ASP.Wm);
}
#pragma endregion
