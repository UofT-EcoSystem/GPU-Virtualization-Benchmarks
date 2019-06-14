/*
  baseMAPQ.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
   mapqVersion
    0 [default] based on BWA-MEM v0.7.9a implementation (which looks unchanged at least through v0.7.17)
    1           based on Bowtie2 implementation
    2           based on Bowtie2 implementation but with somewhat different lookup tables
*/
#include "stdafx.h"

#pragma region static constants
/* In BWA-MEM v0.7.9a, the following constant is set to 50 and the natural logarithm (i.e., ln(50) = 3.912) is saved as an
    integer (i.e., 3).  We use a constant value of 20 and save the natural logarithm (i.e., 2.996) as a floating point value,
    so the resulting computation is pretty much the same here as in BWA-MEM.
*/
const int baseMAPQ::m_mcl = 20;
const double baseMAPQ::m_mcf = log( static_cast<double>(m_mcl) );
#pragma endregion

#pragma region baseMAPQ::Vinfo: constructor/destructor
/// [private] constructor
baseMAPQ::Vinfo::Vinfo()
{
}

/// [public] constructor
baseMAPQ::Vinfo::Vinfo( AriocBase* _pab ) : m_pab(_pab),
                                            N1(0), N2(0),
                                            Vmax1(-1), Vmax2(-1), Vp(0), Vt(0),
                                            nAu(-1), nVmax1(0), nVmax2(0)
{
}

/// destructor
baseMAPQ::Vinfo::~Vinfo()
{
}
#pragma endregion

#pragma region baseMAPQ::Vinfo: public methods
/// [public] method TrackV
void baseMAPQ::Vinfo::TrackV( QAI* pQAI )
{
    // record the total number of unduplicated mappings found by the aligner for the current Q sequence
    this->nAu = pQAI->nAu;

    // record the total number of symbols in the specified read
    this->N1 = pQAI->N;

    // track the highest and second-highest V scores
    if( pQAI->pBH->V >= this->Vmax1 )
    {
        if( pQAI->pBH->V > this->Vmax1 )
        {
            this->Vmax2 = this->Vmax1;
            this->Vmax1 = pQAI->pBH->V;
            this->nVmax2 = this->nVmax1;
            this->nVmax1 = 1;
        }
        else
            this->nVmax1++ ;
    }
    else
    {
        if( pQAI->pBH->V >= this->Vmax2 )
        {
            if( pQAI->pBH->V > this->Vmax2 )
            {
                this->Vmax2 = pQAI->pBH->V;
                this->nVmax2 = 1;
            }
            else
                this->nVmax2++ ;
        }
    }
}

/// [public] method TrackV
void baseMAPQ::Vinfo::TrackV( QAI* pQAI1, QAI* pQAI2 )
{
    if( this->nAu < 0 )
    {
        /* record the minimum of the total number of unduplicated mappings found by the aligner for each of
            the mates in the pair */
        this->nAu = min2(pQAI1->nAu, pQAI2->nAu);
    }

    // record the total number of symbols in both mates
    this->N1 = pQAI1->N;
    this->N2 = pQAI2->N;

    // compute the sum of the V scores for both mates
    INT32 V = static_cast<INT32>(pQAI1->pBH->V) + static_cast<INT32>(pQAI2->pBH->V);

    // track the highest and second-highest V scores
    if( V >= this->Vmax1 )
    {
        if( V > this->Vmax1 )
        {
            this->Vmax2 = this->Vmax1;
            this->Vmax1 = V;
            this->nVmax2 = this->nVmax1;
            this->nVmax1 = 1;
        }
        else
            this->nVmax1++ ;
    }
    else
    {
        if( V >= this->Vmax2 )
        {
            if( V > this->Vmax2 )
            {
                this->Vmax2 = V;
                this->nVmax2 = 1;
            }
            else
                this->nVmax2++ ;
        }
    }
}

/// [public] method FinalizeV
void baseMAPQ::Vinfo::FinalizeV()
{
    if( this->N1 == 0 )
        return;

    // compute a perfect score for the specified mapping
    this->Vp = (this->N1 + this->N2) * m_pab->aas.ASP.Wm;

    // compute the threshold score for the specified mapping
    this->Vt = m_pab->aas.ComputeThresholdScore( this->N1 );
    if( this->N2 )
        this->Vt += m_pab->aas.ComputeThresholdScore( this->N2 );
}
#pragma endregion

#pragma region baseMAPQ: constructor/destructor
/// [private] constructor
baseMAPQ::baseMAPQ()
{
}

/// [public] constructor
baseMAPQ::baseMAPQ( AriocBase* pab ) : m_pab(pab)
{
    // set up the correct member function for computing MAPQ
    switch( m_pab->aas.ACP.mapqVersion )
    {
        case 1:         // based on: BT2 v2
            m_computeMAPQ = &baseMAPQ::computeMAPQ_BT2;
            m_computeMAPQp = &baseMAPQ::computeMAPQ_BT2p;
            break;

        case 2:         // based on: modified BT2 v2
            m_computeMAPQ = &baseMAPQ::computeMAPQ_Ariocx;
            m_computeMAPQp = &baseMAPQ::computeMAPQ_Ariocxp;
            break;

        default:        // based on: BWA-MEM
            m_computeMAPQ = &baseMAPQ::computeMAPQ_BWA;
            m_computeMAPQp = &baseMAPQ::computeMAPQ_BWAp;
            m_altVsec = pab->a21hs.seedWidth * pab->aas.ASP.Wm;
            m_mismatchPenalty = pab->aas.ASP.Wm + pab->aas.ASP.Wx;
            break;
    }
}

/// destructor
baseMAPQ::~baseMAPQ()
{
}
#pragma endregion

#pragma region baseMAPQ: private methods
/// [private] method computeSeedCoverage
INT32 baseMAPQ::computeSeedCoverage( BRLEAheader* pBH )
{
    // point to the BRLEA string
    BRLEAbyte* pbb = reinterpret_cast<BRLEAbyte*>(pBH+1);
    BRLEAbyte* pbbLimit = pbb + pBH->cb;

    // accumulate the total number of reference locations covered by seeds
    INT32 totalMatchRunLength = 0;
    INT32 currentRunLength = 0;
    while( pbb < pbbLimit )
    {
        // accumulate the maximum length of a run of matches
        if( pbb->bbType == bbMatch )
            currentRunLength = (currentRunLength << 6) + pbb->bbRunLength;
        else
        {
            // accumulate the current run length only if it is long enough to be covered by a seed
            if( currentRunLength >= m_pab->a21hs.seedWidth )
                totalMatchRunLength += currentRunLength;
            currentRunLength = 0;
        }

        // iterate
        pbb++ ;
    }

    // accumulate the final run length (if any)
    if( currentRunLength >= m_pab->a21hs.seedWidth )
        totalMatchRunLength += currentRunLength;

    return totalMatchRunLength;
}

/// [private] method computeMAPQ_Ariocx
UINT8 baseMAPQ::computeMAPQ_Ariocx( INT16 V, baseMAPQ::Vinfo& mvi )
{
    /* Return a MAPQ of zero if there is no V-score info.  This can happen with bisulfite-aligned reads where one
        or both of the mates is a reverse-complement duplicate.  See tuClassifyP2::computeMappingQualities().
    */
    if( mvi.N1 == 0 )
        return 0;

    // normalize the range of possible V scores (0 = lowest valid V score)
    INT32 diff = mvi.Vp - mvi.Vt;

    // normalize the V score for the read
    INT32 bestOver = V - mvi.Vt;

    if( (mvi.nVmax1 + mvi.nVmax2) == 1 )
    {
        /* At this point we have a unique mapping. */

        // the LUTs for unpaired and paired-end mappings are a bit different
        static const UINT8 lutUniqueU[] = {  3,  3,  3,  4,  5,  6,  7,  8,  9, 10, // 0.00-0.09
                                            12, 14, 16, 18, 20, 22, 22, 22, 22, 22, // 0.10-0.19
                                            22, 22, 22, 22, 22, 22, 22, 22, 22, 22, // 0.20-0.29
                                            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, // 0.30-0.39
                                            28, 28, 28, 28, 28, 28, 28, 28, 28, 28, // 0.40-0.49
                                            36, 36, 36, 36, 36, 36, 36, 36, 36, 36, // 0.50-0.59
                                            41, 41, 41, 41, 41, 41, 41, 41, 41, 41, // 0.60-0.69
                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, // 0.70-0.79
                                            44, 44, 44, 44, 44, 44, 44, 44, 44, 44, // 0.80-0.89
                                            44, 44, 44, 44, 44, 44, 44, 44, 44, 44, // 0.90-0.99
                                            44 };                                   // 1.00

        static const UINT8 lutUniqueP[] = {  3,  3,  3,  3,  3,  4,  4,  4,  5,  5, // 0.00-0.09
                                             6,  6,  7,  7,  8,  8,  9,  9, 10, 11, // 0.10-0.19
                                            12, 13, 14, 15, 16, 18, 19, 20, 21, 22, // 0.20-0.29
                                            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, // 0.30-0.39
                                            28, 28, 28, 28, 28, 28, 28, 28, 28, 28, // 0.40-0.49
                                            36, 36, 36, 36, 36, 36, 36, 36, 36, 36, // 0.50-0.59
                                            41, 41, 41, 41, 41, 41, 41, 41, 41, 41, // 0.60-0.69
                                            42, 42, 42, 42, 42, 42, 42, 42, 42, 42, // 0.70-0.79
                                            44, 44, 44, 44, 44, 44, 44, 44, 44, 44, // 0.80-0.89
                                            44, 44, 44, 44, 44, 44, 44, 44, 44, 44, // 0.90-0.99
                                            44 };                                   // 1.00

        // look up a MAPQ value in the table
        const UINT8* pLUT = (mvi.N2 ? lutUniqueP : lutUniqueU);
        INT32 i1 = diff ? ((100 * bestOver) / diff) : 100;
        return pLUT[i1];
    }
    
    if( mvi.nVmax1 >= 2 )
    {
        /* There are 2 or more "best" mappings:

            mapQ = static_cast<UINT8>((-10.0 * log10(1.0 - (1.0/mvi.nVmax1) )) + 0.5);

               nAu  MAPQ
                2       3
                3       2
                4-8     1
                9+      0

           Empirically, based on Mason synthetic 100bp and 250bp reads, the probability that the read
            is correctly mapped is nevertheless much better than one might expect in the case where
            there are exactly two possible mappings.  Hence we report a MAPQ of 17 for nAu=2.
        */
        if( mvi.nAu >= 9 )
            return 0;

        static const UINT8 lutDuplicateBest[] = {  0,  0,  17,  2,  1,  1,  1,  1,  1 };
        return lutDuplicateBest[mvi.nAu];
    }

    /* Two or more mappings, but only one highest-scoring mapping */
    static const UINT8 lut2[][11] = { {  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0 },   // [0.0] 0.0 <= bestdiff < 0.1
                                      {  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },   // [0.1] bestdiff > 0         && bestdiff < 0.1*diff
                                      { 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16 },   // [0.2] bestdiff >= 0.1*diff && bestdiff < 0.2*diff
                                      { 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19 },   // [0.3] bestdiff >= 0.2*diff && bestdiff < 0.3*diff
                                      { 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22 },   // [0.4] bestdiff >= 0.3*diff && bestdiff < 0.4*diff
                                      { 23, 23, 23, 23, 23, 25, 25, 25, 25, 25, 32 },   // [0.5] bestdiff >= 0.4*diff && bestdiff < 0.5*diff
                                      { 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37 },   // [0.6] bestdiff >= 0.5*diff && bestdiff < 0.6*diff
                                      { 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38 },   // [0.7] bestdiff >= 0.6*diff && bestdiff < 0.7*diff
                                      { 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39 },   // [0.8] bestdiff >= 0.7*diff && bestdiff < 0.8*diff
                                      { 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40 },   // [0.9] bestdiff >= 0.8*diff && bestdiff < 0.9*diff
                                      { 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40 }    // [1.0] bestdiff >= 0.9*diff 
                                    };

    INT32 bestDiff = mvi.Vmax1 - mvi.Vmax2;

    // compute the lookup-table indexes and hit the lookup table
    INT32 i1 = (10*bestDiff) / diff;
    INT32 i2 = (10*bestOver) / diff;
    return lut2[i1][i2];
}

/// [private] method computeMAPQ_Ariocx
void baseMAPQ::computeMAPQ_Ariocx( QAI* pQAI, baseMAPQ::Vinfo& mvi )
{
    pQAI->mapQ = computeMAPQ_Ariocx( pQAI->pBH->V, mvi );
}

/// [protected] method computeMAPQ_Ariocxp
void baseMAPQ::computeMAPQ_Ariocxp( PAI* pPAI, baseMAPQ::Vinfo& mviCombined, baseMAPQ::Vinfo& mvi1, baseMAPQ::Vinfo& mvi2 )
{
    /* In a paired-end read where both mates mapped:
        - the MAPQ score is computed on the sum of the alignment scores of both mates
        - both mates get the same MAPQ score
    */
    INT32 Vtotal = static_cast<INT32>(pPAI->pQAI1->pBH->V) + static_cast<INT32>(pPAI->pQAI2->pBH->V);
    if( pPAI->arf & arfPrimary )
        pPAI->pQAI1->mapQ = pPAI->pQAI2->mapQ = computeMAPQ_Ariocx( Vtotal, mviCombined );
    else
        pPAI->pQAI1->mapQp2 = pPAI->pQAI2->mapQp2 = computeMAPQ_Ariocx( Vtotal, mviCombined );
}

/// [private] method computeMAPQ_BT2
UINT8 baseMAPQ::computeMAPQ_BT2( INT16 V, baseMAPQ::Vinfo& mvi )
{
    /* Return a MAPQ of zero if there is no V-score info.  This can happen with bisulfite-aligned reads where one
        or both of the mates is a reverse-complement duplicate.  See tuClassifyP2::computeMappingQualities().
    */
    if( mvi.N1 == 0 )
        return 0;

    // normalize the range of possible V scores (0 = lowest valid V score)
    INT32 diff = mvi.Vp - mvi.Vt;

    /* Normalize the V score for the read:
        - the Bowtie2 implementation uses Vmax (i.e. it assumes that the read or pair is the highest-scoring)
        - the current implementation uses V (the actual V score for the mate or pair)
        - the current implementation produces the same results for the highest-scoring read, but a lower
            MAPQ (i.e. a higher probability that the mapping is in the wrong place) for lower-scoring reads
    */
    INT32 bestOver = V - mvi.Vt;

    /* One unique mapping */
    if( (mvi.nVmax1 + mvi.nVmax2) == 1 )
    {
        static const UINT8 lutUnique[] = { 22, 22, 22, 22, 22, 22, 22, 22, 22, 22,  // 0.00-0.09
                                           22, 22, 22, 22, 22, 22, 22, 22, 22, 22,  // 0.10-0.19
                                           22, 22, 22, 22, 22, 22, 22, 22, 22, 22,  // 0.20-0.29
                                           24, 24, 24, 24, 24, 24, 24, 24, 24, 24,  // 0.30-0.39
                                           28, 28, 28, 28, 28, 28, 28, 28, 28, 28,  // 0.40-0.49
                                           36, 36, 36, 36, 36, 36, 36, 36, 36, 36,  // 0.50-0.59
                                           41, 41, 41, 41, 41, 41, 41, 41, 41, 41,  // 0.60-0.69
                                           42, 42, 42, 42, 42, 42, 42, 42, 42, 42,  // 0.70-0.79
                                           44, 44, 44, 44, 44, 44, 44, 44, 44, 44,  // 0.80-0.89
                                           44, 44, 44, 44, 44, 44, 44, 44, 44, 44,  // 0.90-0.99
                                           44 };                                    // 1.00

        // we have a unique mapping
        INT32 i1 = diff ? ((100 * bestOver) / diff) : 100;
        return lutUnique[i1];
    }
    
    /* Two or more mappings (and possibly two or more highest-scoring mappings) */
    static const UINT8 lut2[][11] = { {  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1 },   // [0.0] 0.0 <= bestdiff < 0.1
                                      {  9,  9,  9,  9,  9, 14, 14, 14, 14, 14, 31 },   // [0.1] bestdiff > 0         && bestdiff < 0.1*diff
                                      { 12, 12, 12, 12, 12, 17, 17, 17, 17, 17, 32 },   // [0.2] bestdiff >= 0.1*diff && bestdiff < 0.2*diff
                                      { 16, 16, 16, 16, 16, 18, 18, 18, 18, 18, 33 },   // [0.3] bestdiff >= 0.2*diff && bestdiff < 0.3*diff
                                      { 19, 19, 19, 19, 19, 21, 21, 21, 21, 21, 34 },   // [0.4] bestdiff >= 0.3*diff && bestdiff < 0.4*diff
                                      { 20, 20, 20, 20, 20, 25, 25, 25, 25, 25, 35 },   // [0.5] bestdiff >= 0.4*diff && bestdiff < 0.5*diff
                                      { 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37 },   // [0.6] bestdiff >= 0.5*diff && bestdiff < 0.6*diff
                                      { 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38 },   // [0.7] bestdiff >= 0.6*diff && bestdiff < 0.7*diff
                                      { 39, 39, 39, 39, 39, 39, 39, 39, 39, 39, 39 },   // [0.8] bestdiff >= 0.7*diff && bestdiff < 0.8*diff
                                      { 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40 },   // [0.9] bestdiff >= 0.8*diff && bestdiff < 0.9*diff
                                      { 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40 }    // [1.0] bestdiff >= 0.9*diff 
                                    };

    INT32 bestDiff = (mvi.nVmax1 >= 2) ? 0 : mvi.Vmax1 - mvi.Vmax2;

    // compute the lookup-table indexes and return a value from the lookup table
    INT32 i1 = (10*bestDiff) / diff;
    INT32 i2 = (10*bestOver) / diff;
    return lut2[i1][i2];
}

/// [private] method computeMAPQ_BT2
void baseMAPQ::computeMAPQ_BT2( QAI* pQAI, baseMAPQ::Vinfo& mvi )
{
    pQAI->mapQ = computeMAPQ_BT2( pQAI->pBH->V, mvi );
}

/// [protected] method computeMAPQ_BT2p
void baseMAPQ::computeMAPQ_BT2p( PAI* pPAI, baseMAPQ::Vinfo& mviCombined, baseMAPQ::Vinfo& mvi1, baseMAPQ::Vinfo& mvi2 )
{
    /* In a paired-end read where both mates mapped:
        - the MAPQ score is computed on the sum of the alignment scores of both mates
        - both mates get the same MAPQ score
    */
    INT32 Vtotal = static_cast<INT32>(pPAI->pQAI1->pBH->V) + static_cast<INT32>(pPAI->pQAI2->pBH->V);
    if( pPAI->arf & arfPrimary )
        pPAI->pQAI1->mapQ = pPAI->pQAI2->mapQ = computeMAPQ_BT2( Vtotal, mviCombined );
    else
        pPAI->pQAI1->mapQp2 = pPAI->pQAI2->mapQp2 = computeMAPQ_BT2( Vtotal, mviCombined );
}

/// [protected] method computeMAPQ_BWA
UINT8 baseMAPQ::computeMAPQ_BWA( BRLEAheader* pBH, baseMAPQ::Vinfo& mvi )
{
    /* The following derives from the BWA-MEM v0.7.9a-r786 implementation but with modifications to conform to the way Arioc does things. */

    // the default MAPQ value is zero
    UINT8 mapQ = 0;

    /* Return a MAPQ of zero if there is no V-score info.  This can happen with bisulfite-aligned reads where one
        or both of the mates is a reverse-complement duplicate.  See tuClassifyP2::computeMappingQualities().
    */
    if( mvi.N1 == 0 )
        return mapQ;

    /* For purposes of this computation, we need a "second best" V score:
        - if there are multiple top-scoring mappings, the "second-best" score is the best score
        - if there is only one top-scoring mapping, the second-best score is the next highest mapping's score
        - if there are no second-best mappings, use the "alternative" minimum score (the score associated with one seed's
            worth of coverage)
    */
    INT16 Vsec = (mvi.nVmax1 > 1) ? mvi.Vmax1 :
                 (mvi.nVmax2 > 0) ? mvi.Vmax2 : m_altVsec;

    // if the specified read's V score is no greater than the second-best V score, return MAPQ = 0
    if( pBH->V <= Vsec )
        return mapQ;

    /* at this point we have one read with V = Vmax1 and zero or more reads with V <= Vmax2
    */

    // compute the maximum span of the mapping (i.e., the maximum of the span in R and the span in Q)
    INT16 alnSpan = max2( (pBH->Ir-pBH->Il)+1, pBH->Ma );
    double identity = 1.0 - ((static_cast<double>(alnSpan*m_pab->aas.ASP.Wm - pBH->V) / m_mismatchPenalty) / alnSpan);

    // compute a tentative MAPQ score
    double tmp = (alnSpan < m_mcl) ? 1.0 : (m_mcf / log(static_cast<double>(alnSpan)));
    tmp *= (identity*identity);
    INT32 imapq = d2i32(6.02 * (pBH->V-Vsec) / m_pab->aas.ASP.Wm * tmp * tmp);

    // adjust MAPQ for multiple mappings
    if( pBH->nTBO > 1 )
        imapq -= d2i32(4.343 * log(static_cast<double>(pBH->nTBO)));

    // return a value between 0 and 60
    if( imapq > 0 )
    {
        if( imapq > 60 )
            mapQ = 60;
        else
            mapQ = static_cast<UINT8>(imapq);
    }

    return mapQ;
}

/// [private] method computeMAPQ_BWA
void baseMAPQ::computeMAPQ_BWA( QAI* pQAI, baseMAPQ::Vinfo& mvi )
{
    pQAI->mapQ = computeMAPQ_BWA( pQAI->pBH, mvi );
}

/// [protected] method computeMAPQ_BWAp
void baseMAPQ::computeMAPQ_BWAp( PAI* pPAI, baseMAPQ::Vinfo& mviCombined, baseMAPQ::Vinfo& mvi1, baseMAPQ::Vinfo& mvi2 )
{
    // compute MAPQ for the individual mates
    UINT8 mapQ1 = computeMAPQ_BWA( pPAI->pQAI1->pBH, mvi1 );
    UINT8 mapQ2 = computeMAPQ_BWA( pPAI->pQAI2->pBH, mvi2 );

    // compute the total alignment score for the specified concordant pair
    INT32 Vpair = pPAI->pQAI1->pBH->V + pPAI->pQAI2->pBH->V;

    /* For purposes of this computation, we need a "second best" paired V score:
        - if there are multiple top-scoring mappings, the "second-best" score is the best score
        - if there is only one top-scoring mapping, the second-best score is the next highest mapping's score
        - if there are no second-best mappings, use zero
    */
    INT16 Vsec = (mviCombined.nVmax1 > 1) ? mviCombined.Vmax1 :
                 (mviCombined.nVmax2 > 0) ? mviCombined.Vmax2 : 0;

    // compute the second-best score
    Vsec = max2( Vsec, mvi1.Vmax1+mvi2.Vmax1 ); // use the higher of either the second-best "combined" alignment score or the sum of the highest unpaired scores
    Vsec = min2( mviCombined.Vmax1, Vsec );     // ensure that the combination does not exceed the highest paired alignment score

    // adjust for TLEN (fragment length)
    double d = pPAI->diffTLEN / m_pab->stdevTLEN;    // number of standard deviations from the mean
    INT32 adj = d2i32((1.0/log(4.0)) * log(2.0 * nrc_erfc(d/sqrt(2.0)) * m_pab->aas.ASP.Wm));
    if( adj > 0 )
    {
        Vpair += adj;

        // compute a phred-scaled error probability for the pair
        INT32 Epair = d2i32(6.02 * (Vpair-Vsec) / m_pab->aas.ASP.Wm);
    
        // adjust for multiple mappings
        INT32 nPairedMappings = mviCombined.nVmax1 + mviCombined.nVmax2;
        if( nPairedMappings > 1 )
            Epair -= d2i32(4.343 * log(static_cast<double>(nPairedMappings)));

        // clamp the MAPQ between 0 and 60
        if( Epair < 0 )
            Epair = 0;
        else
        if( Epair > 60 )
            Epair = 60;

        /* For each mate:  if the individual MAPQ is smaller than the estimated error probability for the pair, report the smaller of
            - the estimated error probability for the pair, and
            - the individual MAPQ for the mate plus 40
        */
        if( mapQ1 <= Epair )
            mapQ1 = min2( Epair, mapQ1+40 );

        if( mapQ2 <= Epair )
            mapQ2 = min2( Epair, mapQ2+40 );
    }

    // save the computed MAPQ value for each mate in the pair
    if( pPAI->arf & arfPrimary )
    {
        pPAI->pQAI1->mapQ = mapQ1;
        pPAI->pQAI2->mapQ = mapQ2;
    }
    else
    {
        pPAI->pQAI1->mapQp2 = mapQ1;
        pPAI->pQAI2->mapQp2 = mapQ2;
    }
}
#pragma endregion
