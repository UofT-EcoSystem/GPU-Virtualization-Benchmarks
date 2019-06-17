/*
  baseCountC.cu

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
static __device__ __constant__ __align__(4) UINT32  ccM[AriocDS::SqId::MaxSubId+1];

#if TODO_CHOP_IF_NEW_LOGIC_WORKS
/// [device] function mapDmToForwardStrand
static inline __device__ UINT32 mapDmToForwardStrand( const UINT64 Dm, const INT16 N )
{
    // get the position (J) relative to the start of the strand
    UINT32 pos = static_cast<UINT32>(Dm & 0x7FFFFFFF);

    // if the specified Dm value is on the reverse strand, map it to the corresponding position on the forward strand
    if( Dm & AriocDS::D::maskStrand )
    {
        // map the first aligned position from the reverse strand to the equivalent position on the forward strand
        INT16 subId = static_cast<INT16>(Dm >> 32) & 0x007F;
        pos = static_cast<UINT32>(ccM[subId]-1) - pos;
    }

    // if we have Qrc mapped to the reverse complement strand, use the estimated position of the other end of the mate
    if( Dm & (AriocDS::D::maskStrand|0x1000000000000000) )
        pos -= N;

    return pos;
}
#endif

/// [device] function spanDmForward
static inline __device__ void spanDmForward( INT32& jUpstream, INT32& jDownstream, const UINT64 Dm, const INT16 N )
{
    // get the position (J) relative to the start of the strand
    UINT32 j = static_cast<INT32>(Dm & 0x7FFFFFFF);

    if( Dm & AriocDS::D::maskStrand )
    {
        // Q is mapped to the reverse-complement strand of R
        INT16 subId = static_cast<INT16>(Dm >> 32) & 0x007F;
        jDownstream = static_cast<INT32>(ccM[subId]-1) - j;
        jUpstream = jDownstream - N;
    }
    else
    {
        // Q is mapped to forward strand of R
        jUpstream = j;
        jDownstream = jUpstream + N;
    }
}

/// [device] function swapInt32
static inline __device__ void swapInt32( INT32& x, INT32& y )
{
    INT32 t = x;
    x = y;
    y = t;
}

/// [device] function isTentativeConcordantPair
static inline __device__ INT32 isTentativeConcordantPair( const UINT64 Dm1, const UINT64 Dm2, const INT16 N1, const INT16 N2, const INT32 arfExpectedOrientation, const bool allowDovetail, const INT32 minTLEN, const INT32 maxTLEN )
{
    // if the Dm values reference different subIds (e.g., different chromosomes), they cannot be concordant
    if( (Dm1 ^ Dm2) & AriocDS::D::maskSubId )
        return -1;

    INT32 ju1, jd1, ju2, jd2;
    spanDmForward( ju1, jd1, Dm1, N1 );
    spanDmForward( ju2, jd2, Dm2, N2 );

    // set flags to indicate to which strand each mate is mapped
    bool mappedToRC1 = (((Dm1 >> AriocDS::D::shrRC) ^ Dm1) & AriocDS::D::maskStrand) != 0;
    bool mappedToRC2 = (((Dm2 >> AriocDS::D::shrRC) ^ Dm2) & AriocDS::D::maskStrand) != 0;

    INT32 tlen = -1;
    switch( arfExpectedOrientation )
    {
        case arfOrientationConvergent:

            // the mates must map to opposite strands
            if( mappedToRC1 != mappedToRC2 )
            {
                /* if mate 1 maps to the reverse complement strand and mate 2 to the forward strand, swap the
                    reference positions */
                if( mappedToRC1 )
                {
                    swapInt32( ju1, ju2 );
                    swapInt32( jd1, jd2 );
                }

                // return true if the fragment length falls within the configured limit
                if( allowDovetail )
                {
                    /* we allow for dovetailed mapping and make the computation insensitive to which mate maps upstream
                        by disregarding the sign of the difference between the extreme mapped positions of the mates */
                    tlen = abs(jd2 - ju1);
                }
                else
                {
                    if( jd2 >= ju1 )
                        tlen = jd2 - ju1;
                }
            }
            break;

        case arfOrientationDivergent:
            asm( "brkpt;" );      // TODO: CHOP WHEN DEBUGGED -- THIS CODE HAS NEVER BEEN FULLY TESTED
            // the mates must map to opposite strands
            if( mappedToRC1 != mappedToRC2 )
            {
                /* if mate 1 maps to the forward strand and mate 2 to the reverse complement strand, swap the
                    reference positions */
                if( mappedToRC2 )
                {
                    swapInt32( ju1, ju2 );
                    swapInt32( jd1, jd2 );
                }

                // (as above)
                if( allowDovetail )
                    tlen = abs(jd2 - ju1);
                else
                {
                    if( jd2 >= ju1 )
                        tlen = jd2 - ju1;
                }
            }
            break;

        default:        // arfOrientationSame
            asm( "brkpt;" );      // TODO: CHOP WHEN DEBUGGED -- THIS CODE HAS NEVER BEEN FULLY TESTED
            // the mates must map to the same strand
            if( mappedToRC1 == mappedToRC2 )
                tlen = max2(jd1,jd2) - min2(ju1,ju2);
            break;
    }


#if TODO_CHOP_IF_THE_NEW_LOGIC_WORKS
    // for both D values, get the 0-based position relative to the forward strand
    INT32 j1 = mapDmToForwardStrand( Dm1, N1 );
    INT32 j2 = mapDmToForwardStrand( Dm2, N2 );

    // set flags to indicate to which strand each mate is mapped
    bool mappedToRC1 = (((Dm1 >> AriocDS::D::shrRC) ^ Dm1) & AriocDS::D::maskStrand) != 0;
    bool mappedToRC2 = (((Dm2 >> AriocDS::D::shrRC) ^ Dm2) & AriocDS::D::maskStrand) != 0;

    INT32 tlen = -1;
    switch( arfExpectedOrientation )
    {
        case arfOrientationConvergent:

            // the mates must map to opposite strands
            if( mappedToRC1 != mappedToRC2 )
            {
                /* if mate 1 maps to the reverse complement strand and mate 2 to the forward strand, swap the
                    reference positions so that j1 becomes the position of the forward strand */
                if( mappedToRC1 )
                {
                    INT32 jx = j1;
                    j1 = j2;
                    j2 = jx;
                }

                // return true if the fragment length falls within the configured limit
                if( allowDovetail )
                {
                    /* we allow for dovetailed mapping and make the computation insensitive to which mate maps upstream
                        by disregarding the sign of the difference between the extreme mapped positions of the mates */
                    tlen = abs(j2 - j1);
                }
                else
                {
                    if( j2 >= j1 )
                        tlen = j2 - j1;
                }
            }
            break;

        case arfOrientationDivergent:
            asm( "brkpt;" );      // TODO: CHOP WHEN DEBUGGED -- THIS CODE HAS NEVER BEEN FULLY TESTED
            // the mates must map to opposite strands
            if( mappedToRC1 != mappedToRC2 )
            {
                /* if mate 1 maps to the forward strand and mate 2 to the reverse complement strand, swap the
                    reference positions so that j1 becomes the position of the reverse complement strand */
                if( mappedToRC2 )
                {
                    INT32 jx = j1;
                    j1 = j2;
                    j2 = jx;
                }

                // for divergent mappings the extrema are at the "far end" of each mate's mapping
                j1 -= N1;
                j2 += N2;

                // (as above)
                if( allowDovetail )
                    tlen = abs(j2 - j1);
                else
                {
                    if( j2 >= j1 )
                        tlen = j2 - j1;
                }
            }
            break;

        default:        // arfOrientationSame
            asm( "brkpt;" );      // TODO: CHOP WHEN DEBUGGED -- THIS CODE HAS NEVER BEEN FULLY TESTED
            // the mates must map to the same strand
            if( mappedToRC1 == mappedToRC2 )
            {
                if( j2 > j1 )                               // mate 2 is downstream and...
                {
                    if( mappedToRC2 )
                        tlen = j2 - (j1 - N1);              // ...mapped to the reverse strand
                    else
                        tlen = (j2 + N2) - j1;              // ...mapped to the forward strand
                }
                else                                        // mate 1 is downstream and...
                {
                    if( mappedToRC1 )
                        tlen = j1 - (j2 - N2);              // ...mapped to the reverse strand
                    else
                        tlen = (j1 + N1) - j2;              // ...mapped to the forward strand
                }
            }
            break;
    }
#endif
    
    // return TLEN if its value falls within the configured range; return -1 otherwise
    if( (tlen > maxTLEN) || (tlen < minTLEN) )
        tlen = -1;
    return tlen;
}

/// [kernel] baseCountC_Kernel
static __global__ void baseCountC_Kernel(       UINT64* const   pDmBuffer,              // in,out: pointer to D values
                                          const UINT32          nDm,                    // in: number of D values
                                                Qwarp*  const   pQwBuffer,              // in,out: pointer to Qwarps
                                          const INT32           arfExpectedOrientation, // in: expected paired-end orientation
                                          const bool            allowDovetail,          // in: flag set if "dovetailed" collision between mates is permitted
                                          const INT32           minTLEN,                // in: minimum configured fragment length (TLEN) including soft-clipping adjustment
                                          const INT32           maxTLEN,                // in: maximum configured fragment length (TLEN) including soft-clipping adjustment
                                          const bool            doExclude               // in: flag indicating whether to exclude previously-mapped D values
                                        )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( (tid == 0) || (tid >= nDm) )
        return;

    /* Load the mapped candidate D value for the current CUDA thread; the D value is bitmapped like this:
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
        bit 62: 1
        bit 63: 0
    */
    const UINT64 Dm = pDmBuffer[tid];

    // extract the QID
    UINT32 qid = static_cast<UINT32>(Dm >> 39) & AriocDS::QID::maskQID;

#if TODO_CHOP_WHEN_DEBUGGED
    if( isDebug )
    {
        if( qid > 0 )
            return;

        asm( "brkpt;" );
    }
#endif
    

    // point to the Qwarp
    UINT32 iw = QID_IW(qid);
    INT16 iq = QID_IQ(qid);
    Qwarp* pQw = pQwBuffer + iw;

    /* The goal here is to tentatively identify all mappings in concordantly-mapped pairs. */

    // compute the offset within the Qwarp of the opposite mate
    INT16 iqo = iq ^ 1;

    /* The input Dm list must be ordered by QID, subId, strand, J, and mate, i.e., in order by Df value.  This may
        require the input list to be (re)formatted as Df values, sorted, and again formatted as D values.
    
        Each CUDA thread examines one adjacent pair of Dm values in the list to determine whether they
        come from the same pair, and, if so, whether the pair meets the user-specified distance and orientation
        constraints for a concordantly-mapped pair.
    */

    // Do we have a pair?  For opposite mates in the same pair, the QIDs differ by 1.
    const UINT64 Dmo = pDmBuffer[tid-1];
    UINT32 qido = static_cast<UINT32>(Dmo >> 39) & AriocDS::QID::maskQID;
    if( (qid ^ qido) != 1 )
        return;
        
    /* At this point we need to determine whether the pair conforms to the orientation and distance criteria for
        a concordant paired mapping.  If the pair falls within the criteria for a concordant paired mapping, we
        increment the count of tentatively identified concordant mappings.

        This strategy can underestimate the actual number of concordant mappings in at least two ways:
        - Since the final mapping may be soft-clipped by one or more bases, depending on the Wm and Wx scoring parameters,
            it is possible that a mapping may happen to be soft-clipped so that it (just barely) fits within the user-specified
            fragment length (TLEN).  But we can account for this by expanding the acceptable TLEN criterion by a few extra bases.
        - A mate may map two or more times to the same strand within TLEN distance of the opposite mate; this unusual situation
            will be missed in the following code, where only the pair of mappings with the shortest TLEN will be flagged.  But
            in this situation, the gapped aligner should pick up the missed mapping(s) anyway.

        But this behavior is acceptable because we use this implementation's estimates of whether pairs are concordant to eliminate
         those pairs from subsequent processing.  Not recognizing a few concordant mappings at this stage is better than falsely
         identifying pairs as being concordantly mapped.
    */
           
    // use the parity of the qid to figure out which mapping is for which mate
    INT16 N1;
    INT16 N2;
    UINT64 Dm1;
    UINT64 Dm2;
    if( iq & 1 )
    {
        N1 = pQw->N[iqo];
        N2 = pQw->N[iq];
        Dm1 = Dmo;
        Dm2 = Dm;
    }
    else
    {
        N1 = pQw->N[iq];
        N2 = pQw->N[iqo];
        Dm1 = Dm;
        Dm2 = Dmo;
    }




#if TODO_CHOP_WHEN_DEBUGGED
    if( pQw->sqId[iq] != 0x0000040800000080 )
        return;
#endif


    INT32 tlen = isTentativeConcordantPair( Dm1, Dm2, N1, N2, arfExpectedOrientation, allowDovetail, minTLEN, maxTLEN );
    if( tlen != -1 )
    {
        /* At this point we need to be sure that we aren't dealing with the case where neither mate in the tentatively-
            concordant pair was previously mapped (e.g., by the nongapped aligner).  This can happen when
                - both mates of a pair have been previously mapped but the mapping does not constitute a concordant pair, AND
                - a set of J values (D values) has been loaded for each of the previously-mapped positions, AND
                - one or more of the loaded J values was successfully mapped for each of the mates, AND
                - a pair of those mapped J values happens to meet the criteria for a concordant alignment
                
           Empirically, this happens less than 0.02% of the time.  Nevertheless...
           
           We avoid the problem by ensuring that a tentative concordant pair is counted only if exactly one of the mates was
            previously mapped (i.e., the "exclude" flag is set).
        */
        if( (!doExclude) || ((Dm ^ Dmo) & AriocDS::D::flagX) )
        {
            /* Use a 32-bit PTX "atomic reduction" instruction to increment the tentative concordant mapping counts
                for both mate 1 (at the even-numbered offset) and mate 2 (at the odd-numbered offset).  The Qwarp.nAc array
                is 32-bit aligned and we're on a low-endian machine, so we zero the low-order bit of iq and add 0x00010001
                to increment two adjacent elements in the array.
            */
            atomicAdd( reinterpret_cast<UINT32*>(pQw->nAc+(iq&0xFFFE)), 0x00010001 );

            /* save the computed TLEN; we are only using this to estimate the TLEN distribution so we don't care about
                duplicates for the same QID */
            pQw->tlen[iq] = tlen;
        }

        /* Flag the mappings as not being candidates for subsequent alignment.
        
           This is useful only if the D list contains unique QIDs.  If D values with duplicate QIDs exist, one or more
            of the D values will not form a concordant pair with its neighbor and therefore will not have its "candidate"
            flag unset.  (The baseFilterD implementation can deal with that.)
        */
        pDmBuffer[tid-1] = Dmo & ~AriocDS::D::flagCandidate;
        pDmBuffer[tid] = Dm & ~AriocDS::D::flagCandidate;
    }
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseCountC::initConstantMemory()
{
    CRVALIDATOR;

    // load the offset of each R sequence into constant memory on the current GPU
    CRVALIDATE = cudaMemcpyToSymbol( ccM, m_pab->M.p, m_pab->M.cb );
}

/// [private] method initSharedMemory
UINT32 baseCountC::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseCountC_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseCountC::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    /* Set up concordant paired mapping constraints:
        - expected orientation
        - expected dovetailed collisions
        - maximum TLEN: this can be biased based on user assumptions about soft clipping and the potential size of
            indels; a negative bias decreases the number of mappings that will be flagged as "concordant"
            according to the computed TLEN
    */
    INT32 arfExpectedOrientation = m_pab->aas.ACP.arf & arfMaskOrientation;
    bool allowDovetail = ((m_pab->aas.ACP.arf & arfCollisionDovetail) != 0);
    const INT32 maxTLEN = m_pab->aas.ACP.maxFragLen + m_pab->TLENbias;
    const INT32 minTLEN = m_pab->aas.ACP.minFragLen - m_pab->TLENbias;

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );

#if TODO_CHOP_WHEN_DEBUGGED
    bool isDebug = (m_pdbb->flagRi == riDm);
    if( isDebug )
        isDebug = (0 == strcmp( "tuAlignGs30", m_ptum->Key ));

    if( isDebug )    
    {
        CDPrint( cdpCD0, "%s::baseCountC::launchKernel: isDebug=true allowDovetail=%d", m_ptum->Key, allowDovetail );
        isDebug = false;
    }
#endif



    // execute the kernel
    baseCountC_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pD,                      // in,out: D values for mappings
                                                         m_nD,                      // in: number of mappings
                                                         m_pqb->DB.Qw.p,            // in: Qwarps
                                                         arfExpectedOrientation,    // in: expected paired-end orientation
                                                         allowDovetail,             // in: flag set if "dovetailed" collision between mates is permitted
                                                         minTLEN,                   // in: minimum configured fragment length (TLEN) including soft-clipping adjustment
                                                         maxTLEN,                   // in: maximum configured fragment length (TLEN) including soft-clipping adjustment
                                                         m_doExclude                // in: flag indicating whether to exclude previously-mapped D values
                                                        );
}
#pragma endregion
