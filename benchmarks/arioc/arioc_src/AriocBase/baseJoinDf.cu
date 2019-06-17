/*
  baseJoinDf.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data

/// [device] method setDfFlag
static __device__ __inline__ void setDfFlag( UINT64* pDf )
{
    /* Flag Df2 as a candidate for alignment.  We want to do this:

            *pDf |= AriocDS::Df::flagCandidate;

        but 64-bit atomic operations are not supported prior to compute capability 3.5.
                
        Since the low-order bits of the Df2 value don't change, our workaround is to use the 32-bit version
        of atomicOr on the high-order 32 bits.
    */
#if __CUDA_ARCH__ < 350
    UINT32* pHi32 = reinterpret_cast<UINT32*>(pDf) + 1;
    atomicOr( pHi32, static_cast<UINT32>(AriocDS::Df::flagCandidate >> 32) );
#else
    atomicOr( pDf, AriocDS::Df::flagCandidate );
#endif
}

/// [kernel] baseJoinDf_Kernel
static __global__ void baseJoinDf_Kernel(       UINT64* const   pDfBuffer,              // in,out: pointer to unduplicated Df-list buffer
                                          const UINT32          totalDf,                // in: total number of Df values
                                          const INT64           fragLenRangeMax,        // in: maximum for fragment-length range
                                          const bool            wantOrientationOpposite // in: paired-end orientation
                                        )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( (totalDf == 0) || (tid >= (totalDf-1)) )
        return;

    /* load two Df values and zero the flag bits (which might already have been set in another thread or if
        the Df list contains already-mapped candidates) */
    UINT32 i1 = tid;
    UINT32 i2 = tid + 1;
    UINT64 Df1 = pDfBuffer[i1];
    UINT64 Df2 = pDfBuffer[i2];

    /* Flag both Df values as candidates for alignment; we have a pair of candidates if...
        - we are comparing two mates from the same pair (although we don't distinguish one mate from the other at this point)
        - the Df values are associated with the same subunit ID
        - the Df values have the specified orientation
        - the Df values are within the specified distance of each other
        - the Df values are not identical

       The Df value is bitmapped like this...
            bits 00..00: low-order bit of QID (0: mate 1; 1: mate 2)
            bits 01..01: strand (0: forward; 1: reverse complement)
            bits 02..32: D
            bits 33..39: subId
            bits 40..59: high-order bits of QID
            bits 60..60: rcQ (set if seed derives from reverse complement of Q sequence)
            bits 61..63: flags
        ... so we can do the comparison by shifting the two low-order bits out of the Df values and subtracting

       To determine the strand in order to perform the comparison, we XOR the strand bit (bit 1) with the rcQ bit
        (bit 60).  This happens in either tuXlatToDf or baseLoadJn (and gets undone in tuXlatToD).
    */
    Df1 &= ~(AriocDS::Df::maskFlags | AriocDS::Df::flagRCq);    // zero bits 60..63
    Df2 &= ~(AriocDS::Df::maskFlags | AriocDS::Df::flagRCq);

    /* Compute a value that represents the expected orientation (i.e. bit 1 of the XOR of two Df values):
        - for pairs on the same strand: 0
        - for pairs on opposite strands: 2
    */
    INT64 expectedOrientation = (wantOrientationOpposite ? 2 : 0);

    // set up to loop
    bool orientationOk = false;

    // loop until Df2 lies outside the fragment-length distance from Df1
    UINT64 diff = static_cast<UINT64>(abs( static_cast<INT64>(Df2 >> 2) - static_cast<INT64>(Df1 >> 2) ));
    while( diff <= fragLenRangeMax )
    {
        // check the orientation of Df2 relative to Df1
        if( expectedOrientation == ((Df1 ^ Df2) & 2) )
        {
            // flag Df2 as a candidate for alignment
            setDfFlag( pDfBuffer+i2 );

            // set a flag that indicates the we have found a pair in the expected orientation
            orientationOk = true;
        }

        else    // (Df1 and Df2 do not have the expected orientation)
        {
            /* Fall out of the loop when ...
                -- the current Df2 does not have the expected orientation, AND
                -- we have previously identified at least one Df2 value with the expected orientation
            */
            if( orientationOk )
                break;
        }

        // advance to the next Df value; fall out of the loop at the end of the list
        if( ++i2 == totalDf )
            break;

        Df2 = pDfBuffer[i2];
        Df2 &= ~(AriocDS::Df::maskFlags | AriocDS::Df::flagRCq);

        diff = static_cast<UINT64>(abs( static_cast<INT64>(Df2 >> 2) - static_cast<INT64>(Df1 >> 2) ));
    }

    // if we have a candidate pair, flag Df1
    if( orientationOk )
        setDfFlag( pDfBuffer+i1 );
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseJoinDf::initConstantMemory()
{
}

/// [private] method initSharedMemory
UINT32 baseJoinDf::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseJoinDf_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseJoinDf::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    CRVALIDATOR;

    // set a flag if the mates in a pair are expected to map to opposite strands
    bool wantOrientationOpposite = (m_pab->aas.ACP.arf & static_cast<UINT32>(arfOrientationConvergent|arfOrientationDivergent)) ? true : false;

    /* Compute the fragment-length range for screening the list of Df values; since the Df value represents the position of the first Q-sequence symbol
        on its strand, we need to adjust the expected fragment length according to the orientation of the mates:

            - if the orientation of the mates is the same (i.e. forward-forward or reverse-reverse), the fragment length is adjusted
                by the minimum mapping length:

                    Df               Df
                ····------->·········------->····
                    ^                       ^

            - if the orientation of the mates is divergent, the fragment length is adjusted by 2 times the minimum mapping length:
                           Df        Df
                ····<-------·········------->····
                    ^                       ^
            - if the expected orientation of the mates is convergent, the fragment length is not adjusted

                    Df                      Df
                ····------->·········<-------····
                    ^                       ^

       We don't consider the minimum fragment length at this point; it's more straightforward to deal with it after we have a set of
       successfully-mapped pairs.

       Another consideration is that the Df values represent the diagonals on which seeds are located, NOT the diagonals on which potential
        mappings start, so it seems sensible to widen the maximum fragment-length range to include the worst-case gapped mapping for each
        seed diagonal.  For example, after computing the fragment-length range, we could adjust it as follows:
        
            fragLenRangeMax += 2 * m_pab->aas.ComputeWorstCaseGapSpaceCount( m_pqb->Nmax );

       Doing so, however, has a slight negative effect on the total number of concordant pairs that are found: we seem to be finding more
        mappings whose computed fragment length is greater than the configured maximum, but these mappings are not rejected until after they
        have been excluded from the pipeline.  The tradeoff for not doing this adjustment is to miss 1 mapping or less per million reads, so
        we'll just live with it for now.
    */
    INT32 fragLenRangeMax = m_pab->aas.ACP.maxFragLen;
    switch( m_pab->aas.ACP.arf & arfMaskOrientation )
    {
        case arfOrientationDivergent:
            fragLenRangeMax -= m_pab->aas.ComputeMinimumMappingSize( m_pqb->Nmax );
            break;

        case arfOrientationSame:
            fragLenRangeMax -= 2 * m_pab->aas.ComputeMinimumMappingSize( m_pqb->Nmax );
            break;

        case arfOrientationConvergent:
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected AlignmentResultFlags orientation: 0x%08x", m_pab->aas.ACP.arf & arfMaskOrientation );
    }

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );

    // execute the kernel
    baseJoinDf_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DBj.D.p,            // in, out: unduplicated Df list
                                                         m_pqb->DBj.totalD,         // in: total number of Df values
                                                         fragLenRangeMax,           // in: maximum for fragment-length range                                                
                                                         wantOrientationOpposite    // in: paired-end orientation
                                                       );
}
#pragma endregion
