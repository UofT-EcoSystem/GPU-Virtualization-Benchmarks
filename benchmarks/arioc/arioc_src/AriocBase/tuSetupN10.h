/*
  tuSetupN10.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuSetupN10__

/// <summary>
/// Class <c>tuSetupN10</c> hashes Q sequences for nongapped alignment
/// </summary>
class tuSetupN10 : public tuBaseS, public CudaLaunchCommon
{
    public:
        struct KernelConstants
        {
            UINT64  seedMask;       // bitmask that isolates the low-order symbols in a A21-encoded 64-bit value
            UINT64  hashBitsMask;   // bitmask that isolates the low-order bits of the hash value
            INT16   seedWidth;      // number of adjacent symbols covered by the seed
            INT16   npos;           // number of adjacent positions at which the spaced seed is applied
            INT16   minSeedN;       // number of positions spanned by overlapping spaced seeds
            bool    doCTconversion; // true: do CT conversion of Q sequence bases (bisulfite read alignment)
            UINT32  sps;            // 2: seeding from Q+ and Q-; 1: seeding from Q+ only

            KernelConstants()
            {
            }

            KernelConstants( AriocBase* _pab ) : seedMask(_pab->a21ss.seedMask),
                                                 hashBitsMask(_pab->a21ss.hashBitsMask),
                                                 seedWidth(_pab->a21ss.seedWidth),
                                                 npos(_pab->a21ss.npos),
                                                 minSeedN(_pab->a21ss.seedWidth+_pab->a21ss.npos-1),
                                                 doCTconversion(_pab->a21ss.baseConvert == A21SpacedSeed::bcCT),
                                                 sps(_pab->StrandsPerSeed)
            {
            }
        };

    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        KernelConstants         m_kc;
        UINT32                  m_nCandidates;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    private:
        tuSetupN10( void );
        void main( void );

        void computeGridDimensions( dim3& d3g, dim3& d3b );
        void initConstantMemory( void );
        UINT32 initSharedMemory( void );
        void initGlobalMemory( void );
        void launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void copyKernelResults( void );
        void resetGlobalMemory( void );

    public:
        tuSetupN10( QBatch* pqb );
        virtual ~tuSetupN10( void );
};
