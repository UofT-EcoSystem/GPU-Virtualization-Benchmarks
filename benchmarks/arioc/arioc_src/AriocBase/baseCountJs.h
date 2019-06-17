/*
  baseCountJs.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseCountJs__

/// <summary>
/// Class <c>baseCountJs</c> hashes Q sequences to determine the number of reference locations
/// associated with seed-and-extend seeds
/// </summary>
class baseCountJs : public tuBaseS, public CudaLaunchCommon
{

    public:
        struct KernelConstants
        {
            INT32   maxJg;          // maximum number of positions associated with one seed (i.e., maximum J-list size)
            INT16   seedWidth;      // number of adjacent symbols covered by the seed (i.e., the "k" in k-mer)
            UINT64  seedMask;       // bitmask that isolates the low-order symbols in an A21-encoded 64-bit value
            UINT64  hashBitsMask;   // bitmask that isolates the low-order bits of the hash value
            UINT64  sand;           // (used in MurmurHash implementation)
            INT32   seedMaskShift;  // number of bits to shift to obtain a k-mer in the low-order bits of an A21-encoded value
            bool    doCTconversion; // true: do CT conversion of Q sequence bases (bisulfite read alignment)
            UINT32  sps;            // 2: seeding from Q+ and Q-; 1: seeding from Q+ only

            KernelConstants()
            {
            }

            KernelConstants( AriocBase* _pab ) : maxJg(_pab->aas.ACP.maxJg),
                                                 seedWidth(_pab->a21hs.seedWidth),
                                                 seedMask(_pab->a21hs.seedMask),
                                                 hashBitsMask(_pab->a21hs.hashBitsMask),
                                                 sand(_pab->a21hs.seedWidth),
                                                 seedMaskShift(63-(3*_pab->a21hs.seedWidth)),
                                                 doCTconversion(_pab->a21hs.baseConvert == A21HashedSeed::bcCT),
                                                 sps(_pab->StrandsPerSeed)
            {
            }
        };

    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        KernelConstants         m_kc;
        DeviceBuffersBase*      m_pdbb;
        const UINT32            m_isi;      // seed-interval ("seed iteration") loop index
        UINT32                  m_nSeedPos; // number of seed positions for the current seed iteration
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    protected:
        baseCountJs( void );
        void main( void );

    private:
        void computeGridDimensions( dim3& d3g, dim3& d3b );
        UINT32 initSharedMemory( void );
        void initConstantMemory( void );
        void initGlobalMemory( void );
        void launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void copyKernelResults( void );
        void resetGlobalMemory( void );

    public:
        baseCountJs( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, UINT32 isi );
        virtual ~baseCountJs( void );
};
