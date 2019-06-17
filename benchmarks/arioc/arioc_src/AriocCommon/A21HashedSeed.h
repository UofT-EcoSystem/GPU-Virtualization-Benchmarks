/*
  A21HashedSeed.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __A21HashedSeed__

#if !defined(__A21SeedBase__)
#include "A21SeedBase.h"
#endif

class A21HashedSeed : public A21SeedBase
{
    public:
        static const UINT32 nSPSI = 6;          // number of seed iterations
        static const UINT16 celSPSI = 15000;    // maximum size of SPSI lookup table (see comments in AriocBase/baseCountJs.cu)

    private:
        INT32   m_iHSP;
        Hash    m_Hash;

    public:
        WinGlobalPtr<UINT32>    ofsSPSI;        // offsets of seed positions for seed iterations (one offset per seed iteration)
        WinGlobalPtr<UINT16>    SPSI;           // seed positions for seed iterations

    private:
        void initSI( void );
        void initHSP32( INT16 _hashSeedWidth, INT16 _maxMismatches, INT16 _hashKeyWidth );

    public:
        A21HashedSeed( const char* _si );
        ~A21HashedSeed( void );
        UINT32 ComputeH32( UINT64 u64 );                                    // implements A21SeedBase::ComputeH32( UINT64 )
        UINT32 ComputeH32( UINT64 k1, UINT64 k2 );                          // implements A21SeedBase::ComputeH32( UINT64, UINT64 )
        UINT32 ComputeH32( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4 );    // implements A21SeedBase::ComputeH32( UINT64, UINT64, UINT64, UINT64 )
        UINT64 ComputeH64( UINT64 u64 );                                    // implements A21SeedBase::ComputeH64( UINT64 )
        UINT64 ComputeH64( UINT64 k1, UINT64 k2 );                          // implements A21SeedBase::ComputeH64( UINT64, UINT64 )
        UINT64 ComputeH64( UINT64 k1, UINT64 k2, UINT64 k3, UINT64 k4 );    // implements A21SeedBase::ComputeH64( UINT64, UINT64, UINT64, UINT64 )
        UINT64 ComputeH64_61( UINT64 k1, UINT64 k2 );                       // special-case implementation for 61-bit hash
};
