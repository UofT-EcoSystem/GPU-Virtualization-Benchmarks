/*
  tuSetupN.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuSetupN__

#ifndef __tuSetupN24__
#include "tuSetupN24.h"
#endif


/// <summary>
/// Class <c>tuSetupN</c> obtains J-list offsets and sizes for the seed-and-extend seeds in a given set of Q sequences
/// </summary>
class tuSetupN : public tuBaseS
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        UINT32                  m_nCandidates;
        CudaGlobalPtr<UINT64>   m_Dx;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    protected:
        void main( void );

    private:
        tuSetupN( void );

        void setupN10( void );
        void initGlobalMemory12( void );
        void launchKernel12( void );
        void initGlobalMemory14( void );
        void launchKernel14( void );

        void setupN20( void );
        void launchKernel21( void );
        void resetGlobalMemory21( void );
        void launchKernel22( void );
        void initGlobalMemory26( void );
        void launchKernel26( void );
        void initGlobalMemory28( void );
        void launchKernel28( void );
        void resetGlobalMemory28( void );

    public:
        tuSetupN( QBatch* pqb );
        virtual ~tuSetupN( void );
};
