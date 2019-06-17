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

#if TODO_CHOP_WHEN_THE_ABOVE_WORKS
        void launchKernel11( void );
        void initGlobalMemory12( void );
        void launchKernel12( void );
#endif

        void setupN20( void );
        void launchKernel21( void );
        void resetGlobalMemory21( void );
        void launchKernel22( void );
        
        void setupN30( void );
        void launchKernel31( void );
        void initGlobalMemory32( void );
        void launchKernel32( void );
        void resetGlobalMemory32( void );

        void setupN40( void );
        void launchKernel40( void );
        void initGlobalMemory41( void );
        void launchKernel41( void );
        void resetGlobalMemory41( void );

    public:
        tuSetupN( QBatch* pqb );
        virtual ~tuSetupN( void );
};
