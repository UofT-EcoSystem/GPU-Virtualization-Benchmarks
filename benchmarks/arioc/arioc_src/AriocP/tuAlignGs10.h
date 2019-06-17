/*
  tuAlignGs10.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuAlignGs10__

/// <summary>
/// Class <c>tuAlignGs10</c> extracts QIDs and subId bits for unmapped mates of candidates mapped by the nongapped aligner
/// </summary>
class tuAlignGs10 : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    private:
        tuAlignGs10( void );
        void main( void );

        void computeGridDimensions( dim3& d3g, dim3& d3b );
        void initConstantMemory( void );
        UINT32 initSharedMemory( void );
        void initGlobalMemory( void );
        void launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void copyKernelResults( void );
        void resetGlobalMemory( void );

    public:
        tuAlignGs10( QBatch* pqb );
        virtual ~tuAlignGs10( void );
};
