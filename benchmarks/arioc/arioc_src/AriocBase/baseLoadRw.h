/*
  baseLoadRw.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseLoadRw__

/// <summary>
/// Class <c>baseLoadRw</c> loads interleaved R sequence data for the windowed seed-and-extend gapped aligner
/// </summary>
class baseLoadRw : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*             m_pqb;
        AriocBase*          m_pab;
        UINT64*             m_pD;
        UINT32              m_nD;
        HiResTimer          m_hrt;

    protected:
        AriocTaskUnitMetrics* m_ptum;

    private:
        void computeGridDimensions( dim3& d3g, dim3& d3b );
        UINT32 initSharedMemory( void );
        void initConstantMemory( void );
        void initGlobalMemory( void );
        void launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void copyKernelResults( void );
        void resetGlobalMemory( void );

    protected:
        baseLoadRw( void );
        void main( void );

    public:
        baseLoadRw( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, RiFlags flagRi, UINT32 ofsD = 0, UINT32 nD = _UI32_MAX );
        virtual ~baseLoadRw( void );
};
