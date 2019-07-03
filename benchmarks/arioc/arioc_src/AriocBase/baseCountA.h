/*
  baseCountA.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseCountA__

/// <summary>
/// Class <c>baseCountA</c> copies per-Q mapping counts
/// </summary>
class baseCountA : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        DeviceBuffersBase*      m_pdbb;
        UINT64*                 m_pD;
        UINT32                  m_nD;
        bool                    m_isNongapped;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    protected:
        baseCountA( void );
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
        baseCountA( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, RiFlags flagRi );
        virtual ~baseCountA( void );
};