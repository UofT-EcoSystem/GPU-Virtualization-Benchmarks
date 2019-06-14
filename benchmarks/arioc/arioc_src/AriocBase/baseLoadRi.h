/*
   baseLoadRi.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseLoadRi__

/// <summary>
/// Class <c>baseLoadRi</c> loads interleaved R sequence data into CUDA global memory in preparation for alignment
/// </summary>
class baseLoadRi : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        DeviceBuffersBase*      m_pdbb;
        UINT64*                 m_pD;
        UINT32                  m_nD;
        AriocTaskUnitMetrics*   m_ptum;
        UINT64                  m_usXferR;
        HiResTimer              m_hrt;

    protected:
        baseLoadRi( void );
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
        baseLoadRi( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, RiFlags flagRi );
        virtual ~baseLoadRi( void );

        static INT64 ComputeRiBufsize( INT32 _Mr, UINT32 _nD );
};

