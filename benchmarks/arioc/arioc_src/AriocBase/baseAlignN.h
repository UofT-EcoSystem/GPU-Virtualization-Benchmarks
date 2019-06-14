/*
  baseAlignN.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseAlignN__

/// <summary>
/// Class <c>baseAlignN</c> does nongapped short-read alignment
/// </summary>
class baseAlignN : public tuBaseS, public CudaLaunchCommon
{
    protected:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        DeviceBuffersN*         m_pdbn;
        UINT64*                 m_pD;
        UINT32                  m_nD;
        bool                    m_freeRi;
        bool                    m_baseConvertCT;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    private:
        void computeGridDimensions( dim3& d3g, dim3& d3b );

    protected:
        baseAlignN( void );
        void main( void );
        UINT32 initSharedMemory( void );
        void initConstantMemory( void );
        void initGlobalMemory( void );
        void launchKernel( UINT32 cbSharedPerBlock );
        void copyKernelResults( void );
        void resetGlobalMemory( void );

    public:
        baseAlignN( const char* ptumKey, QBatch* pqb, DeviceBuffersN* pdbn, RiFlags flagRi, bool freeRi );
        virtual ~baseAlignN( void );
};
