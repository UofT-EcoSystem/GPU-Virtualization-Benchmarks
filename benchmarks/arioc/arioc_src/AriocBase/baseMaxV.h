/*
  baseMaxV.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseMaxV__

/// <summary>
/// Class <c>baseMaxV</c> finds the highest dynamic-programming alignment score for a Q sequence
/// </summary>
class baseMaxV : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        UINT32                  m_nDperIteration;
        UINT32                  m_maxDperIteration;
        DeviceBuffersG*         m_pdbg;
        UINT64*                 m_pD;
        CudaGlobalPtr<INT16>*   m_pVmax;
        UINT32                  m_nD;
        CudaGlobalPtr<UINT32>   m_FV;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    protected:
        baseMaxV( void );
        void main( void );

    private:
        UINT32 initSharedMemory( void );
        void initConstantMemory( void );
        void initGlobalMemory( void );
        void launchKernel( UINT32 cbSharedPerBlock );
        void copyKernelResults( void );
        void resetGlobalMemory( void );

    public:
        baseMaxV( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, RiFlags flagRi );
        virtual ~baseMaxV( void );
};
