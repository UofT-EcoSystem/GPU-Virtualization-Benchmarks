/*
  baseMaxVw.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseMaxVw__

/// <summary>
/// Class <c>baseMaxVw</c> implements windowed gapped alignment for unmapped paired-end mates whose opposite mates have nongapped mappings.
/// </summary>
class baseMaxVw : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        CudaGlobalPtr<UINT32>   m_FV;
        DeviceBuffersG*         m_pdbg;
        UINT64*                 m_pD;
        UINT32                  m_nD;
        INT16*                  m_pVmax;
        HiResTimer              m_hrt;

    protected:
        AriocTaskUnitMetrics*   m_ptum;

    private:
        UINT32 initSharedMemory( void );
        void initConstantMemory( void );
        void initGlobalMemory( void );
        void launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void copyKernelResults( void );
        void resetGlobalMemory( void );

    protected:
        baseMaxVw( void );
        void main( void );

    public:
        baseMaxVw( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, RiFlags flagRi, UINT32 ofsD = 0, UINT32 nD = _UI32_MAX );
        virtual ~baseMaxVw( void );
};
