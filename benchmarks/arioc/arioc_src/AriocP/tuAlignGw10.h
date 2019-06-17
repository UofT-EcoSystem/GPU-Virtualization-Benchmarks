/*
  tuAlignGw10.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuAlignGw10__

/// <summary>
/// Class <c>tuAlignGw10</c> extracts QIDs and subId bits for unmapped opposite mates of mapped candidates
/// </summary>
class tuAlignGw10 : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        DeviceBuffersBase*      m_pdbb;
        UINT64*                 m_pD;
        UINT32                  m_nD;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    private:
        tuAlignGw10( void );
        void main( void );

        void computeGridDimensions( dim3& d3g, dim3& d3b );
        void initConstantMemory( void );
        UINT32 initSharedMemory( void );
        void initGlobalMemory( void );
        void launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void copyKernelResults( void );
        void resetGlobalMemory( void );

    public:
        tuAlignGw10( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, RiFlags flagRi );
        virtual ~tuAlignGw10( void );
};
