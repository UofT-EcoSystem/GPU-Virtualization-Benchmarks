/*
  baseSetupJs.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseSetupJs__

/// <summary>
/// Class <c>baseSetupJs</c> groups J values by QID and seed position
/// </summary>
class baseSetupJs : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        DeviceBuffersBase*      m_pdbb;
        UINT32                  m_isi;          // seed-interval ("seed iteration") loop index
        UINT32                  m_nSeedPos;     // number of seed positions for the current seed iteration
        UINT32                  m_iQ;           // index of the first QID for the current iteration
        UINT32                  m_nJlists;      // total number of J lists for the current iteration
        AriocTaskUnitMetrics*   m_ptum;
        bool                    m_baseConvertCT;
        HiResTimer              m_hrt;

    protected:
        baseSetupJs( void );
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
        baseSetupJs( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, UINT32 isi, UINT32 iQ, UINT32 nQ );
        virtual ~baseSetupJs( void );
};
