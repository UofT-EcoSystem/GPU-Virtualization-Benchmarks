/*
  baseLoadJs.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseLoadJs__

/// <summary>
/// Class <c>baseLoadJs</c> loads J values from the seed-and-extend hash table (builds a list of Dj values)
/// </summary>
class baseLoadJs : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        DeviceBuffersBase*      m_pdbb;
        const UINT32*           m_pRuBuffer;    // subId bits
        UINT32                  m_isi;          // seed-interval ("seed iteration") loop index
        UINT32                  m_iSeedPos;     // index of the first seed position for the current seed iteration
        UINT32                  m_nSeedPos;     // number of seed positions for the current seed iteration
        UINT32                  m_nJ;           // total number of J values for the current iteration
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    protected:
        baseLoadJs( void );
        void main( void );

    private:
        void initGlobalMemory( void );
        void computeGridDimensions( dim3& d3g, dim3& d3b );
        void initConstantMemory( void );
        UINT32 initSharedMemory( void );
        void launchKernelD( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void launchKernelDj( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void copyKernelResults( void );
        void resetGlobalMemory( void );

    public:
        baseLoadJs( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, UINT32 isi, UINT32 nJ );
        virtual ~baseLoadJs( void );
};
