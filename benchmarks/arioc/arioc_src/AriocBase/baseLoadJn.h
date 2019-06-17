/*
  baseLoadJn.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseLoadJn__

/// <summary>
/// Class <c>baseLoadJn</c> loads J values from the spaced-seed hash table (builds a list of Df values)
/// </summary>
class baseLoadJn : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        UINT32                  m_npos;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    protected:
        baseLoadJn( void );
        void main( void );

    private:
        void computeGridDimensions( dim3& d3g, dim3& d3b );
        UINT32 initSharedMemory( void );
        void initConstantMemory( void );
        void initGlobalMemory( void );
        void launchKernelDf( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void launchKernelD( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void copyKernelResults( void );
        void resetGlobalMemory( void );

    public:
        baseLoadJn( const char* ptumKey, QBatch* pqb, UINT32 npos );
        virtual ~baseLoadJn( void );
};
