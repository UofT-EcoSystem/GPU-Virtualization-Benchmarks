/*
  tuXlatToDf.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuXlatToDf__

/// <summary>
/// Class <c>tuXlatToDf</c> translates Df values to D values
/// </summary>
class tuXlatToDf : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        UINT64*                 m_pDf;
        UINT32                  m_nD;
        UINT64*                 m_pD;
        UINT64                  m_maskDflags;
        UINT64                  m_newDflags;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    protected:
        void main( void );

    private:
        tuXlatToDf( void );
        void initConstantMemory( void );
        UINT32 initSharedMemory( void );
        void computeGridDimensions( dim3& d3g, dim3& d3b );
        void launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock );
        void copyKernelResults( void );

    public:
        tuXlatToDf( QBatch* pqb, UINT64* pDf, UINT32 nD, UINT64* pD, UINT64 maskDflags, UINT64 newDflags );
        virtual ~tuXlatToDf( void );
};

