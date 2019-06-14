/*
  baseAlignG.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseAlignG__

#pragma region enums
enum TracebackDirection
{
    tdDm = 0x00,   // 00: diagonal (match)
    tdDx = 0x40,   // 01: diagonal (mismatch)
    tdH  = 0x80,   // 10: horizontal
    tdV  = 0xC0,   // 11: vertical
    tdO  = 0x80    // bit 1 only (set if horizontal or vertical)
};

enum TracebackDirection32
{
    td32Dm = 0x00000000,   // 00: diagonal (match)
    td32Dx = 0x40000000,   // 01: diagonal (mismatch)
    td32H  = 0x80000000,   // 10: horizontal
    td32V  = 0xC0000000    // 11: vertical
};
#pragma endregion

/// <summary>
/// Class <c>baseAlignG</c> uses a CUDA kernel to do gapped short-read alignments on a list of Q sequences.
/// </summary>
class baseAlignG : public tuBaseS, public CudaLaunchCommon
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        CudaGlobalPtr<UINT32>   m_SM;
        DeviceBuffersG*         m_pdbg;
        UINT64*                 m_pD;
        UINT32                  m_nD;
        INT16*                  m_pVmax;
        HostBuffers*            m_phb;
        UINT32                  m_nDperIteration;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    protected:
        baseAlignG( void );
        void main( void );

    private:
        UINT32 initSharedMemory( void );
        void initConstantMemory( void );
        void initGlobalMemory( void );
        void computeGridDimensions( dim3& d3g, dim3& d3b, UINT32 nDm );
        void launchKernel( UINT32 cbSharedPerBlock );
        void copyKernelResults( WinGlobalPtr<UINT32>* pwgpBRLEA );
        void resetGlobalMemory( void );

    public:
        baseAlignG( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, HostBuffers* phb, RiFlags flagRi );
        virtual ~baseAlignG( void );
};
