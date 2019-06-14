/*
  tuSortJgpu.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuSortJgpu__

/// <summary>
/// Class <c>tuSortJgpu</c> sorts J lists.
/// </summary>
class tuSortJgpu : public tuBaseS
{
    public:
        struct zeroJvalue8Tag
        {
            __host__ __device__
            UINT64 operator()( UINT64 x )
            {
                return x & 0x0000007FFFFFFFFF;
            }
        };

    private:
        struct JsortChunk
        {
            INT64   ofsJfirst;
            INT64   ofsJlast;
        };

        Jvalue8*                    m_pJ;
        INT64                       m_celJ;
        CudaGlobalPtr<UINT64>*      m_pJbuf;
        CudaDeviceBinding*          m_pcdb;
        CudaGlobalAllocator*        m_pcga;
        INT32                       m_cudaDeviceId;
        INT64                       m_gmemAvailable;
        UINT32                      m_celPerSort;
        WinGlobalPtr<JsortChunk>    m_JSC;
        HiResTimer                  m_hrt;

    protected:
        void main( void );

    private:
        tuSortJgpu( void );

        void saveSortChunkInfo( INT64 _ofsJfirst, INT64 _ofsJlast );
        void defineJlistChunks( void );
        void sortJ10( void );
        void initGlobalMemory10( JsortChunk* _pjsc );
        void launchKernel10( void );
        void copyKernelResults10( JsortChunk* _pjsc );
        void resetGlobalMemory10( void );

    public:
        tuSortJgpu( Jvalue8* _pJ, INT64 _celJ, INT64 _cbCgaReserved );
        virtual ~tuSortJgpu( void );
};
