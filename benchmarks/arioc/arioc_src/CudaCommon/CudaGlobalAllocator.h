/*
  CudaGlobalAllocator.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __CudaGlobalAllocator__

#pragma region enums
enum cgaRegion
{
    cgaLow =    0x0001,
    cgaHigh =   0x0002,
    cgaBoth =   (cgaLow|cgaHigh)
};
#pragma endregion

/// <summary>
/// Class <c>CudaGlobalAllocator</c> allocates all available CUDA global memory and makes it possible to use CudaGlobalPtr<T> to manage suballocations -- the idea
///  being to stay away from repeated calls to NVidia's cudaMalloc/cudaFree APIs, which may lead to memory fragmentation and which (at least as of CUDA v4.2) still
///  leak memory.
/// </summary>
/// <remarks>
/// Suballocations are aligned to a predefined granularity (CudaGlobalAllocator::Granularity) to help with coalesced access to CUDA global memory.
/// WARNING: this implementation is not thread-safe.
/// </remarks>
class CudaGlobalAllocator
{
    private:
        struct cgaAllocation
        {
            void*               p;      // CUDA device pointer
            CudaGlobalPtrBase*  pcgpb;  // CudaGlobalPtrBase that contains this allocation
            size_t              cbCuda; // allocation size (total bytes of CUDA global memory in the allocation)
            bool                inUse;  // flag set if the device pointer has been allocated and not yet freed

            void set( void* _p, CudaGlobalPtrBase* _pcgpb, INT64 _cbCuda )
            {
                p = _p;
                pcgpb = _pcgpb;
                cbCuda = _cbCuda;
                inUse = true;
            }

            void* reuse( CudaGlobalPtrBase* _pcgpb )
            {
                pcgpb = _pcgpb;
                inUse = true;
                return p;
            }

            void reset( void )
            {
                p = NULL;
                pcgpb = NULL;
                cbCuda = 0;
                inUse = false;
            }
        };

    private:
        static const UINT32 MAXINSTANCES = 32;
        static CudaGlobalAllocator* m_Instance[MAXINSTANCES];

    public:
        static const INT64 Granularity = 128;

#if TODO_CHOP_WHEN_DEBUGGED
        static UINT64 msMalloc;
        static UINT64 nMalloc;
#endif

    private:
        WinGlobalPtr<cgaAllocation> m_Alo;  // list of current CGA allocations in the "low" region
        WinGlobalPtr<cgaAllocation> m_Ahi;  // list of current CGA allocations in the "high" region

    public:
        INT8*   p;          // base pointer to global allocation
        INT64   cb;         // size in bytes of global allocation
        INT8*   pLo;        // points to next available byte
        INT8*   pHi;        // points past the last available byte
        INT64   cbFree;     // number of available bytes (i.e. between pLo and pHi)

    private:
        void* allocLo( INT64 cb );
        void* allocHi( INT64 cb );
        void* privateAlloc( CudaGlobalPtrBase* _pcgpb, size_t _cb, cgaRegion _cgar );
        void freeLo( void* p );
        void freeHi( void* p );
        void setAlistAt( void* pSetAt, cgaRegion cgar = cgaLow );
        void dumpAlloc( WinGlobalPtr<cgaAllocation>* pal, UINT32 n, const char* scgar );
        CudaGlobalPtrBase** findCGPref( CudaGlobalPtrBase* const _pcgpb );

    public:
        CudaGlobalAllocator( CudaDeviceBinding& cdb, INT64 cbReserved = 0 );
        virtual ~CudaGlobalAllocator( void );
        INT64 GetAvailableByteCount( void );
        INT64 GetAllocatedByteCount( cgaRegion cgar );
        INT64 GetAllocatedByteCount( void* p );
        void* SetAt( void* pSetAt, cgaRegion cgar );
        void* Alloc( CudaGlobalPtrBase* pcgpb, cgaRegion cgar );    // called by CudaGlobalPtr implementation
        void* Alloc( size_t cb );                                   // "anonymous" allocator (e.g., called by Thrust)
        void Free( void* p, cgaRegion cgar = cgaLow );
        void* GetMruLo( void );                                     // most recent allocation in the low region
        void* GetMruHi( void );                                     // most recent allocation in the high region
        static CudaGlobalAllocator* GetInstance( void );
        void Swap( CudaGlobalPtrBase* pcgpb1, CudaGlobalPtrBase* pcgpb2 );
        void DumpUsage( const char* callerName );
};
