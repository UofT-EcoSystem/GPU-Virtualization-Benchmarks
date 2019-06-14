/*
  CudaPinnedPtr.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __CudaPinnedPtr__

#define USE_CUDAHOSTALLOC 1

/// <summary>
/// Class <c>CudaPinnedPtr</c> allocates a page-locked host pointer that can be mapped into the CUDA global address space.
/// </summary>
template <class T> class CudaPinnedPtr
{
    public:
        T*              p;          // pointer (host address space)
        size_t          cb;         // number of allocated bytes
        size_t          Count;      // number of elements of sizeof(T)
        volatile UINT32 n;          // "tag"

        CudaPinnedPtr( void );
        CudaPinnedPtr( const size_t, const bool );
        CudaPinnedPtr( const CudaPinnedPtr<T>& other );
        ~CudaPinnedPtr( void );

        void Realloc( const size_t, const bool );
        void Free( void );
        T* GetDevicePointer( void );
};


// default constructor
template <class T> CudaPinnedPtr<T>::CudaPinnedPtr() : p(NULL), cb(0), Count(0), n(0)
{
}

/// constructor (size_t, bool): allocates a block of memory in the system heap, sized to contain the specified
///  number of elements of type T
template <class T> CudaPinnedPtr<T>::CudaPinnedPtr( const size_t nElements, const bool zeroAllocatedMemory ) : p(NULL), cb(0), Count(0), n(0)
{
    this->Realloc( nElements, zeroAllocatedMemory );
}

/// copy constructor (CudaPinnedPtr<T>&)
template <class T> CudaPinnedPtr<T>::CudaPinnedPtr( const CudaPinnedPtr<T>& other ) : p(NULL), cb(0), Count(0), n(0)
{
    // allocate memory for this object (and initialize count, p, and cb)
    this->Realloc( other.Count, false );

    // copy the data
    memcpy_s( p, cb, other.p, cb );

    // copy the "tag"
    n = other.n;
}

// destructor
template <class T> CudaPinnedPtr<T>::~CudaPinnedPtr()
{
    this->Free();
}

/// Allocates a block of pagelocked (pinned) memory (or reallocates it if it already exists):
///  - if memory has already been allocated for this CudaPinnedPtr instance, the previously allocated memory is discarded
///  - a new block of memory is allocated, sized to contain the specified number of elements of type T
template <class T> void CudaPinnedPtr<T>::Realloc( const size_t nElements, const bool zeroAllocatedMemory )
{
    CRVALIDATOR;

    // discard any previous memory allocation
    this->Free();

    // save the number of elements
    Count = nElements;

    // allocate memory
    cb = nElements * sizeof(T);

#if USE_CUDAHOSTALLOC
    /* Allocate page-locked memory:
        - cudaHostAllocPortable: allow "portable" access (i.e. from multiple GPUs)
        - cudaHostAllocMapped: map the buffer into the GPU address space so that its contents can be accessed in kernel code (via the PCIx bus)
        - cudaHostAllocWriteCombined: the allocated memory is presumably not cached in the CPU's L1 and L2 caches, so data transfer
            rates are supposedly ~40% faster although we haven't seen any change in performance with or without this flag; in any case, we don't
            use it because we want the CPU to be able to read from these buffers, too
        - FWIW cudaHostAlloc fails in CUDA v4.2 with allocations >= 4GB ...
    */
    try
    {
        RaiiCriticalSection<CudaPinnedPtr> rcs;

        CDPrint( cdpCD0, "%s: cudaHostAlloc( ..., %lld bytes, ... ) on thread %u", __FUNCTION__, cb, GetCurrentThreadId() );
        CRVALIDATE = cudaHostAlloc( &p, cb, cudaHostAllocPortable | cudaHostAllocMapped );
        CDPrint( cdpCD0, "%s: back from cudaHostAlloc()", __FUNCTION__ );
    }
    catch( ... )
    {
        const char details[] = "Unable to allocate page-locked system memory (CUDA \"pinned\" memory).  "
                               "Please ensure that there is sufficient system memory to execute this program.";
        throw new ApplicationException( __FILE__, __LINE__, details );
    }

    // optionally zero the allocated memory
    if( zeroAllocatedMemory )
        memset( p, 0, cb );

#else

    // use the OS to allocate a block of memory of the specified size
    DWORD dwFlags = HEAP_GENERATE_EXCEPTIONS | (zeroAllocatedMemory ? HEAP_ZERO_MEMORY : 0);
    p = (T*)::HeapAlloc( ::GetProcessHeap(), dwFlags, cb );

    // page-lock the memory and map it into the CUDA device address space
    CRVALIDATE = cudaHostRegister( p, cb, cudaHostRegisterPortable | cudaHostRegisterMapped );


#endif
}

/// frees previously allocated CUDA global memory
template <class T> void CudaPinnedPtr<T>::Free()
{
    CRVALIDATOR;

    // do nothing if this CudaPinnedPtr<T> instance is currently uninitialized
    void* pFree = InterlockedExchangePointer( reinterpret_cast<void**>(&this->p), NULL );
    if( pFree == NULL )
        return;

#if USE_CUDAHOSTALLOC
    // free the previously-allocated memory
    CRVALIDATE = cudaFreeHost( pFree );
#else

    CRVALIDATE = cudaHostUnregister( pFree );

    ::HeapFree( ::GetProcessHeap(), 0, pFree );

#endif


    // reset the relevant member variables
    cb = Count = n = 0;
}

/// maps the buffer pointer into the CUDA global address space in the current CUDA context (unneeded as of CUDA v5.0 in 64-bit Windows and Linux)
template <class T> T* CudaPinnedPtr<T>::GetDevicePointer()
{
    CRVALIDATOR;

    T* pCuda = NULL;
    CRVALIDATE = cudaHostGetDevicePointer( &pCuda, p, 0 );
    return pCuda;
}
