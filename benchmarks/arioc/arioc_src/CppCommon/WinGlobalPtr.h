/*
  WinGlobalPtr.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __WinGlobalPtr__

#ifndef __ApplicationException__
#include "ApplicationException.h"
#endif


/// <summary>
/// Class <c>WinGlobalPtrHelper</c> implements a static member variable that limits the maximum size of a persistent
///  <c>WinGlobalPtr</c> allocation.
/// </summary>
class WinGlobalPtrHelper
{
    public:
        static size_t cbmaxWinGlobalAlloc;

    public:
        WinGlobalPtrHelper( void );
        virtual ~WinGlobalPtrHelper( void );

        static size_t EstimateAllocMax( size_t _cb );
};


/// <summary>
/// Class <c>WinGlobalPtr</c> wraps a Windows pointer with "resource acquisition is initialization" functionality.
/// </summary>
template <typename T> class WinGlobalPtr
{
    private:
        static WinGlobalPtrHelper   m_wgph;
        bool                        m_inCppHeap;

    public:
        T*              p;          // pointer
        size_t          cb;         // number of allocated bytes
        size_t          Count;      // number of elements of sizeof(T)
        UINT32 volatile n;          // (unused by the WinGlobalPtr implementation)

        WinGlobalPtr( void );
        WinGlobalPtr( const size_t nElements, const bool zeroAllocatedMemory );
        WinGlobalPtr( const WinGlobalPtr<T>& other );
        virtual ~WinGlobalPtr( void );

        void Realloc( const size_t nElements, const bool zeroAllocatedMemory );
        void Reuse( const size_t nElements, const bool zeroAllocatedMemory );
        void Free( void );
        void MoveTo( WinGlobalPtr<T>& target );
        void New( const size_t nElements, const bool zeroAllocatedMemory );
};


// default constructor
template <typename T> WinGlobalPtr<T>::WinGlobalPtr() : m_inCppHeap(false), p(NULL), cb(0), Count(0), n(0)
{
}

/// constructor (size_t, bool): allocates a block of memory in the system heap, sized to contain the specified
///  number of elements of type T
template <typename T> WinGlobalPtr<T>::WinGlobalPtr( const size_t nElements, const bool zeroAllocatedMemory ) : m_inCppHeap(false), p(NULL), cb(0), Count(0), n(0)
{
    this->Realloc( nElements, zeroAllocatedMemory );
}

/// copy constructor (WinGlobalPtr<T>&)
template <typename T> WinGlobalPtr<T>::WinGlobalPtr( const WinGlobalPtr<T>& other ) : m_inCppHeap(false), p(NULL), cb(0), Count(0), n(0)
{
    // allocate memory for this object (and initialize count, p, and cb)
    if( other.m_inCppHeap )
        this->New( other.Count, false );
    else
        this->Realloc( other.Count, false );

    // copy the data
    memcpy_s( p, cb, other.p, cb );

    // copy the "tag"
    n = other.n;
}

// destructor
template <typename T> WinGlobalPtr<T>::~WinGlobalPtr()
{
    this->Free();
}

/// Allocates a block of Windows global memory (or reallocates it if it already exists):
///  - if memory has already been allocated for this WinGlobalPtr instance, the previously allocated memory
///     is resized (and existing data is unchanged)
///  - otherwise, a new block of memory is allocated, sized to contain the specified number of elements of type T
template <typename T> void WinGlobalPtr<T>::Realloc( const size_t nElements, const bool zeroAllocatedMemory )
{
    // sanity check
    if( m_inCppHeap )
        throw new ApplicationException( __FILE__, __LINE__, "cannot reallocate in the C++ heap" );

    // save the number of elements
    Count = nElements;

    // allocate memory

#ifdef _WIN32
    cb = nElements * sizeof( T );

    HANDLE  hHeap = ::GetProcessHeap();
    DWORD dwFlags = HEAP_GENERATE_EXCEPTIONS | (zeroAllocatedMemory ? HEAP_ZERO_MEMORY : 0);

    if( this->p != NULL )
        this->p = reinterpret_cast<T*>(HeapReAlloc( hHeap, dwFlags, p, cb ));
    else
        this->p = reinterpret_cast<T*>(HeapAlloc( hHeap, dwFlags, cb ));
#endif

#ifdef __GNUC__
    size_t cbPrev = cb;
    cb = nElements * sizeof(T);

    if( this->p != NULL )
    {
        /* We always allocate a nonzero number of bytes, even if the caller specifies a zero-sized allocation.  This
            is because some Linuxen return NULL when malloc and realloc are called with a zero byte count. */
        this->p = reinterpret_cast<T*>(realloc( this->p, max2(cb,1) ));

        if( this->p == NULL )
            throw new ApplicationException( __FILE__, __LINE__, "realloc failed to allocate %lld bytes", cb );

        if( zeroAllocatedMemory && (cb > cbPrev) )
            memset( reinterpret_cast<INT8*>(this->p)+cbPrev, 0, (cb-cbPrev) );
    }
    else
    {
        this->p = reinterpret_cast<T*>(malloc( max2(cb,1) ));

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s: malloc'd cb=%lld bytes at 0x%016llx", __FUNCTION__, cb, this->p );
#endif

        if( this->p == NULL )
        {
            size_t cbAvailable = WinGlobalPtrHelper::EstimateAllocMax( cb );
            throw new ApplicationException( __FILE__, __LINE__, "malloc failed to allocate %lld bytes (%lld bytes available)", cb, cbAvailable );
        }

    	if( zeroAllocatedMemory && cb )
            memset( this->p, 0, cb );
    }
#endif

    CDPrint( cdpCDa, "WinGlobalPtr allocated %lld bytes at 0x%016llx", cb, this->p );
}

/// Reuses or reallocates a previously-allocated block of Windows global memory:
///  - if the currently-allocated size exceeds a threshold size (WinGlobalPtrHelper::cbmaxWinGlobalAlloc), the
///     block is reallocated to the specified size; otherwise
///  - if the currently-allocated size exceeds the specified size, the block is reused (i.e., not reallocated); otherwise
///  - if memory has already been allocated for this WinGlobalPtr instance, the previously allocated memory
///     is resized (and existing data is unchanged); otherwise
///  - a new block of memory is allocated, sized to contain the specified number of elements of type T
template <typename T> void WinGlobalPtr<T>::Reuse( const size_t nElements, const bool zeroAllocatedMemory )
{
    // sanity check
    if( m_inCppHeap )
        throw new ApplicationException( __FILE__, __LINE__, "cannot reallocate in the C++ heap" );

    /* The idea here is to avoid thrashing the heap by frequently reallocating a buffer.  We do this by reusing
        the previous allocation when possible and growing the allocation when necessary.
        
       Unfortunately, this strategy can fail because the largest allocation is the one that persists; in the
        case where extremely large (e.g., multi-gigabyte) allocations occur, this in effect causes the
        application to behave as if it had leaked memory.  We work around this by setting a threshold beyond
        which such large allocations are freed instead of persisted.
    */

    // reallocate if...
    if( (nElements > this->Count) ||                            // the specified size exceeds the currently-allocated size
        (this->cb > WinGlobalPtrHelper::cbmaxWinGlobalAlloc) )  // the currently-allocated size exceeds the threshold
    {
        this->Realloc( nElements, zeroAllocatedMemory );
        return;
    }

    /* At this point we can reuse the current allocation by leaving its size unchanged (i.e., no reallocation occurs
        and the cb and Count fields do not change). */

    // conditionally zero the unused part of the buffer
    if( zeroAllocatedMemory )
    {
        size_t cbnew = nElements * sizeof(T);
        if( cbnew < this->cb )
            memset( reinterpret_cast<INT8*>(this->p)+cbnew, 0, (this->cb-cbnew) );
    }

    CDPrint( cdpCDa, "WinGlobalPtr reused %lld bytes at 0x%016llx", this->cb, this->p );
}

/// frees previously allocated CUDA global memory
template <typename T> void WinGlobalPtr<T>::Free()
{
    // null the current pointer
    void* pFree = InterlockedExchangePointer( reinterpret_cast<void**>(&this->p), NULL );
    if( pFree != NULL )
    {
        // free allocated memory
        if( m_inCppHeap )
        {
            delete[] reinterpret_cast<T*>(pFree);
            m_inCppHeap = false;
            CDPrint( cdpCDa, "WinGlobalPtr freed %lld bytes from the C++ heap at 0x%016llx", cb, pFree );
        }
        else
        {
            // free the previously-allocated memory
#ifdef _WIN32
            HANDLE hHeap = ::GetProcessHeap();
            HeapFree( hHeap, 0, pFree );
#endif

#ifdef __GNUC__
            free( pFree );
#endif

            CDPrint( cdpCDa, "WinGlobalPtr freed %lld bytes at 0x%016llx", cb, pFree );
        }
    }

    // zero the counts
    cb = 0;
    Count = 0;
    n = 0;
}

/// moves a pointer from one WinGlobalPtr&lt;T&gt; instance to another
template <typename T> void WinGlobalPtr<T>::MoveTo( WinGlobalPtr<T>& other )
{
#ifndef __NVCC__                // (NVCC won't compile throw new ApplicationException)
    // sanity check
    if( other.p != NULL )
        throw new ApplicationException( __FILE__, __LINE__, "target WinGlobalPtr<T> has already been allocated" );
#endif

    // copy the current pointer info to the other WinGlobalPtr<T> instance
    other.cb = this->cb;
    other.Count = this->Count;
    other.n = this->n;
    other.p = this->p;

    // reset the current WinGlobalPtr<T> instance
    this->cb = 0;
    this->Count = 0;
    this->n = 0;
    this->p = NULL;
}

/// uses the C++ heap and new/delete semantics
template <typename T> void WinGlobalPtr<T>::New( const size_t nElements, const bool zeroAllocatedMemory )
{
#ifndef __NVCC__                // (NVCC won't compile throw new ApplicationException)
    if( this->p )
        throw new ApplicationException( __FILE__, __LINE__, "WinGlobalPtr<T> at 0x%016llx has already been allocated (%lld bytes)", p, cb );
#endif

    // save the number of elements
    Count = nElements;

    // allocate memory
    cb = nElements * sizeof(T);
    this->p = new T[nElements];

    // optionally zero the allocated memory
    if( zeroAllocatedMemory )
        memset( this->p, 0, cb );

    // set a flag to indicate how the allocation will need to be freed
    m_inCppHeap = true;

    CDPrint( cdpCDa, "WinGlobalPtr allocated %lld bytes in the C++ heap at 0x%016llx", cb, this->p );
}
