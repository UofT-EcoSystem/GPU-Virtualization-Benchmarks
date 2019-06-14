/*
  CudaGlobalPtr.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __CudaGlobalPtr__

/// <summary>
/// Class <c>CudaGlobalPtr</c> wraps a CUDA global device pointer with "resource acquisition is initialization" functionality.
/// </summary>
template <class T> class CudaGlobalPtr : public CudaGlobalPtrBase
{
    public:
        T*  p;      // CUDA device pointer

    public:
        CudaGlobalPtr( void );
        CudaGlobalPtr( const size_t nElements, const bool zeroAllocatedMemory );
        CudaGlobalPtr( CudaGlobalAllocator* pcga );
        virtual ~CudaGlobalPtr( void );

        void CopyToDevice( const void* hostPtr, size_t nElements );
        void CopyToHost( void* hostPtr, size_t nElements );
        void CopyToDeviceAsync( const void* hostPtr, size_t nElements, cudaStream_t streamId );
        void CopyToHostAsync( void* hostPtr, size_t nElements, cudaStream_t streamId );
        void CopyInDevice( void* devicePtr, size_t nElements );

        void Realloc( const size_t nElements, const bool zeroAllocatedMemory );
        void Alloc( const cgaRegion cgar, const size_t nElements, const bool zeroAllocatedMemory );
        void Resize( const size_t nElements );
        void Free( void );

        void Swap( CudaGlobalPtr<T>* other );
        void ShuffleHiLo( bool wantMRUexception = false );
        void ShuffleLoHi( bool wantMRUexception = false );
        void SetTag( const char* ptag );
        void SetTag( WinGlobalPtr<char>& wgptag );
        char* GetTag( void );
};


// default constructor
template <class T> CudaGlobalPtr<T>::CudaGlobalPtr() : CudaGlobalPtrBase(), p(NULL)
{
}

/// constructor (size_t, bool): allocates a block of CUDA global memory, sized to contain the specified number of elements of type T
template <class T> CudaGlobalPtr<T>::CudaGlobalPtr( const size_t nElements, const bool zeroAllocatedMemory ) : CudaGlobalPtrBase(), p(NULL)
{
    this->Realloc( nElements, zeroAllocatedMemory );
}

/// constructor (CudaGlobalAllocator&)
template <class T> CudaGlobalPtr<T>::CudaGlobalPtr( CudaGlobalAllocator* pCGA ) : CudaGlobalPtrBase(pCGA), p(NULL)
{
}

/// destructor
template <class T> CudaGlobalPtr<T>::~CudaGlobalPtr()
{
    m_dtor = true;
    this->Free();
}


/// Reallocates a block of CUDA global memory:
///  - previously allocated CUDA device memory is freed
///  - a new block of CUDA device memory is allocated, sized to contain the specified number of elements of type T
template <class T> void CudaGlobalPtr<T>::Realloc( const size_t nElements, const bool zeroAllocatedMemory )
{
    CRVALIDATOR;

    // sanity check
    if( m_pCGA != NULL )
        throw new ApplicationException( __FILE__, __LINE__, "cannot use Realloc with a CudaGlobalAllocator (use Alloc instead)" );

    // discard any previous memory allocation
    this->Free();

    // save the specified flag, number of elements, and number of bytes in the memory allocation
    m_initZero = zeroAllocatedMemory;
    Count = nElements;
    cb = nElements * sizeof(T);
    m_cbCuda = round2power( cb, CUDAMALLOCGRANULARITY );

#ifdef DEBUG_CUDAGLOBALPTR
char msg[128];
sprintf( msg, "CudaGlobalPtr::Realloc( %08llX ) (thread=%d (0x%x), cb=%lld ...", p, GetCurrentThreadId(), GetCurrentThreadId(), cb );
OutputDebugString( msg );
#endif

    // allocate CUDA global memory
    CRVALIDATE = cudaMalloc( &p, cb );

#ifdef DEBUG_CUDAGLOBALPTR
sprintf( msg, " p = %08llX\r\n", p );
OutputDebugString( msg );
#endif

    // optionally zero the allocated memory
    if( zeroAllocatedMemory )
        CRVALIDATE = cudaMemset( p, 0, cb );
}

/// allocates a sub-block of CUDA global memory from a preallocated block managed by a CudaGlobalAllocator
template <class T> void CudaGlobalPtr<T>::Alloc( cgaRegion cgar, const size_t nElements, const bool zeroAllocatedMemory )
{
    CRVALIDATOR;

    // sanity checks
    if( m_pCGA == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "CudaGlobalPtr<T> constructed without CudaGlobalAllocator" );
    if( this->p )
        throw new ApplicationException( __FILE__, __LINE__, "CudaGlobalPtr<T> previously allocated at 0x%016llx", this->p );

    // save the specified values
    m_initZero = zeroAllocatedMemory;
    this->Count = nElements;
    this->cb = nElements * sizeof(T);

    // grab the current amount of available global memory
    INT64 cbBefore = m_pCGA->cbFree;

    // do the memory allocation
    this->p = static_cast<T*>(m_pCGA->Alloc( this, cgar ));

    // record the actual number of bytes consumed by the allocation (which may be larger than the requested size because of alignment)
    m_cbCuda = cbBefore - m_pCGA->cbFree;
    if( m_cbCuda == 0 )
    {
        // if the number of bytes consumed by CGA allocations did not change, the CGA must have reused a previous allocation...
        m_cbCuda = m_pCGA->GetAllocatedByteCount( this->p );
        if( m_cbCuda == 0 )
            throw new ApplicationException( __FILE__, __LINE__, "unexpected zero-length allocation" );
    }

    // optionally zero the allocated memory
    if( zeroAllocatedMemory )
        CRVALIDATE = cudaMemset( this->p, 0, m_cbCuda );

    CDPrint( cdpCDb, "CudaGlobalPtr<T>::Alloc at 0x%016llx (%s)", this->p, (cgar==cgaLow) ? "cgaLow" : "cgaHigh" );
}

/// resizes a CUDA global memory allocation
template <class T> void CudaGlobalPtr<T>::Resize( const size_t nElements )
{
    // sanity checks
    if( m_pCGA == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "CudaGlobalPtr<T> constructed without CudaGlobalAllocator" );

    if( reinterpret_cast<INT8*>(this->p) >= m_pCGA->pHi )
        throw new ApplicationException( __FILE__, __LINE__, "cannot resize an allocation created with cgaHigh" );

    if( (reinterpret_cast<INT8*>(this->p) + m_cbCuda) != m_pCGA->pLo )
        throw new ApplicationException( __FILE__, __LINE__, "Resize may only be called on the most recent cgaLow allocation" );

    // update the size of the current allocation
    this->Count = nElements;
    this->cb = nElements * sizeof(T);

    // enforce a minimum allocation size
    INT64 cbCuda = max2( this->cb, static_cast<size_t>(CudaGlobalAllocator::Granularity) );
    INT8* pSetAtMin = reinterpret_cast<INT8*>(this->p) + cbCuda;

    // update the CudaGlobalAllocator to resize the current allocation
    INT8* pSetAt = reinterpret_cast<INT8*>(this->p + nElements);
    if( pSetAt > pSetAtMin )
        pSetAt = reinterpret_cast<INT8*>(m_pCGA->SetAt( pSetAt, cgaLow ));
    else
        pSetAt = reinterpret_cast<INT8*>(m_pCGA->SetAt( pSetAtMin, cgaLow ));

    m_cbCuda = pSetAt - reinterpret_cast<INT8*>(this->p);
}

/// frees previously allocated CUDA global memory
template <class T> void CudaGlobalPtr<T>::Free()
{
#ifdef DEBUG_CUDAGLOBALPTR
char msg[128];
sprintf( msg, "CudaGlobalPtr::Free( %08llX ) (thread=%d (0x%x) cb=%lld\r\n", pFree, GetCurrentThreadId(), GetCurrentThreadId(), cb );
OutputDebugString( msg );
#endif

    // get a copy of the current pointer; do nothing if the pointer was never allocated or has already been freed
    void* pFree = InterlockedExchangePointer( reinterpret_cast<void**>(&this->p), NULL );
    if( pFree == NULL )
        return;

    // special case for allocations using a CudaGlobalAllocator
    if( m_pCGA != NULL )
    {
        cgaRegion cgar = (static_cast<INT8*>(pFree) < m_pCGA->pLo) ? cgaLow : cgaHigh;
        m_pCGA->Free( pFree, cgar );
    }
    else
    {
        // free the device global memory
        cudaError_t rval = cudaFree( pFree );
        if( !m_dtor )
        {
            CRVALIDATOR;
            CRVALIDATE = rval;
        }
    }

    // zap the counts
    this->Count = 0;
    this->cb = 0;
    m_cbCuda = 0;
    this->n = 0;
}

/// copies data from host (CPU) memory to CUDA global memory using cudaMemcpy
template <class T> void CudaGlobalPtr<T>::CopyToDevice( const void* hostPtr, size_t nElements )
{
    CRVALIDATOR;

    // verify that the number of elements to copy is valid compared to the amount of allocated device memory
    if( nElements > this->Count )
        throw new ApplicationException( __FILE__, __LINE__, "CudaGlobalPtr::CopyToDevice: %d elements specified for CudaGlobalPtr with Count = %d", nElements, this->Count );

    // copy 
    CRVALIDATE = cudaMemcpy( p, hostPtr, nElements*sizeof(T), cudaMemcpyHostToDevice );
}

/// copies data to host (CPU) memory from CUDA global memory using cudaMemcpy
template <class T> void CudaGlobalPtr<T>::CopyToHost( void* hostPtr, size_t nElements )
{
    CRVALIDATOR;

    // verify that the number of elements to copy is valid compared to the amount of allocated device memory
    if( nElements > this->Count )
        throw new ApplicationException( __FILE__, __LINE__, "CudaGlobalPtr::CopyToHost: %d elements specified for CudaGlobalPtr with Count = %d", nElements, this->Count );

    // copy 
    CRVALIDATE = cudaMemcpy( hostPtr, p, nElements*sizeof(T), cudaMemcpyDeviceToHost );
}

/// copies data from host (CPU) memory to CUDA global memory using cudaMemcpyAsync
template <class T> void CudaGlobalPtr<T>::CopyToDeviceAsync( const void* hostPtr, size_t nElements, cudaStream_t streamId )
{
    CRVALIDATOR;

    // verify that the number of elements to copy is valid compared to the amount of allocated device memory
    if( nElements > this->Count )
        throw new ApplicationException( __FILE__, __LINE__, "CudaGlobalPtr::CopyToDeviceAsync: %d elements specified for Count = %d", nElements, this->Count );

    // copy 
    CRVALIDATE = cudaMemcpyAsync( p, hostPtr, nElements*sizeof(T), cudaMemcpyHostToDevice, streamId );
}

/// copies data to host (CPU) memory from CUDA global memory using cudaMemcpyAsync
template <class T> void CudaGlobalPtr<T>::CopyToHostAsync( void* hostPtr, size_t nElements, cudaStream_t streamId )
{
    CRVALIDATOR;

    // verify that the number of elements to copy is valid compared to the amount of allocated device memory
    if( nElements > this->Count )
        throw new ApplicationException( __FILE__, __LINE__, "CudaGlobalPtr::CopyToHostAsync: %d elements specified for CudaGlobalPtr with Count = %d", nElements, this->Count );

    // copy 
    CRVALIDATE = cudaMemcpyAsync( hostPtr, p, nElements*sizeof(T), cudaMemcpyDeviceToHost, streamId );
}

/// copies contents of the current CudaGlobalPtr<T> instance to the specified CUDA address
template <class T> void CudaGlobalPtr<T>::CopyInDevice( void* devicePtr, size_t nElements )
{
    CRVALIDATOR;

    // verify that the number of elements to copy is valid compared to the amount of allocated device memory
    if( nElements > this->Count )
        throw new ApplicationException( __FILE__, __LINE__, "CudaGlobalPtr::CopyInDevice: %d elements specified for CudaGlobalPtr with Count = %d", nElements, this->Count );

    // copy
    CRVALIDATE = cudaMemcpy( devicePtr, p, nElements*sizeof(T), cudaMemcpyDeviceToDevice );
}

/// swaps the contents of the current CudaGlobalPtr&lt;T&gt; instance with the contents of another specified CudaGlobalPtr&lt;T&gt; instance
template <class T> void CudaGlobalPtr<T>::Swap( CudaGlobalPtr<T>* other )
{
    // copy the member variables to temporaries
    bool temp_initZero =             m_initZero;
    bool temp_dtor =                 m_dtor;
    T* temp_p =                      this->p;
    size_t temp_cb =                 this->cb;
    size_t temp_Count =              this->Count;
    UINT32 temp_n =                  this->n;
    size_t temp_cbCuda =             m_cbCuda;
    CudaGlobalAllocator* temp_pCGA = m_pCGA;

    WinGlobalPtr<char> temp_tag( this->tag.Count, false );
    memcpy_s( temp_tag.p, temp_tag.cb, this->tag.p, this->tag.cb );

    // copy the other instance's member variable values to this instance
    m_initZero = other->m_initZero;
    m_dtor = other->m_dtor;
    this->p = other->p;
    this->cb = other->cb;
    this->Count = other->Count;
    this->n = other->n;
    m_cbCuda = other->m_cbCuda;
    m_pCGA = other->m_pCGA;
    this->SetTag( other->tag );

    // copy this instance's previous member variable values to the other instance
    other->m_initZero = temp_initZero;
    other->m_dtor = temp_dtor;
    other->p = temp_p;
    other->cb = temp_cb;
    other->Count = temp_Count;
    other->n = temp_n;
    other->m_cbCuda = temp_cbCuda;
    other->m_pCGA = temp_pCGA;
    other->SetTag( temp_tag );

    // update the references in the CudaGlobalAllocator
    m_pCGA->Swap( this, other );
}

/// moves the current allocation from the high to the low region
template <class T> void CudaGlobalPtr<T>::ShuffleHiLo( bool wantMRUexception )
{
    // sanity checks
    if( m_pCGA == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "CudaGlobalPtr<T> constructed without CudaGlobalAllocator" );
    if( wantMRUexception && (this->p != m_pCGA->GetMruHi()) )
        throw new ApplicationException( __FILE__, __LINE__, "%s: CudaGlobalPtr 0x%016llx (%s) is not the most recent allocation in the high memory region",
                                                            __FUNCTION__,
                                                            this->p,
                                                            this->tag.cb ? this->tag.p : "(no tag)" );

    // get the number of bytes between the end of the low-memory region and the start of the high-memory region
    size_t cbMaxChunk = m_pCGA->cbFree;

    // free the current (high-region) allocation; the CudaGlobalAllocator::Free implementation does not affect the data in memory
    m_pCGA->Free( this->p, cgaHigh );

    // create a new low-region allocation; again, data in memory is unaffected
    T* pNew = reinterpret_cast<T*>(m_pCGA->Alloc( this, cgaLow ));

    // copy the data; ensure that there is no overlap between the old and new allocations
    INT8* pTo = reinterpret_cast<INT8*>(pNew);
    INT8* pFrom = reinterpret_cast<INT8*>(this->p);
    size_t cbRemaining = m_cbCuda;
    while( cbRemaining )
    {
        CRVALIDATOR;

        // copy a chunk of data
        size_t cb = min2(cbRemaining,cbMaxChunk);
        CRVALIDATE = cudaMemcpy( pTo, pFrom, cb, cudaMemcpyDeviceToDevice );

        // iterate
        cbRemaining -= cb;
        pTo += cb;
        pFrom += cb;
    }

    // update this instance's pointer
    this->p = pNew;
}

/// moves the current allocation from the low to the high region
template <class T> void CudaGlobalPtr<T>::ShuffleLoHi( bool wantMRUexception )
{
    // sanity checks
    if( m_pCGA == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "CudaGlobalPtr<T> constructed without CudaGlobalAllocator" );
    if( wantMRUexception && (this->p != m_pCGA->GetMruLo()) )
        throw new ApplicationException( __FILE__, __LINE__, "%s: CudaGlobalPtr 0x%016llx (%s) is not the most recent allocation in the low memory region",
                                                            __FUNCTION__,
                                                            this->p,
                                                            this->tag.cb ? this->tag.p : "(no tag)" );

    // get the number of bytes between the end of the low-memory region and the start of the high-memory region
    size_t cbMaxChunk = m_pCGA->cbFree;

    // free the current (low-region) allocation
    m_pCGA->Free( this->p, cgaLow );

    // create a new high-region allocation; again, data in memory is unaffected
    T* pNew = reinterpret_cast<T*>(m_pCGA->Alloc( this, cgaHigh ));

    // copy the data; ensure that there is no overlap between the old and new allocations
    INT8* pTo = reinterpret_cast<INT8*>(pNew) + m_cbCuda;
    INT8* pFrom = reinterpret_cast<INT8*>(this->p) + m_cbCuda;
    size_t cbRemaining = m_cbCuda;
    while( cbRemaining )
    {
        CRVALIDATOR;

        // copy a chunk of data
        size_t cb = min2(cbRemaining,cbMaxChunk);
        pTo -= cb;
        pFrom -= cb;
        CRVALIDATE = cudaMemcpy( pTo, pFrom, cb, cudaMemcpyDeviceToDevice );

        // iterate
        cbRemaining -= cb;
    }

    // update this instance's pointer
    this->p = pNew;
}

/// copies the specified null-terminated string into the "tag" field
template <class T> void CudaGlobalPtr<T>::SetTag( const char* ptag )
{
    // zap the previous tag (if any)
    this->tag.Free();

    if( ptag && *ptag )
    {
        // save a copy of the new tag
        size_t cb = strlen( ptag ) + 1;     // number of bytes to save as the new tag
        this->tag.Realloc( cb, false );
        strcpy_s( this->tag.p, this->tag.cb, ptag );
    }
}

/// copies the specified null-terminated string into the "tag" field
template <class T> void CudaGlobalPtr<T>::SetTag( WinGlobalPtr<char>& wgptag )
{
    // zap the previous tag (if any)
    this->tag.Free();

    if( wgptag.cb )
    {
        // save a copy of the new tag
        this->tag.Realloc( wgptag.cb, false );
        memcpy_s( this->tag.p, this->tag.cb, wgptag.p, wgptag.cb );
    }
}

/// returns a pointer to the "tag" field
template <class T> char* CudaGlobalPtr<T>::GetTag()
{
    return this->tag.p;
}

#if defined(_DEBUG)
#define SET_CUDAGLOBALPTR_TAG(cgp,s) cgp.SetTag(s)
#else
#define SET_CUDAGLOBALPTR_TAG(cgp,s)
#endif
