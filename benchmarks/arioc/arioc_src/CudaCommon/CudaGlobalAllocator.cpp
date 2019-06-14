/*
  CudaGlobalAllocator.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variable definitions
CudaGlobalAllocator* CudaGlobalAllocator::m_Instance[MAXINSTANCES];

#if TODO_CHOP_WHEN_DEBUGGED
#include "HiResTimer.h"
UINT64 CudaGlobalAllocator::msMalloc;
UINT64 CudaGlobalAllocator::nMalloc;
#endif

#pragma endregion

#pragma region constructor and destructor
/// default constructor
CudaGlobalAllocator::CudaGlobalAllocator( CudaDeviceBinding& cdb, INT64 cbReserved ) : p(NULL), cb(0), pLo(NULL), pHi(NULL), cbFree(0)
{
    CRVALIDATOR;

    // get available global memory for the CUDA device associated with the current CUDA context.
    this->cb = cdb.GetDeviceFreeGlobalMemory();

    // reserve the specified number of bytes outside this CudaGlobalAllocator instance
    if( this->cb < cbReserved )
        throw new ApplicationException( __FILE__, __LINE__, "Unable to reserve %lld bytes of CUDA global memory (%lld bytes available)", cbReserved, this->cb );
    this->cb -= cbReserved;

    /* The cudaMemGetInfo API reports the total amount of available global memory, not the amount that can actually be allocated by a call to the
        cudaMalloc API.  For this reason we call cudaMalloc repeatedly, with progressively smaller allocation requests, until we succeed. */
    cudaError_t rval = cudaMalloc( &this->p, this->cb );

    for( INT16 n=0; (rval != cudaSuccess) && (n<200); ++n )
    {
        // decrease the memory request by 2MB and try again
        this->cb -= 2*1024*1024;
        rval = cudaMalloc( &this->p, this->cb );
    }

    if( rval != cudaSuccess )
        throw new ApplicationException( __FILE__, __LINE__, "unable to allocate GPU global memory" );

    /* At this point we have presumably allocated all available GPU global memory. */
    CDPrint( cdpCD2, "%s: CudaGlobalAllocator uses %lld bytes", __FUNCTION__, this->cb );

    // initialize global memory
    CRVALIDATE = cudaMemset( this->p, 0xCE, this->cb );

    // initialize the allocation state
    this->pLo = reinterpret_cast<INT8*>(round2power(reinterpret_cast<size_t>(this->p), CudaGlobalAllocator::Granularity));  // align to a multiple of Granularity bytes
    this->cb &= (-CudaGlobalAllocator::Granularity);                                                                        // round down to a multiple of Granularity bytes
    this->pHi = this->p + this->cb;
    this->cbFree = this->cb;
    
    CDPrint( cdpCDb, "[%d] %s: pLo=0x%016llx pHi=0x%016llx cbFree=%lld", cdb.GetDeviceProperties()->cudaDeviceId, __FUNCTION__, this->pLo, this->pHi, this->cbFree );

    // initialize the allocation lists
    m_Alo.Realloc( 32, true );  // start with space for 32 allocations
    m_Ahi.Realloc( 32, true );
    m_Alo.n = m_Ahi.n = 0;      // initialize the count of the number of allocations

    // save a reference to this CGA instance in a static list
    m_Instance[cdb.GetDeviceProperties()->cudaDeviceId] = this;
}

/// destructor
CudaGlobalAllocator::~CudaGlobalAllocator()
{
    // null this instance's entry in the static list
    int deviceId = CudaDeviceBinding::GetCudaDeviceId();
    m_Instance[deviceId] = NULL;

    // zap the allocation lists
    m_Alo.n = m_Ahi.n = 0;

    // free the CUDA global memory (without worrying about error handling)
    cudaFree( this->p );
}
#pragma endregion

#pragma region private methods
/// [private] method allocLo
void* CudaGlobalAllocator::allocLo( INT64 cb )
{
    /* Compute the total number of bytes needed for the allocation:
        - include padding so as to enforce the configured granularity
        - the minimum allocation is one unit of granularity
    */
    INT64 cbNeeded = round2power( cb, CudaGlobalAllocator::Granularity );
    cbNeeded = max2( cbNeeded, CudaGlobalAllocator::Granularity );

    // sanity check
    if( cbNeeded > this->cbFree )
        throw new ApplicationException( __FILE__, __LINE__, "unable to allocate %lld bytes (%lld bytes available)", cbNeeded, this->cbFree );

    // grab a copy of the current pointer to available memory in the "low" region
    void* p = this->pLo;

    // update the member variables
    this->pLo += cbNeeded;
    this->cbFree = this->pHi - this->pLo;

    CDPrint( cdpCDb, "%s returns 0x%016llx: cb=%lld cbNeeded=%lld pLo=0x%016llx pHi=0x%016llx cbFree=%lld", __FUNCTION__, p, cb, cbNeeded, this->pLo, this->pHi, this->cbFree );

    // return the pointer to the allocated memory
    return p;
}

/// [private] method allocHi
void* CudaGlobalAllocator::allocHi( INT64 cb )
{
    /* Compute the total number of bytes needed for the allocation:
        - include padding so as to enforce the configured granularity
        - the minimum allocation is one unit of granularity
    */
    INT64 cbNeeded = round2power( cb, CudaGlobalAllocator::Granularity );
    cbNeeded = max2( cbNeeded, CudaGlobalAllocator::Granularity );

    // sanity check
    if( cbNeeded > this->cbFree )
        throw new ApplicationException( __FILE__, __LINE__, "unable to allocate %lld bytes (%lld bytes available)", cbNeeded, this->cbFree );

    // update the member variables
    this->pHi -= cbNeeded;
    this->cbFree = this->pHi - this->pLo;

    CDPrint( cdpCDb, "%s returns 0x%016llx: cb=%lld cbNeeded=%lld pLo=0x%016llx pHi=0x%016llx cbFree=%lld", __FUNCTION__, this->pHi, cb, cbNeeded, this->pLo, this->pHi, this->cbFree );

    // return the pointer to the allocated memory
    return this->pHi;
}

/// [private] method privateAlloc
void* CudaGlobalAllocator::privateAlloc( CudaGlobalPtrBase* _pcgpb, size_t _cb, cgaRegion _cgar )
{
#ifdef _DEBUG
    // sanity check
    if( _pcgpb && _cb )
        throw new ApplicationException( __FILE__, __LINE__, "invalid call to privateAlloc: one parameter must be null or zero" );
#endif

    HiResTimer hrt;
    void* pNew = NULL;
    INT8* pHiOld = NULL;

    // if a CudaGlobalPtrBase instance is referenced, use its byte count; otherwise, use the specified byte count
    const size_t newcb = _pcgpb ? _pcgpb->cb : _cb;

    switch( _cgar )
    {
        case cgaLow:
            // look for a previously-freed item in the list of low-region allocations that can hold the new allocation
            for( UINT32 n=0; n<m_Alo.n; ++n )
            {
                if( !m_Alo.p[n].inUse && (m_Alo.p[n].cbCuda >= newcb) )
                {
                    CDPrint( cdpCDb, "%s: reusing previous cgaLow allocation at 0x%016llx (requested cb=%lld, actual size %lld)", __FUNCTION__, m_Alo.p[n].p, newcb, m_Alo.p[n].cbCuda );

                    /* reuse the previous allocation; we don't bother to track the leftover bytes if the requested allocation
                        size is smaller than the size of the previous allocation */
                    return m_Alo.p[n].reuse( _pcgpb );
                }
            }

            // at this point we need a new allocation
            pNew = allocLo( newcb );

            // expand the list if necessary
            if( m_Alo.n == static_cast<UINT32>(m_Alo.Count) )
                m_Alo.Realloc( m_Alo.Count+32, true );

            // append to the list of low-region allocations
            m_Alo.p[m_Alo.n].set( pNew, _pcgpb, this->pLo - static_cast<INT8*>(pNew) );
            m_Alo.n++;

#if TODO_CHOP_WHEN_DEBUGGED
            InterlockedExchangeAdd( &msMalloc, hrt.GetElapsed( false ) );
            InterlockedIncrement( &nMalloc );
#endif
            break;

        case cgaHigh:
            // look for a previously-freed item in the list of high-region allocations that can hold the new allocation
            for( UINT32 n=0; n<m_Ahi.n; ++n )
            {
                if( !m_Ahi.p[n].inUse && (m_Ahi.p[n].cbCuda >= newcb ) )
                {
                    CDPrint( cdpCDb, "%s: reusing previous cgaHi allocation at 0x%016llx (requested cb=%lld, actual size %lld)", __FUNCTION__, m_Ahi.p[n].p, newcb, m_Ahi.p[n].cbCuda );
                    return m_Ahi.p[n].reuse( _pcgpb );
                }
            }

            // at this point we need a new allocation
            pHiOld = this->pHi;
            pNew = allocHi( newcb );

            // expand the list if necessary
            if( m_Ahi.n == static_cast<UINT32>(m_Ahi.Count) )
                m_Ahi.Realloc( m_Ahi.Count+32, true );

            // append to the list of low-region allocations
            m_Ahi.p[m_Ahi.n].set( pNew, _pcgpb, pHiOld - this->pHi );
            m_Ahi.n++;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "invalid cgaRegion: %d", _cgar );
    }

    return pNew;
}

/// [private] method freeLo
void CudaGlobalAllocator::freeLo( void* p )
{
    // traverse the list of low-region allocations, starting with the most recent allocation
    bool inList = false;
    for( INT32 m=static_cast<INT32>(m_Alo.n)-1; m>=0; --m )
    {
        // when we find the specified pointer, flag its allocation as not being in use
        if( m_Alo.p[m].p == p )
        {
            m_Alo.p[m].inUse = false;
            inList = true;
            break;
        }
    }

    // sanity check
    if( !inList )
        throw new ApplicationException( __FILE__, __LINE__, "cannot free pointer 0x%016llx", p );

    // remove unused allocations from the end of the list
    for( INT32 m=static_cast<INT32>(m_Alo.n)-1; m>=0; --m )
    {
        // fall out of the loop when we reach an allocation that is still in use
        if( m_Alo.p[m].inUse )
            break;

        // excise the mth allocation
        CDPrint( cdpCDb, "%s: freeing previous allocation at 0x%016llx", __FUNCTION__, p );
        this->SetAt( m_Alo.p[m].p, cgaLow );
    }

    // traverse the list once again to coalesce adjacent unused allocations
    INT32 nLimit = static_cast<INT32>(m_Alo.n) - 1;
    INT32 n = 0;
    while( n < nLimit )
    {
        // do nothing if either the nth or the (n+1)th allocation is in use
        if( m_Alo.p[n].inUse || m_Alo.p[n+1].inUse )
        {
            n++ ;
            continue;
        }

        // merge the (n+1)th entry in the table into the nth entry
        m_Alo.p[n].cbCuda += m_Alo.p[n+1].cbCuda;

        if( (n+1) < nLimit )
        {
            // excise the (n+1)th entry from the table
            size_t cb = sizeof(cgaAllocation) * (nLimit-(n+1));
            memmove_s( m_Alo.p+n+1, cb, m_Alo.p+n+2, cb );
        }

        // null the last entry in the table
        m_Alo.p[nLimit].reset();
        m_Alo.n-- ;

        // update the loop iteration limit
        nLimit-- ;
    }
}

/// [private] method freeHi
void CudaGlobalAllocator::freeHi( void* p )
{
    // traverse the list of high-region allocations, starting with the most recent allocation
    bool inList = false;
    for( INT32 m=static_cast<INT32>(m_Ahi.n)-1; m>=0; --m )
    {
        // when we find the specified pointer, flag its allocation as not being in use
        if( m_Ahi.p[m].p == p )
        {
            m_Ahi.p[m].inUse = false;
            inList = true;
            break;
        }
    }

    // sanity check
    if( !inList )
        throw new ApplicationException( __FILE__, __LINE__, "cannot free pointer 0x%016llx", p );

    // remove unused allocations from the end of the list
    for( INT32 m=static_cast<INT32>(m_Ahi.n)-1; m>=0; --m )
    {
        // fall out of the loop when we reach an allocation that is still in use
        if( m_Ahi.p[m].inUse )
            break;

        // excise the mth allocation
        void* pNext = m ? m_Ahi.p[m-1].p : (this->p+this->cb);
        this->SetAt( pNext, cgaHigh );
    }

    // traverse the list once again to coalesce adjacent unused allocations
    INT32 nLimit = static_cast<INT32>(m_Ahi.n) - 1;
    INT32 n = 0;
    while( n < nLimit )
    {
        // do nothing if either the nth or the (n+1)th allocation is in use
        if( m_Ahi.p[n].inUse || m_Ahi.p[n+1].inUse )
        {
            n++ ;
            continue;
        }

        // merge the (n+1)th entry in the table into the nth entry
        m_Ahi.p[n].p = m_Ahi.p[n+1].p;
        m_Ahi.p[n].cbCuda += m_Ahi.p[n+1].cbCuda;

        if( (n+1) < nLimit )
        {
            // excise the (n+1)th entry from the table
            size_t cb = sizeof(cgaAllocation) * (nLimit-(n+1));
            memmove_s( m_Ahi.p+n+1, cb, m_Ahi.p+n+2, cb );
        }

        // null the last entry in the table
        m_Ahi.p[nLimit].reset();
        m_Ahi.n-- ;

        // update the loop iteration limit
        nLimit-- ;
    }
}

/// [private] method setAlistAt
void CudaGlobalAllocator::setAlistAt( void* pSetAt, cgaRegion cgar )
{
    INT32 n;

    switch( cgar )
    {
        case cgaLow:
            n = static_cast<INT32>(m_Alo.n) - 1;
            while( (n >= 0) && (m_Alo.p[n].p >= pSetAt) )
            {
                m_Alo.p[n].reset();
                m_Alo.n-- ;
                --n;
            }

            if( n >= 0 )
            {
                // update the allocation size of the last item in the list
                m_Alo.p[n].cbCuda = static_cast<INT8*>(pSetAt) - static_cast<INT8*>(m_Alo.p[n].p);
            }
            break;

        case cgaHigh:
            n = static_cast<INT32>(m_Ahi.n) - 1;
            while( (n >= 0) && (m_Ahi.p[n].p < pSetAt) )
            {
                m_Ahi.p[n].reset();
                m_Ahi.n-- ;
                --n;
            }

            if( m_Ahi.n && (n < static_cast<INT32>(m_Ahi.n)) && (m_Ahi.p[n].p != pSetAt) )
                throw new ApplicationException( __FILE__, __LINE__, "invalid high-region address 0x%08llx", pSetAt );
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "invalid value %d for cgaRegion", cgar );
    }
}

/// [private] method dumpAlloc
void CudaGlobalAllocator::dumpAlloc( WinGlobalPtr<cgaAllocation>* pal, UINT32 n, const char* scgar )
{
    cgaAllocation* pcgaa = pal->p + n;
    const char* ptag = NULL;
    const char* sn = NULL;
    char nbuf[24];

    if( pcgaa->pcgpb )
    {
        ptag = pcgaa->pcgpb->tag.Count ? reinterpret_cast<const char*>(pcgaa->pcgpb->tag.p) : "(untagged)";
        sprintf_s( nbuf, sizeof nbuf, "(n=%u)", pcgaa->pcgpb->n );
        sn = reinterpret_cast<const char*>(nbuf);
    }
    else
    {
        ptag = "(anonymous)";
        sn = "";
    }

    const char* sused = pcgaa->inUse ? "" : "(freed)";
    CDPrint( cdpCD0, "%-7s %2u: 0x%016llx %lld %s %s %s", scgar, n, pcgaa->p, pcgaa->cbCuda, ptag, sn, sused );
}

/// private method findCGPref
CudaGlobalPtrBase** CudaGlobalAllocator::findCGPref( CudaGlobalPtrBase* const _pcgpb )
{
    // look in the low region
    for( UINT32 n=0; n<m_Alo.n; ++n )
    {
        if( (m_Alo.p[n].pcgpb == _pcgpb) && m_Alo.p[n].inUse )
            return &m_Alo.p[n].pcgpb;
    }

    // look in the high region
    for( UINT32 n=0; n<m_Ahi.n; ++n )
    {
        if( (m_Ahi.p[n].pcgpb == _pcgpb) && m_Ahi.p[n].inUse )
            return &m_Ahi.p[n].pcgpb;
    }

    return NULL;
}
#pragma endregion

#pragma region public methods
/// [public] method GetAvailableByteCount
INT64 CudaGlobalAllocator::GetAvailableByteCount()
{
    return (this->pHi - this->pLo);
}

/// <summary>
/// Returns the total number of bytes in all allocations.
/// </summary>
INT64 CudaGlobalAllocator::GetAllocatedByteCount( cgaRegion cgar )
{
    INT64 cbTotal = 0;

    if( cgar & cgaLow )
        cbTotal += (this->pLo - this->p);
    if( cgar & cgaHigh )
        cbTotal += ((this->p + this->cb) - this->pHi);

    return cbTotal;
}

/// <summary>
/// Returns the total number of bytes in the specified low-region allocation.
/// </summary>
/// <remarks>This method is used by the CudaGlobalPtr&lt;T&gt; implementation.</remarks>
INT64 CudaGlobalAllocator::GetAllocatedByteCount( void* p )
{
    if( p < this->pLo )
    {
        // look for the specified pointer in the list of low-region allocations
        for( UINT32 n=0; n<m_Alo.n; ++n )
        {
            if( m_Alo.p[n].p == p )
                return m_Alo.p[n].cbCuda;
        }
    }

    else
    {
        // look for the specified pointer in the list of high-region allocations
        for( UINT32 n=0; n<m_Ahi.n; ++n )
        {
            if( m_Ahi.p[n].p == p )
                return m_Ahi.p[n].cbCuda;
        }
    }

    // at this point the specified pointer does not reference a previous allocation
    return 0;
}

/// <summary>
/// Resets an internal pointer to the specified address in memory managed by this CudaGlobalAllocator instance.
/// </summary>
/// <remarks>This method is used by the CudaGlobalPtr&lt;T&gt; implementation.</remarks>
void* CudaGlobalAllocator::SetAt( void* pSetAt, cgaRegion cgar )
{
    switch( cgar )
    {
        case cgaLow:
            // enforce the configured granularity
            pSetAt = reinterpret_cast<INT8*>(round2power(reinterpret_cast<size_t>(pSetAt), CudaGlobalAllocator::Granularity));

            // sanity check
            if( pSetAt > this->pHi )
                throw new ApplicationException( __FILE__, __LINE__, "cannot set the free memory allocation at 0x%016llx (upper limit is 0x%016llx)", pSetAt, this->pHi );

            // reset the allocation to the specified address
            this->pLo = static_cast<INT8*>(pSetAt);     // the specified address becomes the new "free" address
            break;

        case cgaHigh:
            // sanity checks
            if( reinterpret_cast<size_t>(pSetAt) & (CudaGlobalAllocator::Granularity-1) )
                throw new ApplicationException( __FILE__, __LINE__, "cannot set the free memory allocation at 0x%016llx (not aligned to a %lld-byte boundary)", pSetAt, CudaGlobalAllocator::Granularity );

            if( pSetAt <= this->pLo )
                throw new ApplicationException( __FILE__, __LINE__, "cannot set the free memory allocation at 0x%016llx (lower limit is 0x%016llx)", pSetAt, this->pLo );

            // reset the allocation to the specified address
            this->pHi = static_cast<INT8*>(pSetAt);     // the specified address becomes the new "free" address
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "invalid value %d for cgaRegion", cgar );
    }

    // update the allocation list
    setAlistAt( pSetAt, cgar );

    // update the number of available bytes
    this->cbFree = this->pHi - this->pLo;

    CDPrint( cdpCDb, "%s: at 0x%016llx: pLo=0x%016llx pHi=0x%016llx cbFree=%lld", __FUNCTION__, pSetAt, this->pLo, this->pHi, this->cbFree );

    // return the actual set point
    return pSetAt;
}

/// <summary>
/// Allocates a block of memory in the preallocated CUDA global memory buffer
/// </summary>
void* CudaGlobalAllocator::Alloc( CudaGlobalPtrBase* pcgpb, cgaRegion cgar )
{
    // do an allocation associated with the specified CudaGlobalPtrBase instance
    return privateAlloc( pcgpb, 0, cgar );
}

/// <summary>
/// Allocates a block of memory in the preallocated CUDA global memory buffer
/// </summary>
void* CudaGlobalAllocator::Alloc( size_t cb )
{
    // do an "anonymous" allocation in the low region
    return privateAlloc( NULL, cb, cgaLow );
}

/// <summary>
/// Frees the block of memory in the preallocated CUDA global memory buffer
/// </summary>
void CudaGlobalAllocator::Free( void* p, cgaRegion cgar )
{
    switch( cgar )
    {
        case cgaLow:
            freeLo( p );
            break;

        case cgaHigh:
            freeHi( p );
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "invalid value %d for cgaRegion", cgar );
    }
}

/// <summary>
/// Returns a pointer to the most recent allocation in the high region
/// </summary>
void* CudaGlobalAllocator::GetMruHi()
{
    // return a pointer to the last allocation in the list, or null if there is no previous allocation
    return (m_Ahi.n > 0) ? m_Ahi.p[m_Ahi.n-1].p : NULL;
}

/// <summary>
/// Returns a pointer to the most recent allocation in the low region
/// </summary>
void* CudaGlobalAllocator::GetMruLo()
{
    // return a pointer to the last allocation in the list, or null if there is no previous allocation
    return (m_Alo.n > 0) ? m_Alo.p[m_Alo.n-1].p : NULL;
}

/// <summary>
/// Returns a reference to the CudaGlobalAllocator instance associated with the current CUDA device
/// </summary>
/// <remarks>This is a C++ static method.</remarks>
CudaGlobalAllocator* CudaGlobalAllocator::GetInstance()
{
    INT32 deviceId = CudaDeviceBinding::GetCudaDeviceId();
    return m_Instance[deviceId];
}

/// <summary>
/// Swaps a pair CudaGlobalPtrBase references in the low and high allocation lists.
/// </summary>
void CudaGlobalAllocator::Swap( CudaGlobalPtrBase* pcgpb1, CudaGlobalPtrBase* pcgpb2 )
{
    // find the references to the specified CudaGlobalPtrBase instances
    CudaGlobalPtrBase** pp1 = findCGPref( pcgpb1 );
    CudaGlobalPtrBase** pp2 = findCGPref( pcgpb2 );

    /* if either of the specified CudaGlobalPtrBase instances is not found in the allocation lists
        (e.g., it has been freed), null the other instance */
    if( (pp1 == NULL) || (pp2 == NULL) )
    {
        if( pp1 ) *pp1 = NULL;
        if( pp2 ) *pp2 = NULL;
    }

    else
    {
        // swap the references to the CudaGlobalPtrBase instances in the allocation lists
        CudaGlobalPtrBase* ptemp = *pp1;
        *pp1 = *pp2;
        *pp2 = ptemp;
    }
}

/// <summary>
/// Implements a simple dump of the memory usage in the global allocation.
/// </summary>
void CudaGlobalAllocator::DumpUsage( const char* callerInfo )
{
    // emit the specified caller info
    CDPrint( cdpCD0, "%s: %s", __FUNCTION__, callerInfo );

    // dump bytes used and free
    INT64 cbLow = GetAllocatedByteCount( cgaLow );
    INT64 cbHigh = GetAllocatedByteCount( cgaHigh );
    INT64 cbTotal = cbLow + cbHigh;
    INT64 cbAvailable = GetAvailableByteCount();
    CDPrint( cdpCD0, "%s: available=%lld; allocated: %lld+%lld=%lld (%2.1f%%)", __FUNCTION__, cbAvailable, cbLow, cbHigh, cbTotal, 100.0*cbTotal/this->cb );
    
    // dump the allocation lists for the low and high region
    for( UINT32 n=0; n<m_Alo.n; ++n )
        dumpAlloc( &m_Alo, n, "cgaLow" );
    for( INT32 n=static_cast<INT32>(m_Ahi.n)-1; n>=0; --n )
        dumpAlloc( &m_Ahi, n, "cgaHigh" );
}
#pragma endregion
