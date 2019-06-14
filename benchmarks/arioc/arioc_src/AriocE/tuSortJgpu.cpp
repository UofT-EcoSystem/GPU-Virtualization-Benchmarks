/*
  tuSortJgpu.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"
#include <thrust/system_error.h>


#pragma region constructor/destructor
/// [private] constructor
tuSortJgpu::tuSortJgpu()
{
}

/// constructor (Jvalue8*, INT64, INT64)
tuSortJgpu::tuSortJgpu( Jvalue8* _pJ, INT64 _celJ, INT64 _cbCgaReserved ) : m_pJ(_pJ), m_celJ(_celJ),
                                                                            m_pJbuf(NULL),
                                                                            m_pcdb(NULL), m_pcga(NULL), m_cudaDeviceId(-1), m_gmemAvailable(0), m_celPerSort(0)
{
    CRVALIDATOR;

    // associate a GPU context with the current thread
    m_pcdb = new CudaDeviceBinding( CUDADEVICEBINDING_TIMEOUT );

    /* Here we reserve the largest available chunk of contiguous device memory so that we can manage
        our own buffer allocations and avoid fragmentation (as well as any bugs remaining in the CUDA
        runtime memory-management APIs).

        We need to cooperate with the Thrust global-memory allocator as well.  See notes in Windux.h.            
    */
    m_pcga = new CudaGlobalAllocator( *m_pcdb, _cbCgaReserved );
    m_cudaDeviceId = m_pcdb->GetDeviceProperties()->cudaDeviceId;
    m_gmemAvailable = m_pcga->GetAvailableByteCount();

    m_gmemAvailable -= 10 * 1024 * 1024;        // TODO: MAKE THIS AN XPARAM

    // use half of available GPU memory for the J lists; the rest of the GPU memory is reserved for Thrust
    m_pJbuf = new CudaGlobalPtr<UINT64>( m_pcga );
    m_celPerSort = static_cast<UINT32>((m_gmemAvailable/sizeof(UINT64)) / 2);
}

/// destructor
tuSortJgpu::~tuSortJgpu()
{
    if( m_pJbuf )
        delete m_pJbuf;
    if( m_pcga )
        delete m_pcga;
    if( m_pcdb )
        delete m_pcdb;
}
#pragma endregion

#pragma region private methods
/// [private] method saveSortChunkInfo
void tuSortJgpu::saveSortChunkInfo( INT64 _ofsJfirst, INT64 _ofsJlast )
{
    if( m_JSC.n >= static_cast<UINT32>(m_JSC.Count) )
        m_JSC.Realloc( m_JSC.n+1024, true );
    m_JSC.p[m_JSC.n] = { _ofsJfirst, _ofsJlast };
    m_JSC.n++;
}

/// [private] method defineJlistChunks
void tuSortJgpu::defineJlistChunks()
{
    CDPrint( cdpCD3, "%s...", __FUNCTION__ );

    /* Compute a set of "segments" of the J-list table.
    
       - Each segment contains no more than the maximum number of J values that can be sorted in one
          CUDA kernel execution.
       - The maximum number of H values (not J values) is limited to 24 bits.
    */
    
    // replace the J-list tag values with a sequence number (basically just an ordinal for each J list)
    UINT32 ordinal = 0x00FFFFFF;
    UINT32 tag = 0;
    for( INT64 ij=1; ij<m_celJ; ++ij )
    {
        if( m_pJ[ij].tag != tag  )
        {
            // increment the sequence number
            ordinal = (ordinal+1) & 0x00FFFFFF;

            // track the new tag through the J list starting at offset ij
            tag = m_pJ[ij].tag;
        }

        // update the ij'th J value with the ordinal for the current J list
        m_pJ[ij].tag = ordinal;
    }

    /* At this point we need to break the J-list buffer into chunks small enough to sort. */

    // the first J list starts at offset 1 in the buffer
    INT64 ofsJfirst = 1;
    tag = m_pJ[1].tag;

    // iterate through the J lists
    for( INT64 ij=1; ij<m_celJ; ++ij )
    {
        if( m_pJ[ij].tag != tag )
        {
            /* At this point the ij'th J value is the first in its J list.

               We terminate the current chunk if one of the following is true:
               - we have reached the maximum tag value for a chunk
               - the number of J values in the chunk exceeds the number of elements that can be sorted in
               one GPU sort operation
            */
            if( (m_pJ[ij].tag < tag) || ((ij-ofsJfirst) > m_celPerSort) )
            {
                saveSortChunkInfo( ofsJfirst, ij-1 );
                ofsJfirst = ij;
            }

            // track the tag for the current J list
            tag = m_pJ[ij].tag;
        }
    }

    // if the most recently saved chunk has not yet been saved, do it now   THIS IS WRONG:  NO CONDITIONAL NEEDED  ALSO, REALLOC J AFTER COMPACTING!
    saveSortChunkInfo( ofsJfirst, m_celJ-1 );

    // reallocate the chunk list to release unused space
    m_JSC.Realloc( m_JSC.n, false );

    // dump the segment limits
    CDPrint( cdpCD3, "%s: defined %u chunks for segmented J-list sorts", __FUNCTION__, m_JSC.n );
    for( UINT32 n=0; n<m_JSC.n; ++n )
        CDPrint( cdpCD4, "%s: %05u %lld %lld (%lld)", __FUNCTION__, n, m_JSC.p[n].ofsJfirst, m_JSC.p[n].ofsJlast, (m_JSC.p[n].ofsJlast-m_JSC.p[n].ofsJfirst)+1 );

    CDPrint( cdpCD3, "%s completed", __FUNCTION__ );
}

/// [private] method alignN10
void tuSortJgpu::sortJ10()
{
    // sort each segment of the J-list buffer
    for( UINT32 n=0; n<m_JSC.n; ++n )
    {
        initGlobalMemory10( m_JSC.p+n );
        launchKernel10();
        copyKernelResults10( m_JSC.p+n );
        resetGlobalMemory10();
    }
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Flags concordantly-mapped pairs as non-candidates for subsequent windowed gapped alignment
/// </summary>
void tuSortJgpu::main()
{
    CRVALIDATOR;

    CDPrint( cdpCD3, "%s...", __FUNCTION__ );

    try
    {
        /* The idea here is to use a GPU to sort the J lists.
    
           Each element in a J list is a Jvalue8 struct, that is, the low-order 39 bits of a 64-bit value.
            If we use 64-bit integers, CUDA Thrust will do a radix sort, so we create 64-bit values that look
            like this:

                bits 0-38 :  J value (subId, strand, J)
                bit 39    :  (unused)
                bits 40-63:  low-order 24 bits of the hash key (i.e., J-list ID)

           The basic strategy is thus:
            - split the entire J table into 256 segments, each identified by the 8 high-order bits of the hash values
            - for each segment:
                - build a list of 64-bit composite values (24-bit J-list Id, 40-bit J value); the list must be small
                   enough to fit into available GPU memory
                - sort
                - copy the sorted values back into the J table
        */

        defineJlistChunks();
        sortJ10();
    }
    catch( thrust::system_error& ex )
    {
        int cudaErrno = ex.code().value();
        throw new ApplicationException( __FILE__, __LINE__,
                                        "CUDA error %u (0x%08x): %s\r\nCUDA Thrust says: %s",
                                        cudaErrno, cudaErrno, ex.code().message().c_str(), ex.what() );
    }

    CDPrint( cdpCD3, "%s completed", __FUNCTION__ );
}
#pragma endregion
