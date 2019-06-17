/*
   CudaGlobalPtr.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __CudaGlobalPtrBase__

#define CUDAMALLOCGRANULARITY   (static_cast<size_t>(0x00500000))   // 5MB seems to be what's needed for CUDA v4.1

// forward definition
class CudaGlobalAllocator;

/// <summary>
/// Class <c>CudaGlobalPtrBase</c> defines functionality common to all specializations of CudaGlobalPtr<T>
/// </summary>
class CudaGlobalPtrBase
{
protected:
    bool                    m_initZero; // flag set when the allocated memory is to be zeroed
    bool                    m_dtor;     // flag set when a CudaGlobalPtr instance is executing its destructor
    CudaGlobalAllocator*    m_pCGA;     // pointer to global memory allocator
    size_t                  m_cbCuda;   // number of bytes allocated by cudaMalloc (assuming CUDAMALLOCGRANULARITY is the granularity of a cudaMalloc allocation) or by the CudaGlobalAllocator

public:
    size_t                  cb;         // number of allocated bytes
    size_t                  Count;      // number of elements of sizeof(T)
    volatile UINT32         n;          // (unused by the CudaGlobalPtr implementation)
    WinGlobalPtr<char>      tag;        // (unused by the CudaGlobalPtr implementation)

protected:
    CudaGlobalPtrBase( CudaGlobalAllocator* pCGA = NULL ) : m_initZero( false ), m_dtor( false ), m_pCGA( pCGA ), m_cbCuda( 0 ), cb( 0 ), Count( 0 ), n( 0 )
    {
    }

    virtual ~CudaGlobalPtrBase( void )
    {
    }
};
