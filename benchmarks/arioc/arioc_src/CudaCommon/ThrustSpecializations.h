/*
  ThrustSpecializations.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __ThrustSpecializations__

#include <thrust/version.h>
#include <thrust/execution_policy.h>
//#include <thrust/device_malloc_allocator.h>
#include <thrust/system/cuda/memory.h>




/* The following supports the use of the CudaGlobalAllocator implementation by CUDA Thrust.

   The low-level CUDA memory allocator in CUDA Thrust is wrapped in global-scoped functions named
    malloc() and free().  (See <CUDA includes>/include/thrust/system/cuda/detail/malloc_and_free.h
    for these templated function definitions.)

   To override this functionality, we use the "execution policy" mechanism in Thrust, which basically
    uses C++ template function specialization to declare an alternative implementation:
    
    - Create a C++ struct type named eptCGA (ept = "execution policy type") that derives from
      thrust::detail::execution_policy_base<T>.
    - Create the alternative implementation using eptCGA as the template parameter type.
    - Define a global instance of that type named epCGA.
    - Modify every CUDA Thrust API call everywhere to pass epCGA as a parameter so that the C++
       compiler will use the corresponding template function specialization, i.e., the alternative
       implementation.

   This is kind of ugly, but at least we don't have to hack the Thrust code in malloc_and_free.h
    in order to use the CudaGlobalAllocator implementation.
*/


// declare a type derived from thrust::execution_policy
#if THRUST_VERSION < 100900
struct eptCGA : thrust::system::cuda::detail::execution_policy<eptCGA>
#else
struct eptCGA : thrust::cuda::execution_policy<eptCGA>
#endif
{
};

// declare a static instance of the above type with application-global scope
extern eptCGA epCGA;

/// overload the Thrust malloc() template function implementation
template<typename eptCGA> __host__ __device__ void* malloc( eptCGA, size_t n )
{
#ifndef __CUDA_ARCH__
    /* (called from a host thread) */
    return CudaGlobalAllocator::GetInstance()->Alloc( n );              
#else
    /* (called from a device GPU thread) */
#if THRUST_VERSION < 100900
    thrust::system::cuda::detail::execution_policy<thrust::detail::device_t> epDevice;  // thrust::detail::device_t is the type of thrust::device
    return thrust::system::cuda::detail::malloc( epDevice, n );
#else
    thrust::cuda::execution_policy<thrust::detail::device_t> epDevice;  // thrust::detail::device_t is the type of thrust::device
    return thrust::cuda::malloc( epDevice, n );
#endif

#endif
}

/// overload the Thrust free() template function implementation
template<typename eptCGA, typename Pointer> __host__ __device__ void free( eptCGA, Pointer ptr )
{
#ifndef __CUDA_ARCH__
    /* (called from a host thread) */
    CudaGlobalAllocator::GetInstance()->Free( thrust::raw_pointer_cast( ptr ) );
#else
    /* (called from a device GPU thread) */
#if THRUST_VERSION < 100900
    thrust::system::cuda::detail::execution_policy<thrust::detail::device_t> epDevice;  // thrust::detail::device_t is the type of thrust::device
    thrust::system::cuda::detail::free( epDevice, ptr );
#else
    thrust::cuda::execution_policy<thrust::detail::device_t> epDevice;
    thrust::cuda::free( epDevice, ptr );
#endif

#endif
}


#if TODO_JARED_HOBEROCKS_SO_IMPLEMENTATION
// create a custom execution policy by deriving from the existing cuda::execution_policy
struct eptCGA : thrust::cuda::execution_policy<eptCGA>
{
};

// declare a static instance of the above type with application-global scope
extern eptCGA epCGA;

// provide an overload of malloc() for eptCGA
template<typename eptCGA> __host__ __device__ void* malloc(eptCGA, size_t n )
{
  printf("hello, world from my special malloc!\n");

  return thrust::raw_pointer_cast(thrust::cuda::malloc(n));
  //return CudaGlobalAllocator::GetInstance()->Alloc( n );   
}

// provide an overload of free() for eptCGA
template<typename eptCGA, typename Pointer> __host__ __device__ void free(eptCGA, Pointer ptr )
{
  printf("hello, world from my special free!\n");

  thrust::cuda::free( ptr );
  //CudaGlobalAllocator::GetInstance()->Free( thrust::raw_pointer_cast( ptr ) );
}
#endif
