/*
  CudaResult.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

// default constructor
CudaResult::CudaResult() : LineNumber(0)
{
    memset( this->FileName, 0, sizeof this->FileName);
}

// constructor (char*)
CudaResult::CudaResult( const char* fileName ) : LineNumber(0)
{
    // save the parameter value
    strcpy_s( this->FileName, sizeof this->FileName, fileName );
}

// destructor
CudaResult::~CudaResult()
{
}

#pragma region private methods
#if TODO_CHOP_IF_UNUSED
/// [private] method getDeviceApiErrorString
char* CudaResult::getDeviceApiErrorString( CUresult rval )
{
    /*
        There is no device API equivalent of cudaGetErrorString, so here it is...

        These definitions are lifted from v4.0 of cuda.h.
    */
    char* p = NULL;

    switch( rval )
    {
        case CUDA_SUCCESS:                              p = "CUDA_SUCCESS"; break;
        case CUDA_ERROR_INVALID_VALUE:                  p = "CUDA_ERROR_INVALID_VALUE"; break;
        case CUDA_ERROR_OUT_OF_MEMORY:                  p = "CUDA_ERROR_OUT_OF_MEMORY"; break;
        case CUDA_ERROR_NOT_INITIALIZED:                p = "CUDA_ERROR_NOT_INITIALIZED"; break;
        case CUDA_ERROR_DEINITIALIZED:                  p = "CUDA_ERROR_DEINITIALIZED"; break;
        case CUDA_ERROR_PROFILER_DISABLED:              p = "CUDA_ERROR_PROFILER_DISABLED"; break;
        case CUDA_ERROR_PROFILER_NOT_INITIALIZED:       p = "CUDA_ERROR_PROFILER_NOT_INITIALIZED"; break;
        case CUDA_ERROR_PROFILER_ALREADY_STARTED:       p = "CUDA_ERROR_PROFILER_ALREADY_STARTED"; break;
        case CUDA_ERROR_PROFILER_ALREADY_STOPPED:       p = "CUDA_ERROR_PROFILER_ALREADY_STOPPED"; break;
        case CUDA_ERROR_NO_DEVICE:                      p = "CUDA_ERROR_NO_DEVICE"; break;
        case CUDA_ERROR_INVALID_DEVICE:                 p = "CUDA_ERROR_INVALID_DEVICE"; break;
        case CUDA_ERROR_INVALID_IMAGE:                  p = "CUDA_ERROR_INVALID_IMAGE"; break;
        case CUDA_ERROR_INVALID_CONTEXT:                p = "CUDA_ERROR_INVALID_CONTEXT"; break;
        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:        p = "CUDA_ERROR_CONTEXT_ALREADY_CURRENT"; break;
        case CUDA_ERROR_MAP_FAILED:                     p = "CUDA_ERROR_MAP_FAILED"; break;
        case CUDA_ERROR_UNMAP_FAILED:                   p = "CUDA_ERROR_UNMAP_FAILED"; break;
        case CUDA_ERROR_ARRAY_IS_MAPPED:                p = "CUDA_ERROR_ARRAY_IS_MAPPED"; break;
        case CUDA_ERROR_ALREADY_MAPPED:                 p = "CUDA_ERROR_ALREADY_MAPPED"; break;
        case CUDA_ERROR_NO_BINARY_FOR_GPU:              p = "CUDA_ERROR_NO_BINARY_FOR_GPU"; break;
        case CUDA_ERROR_ALREADY_ACQUIRED:               p = "CUDA_ERROR_ALREADY_ACQUIRED"; break;
        case CUDA_ERROR_NOT_MAPPED:                     p = "CUDA_ERROR_NOT_MAPPED"; break;
        case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:            p = "CUDA_ERROR_NOT_MAPPED_AS_ARRAY"; break;
        case CUDA_ERROR_NOT_MAPPED_AS_POINTER:          p = "CUDA_ERROR_NOT_MAPPED_AS_POINTER"; break;
        case CUDA_ERROR_ECC_UNCORRECTABLE:              p = "CUDA_ERROR_ECC_UNCORRECTABLE"; break;
        case CUDA_ERROR_UNSUPPORTED_LIMIT:              p = "CUDA_ERROR_UNSUPPORTED_LIMIT"; break;
        case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:         p = "CUDA_ERROR_CONTEXT_ALREADY_IN_USE"; break;
        case CUDA_ERROR_INVALID_SOURCE:                 p = "CUDA_ERROR_INVALID_SOURCE"; break;
        case CUDA_ERROR_FILE_NOT_FOUND:                 p = "CUDA_ERROR_FILE_NOT_FOUND"; break;
        case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND: p = "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND"; break;
        case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:      p = "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED"; break;
        case CUDA_ERROR_OPERATING_SYSTEM:               p = "CUDA_ERROR_OPERATING_SYSTEM"; break;
        case CUDA_ERROR_INVALID_HANDLE:                 p = "CUDA_ERROR_INVALID_HANDLE"; break;
        case CUDA_ERROR_NOT_FOUND:                      p = "CUDA_ERROR_NOT_FOUND"; break;
        case CUDA_ERROR_NOT_READY:                      p = "CUDA_ERROR_NOT_READY"; break;
        case CUDA_ERROR_LAUNCH_FAILED:                  p = "CUDA_ERROR_LAUNCH_FAILED"; break;
        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:        p = "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES"; break;
        case CUDA_ERROR_LAUNCH_TIMEOUT:                 p = "CUDA_ERROR_LAUNCH_TIMEOUT"; break;
        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:  p = "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING"; break;
        case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:    p = "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED"; break;
        case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:        p = "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED"; break;
        case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:         p = "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE"; break;
        case CUDA_ERROR_CONTEXT_IS_DESTROYED:           p = "CUDA_ERROR_CONTEXT_IS_DESTROYED"; break;
        case CUDA_ERROR_UNKNOWN:                        p = "CUDA_ERROR_UNKNOWN"; break;
        default:                                        p = "(unrecognized CUDA_ERROR code -- see cuda.h)"; break;
    }

    return p;
}
#endif
#pragma endregion

// overload operator=
void CudaResult::operator=( const cudaError_t rval )
{
    // if no error, return immediately
    if( rval == cudaSuccess )
        return;

    // error; throw an exception
    throw new ApplicationException( this->FileName, this->LineNumber, "CUDA runtime API error %d: %s\r\n", rval, cudaGetErrorString(rval) );
}

#if TODO_CHOP_IF_UNUSED
// overload operator=
void CudaResult::operator=( const CUresult rval )
{
    // if no error, return immediately
    if( rval == CUDA_SUCCESS )
        return;

    // error; throw an exception
    throw new ApplicationException( this->FileName, this->LineNumber, "CUDA device API error %d: %s\r\n", rval, getDeviceApiErrorString(rval) );
}
#endif

// method SetDebugInfo
CudaResult& CudaResult::SetDebugInfo( const int lineNumber )
{
    // save the parameter value
    this->LineNumber = lineNumber;

    // return a refererence to this CudaResult instance
    return *this;
}
