/*
  CudaCommon.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variable definitions
CudaCommon CudaCommon::m_instance;
char CudaCommon::m_cudaVersionString[256] = "(CUDA driver and/or runtime version information is not available)";
#pragma endregion

#pragma region constructor/destructor
/// [private] default constructor
CudaCommon::CudaCommon()
{
    cudaError_t rval;

    /* Build a version-info string.

       If no version info is available, we fail silently (and the version-info string contains zeros).  Presumably
        any subsequent CUDA calls will fail as well.
    */
    INT32 vr = 0;
    rval = cudaRuntimeGetVersion( &vr );
    if( rval )
        CDPrint( cdpCD3, "%s: cudaRuntimeGetVersion returned %d (%s)\n", __FUNCTION__, rval, cudaGetErrorString(rval) );

    INT32 vd = 0;
    rval = cudaDriverGetVersion( &vd );
    if( rval )
        CDPrint( cdpCD3, "%s: cudaDriverGetVersion returned %d (%s)\n", __FUNCTION__, rval, cudaGetErrorString(rval) );

    sprintf_s( m_cudaVersionString,
                sizeof m_cudaVersionString,
                "runtime v%d.%d.%d, driver v%d.%d.%d",
                vr/1000, (vr%100)/10, vr%10,
                vd/1000, (vd%100)/10, vd%10 );
}

/// destructor
CudaCommon::~CudaCommon()
{
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Returns the number of physical devices recognized by the CUDA driver
/// </summary>
/// <remarks>static method</remarks>
INT32 CudaCommon::GetDeviceCount()
{
    CRVALIDATOR;

    // count the number of CUDA-enabled devices
    INT32 nCudaDevices = 0;
    CRVALIDATE = cudaGetDeviceCount( &nCudaDevices );
    return nCudaDevices;
}

/// <summary>
/// Returns the CUDA device properties for the specified available device ID
/// </summary>
void CudaCommon::GetDeviceProperties( cudaDevicePropEx* pcdpe, INT32 deviceId )
{
    CRVALIDATOR;

    // fill the specified cudaDevicePropEx struct for the specified CUDA device
    CRVALIDATE = cudaGetDeviceProperties( pcdpe, deviceId );
    pcdpe->cudaDeviceId = deviceId;
}

/// <summary>
/// Returns a string that contains the CUDA runtime and driver version numbers
/// </summary>
/// <remarks>static method</remarks>
char* CudaCommon::GetCudaVersionString( void )
{
    return m_cudaVersionString;
}
#pragma endregion
