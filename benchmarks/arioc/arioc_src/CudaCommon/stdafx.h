/*
  stdafx.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __CudaCommon_Includes__

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#include "targetver.h"
#include <windows.h>
#include <intrin.h>
#endif

#include <stdio.h>
#include <math.h>
#include <string.h>

// CUDA APIs
#include <cuda_runtime.h>               // CUDA runtime API
#include <device_launch_parameters.h>   // C++ defines for CUDA builtins

// CppCommon
#ifndef __CppCommon_Includes__
#include "../CppCommon/stdafx.h"
#endif

// CudaCommon
#include "CudaCommon.h"
#include "CudaResult.h"
#include "CudaDeviceBinding.h"
#include "CudaGlobalPtrBase.h"
#include "CudaGlobalAllocator.h"
#include "CudaGlobalPtr.h"
#include "CudaPinnedPtr.h"
#include "CudaLaunchCommon.h"
#include "ThrustSpecializations.h"
