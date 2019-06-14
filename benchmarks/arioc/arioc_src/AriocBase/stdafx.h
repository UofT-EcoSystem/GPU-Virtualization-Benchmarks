/*
  stdafx.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/

#pragma once
#define __AriocBase_Includes__

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN             // exclude rarely-used stuff from Windows headers
#include "targetver.h"
#include <windows.h>
#include <intrin.h>
#endif

#include <stdio.h>
#include <math.h>

// CUDA APIs
#include <cuda_runtime.h>               // CUDA runtime API
#include <device_launch_parameters.h>   // C++ defines for CUDA builtins

// CppCommon
#ifndef __CppCommon_Includes__
#include "../CppCommon/stdafx.h"
#endif

// CudaCommon
#ifndef __CudaCommon_Includes__
#include "../CudaCommon/stdafx.h"
#endif

// TaskUnit
#ifndef __TaskUnit_Includes__
#include "../TaskUnit/stdafx.h"
#endif

// AriocCommon
#ifndef __AriocCommon_Includes__
#include "../AriocCommon/stdafx.h"
#endif

// AriocBase
#include "TSX.h"
#include "AriocTaskUnitMetrics.h"
#include "AriocAppMetrics.h"
#include "AriocAppMainBase.h"
#include "tuWatchdog.h"
#include "ReferenceDataInfo.h"
#include "baseARowWriter.h"
#include "ARowWriter.h"
#include "SAMHDBuilder.h"
#include "AriocBase.h"
#include "tuLoadLUT.h"
#include "tuUnloadLUT.h"
#include "baseQReader.h"
#include "DeviceBuffers.h"
#include "HostBuffers.h"
#include "QBatch.h"
#include "QBatchPool.h"
#include "tuGpu.h"
#include "baseSetupJn.h"
#include "baseSetupJs.h"
#include "baseLoadRi.h"
#include "baseLoadRix.h"
#include "baseLoadRw.h"
#include "baseLoadJn.h"
#include "baseLoadJs.h"
#include "baseJoinDf.h"
#include "baseAlignN.h"
#include "baseMaxV.h"
#include "baseAlignG.h"
#include "baseCountA.h"
#include "baseCountJs.h"
#include "baseMergeRu.h"
#include "baseMaxVw.h"
#include "baseMapCommon.h"
#include "tuLoadM.h"
#include "baseMAPQ.h"
#include "tuXlatToD.h"
#include "tuXlatToDf.h"
#include "tuSetupN10.h"
#include "tuClassify.h"
#include "tuClassify1.h"
#include "tuFinalizeN.h"
#include "tuFinalizeG.h"
#include "tuFinalizeGs.h"
#include "tuFinalizeGc.h"
#include "tuComputeKMH10.h"
#include "tuComputeKMH30.h"
#include "tuComputeKMH.h"
#include "baseARowBuilder.h"
#include "SAMFormatBase.h"
#include "SAMBuilderBase.h"
#include "SAMBuilderUnpaired.h"
#include "SAMBuilderPaired.h"
#include "SAMHDBuilder.h"
#include "SBFBuilderBase.h"
#include "SBFBuilderUnpaired.h"
#include "SBFBuilderPaired.h"
#include "TSEManifest.h"
#include "TSEFormatBase.h"
#include "TSEBuilderBase.h"
#include "TSEBuilderUnpaired.h"
#include "TSEBuilderPaired.h"
#include "KMHBuilderBase.h"
#include "KMHBuilderUnpaired.h"
#include "KMHBuilderPaired.h"
