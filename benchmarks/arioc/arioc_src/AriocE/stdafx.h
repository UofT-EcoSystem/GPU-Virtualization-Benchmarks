/*
  stdafx.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocE_Includes__

#ifdef _WIN32
#include "targetver.h"
#include <windows.h>
#include <intrin.h>
#endif

#include <stdio.h>
#include <math.h>
#include <ctype.h>

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

// AriocE
#include "AppMain.h"
#include "AppGlobal.h"
#include "AriocEncoderParams.h"
#include "SAMConfigWriter.h"
#include "AriocE.h"
#include "baseEncode.h"
#include "baseValidateA21.h"
#include "tuEncodeFASTA.h"
#include "tuEncodeFASTQ.h"
#include "tuValidateQ.h"
#include "tuSortJgpu.h"
#include "tuLoadR.h"
#include "tuSortJcpu.h"
#include "tuSortBB.h"
#include "tuCountBB.h"
#include "tuValidateSubIds.h"
