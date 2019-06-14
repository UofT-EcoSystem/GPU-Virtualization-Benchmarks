/*
  stdafx.h

    Copyright (c) 2018-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
     in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
     The contents of this file, in whole or in part, may only be copied, modified, propagated, or
     redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __XMC_Includes__

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN             // exclude rarely-used stuff from Windows headers
#include "targetver.h"
#include <windows.h>
#endif

// CppCommon
#ifndef __CppCommon_Includes__
#include "../CppCommon/stdafx.h"
#endif

// TaskUnit
#ifndef __TaskUnit_Includes__
#include "../TaskUnit/stdafx.h"
#endif

// AriocCommon
#ifndef __AriocCommon_Includes__
#include "../AriocCommon/stdafx.h"
#endif

// XMC
#include "AppGlobal.h"
#include "AppMain.h"
#include "tuCount.h"
#include "XMC.h"
