/*
  stdafx.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocU_Includes__

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN             // exclude rarely-used stuff from Windows headers
#include "targetver.h"
#include <windows.h>
#include <intrin.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef __AriocBase_Includes__
#include "../AriocBase/stdafx.h"
#endif

// AriocU
#include "AppMain.h"
#include "AppGlobal.h"
#include "QReaderU.h"
#include "tuLoadMU.h"
#include "baseFilterQu.h"
#include "baseFilterD.h"
#include "baseMapGc.h"
#include "tuSetupN.h"
#include "tuAlignN.h"
#include "tuAlignGs.h"
#include "tuTailU.h"
#include "tuGpuU.h"
#include "AriocU.h"
