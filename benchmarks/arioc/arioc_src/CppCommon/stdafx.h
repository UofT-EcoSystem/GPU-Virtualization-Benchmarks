/*
  stdafx.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __CppCommon_Includes__

#ifndef TINYXML2_INCLUDED
#include "tinyxml2.h"
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#include <windows.h>
#include "targetver.h"
#include <io.h>
#include <direct.h>
#include <share.h>
#endif

#ifdef __GNUC__
#include <string.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <strings.h>
#include <dirent.h>
#include "../Windux/Windux.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <math.h>

// CppCommon
#include "TOD.h"
#include "MT19937.h"
#include "CppCommon.h"
#include "ApplicationException.h"
#include "RaiiPtr.h"
#include "WinGlobalPtr.h"
#include "HiResTimer.h"
#include "RaiiCriticalSection.h"
#include "RaiiMutex.h"
#include "RaiiSyncEventObject.h"
#include "RaiiInverseSemaphore.h"
#include "RaiiSemaphore.h"
#include "RaiiWorkerThread.h"
#include "RaiiDirectory.h"
#include "RaiiFileWorkerF.h"
#include "RaiiFileWorkerP.h"
#include "RaiiFile.h"
#include "AA.h"
#include "Hash.h"
