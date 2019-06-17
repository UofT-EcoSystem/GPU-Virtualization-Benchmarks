/*
  Windux.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
    This include file defines types and aliases for compatibility between Windows and Linux.

    In some cases, we have re-used function signatures from Microsoft Visual Studio libraries, just
     to keep things simple in the calling source code.
*/
#pragma once
#define __Windux__

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <errno.h>
#include <ctype.h>
#include <string.h>
#include <limits.h>
#include <signal.h>
#include <unistd.h>

// some fundamental types
typedef signed      char        INT8;
typedef unsigned    char        UINT8;
typedef signed      short       INT16;
typedef unsigned    short       UINT16;
typedef signed      int         INT32;
typedef unsigned    int         UINT32;
typedef unsigned    int         DWORD;
typedef signed      long long   INT64;
typedef unsigned    long long   UINT64;
typedef void*                   LPVOID;
typedef DWORD*                  LPDWORD;

#define _UI8_MAX    UCHAR_MAX
#define _I16_MAX    SHRT_MAX
#define _I16_MIN    SHRT_MIN
#define _U1I6_MAX   USHRT_MAX
#define _I32_MAX    INT_MAX
#define _UI32_MAX   UINT_MAX
#define _I64_MAX    LLONG_MAX
#define _UI64_MAX   ULLONG_MAX

#define __stdcall

// threading (see also RaiiWorkerThread.h)
#define INFINITE static_cast<DWORD>(-1)
typedef LPVOID (*PTHREAD_START_ROUTINE)( LPVOID lpThreadParameter );
typedef PTHREAD_START_ROUTINE   LPTHREAD_START_ROUTINE;
#define GetCurrentThreadId()    static_cast<UINT32>(syscall(SYS_gettid))
#define Sleep(ms)               usleep((ms)*1000)

// files and directories
typedef int errno_t;
#define _O_BINARY               0 /* does not exist in Linux */
#define _O_CREAT                O_CREAT
#define _O_TRUNC                O_TRUNC
#define _O_SEQUENTIAL           0 /* does not exist in Linux */
#define _O_RDONLY               O_RDONLY
#define _O_RDWR                 O_RDWR
#define _SH_DENYNO              0 /* does not exist in Linux */
#define _SH_DENYWR              0 /* does not exist in Linux */
#define _S_IREAD                (S_IRUSR|S_IRGRP|S_IROTH)
#define _S_IWRITE               (S_IWUSR|S_IWGRP|S_IWOTH)
#define _read(h, b, n)          read(h, b, n)
#define _write(h, b, n)         write(h, b, n)
#define _close(h)               close(h)
#define _lseeki64(h, ofs, mode) lseek64(h, ofs, mode)
#define _mkdir(dirName)         mkdir(dirName,0755)     /* (octal 755) */
#define _chsize_s(h, cb)        ftruncate64(h, cb)
#define _stat64                 stat64
#define MAX_PATH                PATH_MAX
#define _MAX_PATH               PATH_MAX
#define _MAX_FNAME              FILENAME_MAX
#define _MAX_DRIVE              32
#define _MAX_DIR                PATH_MAX
#define _MAX_EXT                FILENAME_MAX
#define MAX_COMPUTERNAME_LENGTH (HOST_NAME_MAX+1)
extern void _splitpath( const char* pQualifiedFilename, char* pDrive, char* pPath, char* pBaseName, char* pExt );
extern void _fullpath( char* pFileSpec, const char* pFileName, size_t cb );

// strings
extern int sprintf_s( char* pTo, size_t cbTo, const char* fmt, ... );
extern void strcpy_s( char* pTo, size_t cbTo, const char* pFrom );
extern void strncpy_s( char* pTo, size_t cbTo, const char* pFrom, size_t cbMax );
extern void* memcpy_s( void* pTo, size_t cbTo, const void* pFrom, size_t cbFrom );
extern void* memmove_s( void* pTo, size_t cbTo, const void* pFrom, size_t cbFrom );
extern char* strcat_s( char* pTo, size_t cbTo, const char* pFrom );
#define vsprintf_s              vsnprintf
#define _strcmpi( s1, s2 )      strcasecmp(s1, s2)
#define _strnicmp( s1, s2, n )  strncasecmp(s1, s2, n)
#define sscanf_s                sscanf
extern void _strupr_s( char* p, size_t cb );
extern void _strlwr_s( char* p, size_t cb );
extern char* _strrev( char * s );

// bit manipulation
#define __popcnt            __builtin_popcount
#define __popcnt64          __builtin_popcountll
#define _byteswap_uint64(u) __builtin_bswap64(u)

#ifndef _rotl
#define _rotl(x,r)  (((x) << (r)) | ((x) >> (32 - (r))))
#endif

#ifndef _rotl64
#define _rotl64(x,r) (((x) << (r)) | ((x) >> (64 - (r))))
#endif


// miscellany
#ifndef GetLastError
extern UINT32 GetLastError( void );         // (defined in Windux.cpp)
#endif

inline void DebugBreak()
{
    raise( SIGTRAP );
}

extern bool GetComputerName( char* pbuf, LPDWORD pcch );


// implementations specific to target architecture
#if !defined(__x86_64__) && !defined(__PPC64__)
#error No Linux implementation for the target architecture.
#endif

#ifdef __x86_64__
#include "../Windux/Windux_x64.h"
#endif

#ifdef __PPC64__
#include "../Windux/Windux_PPC64.h"
#endif
