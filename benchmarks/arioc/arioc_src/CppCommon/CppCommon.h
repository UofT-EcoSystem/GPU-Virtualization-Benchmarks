/*
  CppCommon.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __CppCommon__

#define DEBUGPRINT_BUFSIZE      (1024)
#define DEBUGPRINT_NBUFS        32          // (must be a power of 2)
#define APPNAME_BUFSIZE         16
#define APPVER_BUFSIZE          16
#define APPLEGAL_BUFSIZE        80
#define GIGADOUBLE              (1024.0*1024*1024)

// general C++ razzle-dazzle
#define msizeof(st, m)      (sizeof (reinterpret_cast<st*>(NULL))->m)           // sizeof( struct member )
#define moffsetof(st, m)    (reinterpret_cast<UINT64>(&(reinterpret_cast<st*>(NULL))->m))
#define blockdiv(i, d)      (((i)+(d)-1)/(d))                                   // divide i into blocks of size d
#define round2power(n, p2)  (((n)+(p2)-1) & ((~p2)+1))                          // round n up to next p2, where p2 is a power of 2 ( (~p2)+1 is the same as -p2 )
#define signum(x)           (((x)>0) ? 1 : (((x)<0) ? -1 : 0))                  // 1, 0, or -1 depending on the sign of x
#define suppressLNK4221()   namespace { char suppressLNK4221$__COUNTER__; }     // include in stdafx.cpp to suppress the linker's LNK4221 warning
#define d2i32(d)            static_cast<INT32>((d) + 0.5)                       // pretty good rounding for nonnegative double-to-integer conversion

/* This cryptic definition of arraysize comes from http://demin.ws/blog/english/2011/05/24/safer-sizeof-for-arrays-in-cpp/, although the blog author
    attributes the idea to the Google Chrome source code.

   The goal is to avoid using something like this to determine the number of elements in an array:
        #define arraysize(a)  (sizeof(a)/sizeof(a[0]))
   which will fail silently if you pass it a pointer to an array instead of an array.

   The basic idea here is to define a template function that returns an array of N characters, where N is the number of elements in
    the array whose size is needed.  The C++ compiler knows the size of the specified array at compile time, so it
        - deduces the value of N (i.e., the number of elements in the array whose size is needed)
        - knows that the return value would be a string of N consecutive bytes, and
        - returns the correct value for the sizeof() operator
   It doesn't seem to matter that the template function is never defined -- for our purposes, the C++ compiler knows all it needs to
    know from the function declaration.
*/
template <typename T, size_t N> INT8 (&ArraySizeHelper(T (&array)[N]))[N];
#define arraysize(array) (sizeof(ArraySizeHelper(array)))

/* Definitions of pairwise minimum and maximum macros that don't conflict with anything in the Linux system include files: */
#define max2(a,b) (((a) > (b)) ? (a) : (b))
#define min2(a,b) (((a) < (b)) ? (a) : (b))

// defines that increment a pointer to a multibyte integer value by a specified number of bytes
#define PINT32INCR(p,incr)   (reinterpret_cast<INT32*>( reinterpret_cast<char*>(p) + incr ))
#define PINT64INCR(p,incr)   (reinterpret_cast<INT64*>( reinterpret_cast<char*>(p) + incr ))


#ifdef WIN32
#define FILEPATHSEPARATOR   "\\"
#else
#define FILEPATHSEPARATOR   "/"
#endif

enum CDPrintFlags
{
    cdpNone =       0x00000000,
    cdpMapUser =    0x000000FF, // user-defined settings
    cdpMapDebug =   0x0FFFFF00, // debug
    cdpMapOutput =  0xF0000000, // output options
    
    cdpConsole =    0x80000000,
    cdpDebug =      0x40000000,
    cdpTimestamp =  0x20000000,
    cdpCD =         (cdpConsole | cdpDebug),

    cdpCD0 =        (cdpMapOutput | 0x00000001),    // banners, exceptions
    cdpCD1 =        (cdpMapOutput | 0x00000002),    // basic performance metrics 
    cdpCD2 =        (cdpMapOutput | 0x00000004),    // detailed performance metrics
    cdpCD3 =        (cdpMapOutput | 0x00000008),    // basic trace
    cdpCD4 =        (cdpMapOutput | 0x00000010),    // detailed trace
    cdpCD5 =        (cdpMapOutput | 0x00000020),    // (unused)
    cdpCD6 =        (cdpMapOutput | 0x00000040),    // (unused)
    cdpCD7 =        (cdpMapOutput | 0x00000080),    // (unused)

    cdpCDa =        (cdpMapOutput | 0x00000100),    // trace host memory management
    cdpCDb =        (cdpMapOutput | 0x00000200),    // trace CUDA memory management
    cdpCDc =        (cdpMapOutput | 0x00000400),    // trace file open/close
    cdpCDd =        (cdpMapOutput | 0x00000800),    // trace file I/O
    cdpCDe =        (cdpMapOutput | 0x00001000),    // trace main loop iterations
    cdpCDf =        (cdpMapOutput | 0x00002000),    // Q sequence loading
    cdpCDg =        (cdpMapOutput | 0x00004000),    // encoded read length distribution
    cdpCDh =        (cdpMapOutput | 0x00008000),
    cdpCDi =        (cdpMapOutput | 0x00010000),
    cdpCDj =        (cdpMapOutput | 0x00020000),
    cdpCDk =        (cdpMapOutput | 0x00040000),
    cdpCDl =        (cdpMapOutput | 0x00080000),
    cdpCDm =        (cdpMapOutput | 0x00100000),
    cdpCDn =        (cdpMapOutput | 0x00200000),
    cdpCDo =        (cdpMapOutput | 0x00400000),
    cdpCDp =        (cdpMapOutput | 0x00800000)
};

// CppCommon functions with application-global scope
extern void CDPrint( const CDPrintFlags cdPrintFlags, const char* fmt, ... );
extern const char* XmlGetErrorString( int errorEnum );
extern void XmlFormatErrorInfo( char* buf, int cbBuf, tinyxml2::XMLDocument* pdoc, char* methodName, int errorEnum );
extern void LoadXmlFile( tinyxml2::XMLDocument* pdoc, const char* fileSpec );
extern void SaveXmlFile( const char* fileSpec, tinyxml2::XMLDocument* pdoc );
extern bool SameXmlAttributes( tinyxml2::XMLElement* pel1, tinyxml2::XMLElement* pel2 );
extern double nrc_erfc( double x );
extern INT32 GetAvailableCpuThreadCount( void );
extern UINT64 GetTotalSystemRAM( void );
extern const char* PluralS( const INT32 n );

// CppCommon variables with application-global scope
extern CDPrintFlags CDPrintFilter;
