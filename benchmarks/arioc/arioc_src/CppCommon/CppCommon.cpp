/*
  CppCommon.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region global variable definitions
CDPrintFlags CDPrintFilter = static_cast<CDPrintFlags>(cdpMapOutput | cdpMapUser);  // global flags for CDPrint()
static char sbuf[DEBUGPRINT_NBUFS*DEBUGPRINT_BUFSIZE];
static volatile UINT32 ibuf = 0;           // current output buffer index
extern char* __progname_full;
#pragma endregion

/// Helper function for CDPrint
static int formatTodAndTid( char* s )
{
    // format the current time and thread ID as mm:ss.fff [tid]
    TOD tod;
    return sprintf_s( s, DEBUGPRINT_BUFSIZE, "%02u%02u%02u%03u [%08x] ", tod.hr, tod.mi, tod.se, tod.ms, GetCurrentThreadId() );
}

/// <summary>
/// Writes a formatted string to the console and/or debugger output
/// </summary>
void CDPrint( const CDPrintFlags cdPrintFlags, const char* fmt, ... )
{
    /* Potentially-useful values for cdPrintFlags (written to both debugger output and stderr):
        0xE0000007: basic output and metrics
        0xE800000F: basic output, metrics, basic trace
        0xE800001F: basic output, metrics, detailed trace
    */

    // apply the global filter
    const CDPrintFlags cdpf = static_cast<CDPrintFlags>(CDPrintFilter & cdPrintFlags);

    // do nothing if no output method is specified or if none of the levels bits are set
    if( !(cdpf & cdpCD) || !(cdpf & (cdpMapUser|cdpMapDebug)) )
        return;

    va_list args;
    va_start(args, fmt);

    // get the next available buffer
    INT32 bufferId = InterlockedExchangeAdd( &ibuf, 1 ) & (DEBUGPRINT_NBUFS-1);
    char* s = sbuf + (bufferId * DEBUGPRINT_BUFSIZE);

    int cb = 0;
    if( cdpf & cdpTimestamp )
        cb = formatTodAndTid( s );

    // format the string and append an end-of-line delimiter
    vsprintf_s( s+cb, DEBUGPRINT_BUFSIZE-cb, fmt, args );
    if( strlen(s) < (DEBUGPRINT_BUFSIZE-1) )
        strcat_s( s, DEBUGPRINT_BUFSIZE, "\n" );      // (fputs expands \n to \r\n in Windows)

    // conditionally write the string to the console
    if( cdpf & cdpConsole )
        fputs( s, stderr );
        
#ifdef _WIN32
    // conditionally write the string to the debug output
    if( cdpf & cdpDebug )
        OutputDebugStringA( s );
#endif

    va_end(args);
}

/// [extern] method XmlGetErrorString
const char* XmlGetErrorString( int errorEnum )
{
    using namespace tinyxml2;

    switch( errorEnum )
    {
        case XML_NO_ATTRIBUTE:
            return "no attribute";

        case XML_WRONG_ATTRIBUTE_TYPE:
            return "wrong attribute type";

        case XML_ERROR_FILE_NOT_FOUND:
            return "file not found";

        case XML_ERROR_FILE_COULD_NOT_BE_OPENED:
            return "file could not be opened";

        case XML_ERROR_ELEMENT_MISMATCH:
        case XML_ERROR_MISMATCHED_ELEMENT:
            return "element mismatch";

        case XML_ERROR_PARSING_ELEMENT:
            return "error parsing element";

        case XML_ERROR_PARSING_ATTRIBUTE:
            return "error parsing attribute";

        case XML_ERROR_IDENTIFYING_TAG:
            return "error identifying tag";

        case XML_ERROR_PARSING_TEXT:
            return "error parsing text";

        case XML_ERROR_PARSING_CDATA:
            return "error parsing CDATA";

        case XML_ERROR_PARSING_COMMENT:
            return "error parsing comment";

        case XML_ERROR_PARSING_DECLARATION:
            return "error parsing declaration";

        case XML_ERROR_PARSING_UNKNOWN:
        case XML_ERROR_PARSING:
                return "unknown parsing error";

        case XML_ERROR_EMPTY_DOCUMENT:
            return "empty document";
            
        default:
            return "unspecified XML error";
    }
}

/// [extern] method XmlFormatErrorInfo
void XmlFormatErrorInfo( char* buf, int cbBuf, tinyxml2::XMLDocument* pdoc, char* methodName, int errorEnum )
{
    // copy the method name and error enum value into the specified buffer
    sprintf_s( buf, cbBuf, "%s failed with error %d: %s", methodName, errorEnum, XmlGetErrorString(errorEnum) );

    char* pErrorInfo1 = const_cast<char*>(pdoc->GetErrorStr1());
    if( pErrorInfo1 && *pErrorInfo1 )
    {
        // ignore preceding spaces
        while( isspace( *pErrorInfo1 ) )
            ++pErrorInfo1;

        strcat_s( buf, cbBuf, "\r\nXML error details:\r\n" );
        strcat_s( buf, cbBuf, pErrorInfo1 );

        char* pErrorInfo2 = const_cast<char*>(pdoc->GetErrorStr2());
        if( pErrorInfo2 && *pErrorInfo2 )
        {
            // ignore preceding spaces
            while( isspace( *pErrorInfo2 ) )
                ++pErrorInfo2;

            // append the second error string if it differs from the first
            if( _strcmpi( pErrorInfo1, pErrorInfo2 ) )
            {
                strcat_s( buf, cbBuf, "\r\nAdditional XML error details:\r\n" );
                strcat_s( buf, cbBuf, pErrorInfo2 );
            }
        }
    }
}

/// [extern] method LoadXmlFile
void LoadXmlFile( tinyxml2::XMLDocument* pdoc, const char* fileSpec )
{
    INT32 rval = pdoc->LoadFile( fileSpec );
    if( rval )
    {
        char methodInfo[FILENAME_MAX+64];
        sprintf_s( methodInfo, sizeof methodInfo, "XMLDocument::Loadfile( \"%s\" )", fileSpec );
        char buf[ApplicationExceptionInfo::BUFFER_SIZE-1024];
        XmlFormatErrorInfo( buf, sizeof buf, pdoc, methodInfo, rval );
        throw new ApplicationException( __FILE__, __LINE__, buf );
    }
}

/// [extern] method SaveXmlFile
void SaveXmlFile( const char* fileSpec, tinyxml2::XMLDocument* pdoc )
{
    int rval = pdoc->SaveFile( fileSpec );
    if( rval )
    {
        char methodInfo[FILENAME_MAX+64];
        sprintf_s( methodInfo, sizeof methodInfo, "XMLDocument::Savefile( \"%s\" )", fileSpec );
        char buf[ApplicationExceptionInfo::BUFFER_SIZE-1024];
        XmlFormatErrorInfo( buf, sizeof buf, pdoc, methodInfo, rval );
        throw new ApplicationException( __FILE__, __LINE__, buf );
    }
}

/// [extern] method SameXmlAttributes
bool SameXmlAttributes( tinyxml2::XMLElement* pel1, tinyxml2::XMLElement* pel2 )
{
    using namespace tinyxml2;

    // ensure that the XML elements have the same name
    if( strcmp( pel1->Name(), pel2->Name() ) )
        throw new ApplicationException( __FILE__, __LINE__, "XML element name mismatch: %s != %s", pel1->Name(), pel2->Name() );

    // count the attributes in the first XML element
    UINT32 nat1 = 0;
    const XMLAttribute* pat1 = pel1->FirstAttribute();
    while( pat1 )
    {
        nat1++;
        pat1 = pat1->Next();
    }

    // count the attributes in the second XML element
    UINT32 nat2 = 0;
    const XMLAttribute* pat2 = pel2->FirstAttribute();
    while( pat2 )
    {
        nat2++;
        pat2 = pat2->Next();
    }

    // return false if the XML elements have a different number of attributes
    if( nat1 != nat2 )
        return false;

    // ensure that each attribute exists and has the same value in both XML elements
    pat1 = pel1->FirstAttribute();
    while( pat1 )
    {
        if( !pel2->Attribute( pat1->Name(), pat1->Value() ) )
            return false;

        pat1 = pat1->Next();
    }

    // at this point the XML elements have the same number of attributes, and all of them have the same value
    return true;
}

/// [extern] a good approximation of the error function (from Numerical Recipes in C, Chapter 6)
double nrc_erfc( double x )
{
    /* (per NRinC) Returns the complementary error function erfc(x) with fractional error everywhere less than 1.2E-7 */

    double t;
    double z;
    double ans;

    z = fabs( x );
    t = 1.0 / (1.0 + 0.5*z);
    ans = t * exp( -z*z - 1.26551223 +
                          t * (1.00002368 +
                          t * (0.37409196 +
                          t * (0.09678418 +
                          t * (-0.18628806 +
                          t * (0.27886807 +
                          t * (-1.13520398 +
                          t * (1.48851587 +
                          t * (-0.82215223 +
                          t * 0.17087277)))))))));
    return (x >= 0.0) ? ans : (2.0 - ans);
}

#ifdef _WIN32
/// [extern] GetAvailableCpuThreadCount
INT32 GetAvailableCpuThreadCount()
{
    SYSTEM_INFO si;
    GetSystemInfo( &si );
    return static_cast<INT32>(si.dwNumberOfProcessors);
}

/// [extern] GetTotalSystemRAM
UINT64 GetTotalSystemRAM()
{
    /* Get the total amount of host memory immediately available to this process.  Per MSDN documentation, this is:

        The amount of physical memory currently available, in bytes. This is the amount of physical memory that can be immediately
        reused without having to write its contents to disk first.

       We're interested in physical RAM because:
        - we will page-lock physical RAM for lookup tables
        - although a 64-bit Windows process gets 8TB of virtual memory, for performance reasons we'd prefer to avoid swapping memory pages to disk
    */
    MEMORYSTATUSEX msx;
    msx.dwLength = sizeof msx;
    GlobalMemoryStatusEx( &msx );
    return msx.ullTotalPhys;
}
#endif

#ifdef __GNUC__
/// [extern] GetAvailableCpuThreadCount
INT32 GetAvailableCpuThreadCount()
{
    cpu_set_t cs;
    CPU_ZERO( &cs );
    int rval = sched_getaffinity( 0, sizeof cs, &cs );
    if( rval < 0 )
        throw "GetAvailableCpuThreadCount: error in sched_getaffinity()";

#ifdef CPU_COUNT
    return CPU_COUNT( &cs );
#else
    // treat the cpu_set_t as a bitmap and count the 1 bits
    int nWords = sizeof(cpu_set_t) / sizeof(UINT32);      // number of 32-bit words in a cpu_set_t
    int nThreads = 0;
    UINT32* pcs = reinterpret_cast<UINT32*>( &cs );
    for( int n=0; n<nWords; ++n )
       nThreads += __popcnt( pcs[n] );
    return nThreads;
#endif
}

/// [extern] GetTotalSystemRAM
UINT64 GetTotalSystemRAM()
{
    /* (See comments for the Windows implementation of this function.)
    */
    size_t nPages = sysconf( _SC_PHYS_PAGES );
    size_t cbPage = sysconf( _SC_PAGE_SIZE );
    return nPages * cbPage;
}
#endif

/// [extern] PluralS
const char* PluralS( const INT32 n )
{
    return (n == 1) ? "" : "s";
}
