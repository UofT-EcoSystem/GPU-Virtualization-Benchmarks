/*
  Windux.cpp

    Copyright (c) 2015-2017 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "Windux.h"

/// <summary>
/// Returns the error number associated with the most recent system API call.
/// </summary>
DWORD GetLastError()
{
    return errno;
}

/// <summary>
/// Extracts the drive letter (nonexistent in Linux), directory path, base filename, and extension
///  from a fully-qualified file specification.
/// </summary>
void _splitpath( const char* pQualifiedFilename, char* pDrive, char* pPath, char* pBaseName, char* pExt )
{
    // sanity checks
    if( pQualifiedFilename == NULL )
        throw "_splitpath: qualified filename is null";
    if( *pQualifiedFilename == 0 )
        throw "_splitpath: qualified filename is empty";

    // remove dots, etc.
    char path[PATH_MAX] = { 0 };
    char* rval = realpath( pQualifiedFilename, path );
    if( rval == NULL )
    {
        char errMsg[64];
        sprintf( errMsg, "%s: error %d in realpath()", __FUNCTION__, errno );
        throw errMsg;
    }
    if( *path == 0 )
        throw "_splitpath: qualified filename is empty";

    if( pDrive != NULL )
    {
        // pick off everything up until the colon (if any)
        char* p = strchr( path, ':' );
        if( p != NULL )
        {
            UINT32 cch = static_cast<UINT32>(p - path);
            memcpy( pDrive, path, cch );
            pDrive[cch] = 0;
        }
        else
            *pDrive = 0;
    }

    // split the filename from everything else
    INT32 cch = strlen( path );
    char* p = path + cch - 1;
    if( *p == '/' )
        *(p--) = 0;    // remove trailing separator

    while( (p > path) && (*p != '/') )
        p-- ; 

    if( p > path )
    {
        /* p points to the rightmost separator */
        if( pPath )
        {
            cch = static_cast<UINT32>(p - path) + 1;    // number of characters in the path string (including the rightmost separator)
            memcpy( pPath, path, cch );
            pPath[cch] = 0;
        }

        // point past the separator to the filename
        p++ ;
    }
    else
    {
        /* there is no path, just a filename (and p points to it) */
        if( pPath )
            *pPath = 0;
    }
    
    // split the extension from everything else
    cch = strlen( p );
    char* px = p + cch - 1;
    while( (*px != '.') && (px > p) )
        px-- ;

    if( px > p )
    {
        /* px points to the rightmost separator */

        if( pBaseName )
        {
            cch = static_cast<UINT32>(px - p);
            memcpy( pBaseName, p, cch );
            pBaseName[cch] = 0;
        }

        if( pExt )
        {
            // point past the separator to the extension
            px++ ;

            // copy the extension
            strcpy( pExt, px );
        }
    }
    else
    {
        /* there is no extension, just a base name (and p points to it) */
        if( pBaseName )
            strcpy( pBaseName, p );
        if( pExt )
            *pExt = 0;
    }
}

/// <summary>
/// Returns a fully-qualified file specification for a given file name.
/// </summary>
void _fullpath( char* pFileSpec, const char* pFileName, size_t cb )
{
    char* p = realpath( pFileName, NULL );
    if( p )
    {
        // return a copy of the "canonicalized" file name (file specification)
        strcpy_s( pFileSpec, cb, p );
    }
    else
    {
        // return a copy of the specified file name
        strcpy_s( pFileSpec, cb, pFileName );
    }

    // free the buffer that was malloc'd by realpath()
    free( p );
}

/// <summary>
/// Converts the specified ASCIIZ string to upper case.
/// </summary>
void _strupr_s( char* p, size_t cb )
{
    while( (cb--) && *p )
    {
        *p = toupper( *p );
        p++ ;
    }
}

/// <summary>
/// Converts the specified ASCIIZ string to lower case.
/// </summary>
void _strlwr_s( char* p, size_t cb )
{
    while( (cb--) && *p )
    {
        *p = tolower( *p );
        p++ ;
    }
}


/// <summary>
/// Reverses the specified character string
/// </summary>
char* _strrev( char *s )
{
    int i = static_cast<int>(strlen(s)) - 1;
    int j = 0;

    char c;
    while( i > j)
    {
        c = s[i];
        s[i]= s[j];
        s[j] = c;
        i--;
        j++;
    }

    return s;
}


#if TODO_CHOP_IF_UNUSED
/* Atomic exchange (of various sizes) */
static inline void *xchg_64(void *ptr, void *x)
{
	__asm__ __volatile__("xchgq %0,%1"
				:"=r" ((unsigned long long) x)
				:"m" (*(volatile long long *)ptr), "0" ((unsigned long long) x)
				:"memory");

	return x;
}

static inline unsigned xchg_32(void *ptr, unsigned x)
{
	__asm__ __volatile__("xchgl %0,%1"
				:"=r" ((unsigned) x)
				:"m" (*(volatile unsigned *)ptr), "0" (x)
				:"memory");

	return x;
}

static inline unsigned short xchg_16(void *ptr, unsigned short x)
{
	__asm__ __volatile__("xchgw %0,%1"
				:"=r" ((unsigned short) x)
				:"m" (*(volatile unsigned short *)ptr), "0" (x)
				:"memory");

	return x;
}

/* Test and set a bit */
static inline char atomic_bitsetandtest(void *ptr, int x)
{
	char out;
	__asm__ __volatile__("lock; bts %2,%1\n"
						"sbb %0,%0\n"
				:"=r" (out), "=m" (*(volatile long long *)ptr)
				:"Ir" (x)
				:"memory");

	return out;
}
#endif


/// <summary>
/// strcpy_s
/// </summary>
void strcpy_s( char* pTo, size_t cbTo, const char* pFrom )
{
    // sanity checks
    if( pTo == NULL )
        throw "strcpy_s: null destination pointer";
    if( pFrom == NULL )
        throw "strcpy_s: null source pointer";

    // do nothing if there is nothing to copy
    if( cbTo == 0 )
        return;

    // copy the bytes until a null byte is copied or until the destination buffer is filled
    char* p = pTo;
    size_t cbRemaining = cbTo;

    /* Loop until a null is encountered in the source string, or the number of remaining bytes in the
        destination buffer is zero (which won't happen if a null is encountered). */
    while( ((*(p++) = *(pFrom++)) != 0) && (--cbRemaining) );

    // look for a buffer overrun
    if( cbRemaining == 0 )
        throw "strcpy_s: buffer overrun";
}

/// <summary>
/// strncpy_s
/// </summary>
void strncpy_s( char* pTo, size_t cbTo, const char* pFrom, size_t cbMax )
{
    // sanity checks
    if( pTo == NULL )
        throw "strcpy_s: null destination pointer";
    if( pFrom == NULL )
        throw "strcpy_s: null source pointer";

    // do nothing if there is nowhere to copy the data
    if( cbTo == 0 )
        return;

    /* Loop until...
        - the specified number of bytes has been copied, or
        - a null is copied, or
        - the number of remaining bytes in the destination buffer is zero (which won't happen
           if a null is encountered)
    */
    char* p = pTo;
    size_t cbRemaining = cbTo;
    while( cbMax && ((*(p++) = *(pFrom++)) != 0) && (--cbRemaining) )
        --cbMax;

    // look for a buffer overrun
    if( cbRemaining == 0 )
        throw "strcpy_s: buffer overrun";

    /* append a null byte if...
        - the specified number of bytes has been copied, and
        - at least one byte was copied, and
        - at least one unused byte remains in the target buffer, and
        - a null byte has not yet been copied
    */
    if( (cbMax == 0) && cbRemaining && ((p == pTo) || p[-1]) )
        *p = 0;
}

/// <summary>
/// sprintf_s 
/// </summary>
int sprintf_s( char* pTo, size_t cbTo, const char* fmt, ... )
{
    /* We don't have access to the innards of sprintf(), nor do we want to, but we do want to catch any
        buffer overrun that might occur.  So we simply wrap a call to vsnprintf() with some error handling.
    */
    va_list args;
    va_start(args, fmt);

    /* In case of overrun: vsnprintf truncates the string in the destination buffer but returns the
        total length of the formatted string (excluding a null terminating byte). */
    int rval = vsnprintf( pTo, cbTo, fmt, args );
    if( static_cast<size_t>(rval) >= cbTo )
        throw "sprintf_s: buffer overrun";

    va_end(args);

    return rval;
}

/// <summary>
/// memcpy_s
/// </summary>
void* memcpy_s( void* pTo, size_t cbTo, const void* pFrom, size_t cbFrom )
{
    if( cbFrom > cbTo )
        throw "memcpy_s: buffer overrun";

    return memcpy( pTo, pFrom, cbFrom );
}

/// <summary>
/// memmove_s
/// </summary>
void* memmove_s( void* pTo, size_t cbTo, const void* pFrom, size_t cbFrom )
{
    if( cbFrom > cbTo )
        throw "memmove_s: buffer overrun";

    return memmove( pTo, pFrom, cbFrom );}

/// <summary>
/// strcat_s
/// </summary>
char* strcat_s( char* pTo, size_t cbTo, const char* pFrom )
{
    // sanity checks
    if( pTo == NULL )
        throw "strcat_s: null destination pointer";
    if( pFrom == NULL )
        throw "strcat_s: null source pointer";

    size_t cchTo = strlen( pTo );
    if( cbTo <= cchTo )
        throw "strcat_s: buffer full";

    // copy the bytes until a null byte is copied or until the destination buffer is filled
    char* p = pTo + cchTo;
    size_t cbRemaining = cbTo - cchTo;

    /* Loop until a null is encountered in the source string, or the number of remaining bytes in the
        destination buffer is zero (which won't happen if a null is encountered). */
    while( ((*(p++) = *(pFrom++)) != 0) && (--cbRemaining) );

    // look for a buffer overrun
    if( cbRemaining == 0 )
        throw "strcat_s: buffer overrun";

    return pTo;
}

/// <summary>
/// GetComputerName
/// </summary>
bool GetComputerName( char* pbuf, LPDWORD pcch )
{
    bool rval = true;       // true: success

    /* The call to gethostname() returns 0 to indicate success. */
    if( gethostname( pbuf, *pcch ) )
        pbuf[(*pcch)-1] = 0;                // null-terminate whatever is in the specified buffer

    // return the returned string length
    *pcch = static_cast<DWORD>(strlen( pbuf ));

    return rval;
}
