/*
  RaiiDirectory.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static variable definitions
#ifdef __GNUC__
__thread const char* RaiiDirectory::pattern;
__thread int RaiiDirectory::cchPattern;
#endif
#pragma endregion

#pragma region constructors and destructor
/// default constructor
RaiiDirectory::RaiiDirectory() : m_handle(-1)
{
}

/// constructor
RaiiDirectory::RaiiDirectory( const char* dirPath, const char* filenamePattern ) : m_handle(-1)
{
    this->GetFilenames( dirPath, filenamePattern );
}

// destructor
RaiiDirectory::~RaiiDirectory()
{
#ifdef _WIN32
    if( m_handle != -1 )
        _findclose( m_handle );
#endif
}
#pragma endregion

#pragma region static methods
#ifdef __GNUC__
/// [static] method dirFilter
int RaiiDirectory::dirFilter( const struct dirent* pde )
{
    /* The basic strategy is to traverse the specified directory-entry name, using the specified pattern:
        - * matches 0 or more characters in the name
        - ? matches one character in the name

       We return nonzero to indicate that the name matches the pattern.
    */

    // first, some trivial cases
    if( pde->d_name[0] == '.' )
    {
        if( pde->d_name[1] == 0 )                       // ignore "."
            return 0;
        if( pde->d_name[1] == '.' )                     // ignore ".."
            return 0;
    }

    if( (RaiiDirectory::pattern == NULL) ||             // null or empty pattern ...
        (RaiiDirectory::cchPattern == 0) )              // ... matches anything
        return 1;

    if( (RaiiDirectory::cchPattern == 1) &&
        (RaiiDirectory::pattern[0] == '*') )            // "*" matches anything
        return 1;

    // traverse the pattern
    const char* ix = pde->d_name;
    const char* ip = RaiiDirectory::pattern;
    do
    {
        switch( *ip )
        {
            case '*':
                /* There are two cases here:
                    - The wildcard is at the end of the string: it's a match regardless of what remains in the name.
                    - The wildcard is not at the end of the string: find the next character in the name that
                        matches the next character in the pattern.
                */

                // traverse the pattern until the next non-wildcard character is encountered
                while( *(++ip) && ((*ip == '?') || (*ip == '*')) );
                
                if( *ip == 0 )                      // trailing wildcard: it's a match
                    return 1;

                /* At this point ip points to the character we need to match.  We scan for it in the name. */
                while( *ix != *ip )
                {
                    if( *ix == 0 )                  // we reached the end of the name without finding a match
                        return 0;

                    ++ix;
                }

                /* At this point we have a match. */
                break;

            case '?':                               // any character matches
                break;

            default:
                /* Non-wildcard characters must match exactly.  This logic also handles the case where the pattern string is
                    longer than the name string (i.e. *ip is non-null, *ix is null). */
                if( *ip != *ix )                    
                    return 0;
                break;
        }

        // advance the pointer to the name string
        ++ix;
    }
    while( *(++ip) );

    /* At this point we have reached the end of the pattern, so it's a match if we have also reached the end of the name string. */
    return (*ix == 0);
}
#endif

/// [private static] method filenameComparer
int RaiiDirectory::filenameComparer( const void* a, const void* b )
{
    return _strcmpi( *reinterpret_cast<char* const *>(a), *reinterpret_cast<char* const *>(b) );
}

/// [static] method SetPathSeparator
void RaiiDirectory::SetPathSeparator( char* dirPath )
{
    // replace the separator character (forward shash or backslash) with the character used by the current OS
    char* p = dirPath;
    char rsep = (FILEPATHSEPARATOR[0] == '\\') ? '/' : '\\';
    while( *p )
    {
        if( *p == rsep )
            *p = FILEPATHSEPARATOR[0];

        ++p;
    }
}

/// [static] method ChopTrailingPathSeparator
void RaiiDirectory::ChopTrailingPathSeparator( char* dirPath )
{
    INT32 cb = static_cast<INT32>(strlen( dirPath ));
    char* p = dirPath + cb - 1;
    if( (*p == '\\') || (*p == '/') )
        *p = 0;
}

/// [static] method AppendTrailingPathSeparator
void RaiiDirectory::AppendTrailingPathSeparator( char* dirPath )
{
    INT32 cb = static_cast<INT32>(strlen( dirPath ));
    char* p = dirPath + cb - 1;
    if( (*p != '\\') && (*p != '/') )
        *reinterpret_cast<UINT16*>(p+1) = static_cast<UINT16>(FILEPATHSEPARATOR[0]);
}

/// [static] method IsAbsolutePath
bool RaiiDirectory::IsAbsolutePath( char* dirPath )
{
    // if the specified path is null, return false (since we can't return NULL)
    if( (dirPath == NULL) || (*dirPath == 0) )
        return false;

    // if the first symbol is a path separator, return true
    if( *dirPath == FILEPATHSEPARATOR[0] )
        return true;

    // if the path contains a colon, return true
    if( strchr( dirPath, ':' ) )
        return true;

    // at this point it's not an absolute path
    return false;
}

/// [static] method OpenOrCreateDirectory
void RaiiDirectory::OpenOrCreateDirectory( const char* dirPath )
{
    // make a local copy of the specified path
    char path[FILENAME_MAX];
    strcpy_s( path, FILENAME_MAX, dirPath );

    // ensure that the path is terminated by a path separator
    INT32 cb = static_cast<INT32>(strlen( path ));
    if( path[cb-1] != FILEPATHSEPARATOR[0] )
        strcat_s( path, FILENAME_MAX, FILEPATHSEPARATOR );

    // look for path separators in the specified directory path
    bool isFirstSep = true;
    char* pEnd = path + 2;  // (skip over any drive letter or UNC double-slash)
    while( *pEnd )
    {
        if( *pEnd == FILEPATHSEPARATOR[0] )
        {
            if( isFirstSep )
                isFirstSep = false;
            else
            {
                // temporarily truncate the path at the current separator
                *pEnd = 0;

                // try to create the directory if it doesn't already exist
                if( 0 != _mkdir( path ) )
                {
                    if( errno != EEXIST )       // EEXIST: directory already exists
#pragma warning ( push )
#pragma warning ( disable : 4996 )		// don't nag us about strerror being "deprecated"
                        throw new ApplicationException( __FILE__, __LINE__, "unable to open or create directory %s (errno=%d: %s)", path, errno, strerror(errno) );
#pragma warning ( pop )
                }

                // restore the separator
                *pEnd = FILEPATHSEPARATOR[0];
            }
        }

        // keep scanning the specified path
        pEnd++ ;
    }
}
#pragma endregion

#pragma region public methods
/// [public] method GetFilenames
void RaiiDirectory::GetFilenames( const char* dirPath, const char* filenamePattern )
{
    // save copies of the specified directory path and filename wildcard pattern
    strcpy_s( DirSpec, FILENAME_MAX, dirPath );
    if( filenamePattern )
    {
        strcpy_s( FilenamePattern, FILENAME_MAX, filenamePattern );

#ifdef __GNUC__
        // save a reference to the filename wildcard pattern in thread-local storage
        RaiiDirectory::pattern = filenamePattern;
        RaiiDirectory::cchPattern = static_cast<int>(strlen(filenamePattern));
#endif
    }
    else
        this->FilenamePattern[0] = 0;

    // clean the directory path
    RaiiDirectory::SetPathSeparator( DirSpec );
    RaiiDirectory::ChopTrailingPathSeparator( DirSpec );

#ifdef _WIN32
    // build a fully-qualified filename specification
    char findPattern[FILENAME_MAX];
    sprintf_s( findPattern, FILENAME_MAX, "%s%s%s", DirSpec, FILEPATHSEPARATOR, filenamePattern );

    // look for the first matching filename in the specified directory path
    _finddata_t fd;
    m_handle = _findfirst( findPattern, &fd );
    if( m_handle == -1 )
    {
        if( errno == ENOENT )       // (no matching filenames)
            return;
        else
#pragma warning ( push )
#pragma warning ( disable : 4996 )		// don't nag us about strerror being "deprecated"
            throw new ApplicationException( __FILE__, __LINE__, "unable to find file(s) for '%s' (errno=%d: %s)", dirPath, errno, strerror(errno) );
#pragma warning (pop)
    }

    /* at this point fd contains file info for the first file that matches the directory path specification */

    // allocate space in a buffer to contain filenames
    Buffer.Realloc( DIRBUFCAPACITY*FILENAME_MAX, true );
    INT32 ofsNextInBuf = 0;
    
    // allocate a list of offsets to the filenames in the buffer
    WinGlobalPtr<INT32> fo( DIRBUFCAPACITY, true );
    INT32 nFilenames = 0;

    while( *fd.name )
    {
        // resize the buffers if necessary
        INT32 cbFilename = static_cast<INT32>(strlen( fd.name ) + 1);
        if( (ofsNextInBuf+cbFilename) > Buffer.cb )
        {
            // grow the buffer by 10% and zero the newly-allocated memory
            Buffer.Realloc( 11*Buffer.cb/10, true );
        }

        if( nFilenames == fo.Count )
            fo.Realloc( 1+(11*fo.Count/10), true );

        // copy the filename from the finddata struct into the buffer
        strcpy_s( Buffer.p+ofsNextInBuf, Buffer.cb-ofsNextInBuf, fd.name );
        fo.p[nFilenames++] = ofsNextInBuf;
        ofsNextInBuf += cbFilename;

        // continue to traverse the list of filenames in the directory
        if( _findnext( m_handle, &fd ) )
        {
            if( errno == ENOENT )
                *fd.name = 0;
            else
#pragma warning ( push )
#pragma warning ( disable : 4996 )		// don't nag us about strerror being "deprecated"
                throw new ApplicationException( __FILE__, __LINE__, "_findnext failed for %s (errno=%d: %s)", dirPath, errno, strerror(errno) );
#pragma warning ( pop )
        }
    }

    // get rid of the handle
    _findclose( m_handle );
    m_handle = -1;

    // shrink the buffer that contains the filenames
    Buffer.Realloc( ofsNextInBuf, false );

    // build a list of pointers to the filenames
    Filenames.Realloc( nFilenames, false );
    for( INT32 n=0; n<nFilenames; ++n )
        Filenames.p[n] = Buffer.p + fo.p[n];

    // sort the list
    qsort( Filenames.p, nFilenames, sizeof(char*), filenameComparer );
#endif

#ifdef __GNUC__    
    // scan the directory for matching filenames
    struct dirent** fd;                             // pointer to a list of dirent structs (returned by scandir)

    int nf = scandir( dirPath,                      // directory path
                      &fd,                          // pointer to the pointer to the list of dirent structs
                      RaiiDirectory::dirFilter,     // filter
                      alphasort                     // sort function
                    );

    // the return value is the number of directory entries, or -1 if an error occurred
    if( nf < 0 )
        throw new ApplicationException( __FILE__, __LINE__, "scandir failed for %s (errno=%d: %s)", dirPath, errno, strerror(errno));

    /* at this point fd contains file info for the files that matches the directory path specification */

    // allocate space in a buffer to contain filenames
    Buffer.Realloc( nf*FILENAME_MAX, true );
    INT32 ofsNextInBuf = 0;
    
    // allocate a list of offsets to the filenames in the buffer
    WinGlobalPtr<INT32> fo( DIRBUFCAPACITY, true );
    INT32 nFilenames = 0;

    for( int n=0; n<nf; ++n )
    {
        // resize the buffers if necessary
        INT32 cbFilename = static_cast<INT32>(strlen( fd[n]->d_name ) + 1);
        if( (ofsNextInBuf+cbFilename) > static_cast<INT32>(Buffer.cb) )
        {
            // grow the buffer by 10% and zero the newly-allocated memory
            Buffer.Realloc( 11*Buffer.cb/10, true );
        }

        if( nFilenames == static_cast<INT32>(fo.Count) )
            fo.Realloc( 1+(11*fo.Count/10), true );

        // copy the filename from the dirent struct into the buffer
        strcpy_s( Buffer.p+ofsNextInBuf, Buffer.cb-ofsNextInBuf, fd[n]->d_name );
        fo.p[nFilenames++] = ofsNextInBuf;
        ofsNextInBuf += cbFilename;
    }

    // shrink the buffer that contains the filenames
    Buffer.Realloc( ofsNextInBuf, false );

    // build a list of pointers to the filenames
    Filenames.Realloc( nFilenames, false );
    for( INT32 n=0; n<nFilenames; ++n )
        Filenames.p[n] = Buffer.p + fo.p[n];
#endif
}

/// [public] method GetFileSpecification
void RaiiDirectory::GetFileSpecification( INT32 iFilename, WinGlobalPtr<char>& fileSpec )
{
    /* Allocate enough space for
        - the directory path
        - a separator
        - the filename
        - a trailing null
    */
    fileSpec.Realloc( strlen(this->DirSpec)+1+strlen(Filenames.p[iFilename])+1, false );

    // build the fully-qualified file specification
    sprintf_s( fileSpec.p, fileSpec.cb, "%s%s%s", this->DirSpec, FILEPATHSEPARATOR, Filenames.p[iFilename] );
}
#pragma endregion
