/*
  RaiiDirectory.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiDirectory__

/// <summary>
/// Class <c>RaiiDirectory</c> provides a "resource acquisition is initialization" wrapper for Win32 directory APIs.
/// </summary>
class RaiiDirectory
{
    private:
        static const INT32 DIRBUFCAPACITY = 160;

    private:
        intptr_t            m_handle;

    public:
        char                DirSpec[FILENAME_MAX];
        char                FilenamePattern[FILENAME_MAX];
        WinGlobalPtr<char>  Buffer;
        WinGlobalPtr<char*> Filenames;

    private:
#ifdef __GNUC__
        static __thread const char* pattern;
        static __thread int cchPattern;
        static int dirFilter( const struct dirent* pde );
#endif
        static int filenameComparer( const void*, const void* );

    public:
        RaiiDirectory( void );
        RaiiDirectory( const char* dirPath, const char* filenamePattern );
        ~RaiiDirectory( void );
        void GetFilenames( const char* dirPath, const char* filenamePattern );
        void GetFileSpecification( INT32 iFilename, WinGlobalPtr<char>& fileSpec );
        static void SetPathSeparator( char* filePath );
        static void ChopTrailingPathSeparator( char* filePath );
        static void AppendTrailingPathSeparator( char* filePath );
        static bool IsAbsolutePath( char* filePath );
        static void OpenOrCreateDirectory( const char* dirPath );
};
