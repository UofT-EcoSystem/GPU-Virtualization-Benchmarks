/*
  OutputFileInfo.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// constructor
OutputFileInfo::OutputFileInfo( const char* _basePath, const char* _oftPath, const char* _baseName, OutputFormatType _oft, AlignmentResultFlags _arf, INT64 _maxA ) : maxA( _maxA ), oft( _oft ), arf( _arf )
{
    // save a copy of the base name (to be used to construct output filenames)
    if( _baseName == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "output file base name may not be null" );
    strcpy_s( this->baseName, sizeof this->baseName, _baseName );

    // zero the file path buffers
    memset( this->path, 0, sizeof this->path );
    memset( this->rootPath, 0, sizeof this->rootPath );
    memset( this->relPath, 0, sizeof this->relPath );

    // set a flag to indicate whether a base path was specified
    bool basePathSpecified = (_basePath != NULL) && (*_basePath != 0);
    if( basePathSpecified )
        strcpy_s( this->rootPath, sizeof this->rootPath, _basePath );

    // if there is no path specific to the output format type...
    if( (_oftPath == NULL) || (_oftPath[0] == 0) )
    {
        // if there is a base path, use it
        if( basePathSpecified )
            strcpy_s( this->path, sizeof this->path, _basePath );
    }

    else   // (there is a path specific to the output format type)
    {
        strcpy_s( this->relPath, sizeof this->relPath, _oftPath );

        // use the OFT-specific if it is an absolute path or if no base path is specified
        if( RaiiDirectory::IsAbsolutePath( const_cast<char*>(_oftPath) ) || !basePathSpecified )
            strcpy_s( this->path, sizeof this->path, _oftPath );
        else
        {
            // at this point we need to concatenate the base path and the file path
            strcpy_s( this->path, sizeof this->path, _basePath );
            RaiiDirectory::ChopTrailingPathSeparator( this->path );
            RaiiDirectory::AppendTrailingPathSeparator( this->path );
            strcat_s( this->path, sizeof this->path, _oftPath );
        }
    }

    // special handling for bitbucket output
    if( (0 == _strcmpi( this->path, "nul" )) ||
        (0 == _strcmpi( this->path, "nul:" )) ||
        (0 == _strcmpi( this->path, "/device/null" ))
        )
    {
        this->path[0] = 0;
    }
}

/// copy constructor
OutputFileInfo::OutputFileInfo( const OutputFileInfo& other ) : maxA( other.maxA ), oft( other.oft ), arf( other.arf )
{
    strcpy_s( this->path, sizeof this->path, other.path );
    strcpy_s( this->rootPath, sizeof this->rootPath, other.rootPath );
    strcpy_s( this->relPath, sizeof this->relPath, other.relPath );
    strcpy_s( this->baseName, sizeof this->baseName, other.baseName );
}

/// destructor
OutputFileInfo::~OutputFileInfo()
{
}
#pragma endregion
