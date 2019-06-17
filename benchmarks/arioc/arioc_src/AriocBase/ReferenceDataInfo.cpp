/*
  ReferenceDataInfo.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region class ReferenceFileInfo
/// [public] constructor
ReferenceFileInfo::ReferenceFileInfo( ReferenceFileType _fileType, const char* _path, A21SeedBase& _a21sb ) : FileType(_fileType), MaxSize(0), TotalSize(0)
{
    // special-case handling for a null seed index
    if( _a21sb.IsNull() )
        return;

    // build the subdirectory path specification for lookup tables
    char pathLUT[_MAX_PATH];
    sprintf_s( pathLUT, _MAX_PATH, "%s%s%s", _path, FILEPATHSEPARATOR, _a21sb.IdString );

    switch( _fileType )
    {
        case rftR:
            Dir.GetFilenames( _path, "*a21.*sbf" );
            break;

        case rftH:
            Dir.GetFilenames( pathLUT, "H???.sbf" );
            break;

        case rftJ:
            Dir.GetFilenames( pathLUT, "J???.sbf" );
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected reference file type: %d", _fileType );
    }

    // sanity check
    if( Dir.Filenames.Count == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "missing or empty directory: %s", Dir.DirSpec );

    // traverse the list of files in the specified directory and accumulate the sum of the file sizes
    WinGlobalPtr<char> fileSpec;

    for( INT32 n=0; n<static_cast<INT32>(Dir.Filenames.Count); ++n )
    {
        Dir.GetFileSpecification( n, fileSpec );

        // limit the scope of the RaiiFile object
        {
            RaiiFile rf( fileSpec.p, true );    // true: read only

            // get the file size in bytes
            INT64 cb = rf.FileSize();

            // track the total and maximum file size
            this->TotalSize += cb;
            this->MaxSize = max2( this->MaxSize, cb );

            CDPrint( cdpCDb, "ReferenceFileInfo::ctor: fileSpec=%s cb=%lld TotalSize=%lld MaxSize=%lld", fileSpec.p, cb, this->TotalSize, this->MaxSize );
        }
    }
}

/// [public] destructor
ReferenceFileInfo::~ReferenceFileInfo()
{
}
#pragma endregion
        
#pragma region class ReferenceDataInfo
/// [public] constructor
ReferenceDataInfo::ReferenceDataInfo( const char* _path, A21SeedBase& _a21sb ) : rfiR(rftR,_path,_a21sb),
                                                                                 rfiH(rftH,_path,_a21sb),
                                                                                 rfiJ(rftJ,_path,_a21sb),
                                                                                 HasData(!_a21sb.IsNull())
{
}

/// [public] destructor
ReferenceDataInfo::~ReferenceDataInfo()
{
}
#pragma endregion
