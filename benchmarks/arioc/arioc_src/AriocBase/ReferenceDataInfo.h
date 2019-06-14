/*
  ReferenceDataInfo.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __ReferenceDataInfo__

#pragma region enums
enum ReferenceFileType
{
    rftUnknown =    0,
    rftR =          1,
    rftH =          2,
    rftJ =          4,
    rftHJ =         (rftH|rftJ)
};
#pragma endregion

/// <summary>
/// Class <c>ReferenceFileInfo</c> contains reference file info.
/// </summary>
class ReferenceFileInfo
{
    public:
        ReferenceFileType   FileType;
        RaiiDirectory       Dir;
        INT64               MaxSize;            // maximum file size (buffer size) in bytes
        INT64               TotalSize;          // total of file sizes in bytes

    public:
        ReferenceFileInfo( ReferenceFileType _fileType, const char* _path, A21SeedBase& _a21sb );
        ~ReferenceFileInfo( void );
};

/// <summary>
/// Class <c>ReferenceDataInfo</c> contains reference data info for the R, H, and J tables.
/// </summary>
class ReferenceDataInfo
{
    public:
        ReferenceFileInfo   rfiR;   // reference file info for R sequence data files
        ReferenceFileInfo   rfiH;   // reference file info for H table files
        ReferenceFileInfo   rfiJ;   // reference file info for J table files
        bool                HasData;

    public:
        ReferenceDataInfo( const char* _dir, A21SeedBase& _a21sb );
        ~ReferenceDataInfo( void );
};
