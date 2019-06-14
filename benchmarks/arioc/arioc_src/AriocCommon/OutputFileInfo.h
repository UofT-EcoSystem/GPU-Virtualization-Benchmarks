/*
  OutputFileInfo.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __OutputFileInfo__

struct OutputFileInfo
{
    public:
        char                    path[FILENAME_MAX];     // output file path
        char                    rootPath[FILENAME_MAX]; // base output file path
        char                    relPath[FILENAME_MAX];  // output file path (relative to rootPath)
        char                    baseName[FILENAME_MAX]; // base output filename
        INT64                   maxA;                   // maximum number of alignments to write to a single file
        OutputFormatType        oft;                    // output file format
        AlignmentResultFlags    arf;                    // alignment result flags for file contents

    public:
        OutputFileInfo( const char* _basePath, const char* _oftPath, const char* _baseName, OutputFormatType _oft, AlignmentResultFlags _arf, INT64 _maxA );
        OutputFileInfo( const OutputFileInfo& other );
        virtual ~OutputFileInfo();
};
