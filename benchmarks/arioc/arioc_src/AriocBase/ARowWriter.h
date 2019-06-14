/*
  ARowWriter.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __ARowWriter__

#ifndef __baseARowWriter__
#include "baseARowWriter.h"
#endif

// forward declaration
class AriocBase;

/// <summary>
/// Class <c>ARowWriter</c> writes alignment data to a file.
/// </summary>
class ARowWriter : public baseARowWriter
{
    static const INT64 OUTPUT_BUFFER_SIZE = 10 * 1024*1024;     // 10MB
    static const DWORD BUFFER_LOCK_TIMEOUT = 5 * 60 * 1000;     // 5 minutes

    private:
        void internalClose( void );
        void internalFlush( INT16 iBuf );

    public:
        ARowWriter( AriocBase* pabase, OutputFileInfo* pofi );
        virtual ~ARowWriter( void );
        virtual char* Lock( INT16 iBuf, UINT32 cb );
        virtual void Release( INT16 iBuf, UINT32 cb );
        virtual void Flush( INT16 iBuf );
        virtual void Close( void );
};
