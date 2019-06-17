/*
  ApplicationException.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __ApplicationException__

struct ApplicationExceptionInfo
{
    static const UINT32 BUFFER_SIZE = 10 * 1024;

    char*   pDetails;
    char    File[FILENAME_MAX];
    INT32   Line;
    UINT32  ThreadId;

    ApplicationExceptionInfo() : pDetails(NULL), Line(0), ThreadId(0)
    {
        memset( this->File, 0, FILENAME_MAX );
    }

    ApplicationExceptionInfo( const ApplicationExceptionInfo& other ) : pDetails(NULL), Line(other.Line), ThreadId(other.ThreadId)
    {
        if( other.pDetails )
        {
            this->pDetails = reinterpret_cast<char*>(malloc( BUFFER_SIZE ));
            memcpy_s( this->pDetails, BUFFER_SIZE, other.pDetails, BUFFER_SIZE );
        }

        memcpy_s( this->File, FILENAME_MAX, other.File, FILENAME_MAX );
    }

    virtual ~ApplicationExceptionInfo()
    {
        if( this->pDetails )
            free( this->pDetails );
    }
};

/// <summary>
/// Class <c>ApplicationException</c> wraps basic exception information.
/// </summary>
class ApplicationException
{

    public:
        ApplicationExceptionInfo    ExceptionInfo;
        ApplicationExceptionInfo    CallerExceptionInfo;

    private:
        ApplicationException( void );

    public:
        ApplicationException( const char* _file, const INT32 _line, const char* _fmt, ... );
        ApplicationException( const ApplicationException& other );
        virtual ~ApplicationException( void );
        void SetCallerExceptionInfo( const char* _file, const INT32 _line, const char* _fmt, ... );
        int Dump( void );
};
