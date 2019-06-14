/*
  ApplicationException.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"


#pragma region constructors and destructor
/// [private] default constructor
ApplicationException::ApplicationException()
{
}

/// [public] copy constructor
ApplicationException::ApplicationException( const ApplicationException& other )
{
    this->ExceptionInfo = other.ExceptionInfo;
    this->CallerExceptionInfo = other.CallerExceptionInfo;
}

/// [public] constructor
ApplicationException::ApplicationException( const char* _file, const INT32 _line, const char* _fmt, ... )
{
    // save the exception info in this ApplicationException object
    if( !this->ExceptionInfo.pDetails )
        this->ExceptionInfo.pDetails = reinterpret_cast<char*>(malloc( ApplicationExceptionInfo::BUFFER_SIZE ));
    memset( this->ExceptionInfo.pDetails, 0, ApplicationExceptionInfo::BUFFER_SIZE );

    va_list args;
    va_start(args, _fmt);

    try
    {
        // format the string
        INT32 cch = vsprintf_s( this->ExceptionInfo.pDetails, ApplicationExceptionInfo::BUFFER_SIZE, _fmt, args );

        // if necessary, append a CRLF
        cch-- ; // point to the last character in the string
        if( (this->ExceptionInfo.pDetails[cch] != '\r') && (this->ExceptionInfo.pDetails[cch] != '\n') )
            strcat_s( this->ExceptionInfo.pDetails, ApplicationExceptionInfo::BUFFER_SIZE, "\r\n" );
    }
    catch( ... )
    {
        if( this->ExceptionInfo.pDetails )
        {
            static const char* errMsg = "(error in ApplicationException constructor)";
            strcpy_s( this->ExceptionInfo.pDetails, strlen(errMsg)+1, errMsg );
        }
    }

    va_end(args);

    // save the file info in this ApplicationException object
    strcpy_s( this->ExceptionInfo.File, sizeof this->ExceptionInfo.File, _file );
    this->ExceptionInfo.Line = _line;
    this->ExceptionInfo.ThreadId = GetCurrentThreadId();
}

/// [public] destructor
ApplicationException::~ApplicationException()
{
}
#pragma endregion

#pragma region static methods
/// [static] method Dump
int ApplicationException::Dump()
{
    if( this->ExceptionInfo.pDetails && this->ExceptionInfo.pDetails[0] )
    {
        // trim the "details" string
        for( INT16 n=static_cast<INT16>(strlen(this->ExceptionInfo.pDetails)-1); (n>=0)&&isspace(this->ExceptionInfo.pDetails[n]); --n )
            this->ExceptionInfo.pDetails[n] = 0;

        CDPrint( cdpCD0, "ApplicationException ([0x%08u] %s %d): %s", this->ExceptionInfo.ThreadId, this->ExceptionInfo.File, this->ExceptionInfo.Line, this->ExceptionInfo.pDetails );

        if( this->CallerExceptionInfo.pDetails && this->CallerExceptionInfo.pDetails[0] )
        {
            // trim the "details" string
            for( INT16 n=static_cast<INT16>(strlen(this->CallerExceptionInfo.pDetails)-1); (n>=0)&&isspace(this->CallerExceptionInfo.pDetails[n]); --n )
                this->CallerExceptionInfo.pDetails[n] = 0;

            CDPrint( cdpCD0, " Caller ([0x%08u] %s %d): %s", this->CallerExceptionInfo.ThreadId, this->CallerExceptionInfo.File, this->CallerExceptionInfo.Line, this->CallerExceptionInfo.pDetails );
        }
    }

    // return a "failure" exit code
    return EXIT_FAILURE;
}
#pragma endregion

#pragma region public methods
/// [public] method SetCallerExceptionInfo
void ApplicationException::SetCallerExceptionInfo( const char* _file, const INT32 _line, const char* _fmt, ... )
{
    // do not blot existing caller exception info
    if( this->CallerExceptionInfo.pDetails && this->CallerExceptionInfo.pDetails[0] )
        return;

    // allocate and fill a buffer with a "details" string
    if( !this->CallerExceptionInfo.pDetails )
        this->CallerExceptionInfo.pDetails = reinterpret_cast<char*>(malloc( ApplicationExceptionInfo::BUFFER_SIZE ));

    va_list args;
    va_start(args, _fmt);

    try
    {
        // format the string
        INT32 cch = vsprintf_s( this->CallerExceptionInfo.pDetails, ApplicationExceptionInfo::BUFFER_SIZE, _fmt, args );

        // if necessary, append a CRLF
        cch-- ;
        if( (this->CallerExceptionInfo.pDetails[cch] != '\r') && (this->CallerExceptionInfo.pDetails[cch] != '\n') )
            strcat_s( this->CallerExceptionInfo.pDetails, ApplicationExceptionInfo::BUFFER_SIZE, "\r\n" );
    }
    catch( ... )
    {
        if( this->CallerExceptionInfo.pDetails )
        {
            static const char* errMsg = "(error in ApplicationException::SetCallerExceptionInfo)";
            strcpy_s( this->CallerExceptionInfo.pDetails, strlen(errMsg)+1, errMsg );
        }
    }

    va_end(args);

    // copy the specified filename and line number
    strcpy_s( this->CallerExceptionInfo.File, sizeof this->CallerExceptionInfo.File, _file );
    this->CallerExceptionInfo.Line = _line;
    this->CallerExceptionInfo.ThreadId = GetCurrentThreadId();
}
#pragma endregion
