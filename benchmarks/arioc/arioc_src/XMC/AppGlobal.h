/*
  AppGlobal.h

    Copyright (c) 2018-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
     in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
     The contents of this file, in whole or in part, may only be copied, modified, propagated, or
     redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AppGlobal__


// forward definition
class AppMain;


/// <summary>
/// Class <c>AppGlobal</c> implements application startup functionality
/// </summary>
class AppGlobal : public AppGlobalCommon
{
    protected:
        virtual void parseCommandTail( int argc, char* argv[] );

    private:
        AppGlobal( void );

    public:
        AppGlobal( AppMain* _pam );
        virtual ~AppGlobal( void );
        virtual int Run( int argc, char* argv[], const char* defaultAppName );
};
