/*
  AppGlobalCommon.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AppGlobalCommon__


/// <summary>
/// Global startup for Arioc applications
/// </summary>
class AppGlobalCommon
{
    protected:
        static char     m_fmtDefaultCfgFile[];
        AppMainCommon*  m_pamc;
        char            m_szExeFilespec[_MAX_PATH];
        char            m_szAppName[APPNAME_BUFSIZE];
        char            m_szAppVersion[APPVER_BUFSIZE];
        char            m_szAppCfg[FILENAME_MAX];
        char            m_szAppLegal[APPLEGAL_BUFSIZE];

    public:
        static bool WantEnterAtExit;

    protected:
        AppGlobalCommon( void );
        void getStartupBannerDetails( const char* _argv0 );
        virtual void parseCommandTail( int argc, char* argv[] );

    public:
        AppGlobalCommon( AppMainCommon* pamc );
        virtual ~AppGlobalCommon( void );
        virtual int Run( int argc, char* argv[], const char* defaultAppName );
};
