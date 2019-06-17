/*
  AppGlobalCommon.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#ifdef __GNUC__
#ifndef __AriocVersion_Includes__
#include "../AriocVersion/AriocVersion.h"
#endif
#endif


#pragma region static variable initialization
bool AppGlobalCommon::WantEnterAtExit = false;
char AppGlobalCommon::m_fmtDefaultCfgFile[] =
#ifdef _WIN32
                                  "%s.cfg";
#endif
#ifdef __GNUC__
                                  "./%s.cfg";
#endif
#pragma endregion

#pragma region constructor/destructor
/// [protected] default constructor
AppGlobalCommon::AppGlobalCommon() : m_pamc(NULL)
{
    // do not use this constructor
    throw new ApplicationException( __FILE__, __LINE__, "(not implemented)" );
}

/// constructor (AppMainCommon*)
AppGlobalCommon::AppGlobalCommon( AppMainCommon* pamc ) : m_pamc(pamc)
{
    memset( m_szExeFilespec, 0, FILENAME_MAX );
    memset( m_szAppName, 0, APPNAME_BUFSIZE );
    memset( m_szAppVersion, 0, APPVER_BUFSIZE );
    memset( m_szAppCfg, 0, FILENAME_MAX );
    memset( m_szAppLegal, 0, APPLEGAL_BUFSIZE );
}
#pragma endregion

/// destructor
AppGlobalCommon::~AppGlobalCommon()
{
}
#pragma endregion

#pragma region protected methods
/// [protected] method getStartupBannerDetails
void AppGlobalCommon::getStartupBannerDetails( const char* _argv0 )
{
    /* The executable file specification, application name, and application version are handled a bit differently
        in Windows and in Linux:
        - in Windows, the buffers get updated here with strings extracted from the "fixed file info"
            in the executable file
        - in Linux, the app version is copied from an include file that gets updated in the Windows Visual Studio
            project (i.e., for distribution builds, the Windows version should be compiled before the Linux
            version)
    */

#if defined(_WIN32)
    // extract the file version info
    if( 0 == GetModuleFileName( NULL, m_szExeFilespec, FILENAME_MAX ) )
    {
        DWORD dwErr = GetLastError();
        throw new ApplicationException( __FILE__, __LINE__, "GetModuleFileName returned 0 (GetLastError returned %u)", dwErr );
    }

    DWORD dwHandle;
    UINT32 cb = GetFileVersionInfoSize( m_szExeFilespec, &dwHandle );
    WinGlobalPtr<INT8> vi( cb, true );
    GetFileVersionInfo( m_szExeFilespec, dwHandle, cb, vi.p );
    if( cb == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "missing file version info" );

    // the application name is the "internal name" in the Windows version info
    char* p;
    VerQueryValue( vi.p, "\\StringFileInfo\\040904E4\\InternalName", reinterpret_cast<void**>(&p), &cb );
    strcpy_s( m_szAppName, APPNAME_BUFSIZE, p );

    // the major and minor version numbers are in the "product version" string
    VerQueryValue( vi.p, "\\StringFileInfo\\040904E4\\FileVersion", reinterpret_cast<void**>(&p), &cb );
    strcpy_s( m_szAppVersion, APPVER_BUFSIZE, p );

    // the copyright info is in the "legal copyright" string
    VerQueryValue( vi.p, "\\StringFileInfo\\040904E4\\LegalCopyright", reinterpret_cast<void**>(&p), &cb );
    strcpy_s( m_szAppLegal, APPLEGAL_BUFSIZE, p );
#endif

#ifdef __GNUC__
    strcpy_s( m_szAppVersion, sizeof m_szAppVersion, VER_FILEVERSION_STR );
    strcpy_s( m_szAppLegal, sizeof m_szAppLegal, VER_LEGALCOPYRIGHT_STR );
    realpath( _argv0, m_szExeFilespec );
#endif
}

/// [protected virtual] method parseCommandTail
void AppGlobalCommon::parseCommandTail( int argc, char* argv[] )
{
    /* Parse the command line:    
        argv[0]: executable filename
        argv[1]: configuration filename
    */

    switch( argc )
    {
        case 1:
            sprintf_s( m_szAppCfg, sizeof m_szAppCfg, m_fmtDefaultCfgFile, m_szAppName );
            break;

        case 2:
            strcpy_s( m_szAppCfg, sizeof m_szAppCfg, argv[1] );
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "invalid command-line arguments (syntax: %s [configuration_filename])", m_szAppName );
    }
}
#pragma endregion

#pragma region public methods
/// [public] method Run
int AppGlobalCommon::Run( int argc, char* argv[], const char* defaultAppName )
{
    int rval = 0;

    getStartupBannerDetails( argv[0] );
    if( m_szAppName[0] == 0 )
        strcpy_s( m_szAppName, sizeof m_szAppName, defaultAppName );

    CDPrint( cdpCD0, "%s v%s%s"
                , m_szAppName
                , m_szAppVersion
#ifdef _DEBUG
                , " (debug)"
#else
                , " (release)"
#endif
      
           );

    try
    {
        CDPrint( cdpCD0, m_szAppLegal );
        CDPrint( cdpCD0, " data type sizes      : int=%llu long=%llu *=%llu Jvalue5=%llu Jvalue8=%llu JtableHeader=%llu", sizeof(int), sizeof(long), sizeof(void*), sizeof(Jvalue5), sizeof(Jvalue8), sizeof(JtableHeader) ); 
        CDPrint( cdpCD0, " executable file      : %s", m_szExeFilespec );

        // parse command-line arguments
        parseCommandTail( argc, argv );

        // verify the existence of the configuration file
        char cfgFileSpec[FILENAME_MAX];
        _fullpath( cfgFileSpec, m_szAppCfg, sizeof cfgFileSpec );

        if( !RaiiFile::Exists( cfgFileSpec ) )
            throw new ApplicationException( __FILE__, __LINE__, "cannot open configuration file '%s'", cfgFileSpec );

        // launch the application
        m_pamc->Init( m_szAppName, m_szAppVersion, cfgFileSpec );
        m_pamc->LoadConfig();   // load the XML-formatted config file
        m_pamc->Launch();       // parse the config file and launch the program
    }
    catch( ApplicationException* pex )
    {
        rval = pex->Dump();
    }

    // conditionally wait for the user to press the Enter key
    CDPrintFilter = cdpCD0;
    if( AppGlobalCommon::WantEnterAtExit )
    {
        CDPrint( static_cast<CDPrintFlags>(cdpConsole|0x00000001), "Done.  Press Enter to halt..." );
        getchar();
    }
   
    CDPrint( cdpCD0, "%s ends (%d)", m_szAppName, rval );
    return rval;
}
#pragma endregion
