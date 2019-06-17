/*
  AppGlobal.cpp

    Copyright (c) 2018-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
     in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
     The contents of this file, in whole or in part, may only be copied, modified, propagated, or
     redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

/// <summary>
/// This is the main program entry point.
/// </summary>
int main( int argc, char* argv[] )
{
    AppMain appMain;
    AppGlobal appGlobal( &appMain );
    return appGlobal.Run( argc, argv, "XMC" );
}

/// <summary>
/// Class <c>AppGlobal</c> implements application startup functionality
/// </summary>
#pragma region constructor/destructor
/// default constructor
AppGlobal::AppGlobal()
{
}

/// constructor
AppGlobal::AppGlobal( AppMain* _pam ) : AppGlobalCommon( _pam )
{
}

/// destructor
AppGlobal::~AppGlobal()
{
}
#pragma endregion

#pragma region virtual method implementations
/// [protected virtual] method parseCommandTail
void AppGlobal::parseCommandTail( int argc, char* argv[] )
{
    /* Parse the command line:

        xmc [optional_args] SAM_filename

       Optional arguments are the same as the correspondingly-named arguments in samtools view:

        -f flagbits     include only records where (FLAG & flagbits) = flagbits
        -F flagbits     include only records where (FLAG & flagbits) = 0
        -G flagbits     include only records where (FLAG & flagbits) != flagbits
        -q mapq         include only records where MAPQ >= mapq
        -p nThreads     maximum number of concurrent threads
        -d              report MAPQ distribution

       Optional argument names are preceded by a slash or dash and separated from their corresponding values
        by whitespace.  Examples:
        
            xmc LIBD322.sam
            xmc -F 16 -q 4 LIBD333.sam
    */

    // point to the AppMain specialization
    AppMain* pam = reinterpret_cast<AppMain*>(m_pamc);

    // initialize the maximum number of concurrent CPU threads
    pam->ConcurrentThreadCount = GetAvailableCpuThreadCount();

    char lastArgname = '\0';
    for( int ia=1; ia<argc; ++ia )
    {
        // sanity check
        if( strlen(argv[ia]) < 2 )
            ApplicationException( __FILE__, __LINE__, "%s: invalid argument '%s' in position %d of command tail", __FUNCTION__, argv[ia], ia );

        // get a copy of the first character of the ia'th argument
        char c0 = argv[ia][0];

        // look for the .sam filename
        if( lastArgname == 0 )
        {
            /* At this point we expect either the SAM filename or an argument name (preceded by a dash or slash) */
#ifdef _WIN32
            if( (c0 == '/') || (c0 == '-') )
#endif
#ifdef __GNUC__
            if( c0 == '-' )
#endif
            {
                // look for a boolean argument (i.e., one with no value)
                if( argv[ia][1] == 'd' )
                {
                    pam->Report_d = true;
                    continue;
                }

                if( argv[ia][1] == 'h' )
                {
                    pam->Report_h = true;
                    continue;
                }

                /* At this point we have the argument name and we're expecting the value to follow. */

                // save the second character in the argument
                lastArgname = argv[ia][1];

                // sanity check
                if( !strchr( "fFGqp", lastArgname ) )
                    throw new ApplicationException( __FILE__, __LINE__, "%s: invalid argument name '%s' in position %d of command tail", __FUNCTION__, argv[ia], ia );

                continue;
            }

            /* At this point it ought to be the SAM filename. */

            // sanity check
            if( pam->SamFilename[0] )
                throw new ApplicationException( __FILE__, __LINE__, "%s: unexpected argument '%s' in position %d of command tail", __FUNCTION__, argv[ia], ia );

            // save a copy of the specified filename
            strcpy_s( pam->SamFilename, sizeof pam->SamFilename, argv[ia] );
            continue;
        }    
            
        /* At this point it's the value corresponding to the last argument name. */

        // sanity check
        if( (c0 == '/') || (c0 == '-') )
            throw new ApplicationException( __FILE__, __LINE__, "%s: missing value for argument '%c' in position %d of command tail", __FUNCTION__, lastArgname, ia-1 );
        
        // convert the argument value to a signed 32-bit number
        const char* p = argv[ia];
        int radix = 10;
        if( (p[0] == '0') && ((p[1] == 'x') || (p[1] == 'X')) )
        {
            p += 2;     // skip over the "0x"
            radix = 16;
        }

        // get the argument value
        INT32 val = static_cast<INT32>(strtol( p, NULL, radix ));

        bool isValidValue = (val > 0);
        if( isValidValue )
        {
            switch( lastArgname )
            {
                case 'q':
                    isValidValue = (val <= 255);
                    break;

                case 'f':
                case 'F':
                case 'G':
                    isValidValue = (val < 0x1000);
                    break;

                case 'p':
                    pam->ConcurrentThreadCount = min2( pam->ConcurrentThreadCount, val );
                    break;

                default:
                    break;
            }
        }
         
        if( !isValidValue )
            throw new ApplicationException( __FILE__, __LINE__, "%s: invalid argument value %d for argument '%c' in position %d of command tail", __FUNCTION__, val, lastArgname, ia );

        // save the argument value
        switch( lastArgname )
        {
            case 'f':
                pam->Filter_f = val;
                break;

            case 'F':
                pam->Filter_F = val;
                break;

            case 'G':
                pam->Filter_G = val;
                break;

            case 'q':
                pam->Filter_q = val;
                break;

            default:
                break;
        }

        lastArgname = 0;
    }

    // sanity check
    if( pam->SamFilename[0] == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "missing SAM file name" );
}

/// [public virtual] method Run
int AppGlobal::Run( int argc, char* argv[], const char* defaultAppName )
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
        CDPrint( cdpCD0, " data type sizes      : int=%u long=%u *=%u", static_cast<UINT32>(sizeof(int)), static_cast<UINT32>(sizeof(long)), static_cast<UINT32>(sizeof(void*)) ); 
        CDPrint( cdpCD0, " executable file      : %s", m_szExeFilespec );

        // parse command-line arguments
        parseCommandTail( argc, argv );

        // launch the application
        m_pamc->Launch();
    }
    catch( ApplicationException* pex )
    {
        rval = pex->Dump();
    }

    CDPrint( cdpCD0, "%s ends (%d)", m_szAppName, rval );
    return rval;
}
#pragma endregion
