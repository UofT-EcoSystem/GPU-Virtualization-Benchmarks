/*
  AppMainCommon.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static variable definitions
const double AppMainCommon::MS_PER_SEC = 1000;
const double AppMainCommon::BYTES_PER_MB = 1024*1024;
const double AppMainCommon::BYTES_PER_GB = 1024*1024*1024;
const char* AppMainCommon::m_xaNull[] = { NULL };
#pragma endregion

#pragma region constructor/destructor
/// [protected] default constructor
AppMainCommon::AppMainCommon() : m_pelX(NULL), m_maxBatchSize(AriocDS::QID::limitQID)
{
    memset( this->AppName, 0, sizeof this->AppName );
    memset( this->AppVersion, 0, sizeof this->AppVersion );
    memset( this->ConfigFileName, 0, sizeof this->ConfigFileName );
    memset( this->MachineName, 0, sizeof this->MachineName );
    memset( this->DataSourceInfo, 0, sizeof this->DataSourceInfo );
}

/// destructor
AppMainCommon::~AppMainCommon()
{
}
#pragma endregion

#pragma region protected methods
/**
* Parse a string that represents a 64-bit integer.
*
* The string may have an optional one-character suffix:
*   - K: kilo
*   - M: mega
*   - G: giga
*/
INT64 AppMainCommon::parseInt64GMK( const char* pigmk, const INT64 defaultValue )
{
    // special handling for missing attribute value
    if( pigmk == NULL )
        return defaultValue;

    // special handling for "*" (maximum integer value)
    if( *pigmk == '*' )
        return _I64_MAX;

    INT64 rval = 0;
    INT64 scale = 1;
    char igmk[32];
    strcpy_s( igmk, sizeof igmk, pigmk );

    // special handling for hexadecimal values
    if( tolower(igmk[1]) == 'x' )
    {
        sscanf_s( igmk, "%llx", &rval );
        return rval;
    }

    // if the last character in the specified string is "G", "M", or "K", multiply by 1024**3, 1024**2, or 1024 respectively
    INT32 iLast = static_cast<INT32>(strlen(pigmk)) - 1;
    switch( igmk[iLast] )
    {
        case 'K':
        case 'k':
            scale = 1024;           // kilo
            igmk[iLast] = 0;
            break;

        case 'M':
        case 'm':
            scale = 1024*1024;      // mega
            igmk[iLast] = 0;
            break;

        case 'G':
        case 'g':
            scale = 1024*1024*1024; // giga
            igmk[iLast] = 0;
            break;

        default:
            break;
    }

    sscanf_s( igmk, "%lld", &rval );
    return rval*scale;
}

/**
* Parse a string that represents a 32-bit integer.
*
* The string may have an optional one-character suffix:
*   - K: kilo
*   - M: mega
*   - G: giga
*/
INT32 AppMainCommon::parseInt32GMK( const char* pigmk, const INT32 defaultValue )
{
    // special handling for missing attribute value
    if( pigmk == NULL )
        return defaultValue;

    // special handling for "*" (maximum integer value)
    if( *pigmk == '*' )
        return _I32_MAX;

    return static_cast<INT32>(parseInt64GMK( pigmk, defaultValue ) );
}


/// [protected] method parseRoot
void AppMainCommon::parseRoot( UINT32& gpuMask, INT32& maxDOP, INT32& batchSize )
{
    using namespace tinyxml2;

    // default values
    gpuMask = 0x00000001;
    maxDOP = _I32_MAX;
    batchSize = m_maxBatchSize;
    CDPrintFilter = static_cast<CDPrintFlags>(0xE0000003);

    // parse the attributes
    const XMLAttribute* pa = m_pelRoot->FirstAttribute();
    while( pa )
    {
        bool pending = true;

        if( XMLUtil::StringEqual( pa->Name(), "gpuMask" ) )
        {
            const char* pGpuMask = pa->Value();
            if( pGpuMask )
                sscanf_s( pGpuMask, "%*2s%08X", &gpuMask );                         // (the gpuMask value is formatted as an 8-character hex string, preceded by "0x")
            if( gpuMask == 0 )
                throw new ApplicationException( __FILE__, __LINE__, "the gpuMask attribute value must be nonzero" );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "maxDOP" ) )
        {
            maxDOP = parseInt32GMK( pa->Value(), _I32_MAX );
            if( maxDOP == 0 )
                throw new ApplicationException( __FILE__, __LINE__, "the maxDOP attribute value must be nonzero" );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "batchSize" ) )
        {
            const char* pBatchSize = pa->Value();
            if( pBatchSize )
            {
                batchSize = parseInt32GMK( pBatchSize, m_maxBatchSize );    // default value = (see constructor)
                if( batchSize > m_maxBatchSize )
                    throw new ApplicationException( __FILE__, __LINE__, "maximum supported value for the \"batchSize\" attribute in element <%s> is %d", this->AppName, m_maxBatchSize );
                pending = false;
            }
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "verboseMask" ) )
        {
            const char* pCDPrintFilter = pa->Value();
            if( pCDPrintFilter )
            {
                sscanf_s( pCDPrintFilter, "%*2s%08X", reinterpret_cast<UINT32*>(&CDPrintFilter) );  // (the verboseMask value is formatted as an 8-character hex string, preceded by "0x")
                pending = false;
            }
        }

        // at this point we have either an invalid attribute name or a missing value for a valid attribute name
        if( pending )
            throw new ApplicationException( __FILE__, __LINE__, "invalid attribute \"%s\" for element <%s>", pa->Name(), this->AppName );

        // iterate through the attribute list
        pa = pa->Next();
    }
}

/// [protected] method parseX
void AppMainCommon::parseX()
{
    using namespace tinyxml2;

    // do nothing if there is no <X> element in the config file
    if( m_pelX == NULL )
        return;

    // count the number of attributes in the <X> element
    INT32 nXp = 0;
    INT32 cch = 0;
    const XMLAttribute* patXp = m_pelX->FirstAttribute();
    while( patXp )
    {
        ++nXp;
        cch += static_cast<INT32>(strlen( patXp->Name()) );

        patXp = patXp->Next();
    }

    // build a simple associative array that contains the parameters
    patXp = m_pelX->FirstAttribute();
    for( INT32 n=0; n<nXp; ++n )
    {
        Xparam[patXp->Name()] = parseInt64GMK( patXp->Value(), 0 );
        patXp = patXp->Next();
    }

    // set a flag (defined in AppGlobal.cpp)
    INT32 i = Xparam.IndexOf( "enterAtExit" );
    if( i >= 0 )
        AppGlobalCommon::WantEnterAtExit = (Xparam.Value(i) != 0);
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Initializes the main application instance.
/// </summary>
/// <param name="pAppName">application name</param>
/// <param name="pAppVersion">application version number</param>
/// <param name="pCfgFileName">fully-qualified configuration file name</param>
void AppMainCommon::Init( const char* pAppName, const char* pAppVersion, const char* configFileName )
{
    strcpy_s( this->AppName, sizeof this->AppName, pAppName );
    strcpy_s( this->AppVersion, sizeof this->AppVersion, pAppVersion );
    strcpy_s( this->ConfigFileName, sizeof this->ConfigFileName, configFileName );
}
#pragma endregion
