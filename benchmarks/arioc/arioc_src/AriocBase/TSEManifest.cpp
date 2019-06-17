/*
  TSEManifest.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static variable definitions
#pragma endregion

#pragma region constructors and destructor
/// [private] constructor
TSEManifest::TSEManifest()
{
    /* (do not use this constructor) */
}

/// [public] constructor
TSEManifest::TSEManifest( AriocAppMainBase* paamb, INT32 refId, InputFileGroup* pifgQ ) : m_paamb(paamb), m_buf(INITIAL_BUFFER_SIZE,true)
{
    // look for the smallest data source ID and subunit ID associated with an input file
    INT32 srcId = _I32_MAX;
    UINT8 subId = _UI8_MAX;
    for( UINT32 n=0; n<pifgQ->InputFile.n; n++ )
    {
        if( pifgQ->InputFile.p[n].SrcId < srcId )
        {
            srcId = pifgQ->InputFile.p[n].SrcId;
            subId = pifgQ->InputFile.p[n].SubId;
        }

        else
        {
            if( pifgQ->InputFile.p[n].SrcId == srcId )
                subId = min2( subId, pifgQ->InputFile.p[n].SubId );
        }
    }

    initialize( refId, srcId, subId );
}

/// [public] destructor
TSEManifest::~TSEManifest()
{
}
#pragma endregion

#pragma region private methods
/// [private] method appendToBuffer
void TSEManifest::appendToBuffer( const char* s )
{
    // conditionally expand the buffer
    size_t cb = strlen( s ) + 3;                // CRLF plus a trailing null byte
    if( (m_buf.n + cb) > m_buf.cb )
        m_buf.Realloc( m_buf.cb + 2*cb, true );

    // append the specified string
    strcpy_s( m_buf.p+m_buf.n, cb, s );
    m_buf.n += static_cast<UINT32>(cb - 3);

    // append a Windows-style newline
    strcpy_s( m_buf.p+m_buf.n, 3, "\r\n" );
    m_buf.n += 2;
}

/// [private] method initialize
void TSEManifest::initialize( const INT32 refId, const INT32 srcId, const INT8 subId )
{
    char s[FILENAME_MAX+64];

    // include the subunit ID in the manifest filename only if it is nonzero
    const char* fmt = subId ? "TSEDS.%05d.%03d.manifest" : "TSEDS.%05d.manifest";
    sprintf_s( m_manifestFileName, sizeof m_manifestFileName, fmt, srcId, subId );

    // banner comments
    appendToBuffer( "'" );
    sprintf_s( s, sizeof s, "' %s", m_manifestFileName );
    appendToBuffer( s );
    appendToBuffer( "'" );
    appendToBuffer( "" );

    // refId (for the R sequence)
    AppendI32( "refId", refId );

    // data source and subunit IDs (for Q sequences)
    AppendI32( "srcId", srcId );
    AppendI8( "subId", subId );

    // app name and version
    appendToBuffer( "" );
    sprintf_s( s, sizeof s, "%s v%s", m_paamb->AppName, m_paamb->AppVersion );
    AppendString( "raInfo", s );
    AppendString( "raMachine", m_paamb->MachineName );
}

/// [private] method relativizePath
void TSEManifest::relativizePath( char* buf, const char* basePath, const char* fullPath )
{
    // if the full path does not start with the base path, use the full path as specified
    size_t cbBasePath = strlen( basePath );
    if( memcmp( basePath, fullPath, cbBasePath ) )
        strcpy_s( buf, FILENAME_MAX, fullPath );
    else    // use a relative path
    {
        buf[0] = '.';
        strcpy_s( buf+1, FILENAME_MAX-1, fullPath+cbBasePath );
    }
}

/// [private] method appendPathForManifest
void TSEManifest::appendPathForManifest( const char* basePath, const char* fullPath, const char* baseName, const char* oft, const char* ext )
{
    if( (fullPath == NULL) || (*fullPath == 0) )
        return;

    // copy the output format type and file extension strings
    char oftx[32];
    strcpy_s( oftx, sizeof oftx, oft );
    char extx[32];
    strcpy_s( extx, sizeof extx, ext );

    // use a relative path if possible
    char pathBuf[FILENAME_MAX];
    relativizePath( pathBuf, basePath, fullPath );

    RaiiDirectory::ChopTrailingPathSeparator( pathBuf );
    size_t cb = strlen( pathBuf );
    _strlwr_s( oftx, sizeof oftx );
    _strlwr_s( extx, sizeof extx );
    sprintf_s( pathBuf+cb, (sizeof pathBuf)-cb, "%s%s.%s.*.%s", FILEPATHSEPARATOR, baseName, oftx, extx );

    // build the TSEWF bound variable name for the "assign" statement in the manifest file
    char bvName[16];
    _strupr_s( oftx, sizeof oftx );
    sprintf_s( bvName, sizeof bvName, "%s%s", extx, oftx );

    AppendString( bvName, pathBuf );
}
#pragma endregion

#pragma region public methods
/// [public] method Write
void TSEManifest::Write( WinGlobalPtr<OutputFileInfo>* pofi )
{
    static const AlignmentResultFlags arfCUDR = static_cast<AlignmentResultFlags>(arfConcordant | arfDiscordant | arfRejected | arfUnmapped);

    // write the TSE manifest file to the base path for the first directory that contains TSE output
    char* pManifestDir = NULL;
    char* pBasePath = NULL;
    char* pBaseName = NULL;
    char* pSbfCUDR = NULL;
    char* pSbfCDR = NULL;
    char* pSbfU = NULL;
    char* pKmhU = NULL;
    for( UINT32 n=0; n<pofi->n; ++n )
    {
        switch( pofi->p[n].oft )
        {
            case oftTSE:
                if( pManifestDir == NULL )
                    pManifestDir = pBasePath = pofi->p[n].rootPath;
                if( pBaseName == NULL )
                    pBaseName = pofi->p[n].baseName;

                switch( pofi->p[n].arf & arfCUDR )
                {
                    case arfCUDR:
                        pSbfCUDR = pofi->p[n].path;
                        break;

                    case arfConcordant|arfDiscordant|arfRejected:
                        pSbfCDR = pofi->p[n].path;
                        break;

                    case arfUnmapped:
                        pSbfU = pofi->p[n].path;
                        break;

                    default:
                        break;
                }
                break;

            case oftKMH:
                if( pManifestDir == NULL )
                    pManifestDir = pBasePath = pofi->p[n].rootPath;
                if( pBaseName == NULL )
                    pBaseName = pofi->p[n].baseName;

                pKmhU = pofi->p[n].path;
                break;

            default:
                break;
        }
    }

    // sanity check
    if( pManifestDir == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "no output directory for TSE manifest" );

    // append path information to the manifest file
    appendToBuffer( "" );
    appendPathForManifest( pBasePath, pSbfCUDR, pBaseName, "cudr", "sbf" );
    appendPathForManifest( pBasePath, pSbfCDR, pBaseName, "cdr", "sbf" );
    appendPathForManifest( pBasePath, pSbfU, pBaseName, "u", "sbf" );
    appendPathForManifest( pBasePath, pKmhU, pBaseName, "u", "kmh" );

    if( m_paamb->RGMgr.SBFfileSpec[0] )
    {
        char pathBuf[FILENAME_MAX];
        relativizePath( pathBuf, pBasePath, m_paamb->RGMgr.SBFfileSpec );
        AppendString( "sbfRG", pathBuf );
    }

    // build the manifest file specification
    char fileSpec[FILENAME_MAX];
    strcpy_s( fileSpec, sizeof fileSpec, pManifestDir );
    RaiiDirectory::ChopTrailingPathSeparator( fileSpec );
    strcat_s( fileSpec, sizeof fileSpec, FILEPATHSEPARATOR );
    strcat_s( fileSpec, sizeof fileSpec, m_manifestFileName );
    
    // write the contents of the buffer to the manifest file
    RaiiFile outFile( fileSpec );
    outFile.Write( m_buf.p, m_buf.n );
}

/// [public] method AppendString
void TSEManifest::AppendString( const char* k, const char* v )
{
    // allocate a buffer to contain the formatted string
    WinGlobalPtr<char> s( strlen( k )+strlen( v )+12, true );       // 12: extra space for "assign", enclosing double quotes, whitespace, trailing null

    // conditionally enclose the string value in double quotes
    const char* fmt = strchr( v, ' ' ) ? "assign %s \"%s\"" : "assign %s %s";
    
    // format and emit the string
    sprintf_s( s.p, s.cb, fmt, k, v );
    appendToBuffer( s.p );
}

/// [public] method AppendDouble
void TSEManifest::AppendDouble( const char* k, const double v, const INT32 decimalPlaces )
{
    char fmt[8];
    sprintf_s( fmt, sizeof fmt, "%%%d.%df", decimalPlaces+1, decimalPlaces );

    char s[16];
    sprintf_s( s, sizeof s, fmt, v );
    AppendString( k, s );
}

/// [public] method AppendI8
void TSEManifest::AppendI8( const char* k, const INT8 v )
{
    char s[8];
    sprintf_s( s, sizeof s, "%d", v );
    AppendString( k, s );
}

/// [public] method AppendI32
void TSEManifest::AppendI32( const char* k, const INT32 v )
{
    char s[16];
    sprintf_s( s, sizeof s, "%d", v );
    AppendString( k, s );
}

/// [public] method AppendU64
void TSEManifest::AppendU64( const char* k, const UINT64 v )
{
    char s[24];
    sprintf_s( s, sizeof s, "%llu", v );
    AppendString( k, s );
}
#pragma endregion
