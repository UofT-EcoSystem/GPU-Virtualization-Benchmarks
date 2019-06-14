/*
  TSEManifest.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __TSEManifest__

/// <summary>
/// Class <c>TSEManifest</c> supports the creation of a TSE (Terabase Search Engine) "manifest" file.
/// </summary>
class TSEManifest
{
    private:
        static const UINT32     INITIAL_BUFFER_SIZE = 128;
        AriocAppMainBase*       m_paamb;
        WinGlobalPtr<char>      m_buf;
        char                    m_manifestFileName[48];

    private:
        TSEManifest();
        void appendToBuffer( const char* s );
        void initialize( const INT32 refId, const INT32 srcId, const INT8 subId );
        void relativizePath( char* buf, const char* basePath, const char* fullPath );
        void appendPathForManifest( const char* basePath, const char* fullPath, const char* baseName, const char* oft, const char* ext );

    public:
        TSEManifest( AriocAppMainBase* paamb, INT32 refId, InputFileGroup* pifgQ );
        virtual ~TSEManifest();
        void Write( WinGlobalPtr<OutputFileInfo>* pofi );
        void AppendString( const char* k, const char* v );
        void AppendDouble( const char* k, const double v, const INT32 decimalPlaces = 0 );
        void AppendI8( const char* k, INT8 v );
        void AppendI32( const char* k, INT32 v );
        void AppendU64( const char* k, UINT64 v );
};
