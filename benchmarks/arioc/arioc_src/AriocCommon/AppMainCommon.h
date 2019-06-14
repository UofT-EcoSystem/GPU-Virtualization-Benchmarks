/*
  AppMainCommon.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AppMainCommon__

#if !defined(__A21SpacedSeed__)
#include "A21SpacedSeed.h"
#endif

#if !defined(__A21HashedSeed__)
#include "A21HashedSeed.h"
#endif

#if !defined(__AriocAlignmentScorer__)
#include "AriocAlignmentScorer.h"
#endif


/// <summary>
/// Base implementation for Arioc applications
/// </summary>
class AppMainCommon
{
    protected:
        static const double MS_PER_SEC;
        static const double BYTES_PER_MB;
        static const double BYTES_PER_GB;
        static const INT32  MAX_SRCINFO_LENGTH = 256;
        static const INT32  DEFAULT_KMER_SIZE = 8;

        tinyxml2::XMLDocument   m_xmlDoc;
        tinyxml2::XMLElement*   m_pelRoot;
        tinyxml2::XMLElement*   m_pelX;
        static const char*      m_xaNull[];
        INT32                   m_maxBatchSize;

    public:
        char        AppName[64];
        char        AppVersion[32];
        char        ConfigFileName[FILENAME_MAX];
        char        MachineName[MAX_COMPUTERNAME_LENGTH+2];
        char        DataSourceInfo[MAX_SRCINFO_LENGTH];
        RGManager   RGMgr;
        AA<UINT64>  Xparam;

    protected:
        AppMainCommon( void );
        virtual ~AppMainCommon( void );
        INT64 parseInt64GMK( const char* pigmk, const INT64 defaultValue );
        INT32 parseInt32GMK( const char* pigmk, const INT32 defaultValue );
        void parseRoot( UINT32& gpuMask, INT32& maxDOP, INT32& batchSize );
        void parseX( void );

        virtual void parseXmlElements( void ) = 0;

    public:
        void Init( const char* pAppName, const char* pAppVersion, const char* configFileName );

        virtual void LoadConfig( void ) = 0;
        virtual void Launch( void ) = 0;
};
