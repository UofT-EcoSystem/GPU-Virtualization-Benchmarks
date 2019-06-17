/*
  SAMHDBuilder.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __SAMHDBuilder__

class SAMHDBuilder
{
    static const long           MIN_ENCODER_VERSION_MAJOR = 1;
    static const long           MIN_ENCODER_VERSION_MINOR = 30;
    static const UINT32         HEADER_BUFFER_SIZE = 64*1024;
    static const INT32          AO1 = 5;
    static const INT32          AO2 = 14;
    static const char* const    m_samVersion;
    static const char* const    m_samAttributeOrder[AO1][AO2];

    private:
        RGManager*              m_prgm;
        tinyxml2::XMLDocument   m_docSAM;
        WinGlobalPtr<char>      m_SAMheader;
        char                    m_cmdTail[FILENAME_MAX+32];
        char                    m_appName[64];
        char                    m_appVersion[32];
        bool                    m_suppressHD;
        bool                    m_suppressCfg;

    public:
        char    BigBucketThreshold[16];

    private:
        SAMHDBuilder( void );
        void verifyVN( tinyxml2::XMLDocument* _pxd, char* _baseName );
        void addPG( AriocAppMainBase* paamb );
        void addRG( InputFileGroup* pifgQ );
        void appendSAMtags( char*& p, tinyxml2::XMLElement* el );
        void buildSAMheader( void );
        void getReferenceConfigFileSpec( char* baseName, char* fileSpec, const char* pathCfg );

    public:
        SAMHDBuilder( RGManager* prgm );
        virtual ~SAMHDBuilder( void );
        void Init( const char* pathCfgN, const char* pathCfgG, InputFileGroup* pifgQ, AriocAppMainBase* pamb );
        void WriteHeader( RaiiFile& fileSAM );
};
