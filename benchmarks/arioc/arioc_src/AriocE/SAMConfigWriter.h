/*
  SAMConfigWriter.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __SAMConfigWriter__

#define CONFIG_SUBID_FORMAT "%03d"

class SAMConfigWriter
{
    private:
        const AriocEncoderParams*   m_psip;
        char                        m_configFileName[FILENAME_MAX];
        tinyxml2::XMLDocument       m_doc;
        char                        m_outputFileSpec[FILENAME_MAX];

    private:
        void initXmlTemplate( const char* baseFilename );
        tinyxml2::XMLElement* createSQ( const INT32 subId );
        tinyxml2::XMLElement* getSQ( const INT32 subId );

    public:
        SAMConfigWriter( const AriocEncoderParams* _psip, const char* _stub, const char* _baseName );
        virtual ~SAMConfigWriter( void );
        void AppendReferenceMetadata( const INT32 subId, const char* pMetadata );
        void AppendReferenceLength( const INT32 subId, const INT64 cb );
        void AppendReferenceURI( const INT32 subId, const char* uri );
        void AppendExecutionTime( const INT32 ms );
        void AppendMaxJ( const INT32 maxJ );
        void AppendQualityScoreBias( const INT32 qsb );
        void AppendReadGroupInfo( RGManager* prgm, WinGlobalPtr<char>* pqmdRG, WinGlobalPtr<UINT32>* pofsqmdRG, InputFileGroup::FileInfo* pfi );
        void Write( void );
};
