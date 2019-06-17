/*
  AppMain.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AppMain__

/// <summary>
/// Class <c>AppMain</c> parses input parameters, launches the application, and displays performance metrics.
/// </summary>
class AppMain : public AppMainCommon
{
    public:
        static const char RGID_PLACEHOLDER[];

    private:
        tinyxml2::XMLElement* m_pelDataIn;
        tinyxml2::XMLElement* m_pelDataOut;

        static const char*  m_rgAts;
        static const char*  m_rgAtNull;

    private:
        double computeMbPerSecond( INT64 cb, INT32 ms );
        bool isReadGroupPattern( const char* s );
        bool saveReadGroupInfo( InputFileGroup::FileInfo* pfi, tinyxml2::XMLElement* pelRg, tinyxml2::XMLElement* pelFile );

    protected:
        virtual void parseXmlElements( void );

    public:
        AppMain( void );
        virtual ~AppMain( void );

        virtual void LoadConfig( void );
        virtual void Launch( void );
};
