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
class AppMain : public AriocAppMainBase
{
    public:
        typedef struct
        {
            const char* prefix;
            TumFlags    flags;
            const char* description;
        }
            TumHeadingDesc;

    private:
        static const char*      m_xaGapped[];   // extra attributes for the <gapped> element (not parsed in the base implementation)
        static const char*      m_xaA[];        // extra attributes for the <A> element (not parsed in the base implementation)
        static TumHeadingDesc   m_tumHeading[];

    private:
        void initPairedInputFileGroup( InputFileGroup& ifgQ, tinyxml2::XMLElement* pelPaired0 );
        void parsePairFragmentLength( INT32& minFragLen, INT32& maxFragLen, const char* pFragLen );
        AlignmentResultFlags parsePairCollision( const char* pAlignmentResultFlags );
        void parseAP( INT32& minFragLen, INT32& maxFragLen, AlignmentResultFlags& arfA );

    public:
        AppMain( void );
        virtual ~AppMain( void );
        void Launch( void );
};

