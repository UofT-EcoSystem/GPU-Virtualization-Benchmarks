/*
  AriocAppMainBase.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocAppMainBase__

#if !defined(__A21SpacedSeed__)
#include "A21SpacedSeed.h"
#endif

#if !defined(__A21HashedSeed__)
#include "A21HashedSeed.h"
#endif

#if !defined(__AriocAlignmentScorer__)
#include "AriocAlignmentScorer.h"
#endif

#pragma region enums
enum CIGARfmtType
{
    cftQXIDS =  0,  // = match; X mismatch; I insert into R; D deletion from R; S soft clip
    cftMIDS =   1,  // M match or mismatch; I insert into R; D deletion from R; S soft clip
    cftMID =    2   // M match or mismatch; I insert into R (including soft clipping); D delete from R
};

enum MDfmtType
{
    mftStandard =   0,  // MD string formatted per the SAM Format Specification
    mftCompact =    1   // MD string omits 0 placeholders between mismatched bases
};
#pragma endregion

/// <summary>
/// Base implementation for Arioc applications
/// </summary>
class AriocAppMainBase : public AppMainCommon
{
    protected:
        tinyxml2::XMLElement*   m_pelR;
        tinyxml2::XMLElement*   m_pelNongapped;
        tinyxml2::XMLElement*   m_pelGapped;
        tinyxml2::XMLElement*   m_pelQ;
        tinyxml2::XMLElement*   m_pelA;

    public:
        RGManager   RGMgr;

    protected:
        AriocAppMainBase( void );
        virtual ~AriocAppMainBase( void );
        AlignmentResultFlags parseReport( tinyxml2::XMLElement* pel );
        void parseR( const char** pPathR );
        void parseNongapped( const char*& pSSI, INT32& maxAn, INT32& maxJn, INT32& seedCoverage, INT32& maxMismatches, const char** pxa );
        void parseGapped( const char*& pHSI, INT32& minPosSep, INT32& seedDepth, Wmxgs& w, const char*& pVt, INT32& AtN, INT32& AtG, INT32& maxAg, INT32& maxJg, const char** pxa );
        void parseQ( const char* const elementName, const char*& filePathQ, tinyxml2::XMLElement*& pel );
        void parseA( const char*& pBaseNameA, const char*& pBasePathA, bool& overwriteOutputFiles, INT64& maxAperOutputFile, INT32& mapqUnknown, INT32& mapqVersion, CIGARfmtType& cigarFmtType, MDfmtType& mdFmtType, INT32& k, WinGlobalPtr<OutputFileInfo>& ofi, AlignmentResultFlags& arfSAM, AlignmentResultFlags& arfSBF, AlignmentResultFlags& arfTSE, AlignmentResultFlags& arfKMH, const char** pxa );
        void parsePairedUnpaired( tinyxml2::XMLElement*& pel, INT32& srcId, UINT8& subId, INT64& readIdFrom, INT64& readIdTo, char srcInfo[MAX_SRCINFO_LENGTH] );
        void buildInputFilename( char* inputFilename, const char* baseName );
        const char* decodeMDformatType( MDfmtType _mft );
        const char* decodeCIGARformatType( CIGARfmtType _cft );

        virtual void parseXmlElements( void );

    public:
        virtual void LoadConfig( void );
        virtual void Launch( void ) = 0;
};
