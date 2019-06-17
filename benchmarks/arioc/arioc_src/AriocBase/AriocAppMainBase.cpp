/*
  AriocAppMainBase.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [protected] default constructor
AriocAppMainBase::AriocAppMainBase() : m_pelR(NULL),
                                       m_pelNongapped(NULL),
                                       m_pelGapped(NULL),
                                       m_pelQ(NULL),
                                       m_pelA(NULL)
{
    memset( this->AppName, 0, sizeof this->AppName );
    memset( this->AppVersion, 0, sizeof this->AppVersion );
    memset( this->ConfigFileName, 0, sizeof this->ConfigFileName );
    memset( this->MachineName, 0, sizeof this->MachineName );
    memset( this->DataSourceInfo, 0, sizeof this->DataSourceInfo );
}

/// destructor
AriocAppMainBase::~AriocAppMainBase()
{
}
#pragma endregion

#pragma region protected methods
/// [protected] method parseReport
AlignmentResultFlags AriocAppMainBase::parseReport( tinyxml2::XMLElement* pel )
{
    // ensure that a report= attribute is present
    const char* pAlignmentResultFlags = pel->Attribute( "report" );
    if( pAlignmentResultFlags == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "missing report= attribute for <%s> element", pel->Name() );

    char buf[256];
    strcpy_s( buf, sizeof buf, pAlignmentResultFlags );
    _strlwr_s( buf, sizeof buf );

    // replace each non-alphabetic character with a space and make everything lowercase
    char* p = buf;
    while( *p )
    {
        if( !isalpha(*p) )
            *p = ' ';
        else
            *p = tolower(*p);
        ++p;
    }

    // pick apart the parameters
    char v[4][32];
#ifdef _WIN32
    INT32 nParams = sscanf_s( buf, "%s %s %s %s", v[0], 32, v[1], 32, v[2], 32, v[3], 32 );
#endif
#ifdef __GNUC__
    INT32 nParams = sscanf( buf, "%s %s %s %s", v[0], v[1], v[2], v[3] );
#endif
    if( nParams == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "no value specified for attribute report=\"\" in <%s> element", pel->Name() );

    // map the parameter strings to bit flags
    AlignmentResultFlags arf = arfNone;

    for( INT32 n=0; n<min2(nParams,4); ++n )
    {
        // set a flag if the parameter contains only one-character abbreviations
        INT32 cch = static_cast<INT32>(strlen( v[n] ));
        bool isAbbrev = (static_cast<INT32>(strspn( v[n], "mucdr" )) == cch);

        // point to the first character of the nth parameter
        char* p = v[n];

        do
        {
            // look at the current character in the parameter
            switch( *p )
            {
                case 'c':
                    arf = static_cast<AlignmentResultFlags>(arf | arfReportConcordant);
                    break;

                case 'd':
                    arf = static_cast<AlignmentResultFlags>(arf | arfReportDiscordant);
                    break;

                case 'r':
                    arf = static_cast<AlignmentResultFlags>(arf | arfReportRejected);
                    break;

                case 'm':
                    arf = static_cast<AlignmentResultFlags>(arf | arfReportMapped);
                    break;

                case 'u':
                    arf = static_cast<AlignmentResultFlags>(arf | arfReportUnmapped);
                    break;

                default:
                    throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value report=\"%s\" in <%s> element", v[n], pel->Name() );
            }

            // don't bother with the rest of the string if it does not consist only of one-character abbreviations
            if( !isAbbrev )
                break;
        }
        while( *(++p) );
    }

    return arf;
}

/// [protected] method parseR
void AriocAppMainBase::parseR( const char** pPathR )
{
    // get the R (reference sequence) directory path from the <R> element
    *pPathR = m_pelR->GetText();
    if( (*pPathR == NULL) || (**pPathR == 0) )
        throw new ApplicationException( __FILE__, __LINE__, "missing directory path in <R> element" );
}

/// [protected] method parseNongapped
void AriocAppMainBase::parseNongapped( const char*& pSSI, INT32& maxAn, INT32& maxJn, INT32& seedCoverage, INT32& maxMismatches, const char** pxa )
{
    using namespace tinyxml2;

    // default values
    pSSI = "error";
    maxAn = 10;
    maxJn = _I32_MAX;
    seedCoverage = 7;
    maxMismatches = SSMMDEFAULT;

    // parse the attributes
    const XMLAttribute* pa = m_pelNongapped->FirstAttribute();
    while( pa )
    {
        bool pending = true;

        if( XMLUtil::StringEqual( pa->Name(), "seed" ) )
        {
            pSSI = pa->Value();
            if( pSSI == NULL )
                throw new ApplicationException( __FILE__, __LINE__, "missing attribute in element <nongapped>: seed" );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "maxA" ) )
        {
            maxAn = pa->IntValue();
            if( maxAn < 0 )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value maxA=\"%d\" in element <nongapped>", maxAn );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "maxJ" ) )
        {
            maxJn = parseInt32GMK( pa->Value(), maxJn );   
            if( maxJn < 0 )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value maxJ=\"%d\" in element <nongapped>", maxJn );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "seedCoverage" ) )
        {
            seedCoverage = pa->IntValue();
            if( (seedCoverage < 1) || (seedCoverage > 7) )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value seedCoverage=\"%d\" in element <nongapped>", seedCoverage );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "maxMismatches" ) )
        {
            maxMismatches = pa->IntValue();
            if( maxMismatches < 0 )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value maxMismatches=\"%d\" in element <nongapped>", seedCoverage );
            pending = false;
        }

        // iterate through the list of attribute names that are handled in a specialization of this base class
        const char** px = pxa;
        while( pending && *px )
        {
            pending = (strcmp(pa->Name(), *px) != 0);
            px++ ;
        }

        // at this point we have either an invalid attribute name or a missing value for a valid attribute name
        if( pending )
            throw new ApplicationException( __FILE__, __LINE__, "invalid attribute \"%s\" in element <nongapped>", pa->Name() );

        // iterate through the attribute list
        pa = pa->Next();
    }
}

/// [protected] method parseGapped
void AriocAppMainBase::parseGapped( const char*& pHSI, INT32& minPosSep, INT32& seedDepth, Wmxgs& w, const char*& pVt, INT32& AtN, INT32& AtG, INT32& maxAg, INT32& maxJg, const char** pxa )
{
    using namespace tinyxml2;

    // default values
    pHSI = "error";
    minPosSep = 5;
    seedDepth = 6;
    w = Wmxgs_unknown;
    AtN = 1;
    AtG = 1;
    maxAg = 2;
    maxJg = _I32_MAX;

    // ensure that the seed=, Wmxgs=, and Vt= attributes are present
    if( m_pelGapped->Attribute( "seed" ) == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "missing seed= attribute for <%s> element", m_pelGapped->Name() );
    if( m_pelGapped->Attribute( "Wmxgs" ) == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "missing Wmxgs= attribute for <%s> element", m_pelGapped->Name() );
    if( m_pelGapped->Attribute( "Vt" ) == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "missing Vt= attribute for <%s> element", m_pelGapped->Name() );

    // parse the attributes
    const XMLAttribute* pa = m_pelGapped->FirstAttribute();
    while( pa )
    {
        bool pending = true;

        if( XMLUtil::StringEqual( pa->Name(), "seed" ) )
        {
            pHSI = pa->Value();
            if( pHSI == NULL )
                throw new ApplicationException( __FILE__, __LINE__, "missing attribute in element <gapped>: seed" );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "minPosSep" ) )
        {
            minPosSep = pa->IntValue();
            if( minPosSep < 0 )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value minPosSep=\"%d\" in element <gapped>", minPosSep );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "seedDepth" ) )
        {
            seedDepth = pa->IntValue();
            if( (seedDepth < 1) || (seedDepth > 6) )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value seedDepth=\"%d\" in element <gapped>", seedDepth );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "Wmxgs" ) )
        {
            const char* psw = pa->Value();
            if( psw == NULL )
                throw new ApplicationException( __FILE__, __LINE__, "missing attribute in element <gapped>: Wmxgs" );
            w = AriocAlignmentScorer::StringToWmxgs( psw );
            if( w == Wmxgs_unknown )
                throw new ApplicationException( __FILE__, __LINE__, "unrecognized attribute value Wmxgs=\"%s\" in element <gapped>", psw );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "Vt" ) )
        {
            pVt = pa->Value();
            if( (pVt == NULL) || (*pVt == 0) )
                throw new ApplicationException( __FILE__, __LINE__, "missing attribute value in element <gapped>: Vt" );
            pending = false;
        }
            
        if( pending && XMLUtil::StringEqual( pa->Name(), "AtN" ) )
        {
            AtN = pa->IntValue();
            if( AtN < 0 )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value AtN=\"%d\" in element <gapped>", AtN );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "AtG" ) )
        {
            AtG = pa->IntValue();
            if( AtG < 0 )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value AtG=\"%d\" in element <gapped>", AtG );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "maxA" ) )
        {
            maxAg = pa->IntValue();
            if( maxAg < 0 )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value maxA=\"%d\" in element <gapped>", maxAg );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "maxJ" ) )
        {
            maxJg = parseInt32GMK( pa->Value(), maxJg );
            pending = false;
        }

        // iterate through the list of attribute names that are handled in a specialization of this base class
        const char** px = pxa;
        while( pending && *px )
        {
            pending = (strcmp(pa->Name(), *px) != 0);
            px++ ;
        }

        // at this point we have either an invalid attribute name or a missing value for a valid attribute name
        if( pending )
            throw new ApplicationException( __FILE__, __LINE__, "invalid attribute \"%s\" in element <gapped>", pa->Name() );

        // iterate through the attribute list
        pa = pa->Next();
    }
}

/// [protected] parseQ
void AriocAppMainBase::parseQ( const char* const elementName, const char*& filePathQ, tinyxml2::XMLElement*& pel )
{
    using namespace tinyxml2;

    // default values
    pel = NULL;
    filePathQ = NULL;

    // parse the attributes
    const XMLAttribute* pa = m_pelQ->FirstAttribute();
    while( pa )
    {
        bool pending = true;

        if( pending && XMLUtil::StringEqual( pa->Name(), "filePath" ) )
        {
            filePathQ = pa->Value();
            if( (filePathQ == NULL) || (*filePathQ == 0) )
                throw new ApplicationException( __FILE__, __LINE__, "missing value for attribute \"filePath\" in the <Q> element", elementName );

            pel = m_pelQ->FirstChildElement( elementName );
            if( pel == NULL )
                throw new ApplicationException( __FILE__, __LINE__, "missing <%s> element", elementName );

            pending = false;
        }

        // at this point we have either an invalid attribute name or a missing value for a valid attribute name
        if( pending )
            throw new ApplicationException( __FILE__, __LINE__, "invalid attribute \"%s\" in element <Q>", pa->Name() );

        // iterate through the attribute list
        pa = pa->Next();
    }
}

/// [protected] method parseA
void AriocAppMainBase::parseA( const char*& pBaseNameA,
                               const char*& pBasePathA,
                               bool& overwriteOutputFiles,
                               INT64& maxAperOutputFile,
                               INT32& mapqUnknown,
                               INT32& mapqVersion,
                               CIGARfmtType& cigarFmtType,
                               MDfmtType& mdFmtType,
                               INT32& k,
                               WinGlobalPtr<OutputFileInfo>& ofi,
                               AlignmentResultFlags& arfSAM,
                               AlignmentResultFlags& arfSBF,
                               AlignmentResultFlags& arfTSE,
                               AlignmentResultFlags& arfKMH,
                               const char** pxa )
{
    using namespace tinyxml2;

    // default values
    pBaseNameA = this->AppName;
    pBasePathA = NULL;
    overwriteOutputFiles = false;
    maxAperOutputFile = 1024*1024*1024;
    mapqUnknown = 0;
    mapqVersion = 0;
    cigarFmtType = cftQXIDS;
    mdFmtType = mftStandard;
    k = DEFAULT_KMER_SIZE;
    arfSAM = arfSBF = arfTSE = arfKMH = arfNone;

    // parse the attributes
    const XMLAttribute* pa = m_pelA->FirstAttribute();
    while( pa )
    {
        bool pending = true;

        if( pending && XMLUtil::StringEqual( pa->Name(), "baseName" ) )
        {
            pBaseNameA = pa->Value();
            if( (pBaseNameA == NULL) || (*pBaseNameA == 0) )
                pBaseNameA = this->AppName;
           
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "basePath" ) )
        {
            pBasePathA = pa->Value();

            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "overwrite" ) )
        {
            overwriteOutputFiles = pa->BoolValue();
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "maxA" ) )
        {
            const char* pMaxA = pa->Value();
            if( (pMaxA == NULL) || (*pMaxA == 0) )
                throw new ApplicationException( __FILE__, __LINE__, "missing attribute \"maxA\"in element <A>" );
            maxAperOutputFile = parseInt64GMK( pMaxA, maxAperOutputFile );

            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "mapqUnknown" ) )
        {
            mapqUnknown = pa->IntValue();
            if( (mapqUnknown < 0) || (mapqUnknown > 255) )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value mapqUnknown=\"%s\" in element <A>", pa->Value() );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "mapqVersion" ) )
        {
            mapqVersion = pa->IntValue();
            if( (mapqVersion < 0) || (mapqVersion > 255) )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value mapqVersion=\"%s\" in element <A>", pa->Value() );
            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "cigarFormat" ) )
        {
            if( 0 == strcmp( pa->Value(), "MID" ) )
            {
                cigarFmtType = cftMID;
                pending = false;
            }

            if( pending && (0 == strcmp( pa->Value(), "MIDS" )) )
            {
                cigarFmtType = cftMIDS;
                pending = false;
            }

            if( pending && (0 == strcmp( pa->Value(), "=XIDS" )) )
            {
                cigarFmtType = cftQXIDS;
                pending = false;
            }

            if( pending && pa->Value() )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value cigarFormat=\"%s\" in element <A>", pa->Value() );

            pending = false;
        }

        if( pending && XMLUtil::StringEqual( pa->Name(), "mdFormat" ) )
        {
            if( 0 == strcmp( pa->Value(), "standard" ) )
            {
                mdFmtType = mftStandard;
                pending = false;
            }

            if( pending && (0 == strcmp( pa->Value(), "compact" )) )
            {
                mdFmtType = mftCompact;
                pending = false;
            }

            if( pending && pa->Value() )
                throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value mdFormat=\"%s\" in element <A>", pa->Value() );

            pending = false;
        }

        // iterate through the list of attribute names that are handled in a specialization of this base class
        const char** px = pxa;
        while( pending && *px )
        {
            pending = (strcmp(pa->Name(), *px) != 0);
            px++ ;
        }

        // at this point we have either an invalid attribute name or a missing value for a valid attribute name
        if( pending )
            throw new ApplicationException( __FILE__, __LINE__, "invalid attribute \"%s\" for element <A>", pa->Name() );

        // iterate through the attribute list
        pa = pa->Next();
    }

    /* get the output paths and other info from the subelement(s) of the <A> element */

    // count subelements of the <A> element
    INT32 nOutputPaths = 0;
    XMLElement* pelAchild = m_pelA->FirstChildElement();
    while( pelAchild != NULL )
    {
        nOutputPaths++ ;
        pelAchild = pelAchild->NextSiblingElement();
    }

    // sanity check
    if( nOutputPaths == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "The <A> element must contain at least one child element" );

    // allocate a list of output paths (OutputFileInfo instances)
    ofi.Realloc(nOutputPaths, true);
    ofi.n = 0;

    // SAM
    pelAchild = m_pelA->FirstChildElement( "sam" );
    while( pelAchild )
    {
        AlignmentResultFlags arf = parseReport( pelAchild );                // parse the report= attribute
        ofi.p[ofi.n] = OutputFileInfo( pBasePathA, pelAchild->GetText(), pBaseNameA, oftSAM, arf, maxAperOutputFile );
        arfSAM = static_cast<AlignmentResultFlags>(arfSAM | arf);           // accumulate the alignment result flags across all output files

        if( !overwriteOutputFiles )
        {
            RaiiDirectory dir( ofi.p[ofi.n].path, "*.sam" );                // ensure that files will not be overwritten
            if( dir.Filenames.Count )
                throw new ApplicationException( __FILE__, __LINE__, "one or more SAM files exist in %s (to overwrite, specify overwrite=\"true\" in the <A> element)", ofi.p[ofi.n].path );
        }

        ofi.n++;                                                            // count the number of output paths in the list
        pelAchild = pelAchild->NextSiblingElement( "sam" );
    }

    // SBF
    pelAchild = m_pelA->FirstChildElement( "sbf" );
    while( pelAchild )
    {
        AlignmentResultFlags arf = parseReport( pelAchild );                // parse the report= attribute
        ofi.p[ofi.n] = OutputFileInfo( pBasePathA, pelAchild->GetText(), pBaseNameA, oftSBF, arf, maxAperOutputFile );
        arfSBF = static_cast<AlignmentResultFlags>(arfSBF | arf);           // accumulate the alignment result flags across all output files

        if( !overwriteOutputFiles )
        {
            RaiiDirectory dir( ofi.p[ofi.n].path, "*.sbf" );                // ensure that files will not be overwritten
            if( dir.Filenames.Count )
                throw new ApplicationException( __FILE__, __LINE__, "one or more SBF files exist in %s (to overwrite, specify overwrite=\"true\" in the <A> element)", ofi.p[ofi.n].path );
        }

        ofi.n++ ;                                                           // count the number of output paths in the list
        pelAchild = pelAchild->NextSiblingElement( "sbf" );
    }

    // TSE
    pelAchild = m_pelA->FirstChildElement( "tse" );
    while( pelAchild )
    {
        AlignmentResultFlags arf = parseReport( pelAchild );                // parse the report= attribute
        ofi.p[ofi.n] = OutputFileInfo( pBasePathA, pelAchild->GetText(), pBaseNameA, oftTSE, arf, maxAperOutputFile );
        arfTSE = static_cast<AlignmentResultFlags>(arfTSE | arf);           // accumulate the alignment result flags across all output files

        if( !overwriteOutputFiles )
        {
            RaiiDirectory dir( ofi.p[ofi.n].path, "*.sbf" );                // ensure that files will not be overwritten
            if( dir.Filenames.Count )
                throw new ApplicationException( __FILE__, __LINE__, "one or more SBF files exist in %s (to overwrite, specify overwrite=\"true\" in the <A> element)", ofi.p[ofi.n].path );
        }

        ofi.n++;                                                            // count the number of output paths in the list
        pelAchild = pelAchild->NextSiblingElement( "tse" );
    }

    // KMH
    pelAchild = m_pelA->FirstChildElement( "kmh" );
    while( pelAchild )
    {
        AlignmentResultFlags arf = parseReport( pelAchild );                // parse the report= attribute
        ofi.p[ofi.n] = OutputFileInfo( pBasePathA, pelAchild->GetText(), pBaseNameA, oftKMH, arf, maxAperOutputFile );
        arfKMH = static_cast<AlignmentResultFlags>(arfKMH | arf);           // accumulate the alignment result flags across all output files

        if( !overwriteOutputFiles )
        {
            RaiiDirectory dir( ofi.p[ofi.n].path, "*.kmh" );                // ensure that files will not be overwritten
            if( dir.Filenames.Count )
                throw new ApplicationException( __FILE__, __LINE__, "one or more KMH files exist in %s (to overwrite, specify overwrite=\"true\" in the <A> element)", ofi.p[ofi.n].path );
        }

        // get the kmer size
        INT32 kmerSize = DEFAULT_KMER_SIZE;
        switch( pelAchild->QueryIntAttribute( "k", &kmerSize ) )
        {
            case XML_WRONG_ATTRIBUTE_TYPE:
                throw new ApplicationException( __FILE__, __LINE__, "invalid value %s for attribute k= in element <kmh>", pelAchild->Attribute("k") );

            case XML_NO_ERROR:
                if( (kmerSize < 7) || (kmerSize > 21) )
                    throw new ApplicationException( __FILE__, __LINE__, "invalid value %d for attribute k= in element <kmh>", kmerSize );
                if( (k != DEFAULT_KMER_SIZE) && (k != kmerSize) )
                    throw new ApplicationException( __FILE__, __LINE__, "conflicting values %d and %d specified for attribute k= in element <kmh>", k, kmerSize );

                k = kmerSize;
                break;

            default:
                break;
        }

        ofi.n++;                                                            // count the number of output paths in the list
        pelAchild = pelAchild->NextSiblingElement( "kmh" );
    }

    // if <sbf> and <tse> elements were both specified, ensure that the output is going to different directories
    if( (arfSBF != arfNone) && (arfTSE != arfNone) )
    {
        // compare all pairs in the list
        for( INT32 m=0; m<(nOutputPaths-1); ++m )
        {
            for( INT32 n=0; n<nOutputPaths; ++n )
            {
                // if we have an SBF/TSE pair...
                if( ((ofi.p[m].oft == oftSBF) && (ofi.p[n].oft == oftTSE)) ||
                    ((ofi.p[m].oft == oftTSE) && (ofi.p[n].oft == oftSBF)) )
                {
                    // ...ensure that the directory paths differ
                    if( 0 == _strcmpi( ofi.p[m].path, ofi.p[n].path ) )
                        throw new ApplicationException( __FILE__, __LINE__, "SBF and TSE output may not be placed in the same directory (%s)", ofi.p[m].path );
                }
            }
        }
    }
}

/// [protected] method parsePairedUnpaired
void AriocAppMainBase::parsePairedUnpaired( tinyxml2::XMLElement*& pel, INT32& srcId, UINT8& subId, INT64& readIdFrom, INT64& readIdTo, char srcInfo[MAX_SRCINFO_LENGTH] )
{
    using namespace tinyxml2;


    // get the srcId
    srcId = 0;          // default srcId
    subId = 0;          // default subId
    INT32 val;
    switch( pel->QueryIntAttribute( "srcId", &val ) )
    {
        case XML_NO_ERROR:
            if( val > AriocDS::SqId::MaxSrcId )
                throw new ApplicationException( __FILE__, __LINE__, "invalid value %d for attribute \"srcId\" in element <%s> (maximum %d)", srcId, pel->Name(), AriocDS::SqId::MaxSrcId );
            srcId = val;
            break;

        case XML_WRONG_ATTRIBUTE_TYPE:
            throw new ApplicationException( __FILE__, __LINE__, "invalid value '%s' for integer attribute \"srcId\" in element <%s>", pel->Value(), pel->Name() );
            break;

        default:        // XML_NO_ATTRIBUTE
            break;

    }

    // get the optional subId
    switch( pel->QueryIntAttribute( "subId", &val ) )
    {
        case XML_NO_ERROR:
            if( val > AriocDS::SqId::MaxSubId )
                throw new ApplicationException( __FILE__, __LINE__, "invalid value %d for attribute \"subId\" in element <%s> (maximum %d)", val, pel->Name(), AriocDS::SqId::MaxSubId );
            subId = static_cast<UINT8>(val);
            break;

        case XML_WRONG_ATTRIBUTE_TYPE:
            throw new ApplicationException( __FILE__, __LINE__, "invalid value '%s' for integer attribute \"subId\" in element <%s>", pel->Value(), pel->Name() );
            break;

        default:        // XML_NO_ATTRIBUTE
//            throw new ApplicationException( __FILE__, __LINE__, "missing attribute in element <%s>: subId", pel->Name() );
            break;
    }

    // get the readId range
    readIdFrom = InputFileGroup::DefaultReadIdFrom;
    readIdTo = InputFileGroup::DefaultReadIdTo;

    const char* rir = pel->Attribute( "readId" );
    if( rir != NULL )
    {
        static const char validChars[] = "*0123456789";
        static const char errFmt[] = "invalid readId range specification: %s";

        // parse the range of readId values (no particular error checking here!)
        size_t sepAt = strspn( rir, validChars );
        errno = 0;
        if( sepAt < (strlen( rir )-1) )
        {
            char* endptr = NULL;
            if( *rir != '*' )
            {
                readIdFrom = strtoll( rir, &endptr, 10 );
                if( errno != 0 )
                    throw new ApplicationException( __FILE__, __LINE__, errFmt, rir );
            }

            const char* p2 = rir + sepAt + 1;
            if( *p2 != '*' )
            {
                readIdTo = strtoll( p2, &endptr, 10 );
                if( errno != 0 )
                    throw new ApplicationException( __FILE__, __LINE__, errFmt, rir );
            }
        }
        else
            throw new ApplicationException( __FILE__, __LINE__, errFmt, rir );

        // sanity check
        if( readIdFrom > readIdTo )
            throw new ApplicationException( __FILE__, __LINE__, errFmt, rir );
    }

    // get the data-source info string (or null)
    const char* pval = pel->Attribute( "srcInfo" );
    if( pval )
    {
        // trim trailing whitespace
        const char* pEnd = pval + strlen( pval ) - 1;
        while( iswspace( *pEnd ) && (--pEnd >= pval) );

        // trim leading whitespace
        const char* pStart = pval;
        while( iswspace( *pStart ) && (++pStart < pEnd) );

        // copy the trimmed string
        char* p = srcInfo;
        while( pStart <= pEnd )
            *(p++) = *(pStart++);
        *p = 0;
    }
    else
        srcInfo[0] = 0;     // return an empty string
}

/// [protected] method buildInputFilename
void AriocAppMainBase::buildInputFilename( char* inputFilename, const char* baseName )
{
    // start with a copy of the specified "base name"
    size_t cchBaseName = strlen( baseName );
    memcpy_s( inputFilename, FILENAME_MAX, baseName, cchBaseName+1 );

    // ensure that the filename ends with "$a21.sbf"
    static const char* suffix = "$a21.sbf";
    bool doAppend = (cchBaseName <= 8);
    if( !doAppend )
        doAppend = (0 != _strcmpi( inputFilename+cchBaseName-8, suffix ));
    if( doAppend )
        strcat_s( inputFilename, FILENAME_MAX, suffix );
}

/// [protected] method decodeMDformatType
const char* AriocAppMainBase::decodeMDformatType( MDfmtType _mft )
{
    static const char* decodes[] = { "standard", "compact" };

    switch( _mft )
    {
        case mftStandard:
            return decodes[0];

        case mftCompact:
            return decodes[1];

        default:
            return "???";
    }
}

/// [protected] method decodeCIGARformatType
const char* AriocAppMainBase::decodeCIGARformatType( CIGARfmtType _cft )
{
    static const char* decodes[] = { "=XIDS", "MIDS", "MID" };

    switch( _cft )
    {
        case cftQXIDS:
            return decodes[0];

        case cftMIDS:
            return decodes[1];

        case cftMID:
            return decodes[2];

        default:
            return "???";
    }
}
#pragma endregion

#pragma region virtual method implementations
/// [protected] method parseXmlElements
void AriocAppMainBase::parseXmlElements()
{
    // point to the XML elements in the file
    m_pelRoot = m_xmlDoc.FirstChildElement();       // (root element of the XML document)
    m_pelX = m_pelRoot->FirstChildElement( "X" );
    m_pelR = m_pelRoot->FirstChildElement( "R" );
    m_pelNongapped = m_pelRoot->FirstChildElement( "nongapped" );
    m_pelGapped = m_pelRoot->FirstChildElement( "gapped" );
    m_pelQ = m_pelRoot->FirstChildElement( "Q" );
    m_pelA = m_pelRoot->FirstChildElement( "A" );

    // sanity check
    if( strcmp( m_pelRoot->Name(), this->AppName ) ) throw new ApplicationException( __FILE__, __LINE__, "missing <%s> element", this->AppName );
    if( m_pelR == NULL )            throw new ApplicationException( __FILE__, __LINE__, "missing <R> element" );
    if( m_pelNongapped == NULL )    throw new ApplicationException( __FILE__, __LINE__, "missing <nongapped> element" );
    if( m_pelGapped == NULL )       throw new ApplicationException( __FILE__, __LINE__, "missing <gapped> element" );
    if( m_pelQ == NULL )            throw new ApplicationException( __FILE__, __LINE__, "missing <Q> element" );
    if( m_pelA == NULL )            throw new ApplicationException( __FILE__, __LINE__, "missing <A> element" );
}

/// [public] method LoadConfig
void AriocAppMainBase::LoadConfig()
{
    CDPrint( cdpCD0, " configuration file   : %s", this->ConfigFileName );
    LoadXmlFile( &m_xmlDoc, this->ConfigFileName );
}
#pragma endregion
