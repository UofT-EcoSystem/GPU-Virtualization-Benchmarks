/*
  SAMConfigWriter.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
    - We use the @SQ lines in the SAM header as follows:
        - SN: user-specified subunit ID as a fixed-length decimal value (e.g. "003"; see CONFIG_SUBID_FORMAT)
        - rm: reference metadata from FASTA or FASTQ sequence file (e.g. "gi|251831106|ref|NC_012920.1 ...")
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Writes an XML-formatted configuration file for the specified input file.
/// </summary>
/// <param name="_psip">Encoder parameters for this AriocE instance.</param>
/// <param name="_stub">Output file specification stub</param>
/// <param name="_baseName">Input file base name</param>
SAMConfigWriter::SAMConfigWriter( const AriocEncoderParams* _psip, const char* _stub, const char* _baseName ) : m_psip(_psip), m_doc(true)
{
    // save a copy of the config file name
    strcpy_s( m_configFileName, sizeof m_configFileName, _psip->ConfigFilespec );

    // build the output file specification
    strcpy_s( m_outputFileSpec, sizeof m_outputFileSpec, _stub );
    RaiiDirectory::AppendTrailingPathSeparator( m_outputFileSpec );
    strcat_s( m_outputFileSpec, sizeof m_outputFileSpec, _baseName );
    strcat_s( m_outputFileSpec, sizeof m_outputFileSpec, ".cfg" );

    // initialize the XML document
    initXmlTemplate( _baseName );
}

/// [public] destructor
SAMConfigWriter::~SAMConfigWriter()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initXmlTemplate
void SAMConfigWriter::initXmlTemplate( const char* baseFilename )
{
    // get the current date and time
    TOD tod;

    // in the command-tail string, replace double-quotes with the corresponding XML escape
    char cmdTail[FILENAME_MAX+32];
    char* pFrom = m_configFileName;
    char* pTo = cmdTail;
    while( *pFrom )
    {
        // if the character is not a double-quote ...
        if( *pFrom != '"' )
        {
            // copy one character
            *(pTo++) = *(pFrom++);
            continue;
        }

        /* at this point we need to replace a double-quote with an escape */

        strcpy_s( pTo, 7, "&quot;" );
        pTo += 6;
        pFrom++ ;
    }
    *pTo = 0;

    // build a disambiguated PG:ID
    char PGID[FILENAME_MAX];
    sprintf_s( PGID, sizeof PGID, "%s (%s)",
                m_psip->pam->AppName,
                m_psip->pa21sb->IsNull() ? baseFilename : m_psip->pa21sb->IdString );

    // initialize an XML document whose structure maps to a SAM file header
    WinGlobalPtr<char> buf( 2048, true );
    sprintf_s( buf.p, buf.cb,
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                "<SAM fn=\"%s\">"
                "  <HD VN=\"%s\" />"
                "  <PG ID=\"%s\" PN=\"AriocE\" VN=\"%s\" CL=\"%s\" dt=\"%4u-%02u-%02uT%02u:%02u:%02u\" />"
                "</SAM>",
                baseFilename,
                CONFIG_SAM_VERSION,
                PGID, m_psip->pam->AppVersion, cmdTail, tod.yr, tod.mo, tod.da, tod.hr, tod.mi, tod.se );
    m_doc.Parse( buf.p );
}

/// [private] method createSQ
tinyxml2::XMLElement* SAMConfigWriter::createSQ( const INT32 subId )
{
    using namespace tinyxml2;

    // create a new SQ element
    XMLElement* elSQ = m_doc.NewElement( "SQ" );

    // add an SN ("reference sequence name") attribute to the element; the attribute value is the subunit ID specified in the .cfg file for the encoder
    char buf[8];
    sprintf_s( buf, sizeof buf, CONFIG_SUBID_FORMAT, subId );
    elSQ->SetAttribute( "SN", buf );

    // add rm, UR, and LN attributes here in hopes of keeping the XML attributes in the same order in each SQ element
    elSQ->SetAttribute( "rm", "" );
    elSQ->SetAttribute( "UR", "" );
    elSQ->SetAttribute( "LN", 0 );

    return elSQ;
}

/// [private] method getSQ
tinyxml2::XMLElement* SAMConfigWriter::getSQ( const INT32 subId )
{
    using namespace tinyxml2;

    // build a subId string
    char buf[8];
    sprintf_s( buf, sizeof buf, CONFIG_SUBID_FORMAT, subId );

    // look for an SQ element with the specified subId as an attribute
    XMLElement* elSAM = m_doc.RootElement();
    XMLElement* elSQ = elSAM->FirstChildElement( "SQ" );

    // if there is no SQ element in the document ...
    if( elSQ == NULL )
    {
        // create an SQ element after the HD element
        XMLElement* elHD = elSAM->FirstChildElement( "HD" );
        XMLElement* elSQ = createSQ( subId );
        elSAM->InsertAfterChild( elHD, elSQ );
        return elSQ;
    }

    // look for the specified subId in the set of existing SQ elements; the list is maintained in ascending order of subId
    XMLElement* elPrev;
    do
    {
        const char* rm = elSQ->Attribute( "SN" );
        switch( strcmp( rm, buf ) )
        {
            case 0:                     // return the existing SQ element for the specified subId
                return elSQ;

            case 1:                    // the current existing SQ element's subId is greater than the specified subId
                // create a new SQ element before the current SQ element
                elPrev = elSQ->PreviousSiblingElement( "SQ" );
                if( elPrev == NULL )
                    elPrev = elSAM->FirstChildElement( "HD" );
                elSQ = createSQ( subId );
                elSAM->InsertAfterChild( elPrev, elSQ );
                return elSQ;

            default:                    // the current existing SQ element's subId is less than the specified subId
                break;
        }

        // iterate through the set of SQ elements
        elSQ = elSQ->NextSiblingElement( "SQ" );
    }
    while( elSQ != NULL );

    /* at this point there is no SQ element for the specified subId */

    // create a new SQ element after the last previous SQ element
    elSQ = createSQ( subId );
    elPrev = elSAM->LastChildElement( "SQ" );
    elSAM->InsertAfterChild( elPrev, elSQ );
    return elSQ;
}
#pragma endregion

#pragma region public methods
/// [public] method AppendReferenceMetadata
void SAMConfigWriter::AppendReferenceMetadata( const INT32 subId, const char* pMetadata )
{
    using namespace tinyxml2;

    RaiiCriticalSection<SAMConfigWriter> rcs;

    // get the SQ element
    XMLElement* elSQ = getSQ( subId );

    // update the rm ("reference metadata") attribute
    elSQ->SetAttribute( "rm", pMetadata );
}

/// [public] method AppendReferenceLength
void SAMConfigWriter::AppendReferenceLength( const INT32 subId, const INT64 cb )
{
    using namespace tinyxml2;

    RaiiCriticalSection<SAMConfigWriter> rcs;

    // get the SQ element
    XMLElement* elSQ = getSQ( subId );

    // update the LN attribute
    char buf[20];
    sprintf_s( buf, sizeof buf, "%lld", cb );
    elSQ->SetAttribute( "LN", buf );
}

/// [public] method AppendReferenceURI
void SAMConfigWriter::AppendReferenceURI( const INT32 subId, const char* uri )
{
    using namespace tinyxml2;

    RaiiCriticalSection<SAMConfigWriter> rcs;

    if( uri != NULL )
    {
        // get the SQ element
        XMLElement* elSQ = getSQ( subId );
        
        // add a UR attribute
        elSQ->SetAttribute( "UR", uri );
    }
}

/// [public] method AppendExecutionTime
void SAMConfigWriter::AppendExecutionTime( const INT32 ms )
{
    using namespace tinyxml2;

    // get the PG element
    XMLElement* elSAM = m_doc.FirstChildElement( "SAM" );
    XMLElement* elPG = elSAM->FirstChildElement( "PG" );

    // add a ms attribute
    elPG->SetAttribute( "ms", ms );
}

/// [public] method AppendMaxJ
void SAMConfigWriter::AppendMaxJ( const INT32 maxJ )
{
    using namespace tinyxml2;

    // get the PG element
    XMLElement* elSAM = m_doc.FirstChildElement( "SAM" );
    XMLElement* elPG = elSAM->FirstChildElement( "PG" );

    // add a mJ attribute
    if( maxJ == _I32_MAX )
        elPG->SetAttribute( "mJ", "*" );
    else
        elPG->SetAttribute( "mJ", maxJ );
}

/// [public] method AppendQualityScoreBias
void SAMConfigWriter::AppendQualityScoreBias( const INT32 qsb )
{
    using namespace tinyxml2;

    // get the PG element
    XMLElement* elSAM = m_doc.FirstChildElement( "SAM" );
    XMLElement* elPG = elSAM->FirstChildElement( "PG" );

    // add a qsb attribute
    char buf[24] = { 0 };
    sprintf_s( buf, sizeof buf, "%d", qsb );
    elPG->SetAttribute( "qsb", buf );
}

/// [public] method AppendReadGroupInfo
void SAMConfigWriter::AppendReadGroupInfo( RGManager* prgm, WinGlobalPtr<char>* pqmdRG, WinGlobalPtr<UINT32>* pofsqmdRG, InputFileGroup::FileInfo* pfi )
{
    if( prgm->RG.n == 0 )
        return;

    // create a new XML document for the <RG> XML associated with the specified input file
    tinyxml2::XMLDocument docRG;
    docRG.Parse( prgm->RG.p + prgm->OfsRG.p[pfi->Index] );

    // if we have one or more read groups extracted from FASTQ file deflines...
    // TODO: CHOP if( pofsqmdRG->n )
    if( pfi->RGOrdinal < 0 )
    {
        char rgid[512];
        INT32 rgOrdinal = 0;

        // save the read group IDs (extracted from the FASTQ file) as an ordered list of ID subelements
        for( UINT32 n=0; n<pofsqmdRG->n; ++n )
        {
            tinyxml2::XMLElement* pelRG = docRG.RootElement()->ShallowClone( &m_doc )->ToElement();

            // format the RG ID string as: <srcId>.<subId>;<extracted RG ID string>
            sprintf_s( rgid, sizeof rgid, "%d.%u;%s", pfi->SrcId, pfi->SubId, pqmdRG->p+pofsqmdRG->p[n] );
            pelRG->SetAttribute( "ID", rgid );

            /* Add an explicit ordinal (XML does not guarantee element ordering).
            
               The ordinal is an index into the list of read group IDs discovered by applying the pattern
                to the FASTQ deflines in all of the FASTQ files */
            pelRG->SetAttribute( "ordinal", rgOrdinal++ );

            // move the RG element into position in the XML document
            m_doc.RootElement()->InsertEndChild( pelRG );
        }
    }

    else    // we have a user-specified, per-file read group merged from the <rg> and <file> elements
    {
        // create a new RG element in the XML document that represents the SAM configuration
        tinyxml2::XMLNode* nodRG = docRG.RootElement()->ShallowClone( &m_doc );
        tinyxml2::XMLElement* pelRG = reinterpret_cast<tinyxml2::XMLElement*>(nodRG);

        // the ordinal is the per-file index into the list of read group information
        pelRG->SetAttribute( "ordinal", pfi->RGOrdinal );

        // move the RG element into position in the XML document
        m_doc.RootElement()->InsertEndChild( nodRG );
    }

#if TODO_CHOP_WHEN_DEBUGGED
    tinyxml2::XMLPrinter lpt;
    m_doc.Print( &lpt );
    OutputDebugStringA( lpt.CStr() );
#endif
}

/// [public] method Write
void SAMConfigWriter::Write()
{
#if TODO_CHOP_WHEN_DEBUGGED
    tinyxml2::XMLPrinter lpt;
    m_doc.Print( &lpt );
    OutputDebugStringA( lpt.CStr() );
#endif

    m_doc.SaveFile( m_outputFileSpec );
}
#pragma endregion
