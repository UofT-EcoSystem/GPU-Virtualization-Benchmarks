/*
  RGManager.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// [public] constructor
RGManager::RGManager() : RGOrdinal(0)
{
    memset( this->SBFfileSpec, 0, sizeof SBFfileSpec );
}

/// [public] destructor
RGManager::~RGManager()
{
}
#pragma endregion

#pragma region private methods
/// [private] method isRGforOrdinal
bool RGManager::isRGforOrdinal( tinyxml2::XMLElement* pelRG, UINT32 expectedOrdinal )
{
    const char* sOrdinal = pelRG->Attribute( "ordinal" );
    if( sOrdinal == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "missing 'ordinal=' attribute in <RG> element" );

    // return true if the specified XML element contains the specified value for its ordinal= attribute
    UINT32 actualOrdinal = _UI32_MAX;
    sscanf_s( sOrdinal, "%u", &actualOrdinal );
    return (expectedOrdinal == actualOrdinal);
}

/// [private] method getRGforOrdinal
tinyxml2::XMLElement* RGManager::getRGforOrdinal( tinyxml2::XMLElement* pelRG, UINT32 nextOrdinal )
{
    // start by assuming that the next sibling element contains the specified ordinal
    tinyxml2::XMLElement* pelRGnext = pelRG->NextSiblingElement();
    if( pelRGnext && isRGforOrdinal( pelRGnext, nextOrdinal ) )
        return pelRGnext;

    // since the next sibling RG element does not contain the specified ordinal, look at all of the sibling RG elements
    pelRGnext = pelRG->Parent()->FirstChildElement( "RG" );
    while( pelRGnext )
    {
        if( isRGforOrdinal( pelRGnext, nextOrdinal ) )
            return pelRGnext;

        pelRGnext = pelRGnext->NextSiblingElement();
    }

    /* At this point the specified ordinal does not exist in any of the RG elements. */
    return NULL;
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Loads read-group info
/// </summary>
void RGManager::LoadReadGroupInfo( InputFileGroup* pifg )
{
    using namespace tinyxml2;

    /* AriocE aggregates read-group info for all input files in a group, so read groups with identical
        IDs must be otherwise identical as well.  This should not be a problem for RG info extracted
        from Illumina FASTQ deflines, but it could happen with user-specified RG info.
    */

    // look for a .cfg file associated with each Q-sequence input file
    for( UINT32 i=0; i<pifg->InputFile.n; ++i )
    {
        char fileSpec[FILENAME_MAX];
        InputFileGroup::FileInfo* pfi = pifg->InputFile.p + i;

        // replace "$a21.sbf" with ".cfg"
        strcpy_s( fileSpec, FILENAME_MAX, pfi->Filespec );
        size_t ofsTo = strlen( fileSpec ) - 8;
        strcpy_s( fileSpec+ofsTo, (FILENAME_MAX-ofsTo), ".cfg" );

        // do nothing if there is no .cfg file
        if( !RaiiFile::Exists( fileSpec ) )
            continue;

        // load the XML from the file
        XMLDocument doc;
        LoadXmlFile( &doc, fileSpec );

        // look for read group info (i.e., one or more RG elements)
        XMLElement* pelRG = doc.RootElement()->FirstChildElement( "RG" );
        if( pelRG == NULL )
            return;

        /* Get the ordinal for the first <RG> element.  For read group IDs defined by a pattern, this is
            always zero (see AriocE::SAMConfigWriter::AppendReadGroupInfo()).  Otherwise, the value is
            associated with the order in which per-file RG IDs were encountered in the AriocE .cfg file,
            which is thus the order in which they are defined in the per-file .cfg files emitted by AriocE.
        */
        UINT32 ordinal = pelRG->IntAttribute( "ordinal" );
        while( pelRG )
        {
            // make a local copy of the read group XML for the ith input file
            XMLDocument docRG;
            XMLElement* pelRGcopy = pelRG->ShallowClone( &docRG )->ToElement();

            // excise the ordinal= attribute
            pelRGcopy->DeleteAttribute( "ordinal" );

            // if we have already encountered RG info for the current ordinal...
            if( ordinal < this->OfsRG.n )
            {
                // create an XML DOM representation of the previously-encountered read-group XML string
                char* pRGprev = this->RG.p+this->OfsRG.p[ordinal];
                XMLDocument docRGprev;
                docRGprev.Parse( pRGprev );
                XMLElement* pelRGprev = docRGprev.RootElement();

                // ensure that the attribute values match
                if( !SameXmlAttributes( pelRGcopy, pelRGprev ) )
                    throw new ApplicationException( __FILE__, __LINE__, "inconsistent read group info: ordinal=%u", ordinal );
            }

            else
            {
                // convert the XML element for the new RG info to plain XML text
                XMLPrinter lpt;
                lpt.OpenElement( "RG" );
                const XMLAttribute* patRG = pelRGcopy->FirstAttribute();
                while( patRG )
                {
                    lpt.PushAttribute( patRG->Name(), patRG->Value() );
                    patRG = patRG->Next();
                }
                lpt.CloseElement( true );                           // true: "compact mode" (i.e., no XML text formatting)
                const char* sRG = lpt.CStr();

                // save the read group XML in the lookup tables
                UINT32 cbRG = lpt.CStrSize();
                if( (this->RG.n + cbRG) > static_cast<UINT32>(this->RG.cb) )
                    this->RG.Realloc( this->RG.cb+16*cbRG, true );
                strcpy_s( this->RG.p+this->RG.n, this->RG.cb-this->RG.n, sRG );

                if( this->OfsRG.n == static_cast<UINT32>(this->OfsRG.Count) )
                    this->OfsRG.Realloc( this->OfsRG.Count+16, true );
                this->OfsRG.p[this->OfsRG.n++] = this->RG.n;        // save the offset to the ith RG XML string

                this->RG.n += cbRG;                                 // point past the ith RG XML string
            }

            // delete the local copy of the read group XML
            docRG.DeleteNode( pelRGcopy );

            ++ordinal;
            pelRG = getRGforOrdinal( pelRG, ordinal );
        }
    }

#ifdef _DEBUG
    // dump the RG LUT
    for( UINT32 n=0; n<this->OfsRG.n; ++n )
        CDPrint( cdpCD4, "%s: %d %s", __FUNCTION__, n, this->RG.p+this->OfsRG.p[n] );
#endif
}

/// [public] method SaveRGIDs
void RGManager::SaveRGIDs()
{
    using namespace tinyxml2;

    /* For each read-group XML string, we replace the XML element text with the following layout:
        bytes 0..1: string length
        bytes 2.. : null-terminated read-group ID string
    */
    for( UINT32 n=0; n<this->OfsRG.n; ++n )
    {
        // point to the read-group XML string
        char* pRGinfo = this->RG.p + this->OfsRG.p[n];
        size_t cbRGinfo = strlen( pRGinfo ) + 1;

        // isolate the ID attribute
        XMLDocument docRG;
        docRG.Parse( pRGinfo );
        XMLElement* pelRG = docRG.RootElement();
        const char* pRGID = pelRG->Attribute( "ID" );

        // copy the string length and string data
        *reinterpret_cast<INT16*>(pRGinfo) = static_cast<INT16>(strlen( pRGID ));
        strcpy_s( pRGinfo+sizeof( INT16 ), cbRGinfo-sizeof( INT16 ), pRGID );
    }
}

/// <summary>
/// Writes read-group info to a file in SQL bulk format
/// </summary>
void RGManager::WriteReadGroupInfo( WinGlobalPtr<OutputFileInfo>& ofi )
{
    // do nothing if there is no read-group info
    if( this->OfsRG.n == 0 )
        return;

    // place the RG.sbf file in the same directory as the first TSE output file
    char* filePath = NULL;
    for( size_t n=0; n<ofi.Count; ++n )
    {
        if( ofi.p[n].oft == oftTSE )
        {
            filePath = ofi.p[n].path;
            break;
        }
    }

    // do nothing if we're not generating TSE output
    if( filePath == NULL )
        return;

    // build the file specification
    strcpy_s( this->SBFfileSpec, sizeof this->SBFfileSpec, filePath );
    RaiiDirectory::AppendTrailingPathSeparator( this->SBFfileSpec );
    strcat_s( this->SBFfileSpec, sizeof this->SBFfileSpec, "RG.sbf" );

    /* Copy the read-group info to the output file in the following format:
        byte 0-7 :  sid (srcId/subId/iRGID)
        byte 8-9 :  len(ID)
        byte A-  :  ID string (NOT null terminated)
        [2 bytes]:  len(PL)
                    :  PL
        [2 bytes]:  len(SM)
                    :  SM

       This happens to be the binary format used to import and export SQL Server bulk data to a table with the following schema:
        sid   bigint        not null,     -- contains srcId/subId/iRGID
        ID    varchar(256)  not null,
        PL    varchar(32)   not null,
        SM    varchar(1024) not null
    */
    RaiiFile outFile( this->SBFfileSpec );
    WinGlobalPtr<UINT8> outBuf( 256, false );
    for( UINT32 iRGID=0; iRGID<this->OfsRG.n; ++iRGID )
    {
        // build a tiny XML DOM representation of the nth read-group info string
        tinyxml2::XMLDocument docRG;
        docRG.Parse( this->RG.p+this->OfsRG.p[iRGID] );
        tinyxml2::XMLElement* pelRG = docRG.RootElement();

        // get the attribute values
        const char* pID = pelRG->Attribute( "ID" );
        const char* pPL = pelRG->Attribute( "PL" );
        const char* pSM = pelRG->Attribute( "SM" );

        // sanity check
        if( !pID )
            throw new ApplicationException( __FILE__, __LINE__, "missing RG attribute: ID" );
        if( !pPL )
            throw new ApplicationException( __FILE__, __LINE__, "missing RG attribute: PL" );
        if( !pSM )
            throw new ApplicationException( __FILE__, __LINE__, "missing RG attribute: SM" );

        // get the number of bytes in each of the attribute value strings
        size_t cbID = strlen( pID );
        size_t cbPL = strlen( pPL );
        size_t cbSM = strlen( pSM );

        // grow the output buffer if necessary
        size_t cbRow = sizeof(INT64) + (cbID+2) + (cbPL+2) + (cbSM+2);
        if( cbRow > outBuf.cb )
            outBuf.Realloc( cbRow, false );

        // extract the srcId and subId from the element (which starts with the string "srcId.subId;")
        INT32 srcId = -1;
        INT32 subId = -1;
        sscanf_s( pID, "%d.%d", &srcId, &subId );

        // build the SBF row
        UINT8* p = outBuf.p;
        *reinterpret_cast<UINT64*>(p) = (static_cast<UINT64>(srcId) << 42) | (static_cast<UINT64>(subId) << 35) | iRGID;
        p += sizeof( UINT64 );

        *reinterpret_cast<INT16*>(p) = static_cast<INT16>(cbID);
        p += sizeof( INT16 );        
        memcpy_s( p, outBuf.cb-(p-outBuf.p), pID, cbID );
        p += cbID;

        *reinterpret_cast<INT16*>(p) = static_cast<INT16>(cbPL);
        p += sizeof( INT16 );
        memcpy_s( p, outBuf.cb-(p-outBuf.p), pPL, cbPL );
        p += cbPL;

        *reinterpret_cast<INT16*>(p) = static_cast<INT16>(cbSM);
        p += sizeof( INT16 );
        memcpy_s( p, outBuf.cb-(p-outBuf.p), pSM, cbSM );

#if TODO_CHOP_WHEN_DEBUGGED
        p += cbSM;
        if( static_cast<size_t>(p-outBuf.p) != cbRow )
            DebugBreak();
#endif

        // append to the SBF file
        outFile.Write( outBuf.p, cbRow );
    }

    outFile.Close();
}
#pragma endregion
