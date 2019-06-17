/*
  AppMain.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member definitions
const char AppMain::RGID_PLACEHOLDER[] = "#";
const char* AppMain::m_rgAts = "ID CN DS DT FO KS LB PG PI PL PU SM";
const char* AppMain::m_rgAtNull = "{7A3281BF-38FC-4E4A-BEEF-158520D26B38}";
#pragma endregion

#pragma region constructor/destructor
/// <summary>
/// default constructor
/// </summary>
AppMain::AppMain()
{
}

/// destructor
AppMain::~AppMain()
{
}
#pragma endregion

#pragma region private methods
/// [private] method computeMbPerSecond
double AppMain::computeMbPerSecond( INT64 cb, INT32 ms )
{
	if( ms == 0 )
		return 0.0;

	return static_cast<double>(cb) / (1024.0 * 1024.0 * ms / 1000.0);
}

/// [private] method isReadGroupPattern
bool AppMain::isReadGroupPattern( const char* s )
{
    // We are determining whether the specified string represents a read-group pattern that can be used
    //  to parse read-group information from a FASTQ defline.  Some examples:
    //
    //      (*)
    //      (*) *(/*)
    //      *
    //      * (*:*):
    //      * (*:*:*:*):
    //      *=(*|*|*|*)|
    //
    // It's conceivable that a pattern might contain alphanumeric characters, e.g.:
    //
    //      (*)ACGTTT
    //      SPL(*)
    //
    //  but it seems very unlikely; in any event such patterns could be rewritten with the * wildcard anyway.
    //  So to determine whether we have a pattern, we use the following heuristic: we assume it's a pattern if
    //  the string starts ends with a parenthesis, asterisk, colon, vertical bar, or dot.
    //

    // sanity check
    if( !s || !(*s) )
        throw new ApplicationException( __FILE__, __LINE__, "null or missing read-group ID specification" );

    // return true if the specified string is inferred to represent a Q-sequence metadata pattern
    size_t cb = strlen( s );
    return (strchr("(*:|.",s[0]) != NULL) || (strchr(")*:|.",s[cb-1]) != NULL);
}

/// [private] method saveReadGroupInfo
bool AppMain::saveReadGroupInfo( InputFileGroup::FileInfo* pfi, tinyxml2::XMLElement* pelRg, tinyxml2::XMLElement* pelFile )
{
    // if no read group ID is specified in either the <rg> element or the <file> element...
    if( ((pelRg == NULL) || (pelRg->Attribute("ID") == NULL)) && (pelFile->Attribute("ID") == NULL) )
    {
        // there is no read group info associated with the specified input file
        return false;
    }

    /* At this point there is read group information in the <rg> element and/or the <file> element */
    
    // build a simple XML document that represents the read group info
    tinyxml2::XMLDocument docRG;
    docRG.Parse( "<RG />" );
    tinyxml2::XMLElement* pelRoot = docRG.RootElement();

    /* Add placeholder values to instantiate the order in which the attributes appear:
        - For human readability, we want the ID attribute to appear first in the list.  However,
           attribute order is not part of the XML specification, so we do our best to preserve it here
           by placing it first in the list of potential attributes and depending on the internals of the
           tinyxml implementation to emit the attributes in the order in which they are presented to
           the SetAttribute API.
    */
    for( INT32 i=0; i<static_cast<INT32>(strlen(m_rgAts)); i+=3 )
    {
        char atName[3];
            
        // extract the 2-character read group attribute name from the static list
        atName[0] = m_rgAts[i];
        atName[1] = m_rgAts[i+1];
        atName[2] = 0;

        // add the attribute (with a null value) to the root element
        pelRoot->SetAttribute( atName, m_rgAtNull );
    }


#if TODO_CHOP_WHEN_DEBUGGED
    {
        tinyxml2::XMLPrinter lpt;
        docRG.Print( &lpt );
        CDPrint( cdpCD0, "%s: %s", __FUNCTION__, lpt.CStr() );
    }
#endif

    // replace placeholders with actual attribute values in the <rg> and <file> elements
    const tinyxml2::XMLAttribute* pat;

    if( pelRg )
    {
        // get read group attributes from the <rg> element
        pat = pelRg->FirstAttribute();
        while( pat )
        {
            const char* atName = pat->Name();
            if( strstr( m_rgAts, atName ) )
                pelRoot->SetAttribute( atName, pat->Value() );

            pat = pat->Next();
        }
    }

    // get read group attributes from the nth <file> element; duplicates override attributes from the <rg> element
    pat = pelFile->FirstAttribute();
    while( pat )
    {
        const char* atName = pat->Name();
        if( strstr( m_rgAts, atName ) )
            pelRoot->SetAttribute( atName, pat->Value() );

        pat = pat->Next();
    }

    // excise null attributes
    pat = pelRoot->FirstAttribute();
    do
    {
        /* The following assumes a little bit of "magic", i.e., DeleteAttribute() does a simple excision
            from a forward-chained list of attributes, so the attribute order persists when a null attribute
            is deleted. */
        const tinyxml2::XMLAttribute* patNext = pat->Next();
        if( 0 == strcmp( pat->Value(), m_rgAtNull ) )
            pelRoot->DeleteAttribute( pat->Name() );
        pat = patNext;
    }
    while( pat );

    // sanity check
    const char* pqmdRG = pelRoot->Attribute( "ID" );
    if( (pqmdRG == NULL) || (*pqmdRG == 0) )
        throw new ApplicationException( __FILE__, __LINE__, "%s: missing read group ID", __FUNCTION__ );

#if TODO_CHOP_WHEN_NEW_STUFF_WORKS
    // optionally save the user-specified read-group ID pattern
    if( isReadGroupPattern( pqmdRG ) )
    {
        // save the pattern
        pfi->QmdRG.Realloc( strlen( pqmdRG )+1, false );
        strcpy_s( pfi->QmdRG.p, pfi->QmdRG.cb, pqmdRG );
        pfi->QmdRG.n = true;

        // replace the value of the ID attribute
        pelRoot->SetAttribute( "ID", RGID_PLACEHOLDER );
    }
#endif

    // save the read group ID (either a pattern or a literal string)
    pfi->QmdRG.Realloc( strlen( pqmdRG )+1, false );
    strcpy_s( pfi->QmdRG.p, pfi->QmdRG.cb, pqmdRG );
    pfi->QmdRG.n = true;

    if( isReadGroupPattern( pqmdRG ) )
    {
        // replace the pattern with a placeholder; it will be updated in SAMConfigWriter::AppendReadGroupInfo()
        pelRoot->SetAttribute( "ID", RGID_PLACEHOLDER );
    }
    else
    {
        // track the number of read group IDs declared in the .cfg file (i.e., index the list of RG IDs)
        pfi->RGOrdinal = this->RGMgr.RGOrdinal++;
    }

    // build an XML-formatted string
    tinyxml2::XMLPrinter lpt;
    docRG.Print( &lpt );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: %s", __FUNCTION__, lpt.CStr() );
#endif

    // add the XML-formatted read group information to a buffer
    UINT32 cb = static_cast<UINT32>(lpt.CStrSize());        // CStrSize includes the null terminal byte
    UINT32 cbTotal = this->RGMgr.RG.n + cb;
    if( cbTotal > this->RGMgr.RG.cb )
        this->RGMgr.RG.Realloc( cbTotal+512, true );
    memcpy_s( this->RGMgr.RG.p+this->RGMgr.RG.n, this->RGMgr.RG.cb-this->RGMgr.RG.n, lpt.CStr(), cb );

    // save a reference to the XML string (containing read group information)
    if( this->RGMgr.OfsRG.Count <= static_cast<size_t>(pfi->Index) )
        this->RGMgr.OfsRG.Realloc( pfi->Index+1, true );
    this->RGMgr.OfsRG.p[pfi->Index] = this->RGMgr.RG.n;

    this->RGMgr.RG.n = cbTotal;

    // return true to indicate that read group info was saved
    return true;
}
#pragma endregion

#pragma region virtual method implementations
/// [protected] method parseXmlElements
void AppMain::parseXmlElements()
{
    // point to the XML elements in the configuration file
    m_pelRoot = m_xmlDoc.RootElement();       // (root element of the XML document)
    m_pelX = m_pelRoot->FirstChildElement( "X" );
    m_pelDataIn = m_pelRoot->FirstChildElement( "dataIn" );
    m_pelDataOut = m_pelRoot->FirstChildElement( "dataOut" );

    // sanity check
    if( (m_pelRoot == NULL) || strcmp( m_pelRoot->Name(), this->AppName ) )
        throw new ApplicationException( __FILE__, __LINE__, "missing <%s> element", this->AppName );

    if( m_pelDataIn == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "missing <dataIn> element" );

    if( m_pelDataOut == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "missing <dataOut> element" );
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Parses the config file, launches the application, and displays the results
/// </summary>
void AppMain::Launch()
{
    using namespace tinyxml2;

    /* parse the config file */

    parseXmlElements();

    // parse the attributes of the <AriocE> element
    const char* psi = const_cast<char*>(m_pelRoot->Attribute( "seed" ));

    INT32 seedInterval = (psi ? 1 : 0);                             // default value = 1 (if a seed is specified) or 0 (if no seed is specified)
    m_pelRoot->QueryIntAttribute( "seedInterval", &seedInterval);   // if "seedInterval" is not specified, the default value is used
    
    // sanity check
    if( psi == NULL )
    {
        if( seedInterval )
            throw new ApplicationException( __FILE__, __LINE__, "missing attribute in element <AriocE>: seed" );
    }

    A21SpacedSeed a21ss( psi );
    a21ss.seedInterval = static_cast<INT16>(seedInterval);
    A21HashedSeed a21hs( psi );
    a21hs.seedInterval = static_cast<INT16>(seedInterval);

    // sanity check: verify that the seed interval is zero for a null seed index
    if( a21ss.IsNull() && a21hs.IsNull() && seedInterval )
        throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value seed=\"%s\" for seedInterval=%d in element <AriocE>", psi, seedInterval );

    UINT32 gpuMask = 0;                                                 // default value: no GPU
    const char* pGpuMask = m_pelRoot->Attribute( "gpuMask" );
    if( pGpuMask )
        sscanf_s( pGpuMask, "%*2s%08X", &gpuMask );                     // (the gpuMask value is formatted as an 8-character hex string, preceded by "0x")

    // sanity check
    if( gpuMask )
    {
        // filter the set of available CUDA devices
        CudaDeviceBinding::DeviceFilter_MinComputeCapability( CUDAMINCC );
        CudaDeviceBinding::DeviceFilter_DeviceIdMap( gpuMask );

        // do first-time CUDA initialization on the available devices
        CudaDeviceBinding::InitializeAvailableDevices();

        // get the number of usable CUDA devices
        INT32 nGPUs = CudaDeviceBinding::GetAvailableDeviceCount();

        // sanity check
        if( nGPUs == 0 )
            throw new ApplicationException( __FILE__, __LINE__, "no GPUs with minimum compute capability %s and gpuMask=0x%08x", CUDAMINCC, gpuMask );
    }

    INT32 maxDOP = _I32_MAX;                                            // default value = maximum integer value
    m_pelRoot->QueryIntAttribute( "maxDOP", &maxDOP );                  // if "maxDOP" is not specified, the default value is used
    if( maxDOP == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value for maxDOP in element <AriocE>" );

    INT32 maxJ = parseInt32GMK( m_pelRoot->Attribute("maxJ"), _I32_MAX );   // if "maxJ" is not specified, the default value is the maximum 32-bit signed integer

    bool emitKmers = 0;                                                 // default value = 0
    m_pelRoot->QueryBoolAttribute( "emitKmers", &emitKmers );           // if "emitKmers" is not specified, the default value is used

    const char* pVerboseMask = m_pelRoot->Attribute( "verboseMask" );
    if( pVerboseMask )
        sscanf_s( pVerboseMask, "%*2s%08X", reinterpret_cast<UINT32*>(&CDPrintFilter) );    // (the verboseMask value is formatted as an 8-character hex string, preceded by "0x")
    else
        CDPrintFilter = static_cast<CDPrintFlags>(cdpCD0|cdpCD1);

    // parse the <X> element
    parseX();

    // parse the attributes of the <dataIn> element
    const char* pSequenceType = m_pelDataIn->Attribute( "sequenceType" );
    if( pSequenceType == NULL )
        throw new ApplicationException( __FILE__, __LINE__, "missing attribute in element <dataIn>: sequenceType" );
    char sequenceType = toupper(*pSequenceType);
    if( (sequenceType != 'R') && (sequenceType != 'Q') )
        throw new ApplicationException( __FILE__, __LINE__, "invalid or missing attribute in element <dataIn>: sequenceType (valid values: R, Q)" );
    const Nencoding enc = (sequenceType == 'R') ? NencodingR : NencodingQ;

    INT32 srcId = 0;                                                            // default value = 0
    m_pelDataIn->QueryIntAttribute( "srcId", &srcId );                          // if "srcId" is not specified, the default value is used
    if( srcId > AriocDS::SqId::MaxSrcId )
        throw new ApplicationException( __FILE__, __LINE__, "invalid value %d for attribute srcId in element (valid values: 0-%d)", srcId, AriocDS::SqId::MaxSrcId );

    char* pQNAME = const_cast<char*>(m_pelDataIn->Attribute( "QNAME" ));        // if the "QNAME" attribute is not specified, the default pointer value is null
    
    // sanity check
    if( pQNAME && (enc != NencodingQ) )
        throw new ApplicationException( __FILE__, __LINE__, "attribute QNAME may only be specified for query sequences (reads)") ;

    double samplingRatio = 1.0;                                                 // default value = 1.0
    m_pelDataIn->QueryDoubleAttribute( "samplingRatio", &samplingRatio);        // if "samplingRatio" is not specified, the default value is used

    INT32 qualityScoreBias = -1;                                                // default value = -1 (unknown)
    m_pelDataIn->QueryIntAttribute( "qualityScoreBias", &qualityScoreBias );    // if "qualityScoreBias" is not specified, the default value is used

    const char* filePath = m_pelDataIn->Attribute( "filePath" );
    const char* uriPath = m_pelDataIn->Attribute( "uriPath" );

    // look for the <rg> subelement of the <dataIn> element
    XMLElement* pelRg = m_pelDataIn->FirstChildElement( "rg" );

    // parse the <file> subelements of the <dataIn> element
    INT32 nInputFiles = 0;
    XMLElement* pelFile = m_pelDataIn->FirstChildElement( "file" );
    while( pelFile )
    {
        // count the number of raw input sequence files and verify the presence of the subId attribute on each <file> element
        INT32 subId = -1;
        if( XML_NO_ATTRIBUTE == pelFile->QueryIntAttribute( "subId", &subId ) )
            throw new ApplicationException( __FILE__, __LINE__, "missing attribute in element <file>: subId" );
        if( (subId < 0) || (subId > static_cast<INT32>(AriocDS::SqId::MaxSubId)) )
            throw new ApplicationException( __FILE__, __LINE__, "subId value %d in element <file> is out of bounds (valid values: 0-%d)", subId, AriocDS::SqId::MaxSubId );

        ++nInputFiles;

        pelFile = pelFile->NextSiblingElement( "file" );
    }

    // copy the list of input sequence files to a buffer
    InputFileGroup ifgRaw( filePath, uriPath );
    ifgRaw.Init( nInputFiles );
    pelFile = m_pelDataIn->FirstChildElement( "file" );
    bool hasReadGroupInfo = false;
    for( INT32 n=0; n<nInputFiles; n++ )
    {
        INT32 subId = pelFile->IntAttribute( "subId" );             // get the nth subunit ID
        INT32 mateId = 0;
        pelFile->QueryIntAttribute( "mate", &mateId );                  // if a "mate" attribute is specified, get the nth mate ID (which must be a positive integer)
        const char* uriPathForFile = pelFile->Attribute( "uriPath" );   // if a "uriPath" attribute is specified, get a pointer to it
        ifgRaw.Append( pelFile->GetText(), srcId, subId, InputFileGroup::DefaultReadIdFrom, InputFileGroup::DefaultReadIdTo, mateId, uriPathForFile );

        hasReadGroupInfo |= saveReadGroupInfo( ifgRaw.InputFile.p+n, pelRg, pelFile );

        pelFile = pelFile->NextSiblingElement( "file" );            // point to the next <file> element in the .cfg file
    }

    // clean the pair IDs (if any) and reorder the list by subunit ID
    ifgRaw.CleanMateIds();

    // sanity check
    if( pelRg && !hasReadGroupInfo )
        throw new ApplicationException( __FILE__, __LINE__, "unreferenced <rg> element (missing read group ID)" );

    if( enc != NencodingQ )
    {
        if( ifgRaw.HasPairs )
            throw new ApplicationException( __FILE__, __LINE__, "paired-end attribute may only be specified for query sequences (reads)" );

        if( hasReadGroupInfo )
            throw new ApplicationException( __FILE__, __LINE__, "read group info may only be specified for query sequences (reads)" );
    }

    for( INT32 n=0; n<nInputFiles; n++ )
    {
        char buf[16] = { 0 };
        if( ifgRaw.InputFile.p[n].IsPaired )
            sprintf_s( buf, sizeof buf, " mateId %d", ifgRaw.InputFile.p[n].MateId );
        CDPrint( cdpCD2, " input file %3d     : (%d.%d%s) %s", n, srcId, ifgRaw.InputFile.p[n].SubId, buf, ifgRaw.InputFile.p[n].Filespec );
    }

    // parse the <path> subelement of the <dataOut> element
    XMLElement* pelPath = m_pelDataOut->FirstChildElement( "path" );
    char* outDir = const_cast<char*>(pelPath->GetText());

    // echo the configuration parameters
    CDPrint( cdpCD2, " data source ID     : %d", srcId );
    CDPrint( cdpCD2, " raw sequence       : (%d files)", nInputFiles );
    if( !a21hs.IsNull() )
    {
        CDPrint( cdpCD2, " seed               : %s", a21hs.IdString );
        CDPrint( cdpCD2, "  hashSeedWidth     : %d", a21hs.seedWidth );
        CDPrint( cdpCD2, "  hashKeyWidth      : %d", a21hs.hashKeyWidth );
        CDPrint( cdpCD2, "  maxMismatches     : %d", a21hs.maxMismatches );
        CDPrint( cdpCD2, "  seedInterval      : %d", a21hs.seedInterval );
    }
    else
    if( !a21ss.IsNull() )
    {
        CDPrint( cdpCD2, " seed               : %s", a21ss.IdString );
        CDPrint( cdpCD2, "  hashSeedWidth     : %d", a21ss.seedWidth );
        CDPrint( cdpCD2, "  hashKeyWidth      : %d", a21ss.hashKeyWidth );
        CDPrint( cdpCD2, "  maxMismatches     : %d", a21ss.maxMismatches );
        CDPrint( cdpCD2, "  seedInterval      : %d", a21ss.seedInterval );
    }
    else
        CDPrint( cdpCD2, " seed               : (none)" );

    CDPrint( cdpCD2, " N encoding         : %c", (enc == NencodingR) ? 'R' : 'Q');
    CDPrint( cdpCD2, " output directory   : %s", outDir );
    CDPrint( cdpCD2, " sampling ratio     : %4.2f", samplingRatio );
    if( enc == NencodingQ )
    {
        char buf[24] = { 0 };
        if( qualityScoreBias < 0 )
            strcpy_s( buf, sizeof buf, "(unspecified)" );
        else
            sprintf_s( buf, sizeof buf, "%d", qualityScoreBias );
        CDPrint( cdpCD2, " quality score bias : %s", buf );
        CDPrint( cdpCD2, " QNAME pattern      : %s", pQNAME );
    }

    const char* fmt = (maxDOP == _I32_MAX) ? " max DOP            : *" :
                                             " max DOP            : %d";
    CDPrint( cdpCD2, fmt, maxDOP );

    if( enc == NencodingR )
    {
        fmt = (maxJ == _I32_MAX) ? " maxJ               : *" :
                                   " maxJ               : %d";
        CDPrint( cdpCD2, fmt, maxJ );
        CDPrint( cdpCD2, " emitKmers          : %d (%s)", emitKmers, emitKmers ? "yes" : "no" );
    }

    try
    {
        // set up an AriocE instance for the specified parameters
        A21SeedBase* pa21sb = a21ss.IsNull() ? (A21SeedBase*)&a21hs : (A21SeedBase*)&a21ss;
        AriocE ae( srcId, ifgRaw, enc, pa21sb, outDir, maxDOP, samplingRatio, qualityScoreBias, emitKmers, gpuMask, pQNAME, maxJ, this );

        // examine the input-file contents to determine the data format
        SqFileFormat sff = ae.SniffSqFile( ifgRaw.InputFile.p[0].Filespec );
        switch( sff )
        {
            case SqFormatFASTA:
                break;

            case SqFormatFASTQ:
                qualityScoreBias = ae.Params.QualityScoreBias;
                break;

            default:
                throw new ApplicationException( __FILE__, __LINE__, "unable to identify sequence file format for %s", ifgRaw.InputFile.p[0].Filespec );
        }
    
        /* Encode the sequence data:
            - for reference sequences (NencodingR):
                - each input file is encoded as a pair of output files (Arioc.sbf, Arioc.rc.sbf)
                - one set of lookup tables is generated using the specified seedParam
            - for query sequences (NencodingQ):
                - each input file is encoded as a corresponding output file (Arioc.sbf)
            - each Arioc file has corresponding metadata (sqm.sbf) and raw (raw.sbf) files; for FASTQ input, a quality-score (sqq.sbf) file is also generated
        */
        if( enc == NencodingR )
            ae.ImportR();
        else
            ae.ImportQ();
    }
    catch( ApplicationException* pex )
    {
        exit( pex->Dump() );
    }

    static const double bytesPerGb = 1024*1024*1024;

    CDPrint( cdpCD2, "-------------------" );
    CDPrint( cdpCD2, "sequence import complete" );
    CDPrint( cdpCD2, " concurrent threads             : %d", AriocE::PerfMetrics.nConcurrentThreads );
    CDPrint( cdpCD2, " input files                    : %d", AriocE::PerfMetrics.nInputFiles );
    CDPrint( cdpCD2, " input file format              : %s", AriocE::PerfMetrics.InputFileFormat );
    if( qualityScoreBias > 0 )
        CDPrint( cdpCD2, " quality score bias             : %d", qualityScoreBias );
    CDPrint( cdpCD2, " input sequences read           : %10lld", AriocE::PerfMetrics.nSqIn );
    CDPrint( cdpCD2, " input sequences encoded        : %10lld (%4.2f%%)", AriocE::PerfMetrics.nSqEncoded, 100.0*AriocE::PerfMetrics.nSqEncoded/AriocE::PerfMetrics.nSqIn );
    CDPrint( cdpCD2, " symbols (bases) encoded        : %10lld", AriocE::PerfMetrics.nSymbolsEncoded );
    if( enc == NencodingR )
    {
        CDPrint( cdpCD2, " encoded kmers                  : %10lld", AriocE::PerfMetrics.nKmersUsed );
        CDPrint( cdpCD2, " rejected (N-containing) kmers  : %10lld", AriocE::PerfMetrics.nKmersIgnored );
        CDPrint( cdpCD2, " clamped J lists                : %10lld", AriocE::PerfMetrics.nJclamped );
        CDPrint( cdpCD2, " excluded J values              : %10lld", AriocE::PerfMetrics.nJexcluded );
    }

    if( enc == NencodingR )
    {
        CDPrint( cdpCD2, " null (unused) hash keys        : %10lld", AriocE::PerfMetrics.nNullH );

        CDPrint( cdpCD2, " data sizes" );
        CDPrint( cdpCD2, "  C table (J-list counts)       : %11lld bytes (%3.1fGb)", AriocE::PerfMetrics.cbC, AriocE::PerfMetrics.cbC/bytesPerGb );
        CDPrint( cdpCD2, "  H table (hash keys)           : %11lld bytes (%3.1fGb)", AriocE::PerfMetrics.cbH, AriocE::PerfMetrics.cbH/bytesPerGb );
        CDPrint( cdpCD2, "  J table (buckets)             : %11lld bytes (%3.1fGb)", AriocE::PerfMetrics.cbJ, AriocE::PerfMetrics.cbJ/bytesPerGb );
        CDPrint( cdpCD2, "  R table(s)                    : %11lld bytes (%3.1fGb)", AriocE::PerfMetrics.cbArioc, AriocE::PerfMetrics.cbArioc/bytesPerGb );

        CDPrint( cdpCD2, "-------------------" );
        CDPrint( cdpCD2, " elapsed times (ms)" );
        CDPrint( cdpCD2, "  read raw sequence data        : %7d", AriocE::PerfMetrics.usReadRaw/1000 );
        CDPrint( cdpCD2, "  read encoded sequence data    : %7d", AriocE::PerfMetrics.usReadA21/1000 );
        CDPrint( cdpCD2, "  sequence and kmer encoding    : %7d", AriocE::PerfMetrics.msEncoding );
        CDPrint( cdpCD2, "   count J values               : %7d", AriocE::PerfMetrics.msJlistSizes );
        CDPrint( cdpCD2, "   build J lists                : %7d", AriocE::PerfMetrics.msBuildJ );
        CDPrint( cdpCD2, "   sort J lists                 : %7d", AriocE::PerfMetrics.msSortJ );
        CDPrint( cdpCD2, "   sort big-bucket lists        : %7d", AriocE::PerfMetrics.msSortBB );
        CDPrint( cdpCD2, "   compact H table              : %7d", AriocE::PerfMetrics.msCompactHtable );
        CDPrint( cdpCD2, "   compact J table              : %7d", AriocE::PerfMetrics.msCompactJtable );
        CDPrint( cdpCD2, "   write encoded sequence data  : %7d", AriocE::PerfMetrics.msWriteSq );
        CDPrint( cdpCD2, "   write K table (encoded kmers): %7d", AriocE::PerfMetrics.msWriteKmers );
        CDPrint( cdpCD2, "  write C table (J-list counts) : %7d (%4.1fMb/s)", AriocE::PerfMetrics.msWriteC, computeMbPerSecond(AriocE::PerfMetrics.cbC,AriocE::PerfMetrics.msWriteC) );
        CDPrint( cdpCD2, "  write H table                 : %7d (%4.1fMb/s)", AriocE::PerfMetrics.msWriteH, computeMbPerSecond(AriocE::PerfMetrics.cbH,AriocE::PerfMetrics.msWriteH) );
        CDPrint( cdpCD2, "  write J table                 : %7d (%4.1fMb/s)", AriocE::PerfMetrics.msWriteJ, computeMbPerSecond(AriocE::PerfMetrics.cbJ,AriocE::PerfMetrics.msWriteJ) );
        CDPrint( cdpCD2, "  validate lookup tables        : %7d", AriocE::PerfMetrics.msValidateHJ );
        CDPrint( cdpCD2, "  validate subunit IDs          : %7d", AriocE::PerfMetrics.msValidateSubIds );
        CDPrint( cdpCD2, "  total                         : %7d", AriocE::PerfMetrics.msApp );
    }
    else
    {
        CDPrint( cdpCD2, " data sizes" );
        CDPrint( cdpCD2, "  encoded sequence data         : %11lld bytes (%3.1fGb)", AriocE::PerfMetrics.cbArioc, AriocE::PerfMetrics.cbArioc/bytesPerGb );

        CDPrint( cdpCD2, "-------------------" );
        CDPrint( cdpCD2, " elapsed times (ms)" );
        CDPrint( cdpCD2, "  sequence and kmer encoding    : %8d", AriocE::PerfMetrics.msEncoding );
        CDPrint( cdpCD2, "   write encoded sequence data  : %8d", AriocE::PerfMetrics.msWriteSq );
        CDPrint( cdpCD2, "   write K table (encoded kmers): %8d", AriocE::PerfMetrics.msWriteKmers );
        CDPrint( cdpCD2, "  total                         : %8d", AriocE::PerfMetrics.msApp );
    }
}

/// [public] method LoadConfig
void AppMain::LoadConfig()
{
    CDPrint( cdpCD0, " configuration file : %s", this->ConfigFileName );
    LoadXmlFile( &m_xmlDoc, this->ConfigFileName );
}
#pragma endregion
