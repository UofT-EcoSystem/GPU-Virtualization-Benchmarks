/*
  AppMain.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static variable definitions
AppMain::TumHeadingDesc AppMain::m_tumHeading[] =
                        { { "baseAlignG",   tumCandidatesPerSecond, "GPU: seed-and-extend gapped aligner (with traceback)" },
                          { "baseAlignN",   tumCandidatesPerSecond, "GPU: spaced-seed nongapped aligner" },
                          { "baseCountA",   tumDefault,             "GPU: count per-read mappings" },
                          { "baseFilterD",  tumDefault,             "GPU: filter mapped candidate D values" },
                          { "baseLoadJn",   tumCandidatesPerSecond, "CPU --> GPU: J values (nongapped aligner)" },
                          { "baseLoadJs",   tumCandidatesPerSecond, "CPU --> GPU: J values (gapped aligner)" },
                          { "baseLoadRi",   tumCandidatesPerSecond, "CPU --> GPU: load interleaved R sequence data" },
                          { "baseMapGc",    tumDefault,             "GPU: find seed-and-extend mappings (seed coverage priority)" },
                          { "baseMaxV",     tumCandidatesPerSecond, "GPU: seed-and-extend gapped aligner (Vmax only)" },
                          { "baseQReader",  tumDefault,             "file --> CPU: load Q sequences (reads)" },
                          { "tuAlignGs",    tumDefault,             "GPU: seed-and-extend gapped aligner" },
                          { "tuAlignN",     tumDefault,             "GPU: spaced-seed nongapped aligner" },
                          { "tuClassifyU",  tumDefault,             "CPU: classify unpaired mappings" },
                          { "tuFinalizeG",  tumDefault,             "CPU: post-process gapped mappings" },
                          { "tuFinalizeN",  tumDefault,             "CPU: post-process nongapped mappings" },
                          { "tuLoadM",      tumDefault,             "file --> CPU: load Q-sequence metadata" },
                          { "tuSetupG",     tumElapsed,             "GPU: setup for gapped alignment" },
                          { "tuSetupN",     tumElapsed,             "GPU: setup for nongapped alignment" },
                          { "tuWrite",      tumDefault,             "CPU: emit alignment results" },
                          { "tuXlatToD",    tumDefault,             "GPU: translate D values" },
                          { "tuComputeKMH", tumDefault,             "GPU: kmer hashing" }
                        };
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
/// [private] method initUnpairedInputFileGroup
void AppMain::initUnpairedInputFileGroup( InputFileGroup& ifgQ, tinyxml2::XMLElement* pelUnpaired0 )
{
    using namespace tinyxml2;

    INT32 nQfiles = 0;

    // count files and accumulate the total size of the file specifications
    XMLElement* pelUnpaired = pelUnpaired0;
    while( pelUnpaired )
    {
        // parse the <file> subelements of the <unpaired> element
        XMLElement* pelFile = pelUnpaired->FirstChildElement( "file" );
        if( pelFile == NULL )
            throw new ApplicationException( __FILE__, __LINE__, "empty <unpaired> element (at least one <file> elements must be specified)" );

        do
        {
            // count 'em
            nQfiles++;

            // move to the next <file> element
            pelFile = pelFile->NextSiblingElement( "file" );
        }
        while( pelFile );

        // move to the next <unpaired> element
        pelUnpaired = pelUnpaired->NextSiblingElement( "unpaired" );
    }

    // initialize the input file group
    ifgQ.Init( nQfiles );
    pelUnpaired = pelUnpaired0;
    while( pelUnpaired )
    {
        // get the srcId, subId, and readId range for the file pair
        INT32 srcId;
        UINT8 subId;
        INT64 readIdFrom, readIdTo;
        parsePairedUnpaired( pelUnpaired, srcId, subId, readIdFrom, readIdTo, this->DataSourceInfo );

        // get the file info
        XMLElement* pelFile = pelUnpaired->FirstChildElement( "file" );
        while( pelFile )
        {
            char inputFilename[FILENAME_MAX];
            buildInputFilename( inputFilename, pelFile->GetText() );    // build a filename as "*$a21.sbf"
            
            // for unpaired reads, different files may have different subIds
            INT32 val = subId;
            pelFile->QueryIntAttribute( "subId", &val );
            if( val > AriocDS::SqId::MaxSubId )
                throw new ApplicationException( __FILE__, __LINE__, "invalid value %d for subId= attribute in <file> element", val );
            subId = static_cast<UINT8>(val);

            ifgQ.Append( inputFilename, srcId, subId, readIdFrom, readIdTo, 0, NULL );
            pelFile = pelFile->NextSiblingElement( "file" );
        }

        pelUnpaired = pelUnpaired->NextSiblingElement( "unpaired" );
    }
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Parses the config file, launches the application, and displays the results
/// </summary>
void AppMain::Launch()
{
    using namespace tinyxml2;

    // point to the XML elements in the file
    parseXmlElements();

    /* parse the XML attributes */
    UINT32 gpuMask;
    INT32 maxDOP, batchSize;
    parseRoot( gpuMask, maxDOP, batchSize );
    parseX();

    const char* pathR = NULL;
    parseR( &pathR );

    INT32 maxAn, maxJn, seedCoverageN, maxMismatches;
    const char* pSSI;
    parseNongapped( pSSI, maxAn, maxJn, seedCoverageN, maxMismatches, m_xaNull );

    Wmxgs w;
    INT32 minPosSep, seedDepth, AtN, AtG, maxAg, maxJg;
    const char* pVt;
    const char* pHSI;
    parseGapped( pHSI, minPosSep, seedDepth, w, pVt, AtN, AtG, maxAg, maxJg, m_xaNull );

    XMLElement* pel;
    const char* filePathQ;
    parseQ( "unpaired", filePathQ, pel );

    bool overwriteOutputFiles;
    const char* pBaseNameA;
    const char* pBasePathA;
    INT64 maxAperOutputFile;
    INT32 mapqUnknown, mapqVersion, k;
    CIGARfmtType cft;
    MDfmtType mft;
    WinGlobalPtr<OutputFileInfo> ofi;
    AlignmentResultFlags arfSAM, arfSBF, arfTSE, arfKMH;
    parseA( pBaseNameA, pBasePathA, overwriteOutputFiles, maxAperOutputFile, mapqUnknown, mapqVersion, cft, mft, k, ofi, arfSAM, arfSBF, arfTSE, arfKMH, m_xaNull );

    // initialize seed parameters
    A21SpacedSeed a21ss( pSSI, maxMismatches );
    A21HashedSeed a21hs( pHSI );

    // sanity checks
    if( a21ss.IsNull() )
        throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value seed=\"%s\" in element <nongapped>", pSSI );
    if( a21hs.IsNull() )
        throw new ApplicationException( __FILE__, __LINE__, "invalid attribute value seed=\"%s\" in element <gapped>", pHSI );

    // initialize the query-sequence input file information
    InputFileGroup ifgQ( filePathQ, NULL );
    initUnpairedInputFileGroup( ifgQ, pel );
    ifgQ.CleanMateIds();
    ifgQ.ValidateSubunitIds();

    // initialize alignment scoring parameters
    ScoreFunctionType sft;
    double sfCoeff;
    double sfConst;
    AriocAlignmentScorer::StringToScoreFunction( pVt, sft, sfCoeff, sfConst );
    
    AlignmentScoreParameters asp( a21ss.maxMismatches, w, sft, sfCoeff, sfConst, a21hs.baseConvert );

    // build the alignment control parameters
    AlignmentControlParameters acp( maxJn, maxAn, seedCoverageN, AtN, AtG, maxJg, maxAg,
                                    seedDepth, minPosSep,
                                    0, 0, arfSAM, arfSBF, arfTSE, arfKMH, mapqUnknown, mapqVersion );

    // set up an AriocU instance for the specified parameters
    AriocU au( pathR, a21ss, a21hs, acp, asp,   // parameters for nongapped (spaced-seed) and gapped (seed-and-extend dynamic-programming) alignment
               &ifgQ,                           // parameters for Q sequences
               ofi,                             // parameters for alignment reporting
               gpuMask, maxDOP, batchSize, k, cft, mft,
               this );

    // echo the configuration parameters
    CDPrint( cdpCD0, " computer name        : %s", au.paamb->MachineName );

    CDPrintFlags cdp = cdpCD2;
    const char* fmt = (maxDOP == _I32_MAX) ? " max CPU threads      : *" :
                                             " max CPU threads      : %d";
    CDPrint( cdp, fmt, maxDOP );
    CDPrint( cdp, " number of GPUs       : %d (mask: 0x%08x)", __popcnt(gpuMask), gpuMask );
    fmt = (batchSize == _I32_MAX) ? " batch size           : (default)" :
                                    " batch size           : %s (%d)";
    CDPrint( cdp, fmt, m_pelRoot->Attribute("batchSize"), batchSize );
    CDPrint( cdp, " R sequence path      : %s", pathR );
    CDPrint( cdp, "  big-bucket threshold: %s", au.SAMhdb.BigBucketThreshold );
    const char* bc = (au.a21ss.baseConvert == A21SpacedSeed::bcCT) ? "CT" : "(none)";
    CDPrint( cdp, "  DNA base conversion : %s", bc );
    CDPrint( cdp, " nongapped aligner:" );
    CDPrint( cdp, "  seed                : %s", m_pelNongapped->Attribute("seed") );
    CDPrint( cdp, "   hashSeedWidth      : %d", a21ss.seedWidth );
    CDPrint( cdp, "   spacedSeedWeight   : %d", a21ss.spacedSeedWeight );
    CDPrint( cdp, "   maxMismatches      : %d", a21ss.maxMismatches );
    CDPrint( cdp, "   hashKeyWidth       : %d", a21ss.hashKeyWidth );
    CDPrint( cdp, "  maxA                : %d", acp.maxAn );
    fmt = (acp.maxJn == _I32_MAX) ? "  maxJ                : *" :
                                    "  maxJ                : %d";
    CDPrint( cdp, fmt, acp.maxJn );
    CDPrint( cdp, "  seedCoverage        : %d", acp.seedCoverageN );

    CDPrint( cdp, " gapped aligner:" );
    CDPrint( cdp, "  seed                : %s", m_pelGapped->Attribute("seed") );
    CDPrint( cdp, "   hashSeedWidth      : %d", a21hs.seedWidth );
    CDPrint( cdp, "   hashKeyWidth       : %d", a21hs.hashKeyWidth );
    CDPrint( cdp, "  mapqVersion         : %d", acp.mapqVersion );
    CDPrint( cdp, "  W scores/penalties  : %s", m_pelGapped->Attribute("Wmxgs") );
    CDPrint( cdp, "   Wm (match)         : %d", asp.Wm );
    CDPrint( cdp, "   Wx (mismatch)      : %d", asp.Wx );
    CDPrint( cdp, "   Wg (gap)           : %d", asp.Wg );
    CDPrint( cdp, "   Ws (gap space)     : %d", asp.Ws );
    CDPrint( cdp, "   Vt (threshold)     : %s", AriocAlignmentScorer::ScoreFunctionToString( asp.sft, asp.sfA, asp.sfB ) );
    fmt = (acp.maxJg == _I32_MAX) ? "  maxJ                : *" :
                                    "  maxJ                : %d";
    CDPrint( cdp, fmt, acp.maxJg );
    CDPrint( cdp, "  AtN, AtG            : %d,%d", acp.AtN, acp.AtG );
    CDPrint( cdp, "  maxA                : %d", acp.maxAg );
    CDPrint( cdp, "  seedDepth           : %d", acp.seedDepth );
    CDPrint( cdp, "  minPosSep           : %d", acp.minPosSep );

    CDPrint( cdp, " unpaired reads" );
    CDPrint( cdp, "  reporting           : %s", AriocAlignmentScorer::AlignmentResultFlagsToString( acp.arf, arfMaskReport ) );
    CDPrint( cdp, "  seed rev complement : %s", (a21ss.baseConvert||a21hs.baseConvert) ? "true" : "false" );

    CDPrint( cdp, " Q sequence files     : (%u files, unpaired)", ifgQ.InputFile.n );
    for( UINT32 n=0; n<ifgQ.InputFile.n; n++ )
    {
        char sRange[48] = "";
        INT32 rangeCase = ((ifgQ.InputFile.p[n].ReadIdFrom == InputFileGroup::DefaultReadIdFrom) ? 1 : 0) +
            ((ifgQ.InputFile.p[n].ReadIdTo == InputFileGroup::DefaultReadIdTo) ? 2 : 0);
        switch( rangeCase )
        {
        case 0:
            sprintf_s( sRange, sizeof sRange, " (%lld-%lld)", ifgQ.InputFile.p[n].ReadIdFrom, ifgQ.InputFile.p[n].ReadIdTo );
            break;

        case 1:
            sprintf_s( sRange, sizeof sRange, " (0-%lld)", ifgQ.InputFile.p[n].ReadIdTo );
            break;

        case 2:
            sprintf_s( sRange, sizeof sRange, " (%lld-*)", ifgQ.InputFile.p[n].ReadIdFrom );
            break;

        default:
            break;
        }

        CDPrint( cdp, "  %05d.%03u           : %s%s", ifgQ.InputFile.p[n].SrcId, ifgQ.InputFile.p[n].SubId, ifgQ.InputFile.p[n].Filespec, sRange );
    }

    const INT32 nOutputPaths = static_cast<INT32>(ofi.Count);
    CDPrint( cdp, " A (alignment output) : (%d path%s, %lld mappings per output file, output file base name: %s)",
                    nOutputPaths, PluralS(nOutputPaths),
                    maxAperOutputFile, pBaseNameA );
    for( INT32 n=0; n<nOutputPaths; ++n )
    {
        const char* pFormatType = (ofi.p[n].oft == oftSAM) ? "SAM" :
                                  (ofi.p[n].oft == oftSBF) ? "SBF" :
                                  (ofi.p[n].oft == oftTSE) ? "TSE" :
                                  (ofi.p[n].oft == oftKMH) ? "KMH" :
                                  "(***error***)";
        CDPrint( cdp, "  %-20s: %s (%s)", pFormatType,ofi.p[n].path, AriocAlignmentScorer::AlignmentResultFlagsToString(ofi.p[n].arf, arfMaskReport) );
    }

    if( *RGMgr.SBFfileSpec )
        CDPrint( cdp, "  read groups         : %s", RGMgr.SBFfileSpec );
    CDPrint( cdp, "  cigarFormat         : %s", decodeCIGARformatType(cft) );
    CDPrint( cdp, "  mdFormat            : %s", decodeMDformatType(mft) );

    if( arfKMH != arfNone )
        CDPrint( cdp, " K (for kmer hash)    : %d", k );

    // X parameters
    if( Xparam.Count )
    {
        CDPrint( cdp, " X parameters" );
        for( size_t n=0; n<Xparam.Count; ++n )
              CDPrint( cdp, "  %-20s: 0x%016llx (%lld)", Xparam.Key(n), Xparam.Value(n), Xparam.Value(n) );
    }

    // run the aligner
    au.Main();

    // emit performance metrics
    CDPrint( cdp, "" );
    CDPrint( cdp, "----------------------------" );
    AriocBase::aam.GPUmask = gpuMask;
    AriocBase::aam.sam.nCandidateQ = AriocBase::aam.n.Q = AriocBase::tum["tuClassifyU"].n.CandidateQ;
   
    CDPrint( cdp, "" );
    AriocBase::aam.Print( cdp, "Summary", 1, 1, 32 );

#if TODO_CHOP_WHEN_DEBUGGED
    for( UINT32 n=0; n<512; ++n )
        CDPrint( cdpCD0, "debugHistogram[%4d]: %u", n-256, debugHistogram[n] );

    CDPrint( cdpCD0, "debugMultipleBestMappings: %u", debugMultipleBestMappings );
    CDPrint( cdpCD0, "debugMultipleBestMappingsNonzeroMapq: %u", debugMultipleBestMappingsNonzeroMAPQ );
    CDPrint( cdpCD0, "debugVtooLow: %u", debugVtooLow );
    CDPrint( cdpCD0, "debugUniqueMapping: %u", debugUniqueMapping );
#endif

    // emit task unit metrics for which descriptive headings have been defined
    CDPrint( cdp, "" );
    for( INT32 ih=0; ih<static_cast<INT32>(arraysize(m_tumHeading)); ++ih )
    {
        // emit the ih'th heading
        CDPrint( cdp, " %s:", m_tumHeading[ih].description );

        for( UINT32 ii=0; ii<AriocBase::tum.Count; ++ii )
        {
            const char* key = AriocBase::tum.Key( ii );
            if( 0 == strncmp( m_tumHeading[ih].prefix, key, strlen(m_tumHeading[ih].prefix)) )
            {
                // emit the ii'th metric(s)
                AriocBase::tum.Value(ii).Print( cdp, NULL, m_tumHeading[ih].flags, 2, 1, 32 );
            }
        }
    }

    // emit task unit metrics for which no descriptive headings have been defined
    bool needMiscHeading = true;
    for( UINT32 ii=0; ii<AriocBase::tum.Count; ++ii )
    {
        // conditionally emit the ii'th task unit metric
        if( !AriocBase::tum.Value(ii).emitted )
        {
            if( needMiscHeading )
            {
                CDPrint( cdp, " Miscellaneous task unit metrics:" );
                needMiscHeading = false;
            }        
        
            AriocBase::tum.Value(ii).Print( cdp, NULL, tumDefault, 2, 1, 32 );
        }
    }

    CDPrint( cdp, " Task unit metrics                 : %u", AriocBase::tum.Count );

    // conditionally emit a TSE manifest file
    if( arfTSE != arfNone )
    {
        TSEManifest tsem( this, au.RefId, &ifgQ );

        tsem.AppendU64( "raQ", AriocBase::aam.n.Q );
        tsem.AppendDouble( "raQperSec", 1000.0 * AriocBase::aam.n.Q / AriocBase::aam.ms.Aligners );

        if( this->DataSourceInfo[0] )
            tsem.AppendString( "srcInfo", this->DataSourceInfo );

        tsem.Write( &ofi );
    }
}
#pragma endregion
