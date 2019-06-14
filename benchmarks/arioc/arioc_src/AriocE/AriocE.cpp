/*
  AriocE.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variables
AppPerfMetrics AriocE::PerfMetrics;
#pragma endregion

#pragma region constructors and destructor
/// [public] constructor
AriocE::AriocE( INT32 srcId, InputFileGroup& ifgRaw, const Nencoding enc,
                A21SeedBase* pa21sb, char* outDir, const INT32 maxThreads, double samplingRatio, char qualityScoreBias,
                bool emitKmers, UINT32 gpuMask, char* qmdQNAME, INT32 maxJ, AppMain* pam ) :
                            m_samplingRatio(samplingRatio),
                            m_celJ(0), m_nJ(0),
                            m_maxJ(maxJ), m_iCshort(0), m_iCzero(0), m_nBBHJ(0), m_nBBRI(0),
                            m_encoderWorkerThreadTimeout(ENCODERWORKERTHREAD_TIMEOUT),
                            m_pam(pam),
                            Params(pam, enc, maxThreads, srcId, pam->ConfigFileName, ifgRaw, outDir, samplingRatio, qualityScoreBias, emitKmers, gpuMask, qmdQNAME, pa21sb)
{
    // copy and clean the specified strings
    for( UINT32 n=0; n<ifgRaw.InputFile.n; ++n )
        RaiiDirectory::SetPathSeparator( this->Params.ifgRaw.InputFile.p[n].Filespec );
    strcpy_s( this->Params.OutDir, sizeof this->Params.OutDir, outDir );
    RaiiDirectory::SetPathSeparator( this->Params.OutDir );
    RaiiDirectory::ChopTrailingPathSeparator( this->Params.OutDir );

    // ensure that the output directory exists
    RaiiDirectory::OpenOrCreateDirectory( this->Params.OutDir );

    if( enc == NencodingR )
    {
        // build the output directory path for lookup table files
        sprintf_s( this->Params.OutDirLUT, sizeof this->Params.OutDirLUT, "%s%s%s", this->Params.OutDir, FILEPATHSEPARATOR, pa21sb->IdString );

        // ensure that the directory exists
        RaiiDirectory::OpenOrCreateDirectory( this->Params.OutDirLUT );
    }

    // build a file specification "stub" for R sequence files and all Q sequence files
    strcpy_s( this->Params.OutFilespecStubSq, FILENAME_MAX, this->Params.OutDir );
    strcat_s( this->Params.OutFilespecStubSq, FILENAME_MAX, FILEPATHSEPARATOR );

    // build a file specification "stub" for lookup table files (C, H, J)
    strcpy_s( this->Params.OutFilespecStubLUT, FILENAME_MAX, this->Params.OutDirLUT);
    strcat_s( this->Params.OutFilespecStubLUT, FILENAME_MAX, FILEPATHSEPARATOR );

    // build a 3-digit string that represents the number of mismatches and the number of hashkey bits
    sprintf_s( this->Params.LUTtypeStub, sizeof this->Params.LUTtypeStub, "%1d%02d", pa21sb->maxMismatches, pa21sb->hashKeyWidth );

    // initialize the encodings for the ASCII symbols that represent DNA bases in the specified sequence (ACGTN)
    memset( this->Params.SymbolEncode, enc, sizeof this->Params.SymbolEncode ); // 010 (NencodingQ) or 011 (NencodingR)
    this->Params.SymbolEncode['A'] = this->Params.SymbolEncode['a'] = 0x04;     // 100
    this->Params.SymbolEncode['C'] = this->Params.SymbolEncode['c'] = 0x05;     // 101
    this->Params.SymbolEncode['G'] = this->Params.SymbolEncode['g'] = 0x06;     // 110
    this->Params.SymbolEncode['T'] = this->Params.SymbolEncode['t'] = 0x07;     // 111

    // initialize the complement lookup table
    this->Params.DnaComplement['A'] = this->Params.DnaComplement['a'] = 'T';
    this->Params.DnaComplement['C'] = this->Params.DnaComplement['c'] = 'G';
    this->Params.DnaComplement['G'] = this->Params.DnaComplement['g'] = 'C';
    this->Params.DnaComplement['T'] = this->Params.DnaComplement['t'] = 'A';
    this->Params.DnaComplement['N'] = this->Params.DnaComplement['n'] = 'N';

    // get the number of available "logical processors" (concurrent independent CPU threads or hyperthreads)
    INT32 nAvailableCpuThreads = GetAvailableCpuThreadCount();

    // clamp the maximum number of "logical processors" (CPU cores or hyperthreads)
    this->Params.nLPs = min2( maxThreads, nAvailableCpuThreads );

    // look for a user-specified override for the worker-thread timeouts
    INT32 i = m_pam->Xparam.IndexOf( "workerThreadLaunchTimeout" );
    if( i >= 0 )
        tuBaseA::WorkerThreadLaunchTimeout = static_cast<DWORD>(m_pam->Xparam.Value(i));
    i = m_pam->Xparam.IndexOf( "encoderWorkerThreadTimeout" );
    if( i >= 0 )
        m_encoderWorkerThreadTimeout = static_cast<DWORD>(m_pam->Xparam.Value(i));
}

/// [public] destructor
AriocE::~AriocE()
{
}
#pragma endregion

#pragma region private methods
/// [private] method findSOL
char* AriocE::findSOL( char* p, char* pLimit )
{
    while( (p < pLimit) && (*p <= ' ') )  // skip consecutive control characters and spaces
        ++p;
    return ((p < pLimit) ? p : NULL);
}

/// [private] method findEOL
char* AriocE::findEOL( char* p, char* pLimit )
{
    char* pEOL = strpbrk( p, "\r\n\x1A" );      // --> first end-of-line byte (0x0D or 0x0A) or end-of-file byte (0x1A)
    if( pEOL == NULL )                          // if there is no end-of-line byte, assume that the end of the buffer is the end of the "line"
        pEOL = pLimit;
    return pEOL;
}
#pragma endregion

#pragma region public methods
/// [public] method SniffSqFile
SqFileFormat AriocE::SniffSqFile( char* fileSpec )
{
    /* We "sniff" the format of the sequence file by reading several lines of data from the beginning of the file.
    */
    RaiiFile fileSq( fileSpec, true );          // true: read-only

    // read the first 32Kb into a buffer
    m_buf.Realloc( 32*1024, false );
    INT32 cb = static_cast<INT32>(fileSq.Read( m_buf.p, m_buf.cb-1 ));
    m_buf.p[cb] = 0;                                // append a null terminating byte

    /* Look at the data pattern in the file to determine the file format.  We assume that lines can be terminated by any of the following
        end-of-line (EOL) delimiters:
            - CR (0x0D)
            - LF (0x0A)
            - EOF (0x1A)
            - end of data
    
       FASTA pattern:  > [description] EOL
                       [ACGTN]+ EOL

       FASTQ pattern:  @ [description] EOL
                       [ACGT]+ EOL
                       + [description] EOL
                       [quality scores] EOL
    */
    bool isSqData = true;
    char* p = m_buf.p;                          // --> start of buffer
    char* pLimit = m_buf.p + cb;                // --> null terminating byte in buffer
    p = findSOL( p, pLimit );                   // --> first byte of non-null data
    char* pEOL = findEOL( p, pLimit );          // --> the byte after the last byte of data in the current line

    switch( *p )
    {
        case '>':
            // try to confirm a FASTA pattern
            p = findSOL( pEOL+1, pLimit );
            if( p == NULL )
                break;

            // look for a run of ACGTN
            pEOL = findEOL( p, pLimit );
            while( isSqData && (p < pEOL) )
                isSqData = (strchr("AaCcGgTtNn", *(p++)) != NULL);

            // if the line contained only ACGTN, assume we have a FASTA file; otherwise, we don't know the file format
            if( isSqData )
                this->Params.InputFileFormat = SqFormatFASTA;
            break;

        case '@':
            // try to confirm a FASTQ pattern
            p = findSOL( pEOL+1, pLimit );
            if( p == NULL )
                break;

            // look for a run of ACGTN
            pEOL = findEOL( p, pLimit );
            while( isSqData && (p < pEOL) )
                isSqData = (strchr("AaCcGgTtNn", *(p++)) != NULL);

            // if the line contained only ACGTN, assume we have a FASTQ file; otherwise, we don't know the file format
            if( !isSqData )
                break;

            // if the next line starts with a "+", assume we have a FASTQ file; otherwise, we don't know the file format
            p = findSOL( pEOL+1, pLimit );
            if( *p == '+' )
                this->Params.InputFileFormat = SqFormatFASTQ;
            break;

        default:
            break;
    }

    // we're all done if it's not FASTQ or if a quality score bias was specified
    if( (this->Params.InputFileFormat != SqFormatFASTQ) || (this->Params.QualityScoreBias > 0) )
        return this->Params.InputFileFormat;
    
    // try to figure out the quality score bias from the first few quality scores in the file
    p = m_buf.p;                    // --> start of buffer
    pLimit = m_buf.p + cb;          // --> null terminating byte in buffer
    p = findSOL( p, pLimit );       // --> first byte of non-null data
    pEOL = findEOL( p, pLimit );    // --> the byte after the last byte of data in the current line
    INT8 minQ = 0x7F;
    INT8 maxQ = -1;
    bool expectQ = false;
    do
    {
        if( (*p == '+') && !expectQ )       // FASTQ line 3
            expectQ = true;             

        else
        {
            if( expectQ )
            {
                for( INT8* pq=reinterpret_cast<INT8*>(p); pq<reinterpret_cast<INT8*>(pEOL); ++pq )
                {
                    minQ = min2( minQ, *pq );
                    maxQ = max2( maxQ, *pq );
                }

                expectQ = false;
            }
        }

        // advance to the next full line of text
        p = findSOL( pEOL+1, pLimit );
        pEOL = p ? findEOL( p, pLimit ) : NULL;
    }
    while( p && (pEOL < pLimit) );

    /* 59-104: Solexa/Illumina 1.0
       64-104: Illumina 1.3+
       66-104: Illumina 1.5+
    */
    if( (minQ >= 59) && (maxQ <= 104) )
    {
        if( minQ < 64 )
            throw new ApplicationException( __FILE__, __LINE__, "warning: Solexa/Illumina 1.0 quality scores not supported" );

        this->Params.QualityScoreBias = 64;
        this->Params.QSB8 = 0x4040404040404040;
    }

    /* 33-73: Sanger
       33-74: Illumina 1.8+
    */
    else
    if( (minQ >= 33) && (maxQ <= 74) )      // Illumina v1.8+, Sanger
    {
        this->Params.QualityScoreBias = 33;
        this->Params.QSB8 = 0x2121212121212121;
    }
    else
        throw new ApplicationException( __FILE__, __LINE__, "warning: unknown quality score bias (detected scores %d (%c) through %d (%c)", minQ, minQ, maxQ, maxQ );

    return this->Params.InputFileFormat;
}

/// <summary>
/// Computes a 32-bit hash of a 64-bit value
/// </summary>
/// <remarks>This is a static method.</remarks>
UINT32 AriocE::Hash6432( UINT64 x )
{
    /* Computes a 32-bit value by hashing the specified 64-bit value.
        
       The hash implementation is the 32-bit version of MurmurHash3 from http://code.google.com/p/smhasher/wiki/MurmurHash.
    */

    // constants
    UINT32 c1 = 0xcc9e2d51;
    UINT32 c2 = 0x1b873593;
    UINT32 seed = 0xb0f57ee3;

    // b2164 bits 0-31
    register UINT32 k1 = static_cast<UINT32>(x);

    k1 *= c1;
    k1 = _rotl( k1, 15 );
    k1 *= c2;

    UINT32 h = seed ^ k1;
    h = _rotl( h, 13 );
    h = 5*h + 0xe6546b64;

    // b2164 bits 32-63
    k1 = static_cast<UINT32>(x >> 32);
    k1 *= c1;
    k1 = _rotl( k1, 15 );
    k1 *= c2;

    h ^= k1;
    h = _rotl( h, 13 );
    h = 5*h + 0xe6546b64;

    // avalanche
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return h;
}
#pragma endregion
