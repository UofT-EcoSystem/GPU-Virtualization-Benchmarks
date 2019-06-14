/*
  XMC.cpp

    Copyright (c) 2018-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
     in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
     The contents of this file, in whole or in part, may only be copied, modified, propagated, or
     redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// [public] constructor
XMC::XMC( AppMain* _pam ) : m_pam(_pam), SamRowOffset(0)
{
}

/// [public] destructor
XMC::~XMC()
{
}
#pragma endregion

#pragma region private methods
/// [private] method sniffSamFile
void XMC::sniffSamFile()
{
    RaiiFile fileSAM( m_pam->SamFilename, true );          // true: read-only

    // read the first 128Kb into a buffer (which should be enough to contain the entire set of SAM header records)
    m_buf.Realloc( 128*1024, false );
    INT32 cb = static_cast<INT32>(fileSAM.Read( m_buf.p, m_buf.cb-1 ));
    m_buf.p[cb] = 0;                                // append a null terminating byte

    /* SAM headers
    
       We're not interested in the content of the SAM headers, but we do need to know where the alignment records start.
    */
    if( m_pam->Report_h )
        CDPrint( cdpCD0, "SAM headers:" );

    char* p = m_buf.p;                          // --> start of buffer
    char* pLimit = m_buf.p + cb;                // --> null terminating byte in buffer
    p = FindSOL( p, pLimit );                   // --> first byte of non-null data

    while( p && *p && (*p == '@') )
    {
        char* pEOL = FindEOL( p, pLimit );          // --> the byte after the last byte of data in the current line

        if( (pEOL >= p) && m_pam->Report_h )
            CDPrint( cdpCD0, "%s: %.*s", __FUNCTION__, static_cast<INT32>(pEOL-p), p );

        // advance to the next line
        p = FindSOL( pEOL, pLimit );
    }

    if( m_pam->Report_h )
        CDPrint( cdpCD0, "" );

    /* At this point p --> the first SAM alignment record. */

    // sanity check
    if( !p || !(*p) )
        throw new ApplicationException( __FILE__, __LINE__, "%s: end of input buffer at %lld bytes", __FUNCTION__, m_buf.cb );

    // save the offset of the first SAM alignment record
    this->SamRowOffset = p - m_buf.p;

#if TODO_ASAP
    inHeaders = false;

            // try to confirm a FASTA pattern
            p = FindSOL( pEOL+1, pLimit );
            if( p == NULL )
                break;

            // look for a run of ACGTN
            pEOL = FindEOL( p, pLimit );
            while( isSqData && (p < pEOL) )
                isSqData = (strchr("AaCcGgTtNn", *(p++)) != NULL);

            // if the line contained only ACGTN, assume we have a FASTA file; otherwise, we don't know the file format
            if( isSqData )
                this->Params.InputFileFormat = SqFormatFASTA;
            break;

        case '@':
            // try to confirm a FASTQ pattern
            p = FindSOL( pEOL+1, pLimit );
            if( p == NULL )
                break;

            // look for a run of ACGTN
            pEOL = FindEOL( p, pLimit );
            while( isSqData && (p < pEOL) )
                isSqData = (strchr("AaCcGgTtNn", *(p++)) != NULL);

            // if the line contained only ACGTN, assume we have a FASTQ file; otherwise, we don't know the file format
            if( !isSqData )
                break;

            // if the next line starts with a "+", assume we have a FASTQ file; otherwise, we don't know the file format
            p = FindSOL( pEOL+1, pLimit );
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
    p = FindSOL( p, pLimit );       // --> first byte of non-null data
    pEOL = FindEOL( p, pLimit );    // --> the byte after the last byte of data in the current line
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
        p = FindSOL( pEOL+1, pLimit );
        pEOL = p ? FindEOL( p, pLimit ) : NULL;
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
#endif

}

/// [private] method partitionSamFile
void XMC::partitionSamFile()
{
    RaiiFile fileSAM( m_pam->SamFilename, true );          // true: read-only
    INT64 cbSamFile = fileSAM.FileSize();

    // if the file size is less than 1 megabyte, we use only one worker thread
    INT32 nThreads = (cbSamFile < 1024*1024*1024) ? 1 : m_pam->ConcurrentThreadCount;

    /* At this point we need to define one partition for each concurrent worker thread.  The partitions don't have to
        be exactly the same size, but they do need to start and end in the right place. */

    // estimate the number of bytes in one partition
    INT64 cbEstimated = blockdiv( cbSamFile-this->SamRowOffset, nThreads );

    this->SFPI.Realloc( nThreads, true );
    for( INT32 ip=0; ip<nThreads; ++ip )
    {
        // find the file offset of the start of the first SAM record in the partition
        if( ip == 0 )
            this->SFPI.p[ip].ofsStart = this->SamRowOffset;

        else
        {
            INT64 ofs = this->SamRowOffset + (ip * cbEstimated);

            // find the previous EOL in the file
            const INT64 cbChunk = 32 * 1024;                // we assume that a SAM record is no longer than 32Kb
            INT64 fp = max2( 0, ofs-cbChunk );
            fp = fileSAM.Seek( fp, SEEK_SET );
            INT64 cb = fileSAM.Read( m_buf.p, cbChunk );
            m_buf.p[cb] = 0;                                // append a null terminating byte

            char* p = m_buf.p + (cb-1);
            while( (*p != '\x0A') && (*p != '\x0D') )       // we assume that SAM records are separated by LF or CRLF
            {
                --p;

                // don't underflow
                if( p == m_buf.p )
                    throw new ApplicationException( __FILE__, __LINE__, "%s: cannot identify previous end of line", __FUNCTION__ );
            }

            /* At this point p points to the character before the start of a SAM record. */
            this->SFPI.p[ip].ofsStart = (ofs - (cb - (p - m_buf.p))) + 1;
        }

        // the ip'th start offset is the (ip-1)th limit offset
        if( ip )
            this->SFPI.p[ip-1].ofsLimit = this->SFPI.p[ip].ofsStart;
    }

    // the final partition's limit offset is the end of the file
    this->SFPI.p[this->SFPI.Count-1].ofsLimit = cbSamFile;
    INT32 cch = static_cast<INT32>(log10(static_cast<float>(cbSamFile))) + 1;

    // emit SAM file offsets
    CDPrint( cdpCD0, "SAM file size         : %lld", fileSAM.FileSize() );
    CDPrint( cdpCD0, " Headers              : %*lld-%*lld", cch, 0, cch, this->SamRowOffset );
    for( INT32 ip=0; ip<static_cast<INT32>(this->SFPI.Count); ++ip )
    {
        const char* pad = (ip < 10) ? " " : "";
        CDPrint( cdpCD0, " Partition %d%s         : %*lld-%*lld", ip, pad, cch, this->SFPI.p[ip].ofsStart, cch, this->SFPI.p[ip].ofsLimit );

#if TODO_CHOP_WHEN_DEBUGGED        
        // emit the first and last 64 characters in the partition
        fileSAM.Seek( this->SFPI.p[ip].ofsStart, SEEK_SET );
        char bufHead[80];
        fileSAM.Read( bufHead, 64 );
        bufHead[64] = 0;
        CDPrint( cdpCD0, "     Head: %s", bufHead );

        fileSAM.Seek( this->SFPI.p[ip].ofsLimit-64, SEEK_SET );
        char bufTail[80];
        fileSAM.Read( bufTail, 64 );
        bufTail[64] = 0;
        CDPrint( cdpCD0, "     Tail: %s", bufTail );
#endif

    }
}
#pragma endregion

#pragma region public methods
/// [public static] method FindSOL
char* XMC::FindSOL( char* p, char* pLimit )
{
    while( (p < pLimit) && (*p <= ' ') )  // skip consecutive control characters and spaces
        ++p;
    return ((p < pLimit) ? p : NULL);
}

/// [public static] method FindEOL
char* XMC::FindEOL( char* p, char* pLimit )
{
    char* pEOL = strpbrk( p, "\r\n\x1A" );      // --> first end-of-line byte (0x0D or 0x0A) or end-of-file byte (0x1A)
    if( pEOL == NULL )                          // if there is no end-of-line byte, assume that the end of the buffer is the end of the "line"
        pEOL = pLimit;
    return pEOL;
}

/// [public] method Main
void XMC::Main()
{
    sniffSamFile();
    partitionSamFile();

    /* Start a TaskUnit instance for each SAM file partition:
        - Each partition is associated with a 0-based ordinal that identifies the subset of the input data that is
            processed on an individual worker thread.
        - There is no RAII here so these tuGpu instances must be explicitly destructed later on.
    */
    INT32 nThreads = static_cast<INT32>(this->SFPI.Count);
    WinGlobalPtr<tuCount*> ptuCount( nThreads, true );
    for( INT16 ip=0; ip<nThreads; ++ip )
    {
        ptuCount.p[ip] = new tuCount( m_pam, this, ip );
        ptuCount.p[ip]->Start();
    }

    // wait for all work to complete
    for( INT16 ip=0; ip<nThreads; ++ip )
        ptuCount.p[ip]->Wait( INFINITE );

    // destruct the tuCount instances
    for( INT16 ip=0; ip<nThreads; ++ip )
        delete ptuCount.p[ip];

    // accumulate counts
    for( INT16 ip=0; ip<nThreads; ++ip )
        m_pam->SFPI.AccumulateCounts( this->SFPI.p[ip] );
}
#pragma endregion
