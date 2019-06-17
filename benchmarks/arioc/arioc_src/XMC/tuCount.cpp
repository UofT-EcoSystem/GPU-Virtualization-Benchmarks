/*
  tuCount.cpp

    Copyright (c) 2018-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
     in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
     The contents of this file, in whole or in part, may only be copied, modified, propagated, or
     redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] default constructor
tuCount::tuCount()
{
}

/// constructor (AppMain*, XMC*, INT32)
tuCount::tuCount( AppMain* _pam, XMC* _pxmc, INT32 _ip ) : m_pxmc(_pxmc ), m_ip(_ip), m_psfpi(_pxmc->SFPI.p+_ip),
                                                           m_filter_f(_pam->Filter_f), m_filter_F(_pam->Filter_F), m_filter_G(_pam->Filter_G), m_filter_q(_pam->Filter_q), m_report_d(_pam->Report_d),
                                                           m_fileSAM(_pam->SamFilename, true), m_ofsBuf(-1), m_cbBuf(0), m_pBufLimit(NULL)
{
    // allocate an input buffer
    INT64 cb = min2(m_psfpi->ofsLimit-m_psfpi->ofsStart, SAMFILEBUFSIZE);
    m_buf.Realloc( cb+1, false );
    m_buf.n = static_cast<UINT32>(cb);
    m_pBufLimit = m_buf.p + cb;
    *m_pBufLimit = 0;                                                // zero the last byte in the buffer

    // seek to the first byte for the current SAM file partition
    if( m_psfpi->ofsStart != m_fileSAM.Seek( m_psfpi->ofsStart, SEEK_SET ) )
        throw new ApplicationException( __FILE__, __LINE__, "%s: cannot seek to file offset %lld in %s", __FUNCTION__, m_psfpi->ofsStart, m_fileSAM.FileSpec.p );
}

/// destructor
tuCount::~tuCount()
{
}
#pragma endregion

#pragma region private methods
/// [private] method refreshInputBuffer
INT64 tuCount::refreshInputBuffer( INT64 _ofs )
{
    // copy remaining buffer contents to the start of the buffer
    memcpy_s( m_buf.p, m_buf.n, m_buf.p+m_ofsBuf, m_cbBuf );

    // compute the number of bytes to read from the SAM file
    INT64 cbToRead = min2(m_buf.n-m_cbBuf, m_psfpi->ofsLimit-_ofs);

    // if there is nothing left to read...
    if( cbToRead <= 2 )
    {
        // sanity check
        if( strlen(m_buf.p) > 2 )
            throw new ApplicationException( __FILE__, __LINE__, "%s: unparsed data in SAM file partition %d at offset %lld: %s",
                                                                __FUNCTION__, m_ip, _ofs, m_buf.p );

        // set the return value to indicate that we're parsed all of the SAM records in the current partition
        return m_psfpi->ofsLimit;
    }
    
    // read bytes into the rest of the buffer
    INT64 cb = m_fileSAM.Read( m_buf.p+m_cbBuf, cbToRead );
    if( cb != cbToRead )
        throw new ApplicationException( __FILE__, __LINE__, "%s: read %lld/%lld bytes", __FUNCTION__, cb, cbToRead );

    // update the number of bytes available in the buffer
    m_cbBuf += static_cast<INT32>(cb);

    // update the buffer-limit pointer
    m_pBufLimit = m_buf.p + m_cbBuf;

    // null terminate the buffer data
    m_buf.p[m_cbBuf] = 0;

    // reset the buffer offset
    m_ofsBuf = 0;

    // return an updated file offset
    return _ofs + cb;
}

/// [private] method parseSamRecord
INT64 tuCount::parseSamRecord( INT64 _ofs )
{
    // if necessary, fill the input buffer
    if( m_cbBuf < MINSAMRECORDSIZE )
    {
        // refill the buffer
        _ofs = refreshInputBuffer( _ofs );
        if( _ofs < 0 )
            return _ofs;    
    }

    // point to the start of the current SAM record
    char* p = m_buf.p + m_ofsBuf;

    // look for EOL
    char* pEOL = XMC::FindEOL( p, m_pBufLimit );
    if( pEOL < m_pBufLimit )
        *pEOL = 0;
    else
    {
        /* At this point there is no EOL in the remaining data in the buffer */
        _ofs = refreshInputBuffer( _ofs );
        if( _ofs < 0 )
            throw new ApplicationException( __FILE__, __LINE__, "%s: unexpected end of file in partition %d", __FUNCTION__, m_ip );

        // update the pointers to the start and end of the current SAM record
        p = m_buf.p;
        pEOL = XMC::FindEOL( p, m_pBufLimit );
        if( pEOL == m_pBufLimit )
            throw new ApplicationException( __FILE__, __LINE__, "%s: no end-of-line for SAM record", __FUNCTION__ );

        *pEOL = 0;
    }


#if TODO_CHOP_WHEN_DEBUGGED
    //CDPrint( cdpCD0, "%s: %s", __FUNCTION__, p );
    if( strncmp( "ST-", p, 3) )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif

    try
    {
        parseXM( p );
    }
    catch( ApplicationException* _pex )
    {
        char* pQNAME = p;
        while( *(++p) != '\t' );
        _pex->SetCallerExceptionInfo( __FILE__, __LINE__, "QNAME: %.*s", static_cast<INT32>(p-pQNAME), pQNAME );
        throw _pex;    
    }

    // increment the count of SAM records
    ++m_psfpi->TotalRows;

    // "consume" the bytes in the SAM record
    char* pSOL = XMC::FindSOL( pEOL+1, m_pBufLimit );
    INT32 cbConsumed = pSOL ? static_cast<INT32>(pSOL - p) : m_cbBuf;
    m_ofsBuf += cbConsumed;
    m_cbBuf -= cbConsumed;

    // return the file offset
    return _ofs;
}

/// [private] method parseXM
void tuCount::parseXM( char* _p )
{
    // QNAME
    char* pSOF = _p;
    char* pEOF = _p;
    while( *(++pEOF) != '\t' );

    // FLAG
    pSOF = pEOF + 1;
    pEOF = pSOF;
    while( *(++pEOF) != '\t' );

    // convert to integer
    *pEOF = 0;
    INT32 FLAG = strtol( pSOF, NULL, 10 );

    // apply filters
    if( ((FLAG & m_filter_f) != m_filter_f) ||                  // -f: only include reads with all of the FLAGs present 
        (FLAG & m_filter_F) ||                                  // -F: only include reads with none of the FLAGS present
        (m_filter_G && ((FLAG & m_filter_G) == m_filter_G)) )   // -G: only exclude reads with all of the FLAGs present
    {
        return;
    }

    // RNAME
    pSOF = pEOF + 1;
    pEOF = pSOF;
    while( *(++pEOF) != '\t' );

    // POS
    pSOF = pEOF + 1;
    pEOF = pSOF;
    while( *(++pEOF) != '\t' );

    // MAPQ
    pSOF = pEOF + 1;
    pEOF = pSOF;
    while( *(++pEOF) != '\t' );

    // convert to integer
    *pEOF = 0;
    INT32 MAPQ = strtol( pSOF, NULL, 10 );

    // apply filter
    if( MAPQ < m_filter_q )                     // -q: only include reads with mapping quality >= q
        return;

    // optionally accumulate MAPQ counts
    if( m_report_d )
        m_psfpi->MapqCount[MAPQ]++;

    /* At this point we can skip to the XM field.

       We assume that we can skip at least MINSAMRECORDSIZE bytes for the remaining required SAM fields (CIGAR, RNEXT, PNEXT, TLEN, SEQ, QUAL).
    */
    pSOF = strstr( pEOF+MINSAMRECORDSIZE, "XM:Z:" );
    if( pSOF )
    {
        // point to the first character in the XM map
        char* p = pSOF + 5;     // 5: strlen( "XM:Z:" )

        // traverse the XM map and count Cs in context
        while( *p && (*p != '\t') )
        {
            if( *p != '.' )
            {
                // if bit 5 is set, it's a lower-case character (i.e., unmethylated)
                if( *p & 0x20 )
                {
                    switch( *p )
                    {
                        case 'z':       // CpG context
                            m_psfpi->C_CpG++;
                            break;

                        case 'x':       // CHG context
                            m_psfpi->C_CHG++;
                            break;

                        case 'h':       // CHH context
                            m_psfpi->C_CHH++;
                            break;

                        case 'u':       // unknown context
                            m_psfpi->C_unknown++;
                            break;

                        default:
                            throw new ApplicationException( __FILE__, __LINE__, "%s: invalid XM character '%c' in %s", __FUNCTION__, *p, pSOF );
                    }
                }
                else    // bit 5 is not set, so it's an upper-case character (i.e., methylated)
                {
                    switch( *p )
                    {
                        case 'Z':       // CpG context
                            m_psfpi->Cm_CpG++;
                            break;

                        case 'X':       // CHG context
                            m_psfpi->Cm_CHG++;
                            break;

                        case 'H':       // CHH context
                            m_psfpi->Cm_CHH++;
                            break;

                        case 'U':       // unknown context
                            m_psfpi->Cm_unknown++;
                            break;

                        default:
                            throw new ApplicationException( __FILE__, __LINE__, "%s: invalid XM character '%c' in %s", __FUNCTION__, *p, pSOF );
                    }
                }
            }

            // advance to the next character in the XM map
            ++p;
        }

        // update the counter
        m_psfpi->TotalXM++;
    }
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Categorizes and counts cytosines in a BS-seq SAM file
/// </summary>
/// <remarks>There is one <c>tuCount</c> instance per SAM file partition.</remarks>
void tuCount::main()
{
    INT64 ofs = m_psfpi->ofsStart;
    do
    {
        ofs = parseSamRecord( ofs );
    }
    while( (ofs < m_psfpi->ofsLimit) || (m_cbBuf >= MINSAMRECORDSIZE) );


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: %lld SAM records parsed in partition %d", __FUNCTION__, m_psfpi->TotalRows, m_ip );
#endif
}
