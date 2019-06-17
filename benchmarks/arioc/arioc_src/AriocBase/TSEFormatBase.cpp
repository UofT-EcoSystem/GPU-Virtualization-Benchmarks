/*
  TSEFormatBase.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Implements functionality common to TSE SBF format and TSE KMH format.
/// </summary>
TSEFormatBase::TSEFormatBase( QBatch* pqb, UINT32 cbFixedFields, INT16 nEmitFields ) : baseARowBuilder( pqb, nEmitFields, false ), m_mapCat( mcNull )
{
    /* Compute the amount of buffer space required to build each TSE row.  A worst-case (maximum) size is
        assumed for each field.

        We don't use SQL null values in any column, so the size in the SBF data corresponds to the binary data size.
    */

    // sanity check
    if( pqb->Nmax > m_maxNforTSE )
        throw new ApplicationException( __FILE__, __LINE__, "cannot represent sequence of length %d in TSE data (maximum length = %d)", pqb->Nmax, m_maxNforTSE );

    // save the worst-case row size (although the BRLEE and BRLEQ strings should actually be much smaller than the original data)
    m_cbRow = cbFixedFields + 2 * min2( m_maxNforTSE, pqb->Nmax );

    if( pqb->pab->pifgQ->HasPairs )
        m_cbRow *= 2;       // double the buffer-space estimate if we're writing paired-end reads

#if TODO_CHOP_IF_UNUSED
    // compute the size of the reverse BRLEA buffer
    INT32 cbBRLEA = pqb->pab->aas.ComputeWorstCaseBRLEAsize( pqb->Nmax );

    // allocate the buffer
    m_revBRLEA.Realloc( cbBRLEA, false );
#endif
}

/// [public] destructor
TSEFormatBase::~TSEFormatBase()
{
}
#pragma endregion

#pragma region private methods
/// [private] method getFirstBQS
UINT8 TSEFormatBase::getFirstBQS( UINT64** ppBQS, UINT8* pnConsumed, char* pMbuf )
{
    *ppBQS = reinterpret_cast<UINT64*>(pMbuf)-1;    // point before the start of the specified metadata (quality score) buffer
    *pnConsumed = 8;                                // initialize a state variable used in getNextBQS()

    // return the first BQS in the buffer
    return static_cast<UINT8>(*pMbuf);
}

/// [private] method getNextBQS
UINT8 TSEFormatBase::getNextBQS( UINT64** ppBQS, UINT64* pBQS8, UINT8* pnConsumed )
{
    if( *pnConsumed < 8 )
    {
        *pBQS8 >>= 8;           // move the next BQS into the low-order byte
        (*pnConsumed)++;        // increment the number of BQS consumed
    }
    else
    {
        *ppBQS = *ppBQS + 1;    // point to the next 8 BQSs
        *pBQS8 = **ppBQS;       // load the next 8 BQSs
        *pnConsumed = 1;        // reset the number of BQS consumed
    }

    // return the low-order byte
    return static_cast<UINT8>(*pBQS8);
}

/// [private] method: appendBRLEQRunLength
INT32 TSEFormatBase::appendBRLEQRunLength( UINT8* pBRLEQ, INT16& cbBRLEQ, INT32 runLength )
{
    if( runLength <= 0x7F )         // 7 bits
        pBRLEQ[cbBRLEQ++] = runLength;

    else
        if( runLength <= 0x3FFF )       // 14 bits
        {
        pBRLEQ[cbBRLEQ++] = runLength >> 7;
        pBRLEQ[cbBRLEQ++] = (runLength & 0x7F);
        }

        else                            // up to 21 bits (more than enough)
        {
            pBRLEQ[cbBRLEQ++] = runLength >> 14;
            pBRLEQ[cbBRLEQ++] = (runLength >> 7) & 0x7F;
            pBRLEQ[cbBRLEQ++] = runLength & 0x7F;
        }

    return runLength;
}

/// <summary>
/// private method: appendBRLEQBinValues
/// </summary>
INT32 TSEFormatBase::appendBRLEQBinValues( UINT8* pBRLEQ, INT16& cbBRLEQ, const UINT8* pbv, INT32 i, INT32 j )
{
    /* Decide whether to append 2-bit or 3-bit values:
        - 2-bit: 3 consecutive bin values <= 3
        - 3-bit: 2 or fewer values, or one of the next 3 values >= 4
    */
    INT32 n = j - i;

    switch( n )
    {
        case 3:
            if( (pbv[i] <= 3) && (pbv[i+1] <= 3) && (pbv[i+2] <= 3) )
                pBRLEQ[cbBRLEQ++] = m_2bpq | pbv[i] | (pbv[i+1] << 2) | (pbv[i+2] << 4);    // 3 2-bit values
            else
            {
                pBRLEQ[cbBRLEQ++] = m_3bpq | pbv[i] | (pbv[i+1] << 3);                      // 2 3-bit values
                n = 2;
            }
            break;

        case 2:
            pBRLEQ[cbBRLEQ++] = m_3bpq | pbv[i] | (pbv[i+1] << 3);                          // 2 3-bit values
            break;

        case 1:
            pBRLEQ[cbBRLEQ++] = m_3bpq | pbv[i];                                            // 1 3-bit value
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for n=%d j=%d i=%d", n, j, i );
    }

    // return the number of bin values appended
    return n;
}
#pragma endregion

#pragma region protected methods
#pragma warning ( push )
#pragma warning( disable:4996 )     // (don't nag us about strcpy being "unsafe")

/// [protected] method emitSqId
UINT32 TSEFormatBase::emitSqId( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<UINT64>( pbuf, sqId );
}

/// [protected] method emitMapCat
UINT32 TSEFormatBase::emitMapCat( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<UINT8>( pbuf, m_mapCat );
}

/// [protected] method emitHash32
UINT32 TSEFormatBase::emitHash32( char* pbuf, INT64 sqId, QAI* pQAI )
{
    UINT32 hash32 = computeA21Hash32( pQAI );
    return intToSBF<UINT32>( pbuf, hash32 );
}

/// [protected] method emitHash64
UINT32 TSEFormatBase::emitHash64( char* pbuf, INT64 sqId, QAI* pQAI )
{
    UINT64 hash64 = computeA21Hash64( pQAI );
    return intToSBF<UINT64>( pbuf, hash64 );
}

/// [protected] method emitBRLEQfr
UINT32 TSEFormatBase::emitBRLEQfr( char* pbuf, QAI* pQAI, UINT32 iMate, UINT32 iMetadata )
{
    /* This method stores the following data in the output buffer:
        byte 0..0: minimum BQS
        byte 1..1: maximum BQS
        byte 2..3: length of BRLEQ string
        byte 3.. : BRLEQ string
    */

    // if there are no quality scores...
    if( !m_pqb->MFBq[0].buf.Count )
    {
        *(INT32*)pbuf = 0;      // store zero for minimum BQS (1 byte), maximum BQS (1 byte), and length (2 bytes)
        return sizeof( INT32 );
    }

    // point to the quality score metadata for the current Q sequence
    char* pMbuf = m_pqb->MFBq[iMate].buf.p + m_pqb->MFBq[iMate].ofs.p[iMetadata] + sizeof( MetadataRowHeader );

    // find the minimum and maximum BQSs (base quality scores)
    UINT64* pBQS;
    UINT64 bqs8 = 0;
    UINT8 nConsumed;
    UINT8 maxBQS = getFirstBQS( &pBQS, &nConsumed, pMbuf );
    UINT8 minBQS = 0xFF;
    for( INT16 n=0; n<pQAI->N; ++n )
    {
        UINT8 bqs = getNextBQS( &pBQS, &bqs8, &nConsumed );
        if( bqs < minBQS )
            minBQS = bqs;
        else
        {
            if( bqs > maxBQS )
                maxBQS = bqs;
        }
    }

    /* Compute a lookup table to map BQSs to bins:
        - we use floating-point so that we can round to the nearest cutpoint
        - the highest BQSs are encoded with the smallest values
    */

    // set up a lookup table to convert BQS values
    WinGlobalPtr<UINT8> lut( maxBQS+1, true );
    UINT8 nBQS = (maxBQS-minBQS) + 1;

    // initialize the table
    if( nBQS <= m_nBQSbins )
    {
        // each BQS defines a different bin
        for( UINT8 n=0; n<nBQS; ++n )
            lut.p[maxBQS-n] = n;
    }
    else
    {
        double bqsIncr = (double)m_nBQSbins / nBQS;
        double bqsCum = -0.49999;

        for( INT32 n=maxBQS; n>=minBQS; --n )
        {
            lut.p[n] = static_cast<UINT8>(bqsCum+0.5);
            bqsCum += bqsIncr;
        }
    }

    /*
        BRLEQ -- Binary Run-Length Encoded Quality score string

        The goals of the BRLEQ encoding are:
            - conserve space in representing a set of base quality score (BQS) values
            - reconstruct a set of BQSs from the BRLEQ

        The BRLEQ format records binned ("quantized") scores using 3-bit (8 bin) resolution.  This should
        be enough to preserve whatever information is in the raw (6-bit) BQSs.  That is, the reconstructed
        BQSs should work just as well as the raw BQSs when used by variant callers or other analysis tools
        that rely on numerical BQS values.

        A total length (number of BRLEQ bytes) is associated with each BRLEQ.  The length may be zero, but
        the meaning of a zero-length BRLEQ string is undefined.  In SQL Server, the length is a property of
        the SqlBinary type and does not need to be represented explicitly.

        There is no end-of-string delimiter.  To decompress a BRLEQ string, all three of the following must
        be known:
            - minimum BQS (in the original BQS string)
            - maximum BQS (in the original BQS string)
            - string length of the original BQS string (same as the length of the associated read sequence)

        Where binned values are recorded in a BRLEQ string, they may be represented as either 2-bit
        or 3-bit values.  Bin numbers are assigned so that the lowest (2-bit) values represent higher BQS;
        this heuristic favors a more compact representation because higher BQS are more common than lower
        BQS in the raw sequencer data.

        There is no length or count associated with a series of bin values.  Instead, every BRLEQ byte
        that contains bin values is assumed to contain either two 3-bit values or three 2-bit values.
        When the last byte in a BRLEQ contains bin values, the total number of BQSs in the reconstructed
        binary string is determined by the length of the corresponding read sequence; there is one BQS for
        each symbol in a read sequence.

        Bit mapping:
            Bits 6 and 7 of each BRLEQ indicate the BRLEQ byte type, which determines the meaning of the
            remaining bits in the byte.

                bit   7   6   5   4   3   2   1   0
                +---+---+---+---+---+---+---+---+
                + 0 |       run length          |    run length
                +---+---+---+---+---+---+---+---+

                +---+---+---+---+---+---+---+---+
                + 1 | 0 |  BQS  |  BQS  |  BQS  |    three 2-bit values
                +---+---+---+---+---+---+---+---+

                +---+---+---+---+---+---+---+---+
                + 1 | 1 |    BQS    |    BQS    |    two 3-bit values
                +---+---+---+---+---+---+---+---+

            Bit 7 determines whether a byte contains a run length or BQS bin values.
            When bit 7 is set, bit 6 indicates the number of bits in each bin value.

        The largest run length that can fit into a single byte is 127 (7 bits).  For
        longer runs, the bits are concatenated from successive bytes, with the high-
        order bits of the run length derived from the first byte encountered.

        The bin value for a run is determined by the most recently encountered previous
        bin value.  (Every BRLEQ must therefore start with a set of explicit bin values.)

        BQS bin values are stored in little-endian order.

        In cases where there are alternate encodings (see below) the bin-value byte type is
        preferred.

        Examples:
            1 1 1 1         10 01 01 01   10 00 00 01                 0x95 0x81       (preferred)
       <or> 1 1 1 1         10 01 01 01   0 0000001                   0x95 0x01
            1 1 1 1 1 1 1   10 01 01 01   0 0000101                   0x81 0x05
            3 1 2           10 10 01 11                               0xA7
            4 1 2           11 001 100    10 00 00 10                 0xCC 0x82
            4 1 5 5 5 5     11 001 100    11 101 101   11 101 101     0xCC 0xEB 0xEB  (preferred)
       <or> 4 1 5 5 5 5     11 001 100    11 101 101   0 00000010     0xCC 0xEB 0x02
    */

    // do the binning
    WinGlobalPtr<UINT8> bv( pQAI->N, true );
    getFirstBQS( &pBQS, &nConsumed, pMbuf );
    for( INT16 n=0; n<pQAI->N; ++n )
    {
        UINT8 bqs = getNextBQS( &pBQS, &bqs8, &nConsumed );
        bv.p[n] = lut.p[bqs];
    }

    // prepare to place the BRLEQ bytes in the output buffer
    UINT8* pBRLEQ = reinterpret_cast<UINT8*>(pbuf)+4;     // point 4 bytes past the start of the buffer
    INT16 cbBRLEQ = 0;

    // encode the beginning of the binned-value string
    INT32 i = appendBRLEQBinValues( pBRLEQ, cbBRLEQ, bv.p, 0, min2( pQAI->N, 3 ) );
    UINT8 bvTail = bv.p[i-1];

    // loop until the end of the BQS string
    while( i < pQAI->N )
    {
        // traverse the bin values to find the end of the run that starts with the current tail value
        INT32 j = i;
        while( (j<pQAI->N) && (bv.p[j]==bvTail) )
            j++;

        /* At this point, j points past the end of the run.

           Now we need to encode runs wherever they use less space than encoding bin values.
        */
        INT32 runLength = j - i;

        if( runLength >= 4 )
        {
            // a run length of 4 or more is always more compact than encoded bin values
            i += appendBRLEQRunLength( pBRLEQ, cbBRLEQ, runLength );
        }
        else
        {
            // append the next 3 bin values (or as many as remain in the binned-value string)
            i += appendBRLEQBinValues( pBRLEQ, cbBRLEQ, bv.p, i, min2( i+3, pQAI->N ) );
        }

        // update the tail value
        bvTail = bv.p[i-1];
    }

    // fill in the minimum and maximum BQSs and the BRLEQ string length
    pbuf[0] = minBQS;
    pbuf[1] = maxBQS;
    *reinterpret_cast<INT16*>(pbuf+2) = cbBRLEQ;

    // return the total number of bytes used
    return 2*sizeof( UINT8 ) + sizeof( INT16 ) + cbBRLEQ;
}

/// [protected] method emitSqLen
UINT32 TSEFormatBase::emitSqLen( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT16>( pbuf, pQAI->N );
}

/// [protected] method emitSEQ
UINT32 TSEFormatBase::emitSEQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    char* p = pbuf + sizeof( INT16 );
    INT16 cb = stringizeSEQf( p, pQAI );
    *reinterpret_cast<INT16*>(pbuf) = cb;
    return cb + sizeof( INT16 );
}

/// [private] method getRGIDfromReadMetadata
UINT8 TSEFormatBase::getRGIDfromReadMetadata( QAI* pQAI, UINT32 iMate, UINT32 iMetadata )
{
    // point to the Q sequence metadata
    char* pMetadataBytes = m_pqb->MFBm[iMate].buf.p;
    pMetadataBytes += m_pqb->MFBm[iMate].ofs.p[iMetadata];
    MetadataRowHeader* pmrh = reinterpret_cast<MetadataRowHeader*>(pMetadataBytes);

    // return the index into the list of read-group info strings
    return pmrh->byteVal;
}

#pragma warning ( pop )
#pragma endregion
