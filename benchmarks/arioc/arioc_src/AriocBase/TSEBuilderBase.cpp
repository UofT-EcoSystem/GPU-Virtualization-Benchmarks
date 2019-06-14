/*
  TSEBuilderBase.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variables
/* The following defines the fields in the SBF (SQL binary format) file for subsequent bulk insert into a SQL table:
    
        sqId    bigint          not null,     -- bitmap contains srcId (see AriocDS.h)
        FLAG    smallint        not null,     -- SAM FLAG value
        MapCat  tinyint         not null,     -- 0x00: (no mapping type); 0x01: concordant pair; 0x02: discordant pair; 0x04: rejected pair (both ends mapped but not concordant)
        MAPQ    tinyint         not null,     -- SAM MAPQ value
        rgId    tinyint         not null,     -- read group ID (offset into table of read-group info strings)
        subId   tinyint         not null,     -- subunit ID (e.g., chromosome number)
        J       int             not null,     -- 1-based position of the first (upstream) mapped symbol, relative to the strand on which it is mapped
        POS     int             not null,     -- 1-based position of the first (upstream) mapped symbol, relative to the forward strand (i.e., SAM POS)
        spanR   smallint        not null,     -- number of reference positions spanned by the mapping
        sqLen   smallint        not null,     -- read sequence length
        hash64  bigint          not null,     -- 64-bit hash of uppercase ASCII read sequence symbols
        V       smallint        not null,     -- Smith-Waterman alignment score
        NM      smallint        not null,     -- edit distance
        NA      smallint        not null,     -- number of mappings found by the aligner
        minBQS  tinyint         not null,     -- minimum base quality score
        maxBQS  tinyint         not null,     -- maximum base quality score
        BRLEQ   varbinary(150)  not null,     -- binary run-length encoded base quality scores
        BRLEE   varbinary(150)  not null,     -- binary run-length encoded read sequence
        SEQ     varchar(300)    not null      -- sequence
*/
TSEBuilderBase::mfnEmitField TSEBuilderBase::m_emitField[] = { &TSEBuilderBase::emitSqId,
                                                               &TSEBuilderBase::emitFLAG,
                                                               &TSEBuilderBase::emitMapCat,
                                                               &TSEBuilderBase::emitMAPQ,
                                                               &TSEBuilderBase::emitRGID,
                                                               &TSEBuilderBase::emitSpanR,   // subId, J, POS, spanR
                                                               &TSEBuilderBase::emitSqLen,
                                                               &TSEBuilderBase::emitHash64,
                                                               &TSEBuilderBase::emitV,
                                                               &TSEBuilderBase::emitNM,
                                                               &TSEBuilderBase::emitNA,
                                                               &TSEBuilderBase::emitBRLEQ,   // minBQS, maxBQS, BRLEQ
                                                               &TSEBuilderBase::emitBRLEE,
                                                               &TSEBuilderBase::emitSEQ };

// the following list is ordered so that each array index corresponds to a value in enum BRLEAbyteType:
TSEBuilderBase::mfnEmitBRLEE TSEBuilderBase::m_emitBRLEE[] = { &TSEBuilderBase::emitBRLEEmatch,
                                                               &TSEBuilderBase::emitBRLEEgapQ,
                                                               &TSEBuilderBase::emitBRLEEmismatch,
                                                               &TSEBuilderBase::emitBRLEEgapR };

// map Q symbols to BRLEE symbols                                              A21             BRLEE
UINT8 TSEBuilderBase::m_BRLEEsymbolEncode[] = { 0x00,                       // 0x00 (null) --> 0x00 (null)
                                                0x00,                       // 0x01 (null) --> 0x00 (null)
                                                0x03,                       // 0x02 (N)    --> 0x03 (N)
                                                0x03,                       // 0x03 (N)    --> 0x03 (N)
                                                0x04, 0x05, 0x06, 0x07 };   // ACGT        --> ACGT

#pragma endregion

#pragma region constructors and destructor
/// <summary>
/// Implements functionality for reporting alignments for the Terabase Search Engine in SQL Server binary data import/export format
/// </summary>
TSEBuilderBase::TSEBuilderBase( QBatch* pqb ) : TSEFormatBase(pqb, sizeof(TSE_FIXED_FIELDS), arraysize(m_emitField))
{
}

/// [public] destructor
TSEBuilderBase::~TSEBuilderBase()
{
}
#pragma endregion

#pragma region private methods
/// [private] method getQiSymbol
UINT8 TSEBuilderBase::getQiSymbol( const UINT64* pQ, const INT32 i )
{
    // compute the byte offset and the bit offset of the ith symbol
    div_t qr = div( i, 21 );

    // load the 64-bit value that contains the symbol from the interleaved Q sequence data
    UINT64 b21 = pQ[CUDATHREADSPERWARP*qr.quot];

    // shift the symbol into position, mask it, encode it, and return
    return m_BRLEEsymbolEncode[(b21 >> (3*qr.rem)) & 0x07];
}

/// [private] method getQiSymbolPair
UINT8 TSEBuilderBase::getQiSymbolPair( const UINT64* pQ, const INT32 i )
{
    // compute the byte offset and the bit offset of the first symbol
    div_t qr = div( i, 21 );

    // load the 64-bit value that contains the first symbol from the interleaved Q sequence data
    UINT64 b21 = pQ[CUDATHREADSPERWARP*qr.quot];

    UINT8 symbolPair;

    // if the second symbol is in the same 64-bit value, shift the symbol pair into position and mask it
    if( qr.rem < 20 )
        symbolPair = (b21 >> (3*qr.rem)) & 0x3F;

    else        // the second symbol is in the subsequent 64-bit value for the sequence
    {
        UINT64 b21x = pQ[CUDATHREADSPERWARP*(qr.quot+1)];

        symbolPair = static_cast<UINT8>((b21 >> 60) | ((b21x << 3) & 0x38));
    }

    // return the encoded symbols
    return m_BRLEEsymbolEncode[symbolPair & 0x07] | (m_BRLEEsymbolEncode[symbolPair>>3] << 3);
}

/// [private] method: emitBRLEE2bps
UINT32 TSEBuilderBase::emitBRLEE2bps( char* pbuf, INT16 N, UINT64* pQ )
{
    char* p = pbuf;
    UINT64 b21 = 0;
    UINT8 c4 = 0;
    UINT32 shl;

    /* Emit the first BRLEE byte:
        - bits 6..7:  01  ("gap in Q")
        - bit 5    :   0  (2 bits/symbol)
        - bits 3..4:      (unused)
        - bits 0..2:      number of symbols in the final BRLEE byte
    */
    const INT32 cchFinal = 4 - ((4-N) & 3);
    *(p++) = static_cast<char>(0x40 | cchFinal);

    // emit symbols in chunks of 4
    for( INT16 i=0; i<N; i+=4 )
    {
        UINT32 celInChunk = min2( 4, N-i );
        for( shl=0; shl<(2*celInChunk); shl+=2 )
        {
            // if b21 is empty, get the next 21 symbols
            if( b21 == 0 )
            {
                b21 = *pQ;
                pQ += CUDATHREADSPERWARP;
            }

            // abort if the symbol 'N' is encountered
            if( (b21 & 7) == NencodingQ )
                return _UI32_MAX;

            // emit one symbol
            c4 |= (b21 & 3) << shl;

            // shift the next symbol into position
            b21 >>= 3;
        }

        // copy symbols to the target buffer
        *(p++) = c4;
        c4 = 0;
    }

    // return the BRLEE byte count
    return static_cast<UINT32>(p-pbuf);
}

/// [private] method: emitBRLEE3bps
UINT32 TSEBuilderBase::emitBRLEE3bps( char* pbuf, INT16 N, UINT64* pQ )
{
    char* p = pbuf;

    /* Emit the first BRLEE byte:
        - bits 6..7:  01  ("gap in Q")
        - bit 5    :   1  (3 bits/symbol)
        - bits 0..4:      (unused)
    */
    *(p++) = 0x60;

    // copy the A21-encoded Q sequence to the output buffer
    INT16 cel = blockdiv( N, 21 );
    for( INT16 i=0; i<cel; i++ )
    {
        *reinterpret_cast<UINT64*>(p) = *pQ;
        p += sizeof(UINT64);
        pQ += CUDATHREADSPERWARP;
    }

    // trim trailing zero bytes
    while( p[-1] == 0 )
        --p;

    // return the BRLEE byte count
    return static_cast<UINT32>(p-pbuf);
}

/// [private] method: emitBRLEErunLength
void TSEBuilderBase::emitBRLEErunLength( char*& p, BRLEAbyteType bbType, INT32 runLength )
{
    if( runLength <= 0x3F )         // 6 bits
        *(p++) = (bbType << 6) | runLength;

    else
    if( runLength <= 0x0FFF )       // 12 bits
    {
        *(p++) = (bbType << 6) | (runLength >> 6);
        *(p++) = (bbType << 6) | (runLength & 0x3F);
    }

    else                            // up to 18 bits (more than enough)
    {
        *(p++) = (bbType << 6) | (runLength >> 12);
        *(p++) = (bbType << 6) | ((runLength >> 6) & 0x3F);
        *(p++) = (bbType << 6) | (runLength & 0x3F);
    }
}

/// [private] method emitBRLEEsymbols
void TSEBuilderBase::emitBRLEEsymbols( char*& p, BRLEAbyteType bbType, INT32 runLength, const UINT64* pQ, INT32 i )
{
    while( runLength > 1 )
    {
        // emit a pair of symbols
        *(p++) = static_cast<char>((bbType << 6) | getQiSymbolPair( pQ, i ));

        // iterate
        i += 2;         // point to the next symbol in the Q sequence
        runLength -= 2;
    }

    if( runLength )
    {
        // emit the final symbol
        *(p++) = static_cast<char>((bbType << 6)) | getQiSymbol( pQ, i );
    }
}

/// [private] method emitBRLEEmatch
void TSEBuilderBase::emitBRLEEmatch( char*& p, INT32 runLength, const UINT64* pQ, INT32& i )
{
    emitBRLEErunLength( p, bbMatch, runLength );
    i += runLength;
}

/// [private] method emitBRLEEmismatch
void TSEBuilderBase::emitBRLEEmismatch( char*& p, INT32 runLength, const UINT64* pQ, INT32& i )
{
    emitBRLEEsymbols( p, bbMismatch, runLength, pQ, i );
    i += runLength;
}

/// [private] method emitBRLEEgapR
void TSEBuilderBase::emitBRLEEgapR( char*& p, INT32 runLength, const UINT64* pQ, INT32& i )
{
    /* The current BRLEA run represents a gap in the R sequence.  The BRLEE needs to contain the
        corresponding Q symbols so that the gap in R can be "filled" when using the BRLEE to
        reconstruct Q from R.
    */
    emitBRLEEsymbols( p, bbGapR, runLength, pQ, i );
    i += runLength;
}

/// [private] method emitBRLEEgapQ
void TSEBuilderBase::emitBRLEEgapQ( char*& p, INT32 runLength, const UINT64* pQ, INT32& i )
{
    /* The current BRLEA run represents a gap in the Q sequence.  The BRLEE need only contain
        a count of these symbols since they will simply be skipped when using the BRLEE to
        reconstruct Q from R.
    */
    emitBRLEErunLength( p, bbGapQ, runLength );
}
#pragma endregion

#pragma region protected methods
#pragma warning ( push )
#pragma warning( disable:4996 )     // (don't nag us about strcpy being "unsafe")

/// [protected] method emitMAPQ
UINT32 TSEBuilderBase::emitMAPQ( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<UINT8>( pbuf, pQAI->mapQ );
}

/// [protected] method emitSpanR
UINT32 TSEBuilderBase::emitSpanR( char* pbuf, INT64 sqId, QAI* pQAI )
{
    UINT32 cb = intToSBF<UINT8>( pbuf, pQAI->subId );   // subId
    cb += intToSBF<INT32>( pbuf+cb, pQAI->pBH->J+1 );   // J
    cb += intToSBF<INT32>( pbuf+cb, pQAI->Jf+1 );       // Jf
    cb += intToSBF<INT16>( pbuf+cb, pQAI->pBH->Ma );    // spanR
    return cb;
}

/// [protected] method emitV
UINT32 TSEBuilderBase::emitV( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT16>( pbuf, pQAI->pBH->V );
}

/// [protected] method emitNM
UINT32 TSEBuilderBase::emitNM( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT16>( pbuf, pQAI->editDistance );
}

/// [protected] method emitNA
UINT32 TSEBuilderBase::emitNA( char* pbuf, INT64 sqId, QAI* pQAI )
{
    return intToSBF<INT16>( pbuf, pQAI->nAu );
}

/// [protected] method emitBRLEE
UINT32 TSEBuilderBase::emitBRLEE( char* pbuf, INT64 sqId, QAI* pQAI )
{
    /*
       BRLEE -- Binary Run-Length Encoded Edit string

       The goals of the BRLEE encoding are:
        - conserve space in representing a read (AKA "query" or "Q") sequence
        - reconstruct a Q sequence given a start location in a reference ("R") sequence

       The BRLEE format records matching symbols and deletions in Q (symbols "skipped" in R when using the
        BRLEE to construct a Q sequence) as run lengths.  It records mismatched symbols and insertions
        (symbols inserted directly into Q when using the BRLEE to construct a Q) in bit fields.
  
       A total length (number of BRLEE bytes) is associated with each BRLEE.  The length may be zero, but
        the meaning of a zero-length BRLEE string is undefined.

       Symbols are represented as 3-bit values, as follows:
            000 (null)
            001 (null)
            010 (null)
            011 N
            100 A
            101 C
            110 G
            111 T

       There is no length or count associated with a series of symbols.  Instead, the end of the series is indicated
        either with a null symbol value (e.g., 000) or by encountering a different BRLEE byte type.

       Bit mapping:
            Bits 6 and 7 of each BRLEE byte determine the meaning of bits 0 through 5.
  
            bits  value       description                     bits 0..5
            6..7   00         match                           run length
                   01         gap in Q (deletion from R)      run length
                   10         mismatch                        bits 0..2: symbol 1; bits 3..5: symbol 2
                   11         gap in R (insertion into R)     bits 0..2: symbol 1; bits 3..5: symbol 2
              
            A compact 2 bits/symbol representation is used to represent a single run of ACGT symbols (i.e., to
            encode an unmapped sequence that contains only A, C, G, and T).  A similar 3 bits/symbol
            representation is used when the sequence contains one or more Ns.  Since a mapping cannot start
            with a gap in the query sequence, the initial byte of both the 2 bits/symbol and the 3 bits/symbol
            packings is formatted as a "gap in Q" byte type, as follows:
              
            bits  value  description 
            6..7   01    ("gap in Q")
            5..5    0    2 bits/symbol (4 symbols/byte)
                    1    3 bits/symbol (2 symbols/byte)
            3..4   00
            0..2         number of symbols in the final BRLEE byte (used only for 2 bits/symbol encoding)

            All subsequent bytes contain encoded symbols.  For 2-bit encoded symbols (00: A; 01: C; 10: G; 11: T)
            the bitmap is:
            6..7    symbol 4
            4..5    symbol 3
            2..3    symbol 2
            0..1    symbol 1
            For 3-bit encoded symbols, the bitmap is:
            6..7    00
            3..5    symbol 2
            0..2    symbol 1
             
            The number of symbols represented in the final byte is specified by bits 0..2 in the initial byte.

       Run length is recorded for the following two types of runs:
        - matching symbols (BRLEE byte type = 0)
        - deletions in the reference (BRLEE byte type = 1)
       For these types of runs, the run length is accumulated by concatenating the 6-bit values in big-endian order.
       Run-length accumulation is terminated either by a byte with a different type or by the end of the BRLEE string.
              
       Examples:
            length  binary                                      hexadecimal   description
            1       00 010000                                   10            16 matching symbols
            2       00 000001  00 100100                        01 24         100 matching symbols
            4       00 001010  10 110101  00 000001  00 100010  0A B5 01 22   10 matching symbols, 2 mismatches (CG), 98 matching symbols
            4       01 000011  10011100 00010111 00010111       43 9C 17 17   ATCGTCCATCC 
    */
    UINT32 cb = 0;

    // if the read is mapped...
    if( pQAI->flags & qaiMapped )
    {
        // if the read is a perfect match, emit an empty string (i.e., a 16-bit "length prefix" of zero)
        if( pQAI->pBH->V == (m_pab->aas.ASP.Wm * pQAI->N) )
            return intToSBF<INT16>( pbuf, 0 );

        // the read is not a perfect match, so emit a BRLEE
        cb = emitBRLEEfr( pbuf+sizeof(INT16), pQAI );
    }

    // if the read is unmapped or if it could not be run-length encoded
    if( cb == 0 )
        cb = emitBRLEEu( pbuf+sizeof(INT16), pQAI );    // emit a "non-RLE" BRLEE

    // save the length prefix and return the total number of bytes appended to the output buffer
    *reinterpret_cast<INT16*>(pbuf) = static_cast<INT16>(cb);
    return cb + sizeof(INT16);
}

/// [private] method emitBRLEEfr
UINT32 TSEBuilderBase::emitBRLEEfr( char* pbuf, QAI* pQAI )
{
    // point to the output buffer
    char* p = pbuf;

    // point to the BRLEE
    BRLEAheader* const pBH = pQAI->pBH;
    BRLEAbyte* pBB = reinterpret_cast<BRLEAbyte*>(pBH+1);
    BRLEAbyte* pBBlimit = pBB + pBH->cb;

    // point to the encoded interleaved sequence data
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(pQAI->qid);
    UINT64* pQ = m_pqb->QiBuffer.p + pQw->ofsQi + QID_IQ(pQAI->qid);
    INT32 i = 0;

    // point to the start of the reference sequence for the specified read
    const UINT64* pR = const_cast<UINT64*>(m_pab->R);
    pR += (pBH->J & 0x80000000) ? m_pab->ofsRminus.p[pQAI->subId] : m_pab->ofsRplus.p[pQAI->subId];

    // look for "soft clipping" at the start of the alignment
    INT32 nClipped = pBH->Il;
    if( nClipped )
        emitBRLEEgapR( p, nClipped, pQ, i );

    // traverse the BRLEE
    BRLEAbyteType bbPrev = bbMatch;     // (a BRLEA always starts with a match)
    INT32 runLength = 0;

    while( pBB < pBBlimit )
    {
        if( pBB->bbType != bbPrev )
        {
            // emit the previous BRLEE run
            (this->*(m_emitBRLEE[bbPrev]))(p, runLength, pQ, i);

            // reset the run length
            runLength = 0;
            bbPrev = static_cast<BRLEAbyteType>(pBB->bbType);
        }

        // accumulate the run length
        runLength = (runLength << 6) | pBB->bbRunLength;

        pBB++;
    }

    // emit the final BRLEE run
    (this->*(m_emitBRLEE[(pBBlimit-1)->bbType]))(p, runLength, pQ, i);

    // look for "soft clipping" at the end of the alignment
    nClipped = pQAI->N - (pBH->Ir + 1);
    if( nClipped )
        emitBRLEEgapR( p, nClipped, pQ, i );

    return static_cast<UINT32>(p - pbuf);
}

/// [private] method emitBRLEEu
UINT32 TSEBuilderBase::emitBRLEEu( char* pbuf, QAI* pQAI )
{
    // point to the encoded interleaved sequence data
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(pQAI->qid);
    UINT64* pQ = m_pqb->QiBuffer.p + pQw->ofsQi + QID_IQ(pQAI->qid);

    /* An unmapped read has no reference sequence to edit, so we emit a string of encoded symbols instead.

       If the Q sequence contains only ACGT, we can encode using 2 bits/symbol; otherwise, we use A21 encoding
        (i.e., 3 bits/symbol).  But because Q sequences containing Ns are expected to be relatively scarce, we
        always start with the 2bps encoder and switch to A21 encoding if an N is encountered.
    */
    UINT32 cb = emitBRLEE2bps( pbuf, pQAI->N, pQ );
    if( cb == _UI32_MAX )
        cb = emitBRLEE3bps( pbuf, pQAI->N, pQ );

    return cb;
}
#pragma warning ( pop )
#pragma endregion
