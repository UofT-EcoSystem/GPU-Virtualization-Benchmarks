/*
  SAMFormatBase.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"


#if TODO_CHOP_WHEN_DEBUGGED
static bool isDebug = false;
static int nDebugHit = 0;
#endif



#pragma region static member variables
// the following list is ordered so that each array index corresponds to a value in enum BRLEAbyteType:
SAMFormatBase::mfnEmitXM SAMFormatBase::m_emitXM[] = { &SAMFormatBase::emitXMmatch,
                                                       &SAMFormatBase::emitXMgapQ,
                                                       &SAMFormatBase::emitXMmismatch,
                                                       &SAMFormatBase::emitXMgapR };

// the following lookup table converts raw XM info into Bismark-compatible XM symbols (see initializer() and convertXM())
INT32 SAMFormatBase::initRval = SAMFormatBase::initializer();
char SAMFormatBase::m_ctxXM[512] = { 0 };
#pragma endregion

#pragma region constructors and destructor
/// <summary>
/// Implements base functionality for paired and unpaired SAM-formatted file output
/// </summary>
SAMFormatBase::SAMFormatBase( QBatch* pqb, INT16 nEmitFields, bool emitA21TraceFields ) : baseARowBuilder( pqb, nEmitFields, emitA21TraceFields )
{
    // set up to emit CIGAR strings in the user-specified format
    switch( pqb->pab->CFT )
    {
        case cftQXIDS:
            m_CIGARsymbolMID = "=DXI";
            m_CIGARsymbolS = 'S';
            break;

        case cftMIDS:
            m_CIGARsymbolMID = "MDMI";
            m_CIGARsymbolS = 'S';
            break;

        case cftMID:
            m_CIGARsymbolMID = "MDMI";
            m_CIGARsymbolS = 'I';
            break;
    
        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for CIGAR format type: %d", pqb->pab->CFT );
    }

    // set up to emit MD strings in the user-specified format
    switch( pqb->pab->MFT )
    {
        case mftStandard:
            m_emitMDsymbols = &SAMFormatBase::emitMDsymbolsStandard;
            m_emitMDstandard = true;
            break;

        case mftCompact:
            m_emitMDsymbols = &SAMFormatBase::emitMDsymbolsCompact;
            m_emitMDstandard = false;
            break;
    
        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for MD format type: %d", pqb->pab->MFT );
    }
};

/// [public] destructor
SAMFormatBase::~SAMFormatBase()
{
#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: nDebugHit=%d", __FUNCTION__, nDebugHit );
#endif
}
#pragma endregion

#pragma region private methods
/// [private static] method initializer
INT32 SAMFormatBase::initializer()
{
    /* The idea is that three consecutive symbols constitute the "context" for each position in
        a SAM XM attribute string.  When three consecutive symbols s0, s1, and s2 are represented
        in 3 3-bit values, there are 512 possible contexts for s0.

       The 3-bit values in the raw XM info are:     
        0x00  000  (error)
        0x01  001  N
        0x02  010  A
        0x03  011  C
        0x04  100  Cm (methylcytosine)
        0x05  101  G
        0x06  110  T
    */
    for( INT16 n=0; n<static_cast<INT16>(arraysize(SAMFormatBase::m_ctxXM)); ++n )
    {
        INT16 nn = n & 7;                           // isolate bits 0-2

        if( (nn == 0) || ((n & 0x0038) == 0) || ((n & 0x01C0) == 0) )
        {
            // if any of the three symbols is zero, we have an error
            m_ctxXM[n] = '?';
            continue;
        }

        if( (nn != 3) && (nn != 4) )
        {
            // it's not C or Cm, so there is no methylation context
            m_ctxXM[n] = '.';
            continue;
        }

        if( (n & 0x0038) == 0x0028 )                // 0x0028 = 101xxxb = G in bits 3-5
        {
            // CG (CpG) context
            m_ctxXM[n] = (nn == 3) ? 'z' : 'Z';
            continue;
        }

        if( ((n & 0x0038) == 0x0008) || ((n & 0x01C0) == 0x0040) )  // 0x0008 = 001xxx = N in bits 3-5; 0x0040 = 001xxxxxx = N in bits 6-8
        {
            // unknown (CN or CHN) context
            m_ctxXM[n] = (nn == 3) ? 'u' : 'U';
            continue;
        }

        /* At this point we have either CHG or CHH. */
        if( (n & 0x01C0) == 0x0140 )                // 0x0140 = 101xxxxxxb = G in bits 6-8
            m_ctxXM[n] = (nn == 3) ? 'x' : 'X';
        else
            m_ctxXM[n] = (nn == 3) ? 'h' : 'H';
    }

    return 0;
}

/// [private] method getRunLengthBRLEA
inline INT16 SAMFormatBase::getRunLengthBRLEA( BRLEAbyte*& pB, const BRLEAbyte* pBlimit )
{
    // get the run length from the first BRLEA byte and point to the subsequent BRLEA byte
    INT16 cch = (pB++)->bbRunLength;

    // loop while the BRLEA byte type remains the same
    while( (pB[0].bbType == pB[-1].bbType) && (pB < pBlimit) )
        cch = (cch << 6) | (pB++)->bbRunLength;  // accumulate the run length

    return cch;
}

/// [private] method emitMDsymbolsStandard
void SAMFormatBase::emitMDsymbolsStandard( char*& p, UINT32 j, INT16 cchRun, const UINT64* pR0 )
{
    /* This implementation emits MD symbols that conform to the regex in the SAM Format specification:
            [0-9]+(([A-Z]|\^[A-Z]+)[0-9]+)*
    */
    // isolate the 3-bit representation of the current R symbol
    div_t qr = div( j, 21 );                    // quotient: offset of the 64-bit value that contains the symbol; remainder: 0-based position of symbol
    UINT64 b21 = pR0[qr.quot] >> (3*qr.rem);    // shift the symbol into bits 0-2

    // emit the first R symbol
    *(p++) = m_symbolDecode[b21 & 7];

    // emit the remaining R symbols
    while( --cchRun )
    {
        b21 >>= 3;                                      // shift the next 3-bit symbol into bits 0-2
        if( b21 == 0 )                                  // if no symbols remain in the 64-bit value ...
            b21 = pR0[++qr.quot];                       // ... get the next 64-bit value

        *(p++) = '0';                                   // emit a zero
        *(p++) = m_symbolDecode[b21 & 7];               // emit the next R symbol
    }
}

/// [private] method emitMDsymbolsCompact
void SAMFormatBase::emitMDsymbolsCompact( char*& p, UINT32 j, INT16 cchRun, const UINT64* pR0 )
{
    /* This implementation omits extraneous zeroes from the MD string, but it does    
        not conform to the regex in the SAM Format specification. */

    // isolate the 3-bit representation of the current R symbol
    div_t qr = div( j, 21 );                    // quotient: offset of the 64-bit value that contains the symbol; remainder: 0-based position of symbol
    UINT64 b21 = pR0[qr.quot] >> (3*qr.rem);    // shift the symbol into bits 0-2

    // emit the first R symbol
    *(p++) = m_symbolDecode[b21 & 7];

    // emit the remaining R symbols
    while( --cchRun )
    {
        b21 >>= 3;                                      // shift the next 3-bit symbol into bits 0-2
        if( b21 == 0 )                                  // if no symbols remain in the 64-bit value ...
            b21 = pR0[++qr.quot];                       // ... get the next 64-bit value

        *(p++) = m_symbolDecode[b21 & 7];               // emit the next R symbol
    }
}

/// [private] method emitCIGARsymbols
void SAMFormatBase::emitCIGARsymbols( char*& p, BRLEAbyte* pB, INT16 cbBRLEA )
{
    BRLEAbyte* pBlimit = pB + cbBRLEA;

    /* Map values from enum BRLEAbyteType to CIGAR symbols:
                                // "old" format            "new" format
        bbMatch     =   0x00,   // M (aligned)             =
        bbGapQ      =   0x01,   // D (deletion from R)     D
        bbMismatch  =   0x02,   // M (aligned)             X
        bbGapR      =   0x03    // I (insertion into R)    I
    */

    // get the first run type and length
    char cRun = m_CIGARsymbolMID[pB->bbType];
    INT16 cchRun = getRunLengthBRLEA( pB, pBlimit );

    // iterate through the runs in the BRLEA
    while( pB < pBlimit )
    {
        // get the next run type and length
        char cNext = m_CIGARsymbolMID[pB->bbType];
        INT16 cchNext = getRunLengthBRLEA( pB, pBlimit );

        // if adjacent runs are of the same type (i.e., either bbMatch and bbMismatch), accumulate the run lengths
        if( cNext == cRun )
            cchRun += cchNext;
        else
        {
            // emit the current run
            p += u32ToString( p, cchRun );
            *(p++) = cRun;

            // reset the run type and length
            cRun = cNext;
            cchRun = cchNext;
        }
    }

    // emit the final run
    p += u32ToString( p, cchRun );
    *(p++) = cRun;
}

/// [private] method getNextR
inline UINT8 SAMFormatBase::getNextR( UINT64& R, const UINT64*& pR )
{
    /* At this point:
        - R contains the MRU symbol in bits 0-2
        - pR points to the next A21 value
    */

    // get the next R symbol into bits 0-2 of the A21-formatted R symbols
    R = ((R > 7) ? (R >> 3) : *(pR++));

    // return bits 0-2 of the A21-formatted R symbols
    return R & 7;
}

/// [private] method getNextQiF
inline UINT8 SAMFormatBase::getNextQiF( UINT64& Qi, const UINT64*& pQi )
{
    // get the next Q symbol into bits 0-2 of the A21-formatted Q symbols
    if( Qi > 7 )
        Qi >>= 3;
    else
    {
        Qi = *pQi;
        pQi += CUDATHREADSPERWARP;
    }

    // return bits 0-2 of the A21-formatted Q symbols
    return Qi & 7;
}

/// [private] method getNextQiRC
inline UINT8 SAMFormatBase::getNextQiRC( UINT64& Qi, const UINT64*& pQi )
{
    // get the next Q symbol into bits 0-2 of the A21-formatted Q symbols
    if( Qi > 7 )
        Qi >>= 3;
    else
    {
        // get the reverse complement of the specified A21-formatted 64-bit value
        Qi = AriocCommon::A21ReverseComplement( *pQi );

        // shift the rightmost non-null 3-bit symbol into the least significant bits
        UINT32 lsb = 0;
        _BitScanForward64( reinterpret_cast<DWORD*>(&lsb), Qi );
        if( lsb )
            Qi >>= 3 * (lsb/3);

        pQi -= CUDATHREADSPERWARP;
    }

    // return bits 0-2 of the A21-formatted Q symbols
    return Qi & 7;
}

/// [private] method emitXMmatch
void SAMFormatBase::emitXMmatch( char*& p, INT32 runLength, UINT64& R, const UINT64*& pR, UINT64& Qi, const UINT64*& pQi, INT32& i, mfnGetNextQi getNextQi )
{
    /* The current BRLEA run represents one or more consecutive matches between the Q and R sequences.
        The only interesting match is between C in R and T in Q, which represents a bisulfite-converted C.
        
                binary  decimal  symbol     raw XM
                000     0        (null)     \x00 (error) 
                001     1        (null)     \x00 (error)
                010     2        Nq         \x01 (N)
                011     3        Nr         \x01 (N)
                100     4        A          \x02 (A)
                101     5        C          \x04 (methylcytosine)
                110     6        G          \x05 (G)
                111     7        T          \x03 (C) if mismatched to C in R; otherwise: 0x06 (T)
    */
    i += runLength;

    while( runLength-- )
    {
        UINT8 q = (this->*getNextQi)( Qi, pQi );
        UINT8 r = getNextR( R, pR );

#if TODO_CHOP_WHEN_DEBUGGED
        //if( isDebug )
            CDPrint( cdpCD0, "%s: i=%3d runLength=%d R=%c Q=%c", __FUNCTION__, i, runLength, "..NNACGT"[r], "..NNACGT"[q] );
#endif

        *(p++) = (r == 5) ? "\x00\x00\x01\x01\x02\x04\x05\x03"[q] : "\x00\x00\x01\x01\x02\x04\x05\x06"[q];
    }
}

/// [private] method emitXMmismatch
void SAMFormatBase::emitXMmismatch( char*& p, INT32 runLength, UINT64& R, const UINT64*& pR, UINT64& Qi, const UINT64*& pQi, INT32& i, mfnGetNextQi getNextQi )
{
    /* The current BRLEA run represents one or more consecutive mismatches between the Q and R sequences.
        The only interesting mismatch is between C in R and T in Q, which represents a bisulfite-converted C.
        
                binary  decimal  symbol     raw XM
                000     0        (null)     \x00 (error) 
                001     1        (null)     \x00 (error)
                010     2        Nq         \x01 (N)
                011     3        Nr         \x01 (N)
                100     4        A          \x02 (A)
                101     5        C          \x04 (methylcytosine)
                110     6        G          \x05 (G)
                111     7        T          \x03 (C) if mismatched to C in R; otherwise: 0x06 (T)
    */
    i += runLength;

    while( runLength-- )
    {
        UINT8 q = (this->*getNextQi)( Qi, pQi );
        UINT8 r = getNextR( R, pR );

#if TODO_CHOP_WHEN_DEBUGGED
   // if( isDebug )
        CDPrint( cdpCD0, "%s: i=%3d runLength=%d R=%c Q=%c", __FUNCTION__, i, runLength, "..NNACGT"[r], "..NNACGT"[q] );
#endif
    
        *(p++) = (r == 5) ? "\x00\x00\x01\x01\x02\x00\x05\x03"[q] : "\x00\x00\x01\x01\x02\x04\x05\x06"[q];
    }
}

/// [private] method emitXMgapR
void SAMFormatBase::emitXMgapR( char*& p, INT32 runLength, UINT64& R, const UINT64*& pR, UINT64& Qi, const UINT64*& pQi, INT32& i, mfnGetNextQi getNextQi )
{
    /* The current BRLEA run represents a gap in the R sequence.  The Q symbols have no corresponding
        R symbols, so we have the following mappings:
        
                binary  decimal  symbol     raw XM
                000     0        (null)     \x00 (error) 
                001     1        (null)     \x00 (error)
                010     2        Nq         \x01 (N)
                011     3        Nr         \x01 (N)
                100     4        A          \x02 (A)
                101     5        C          \x04 (methylcytosine)
                110     6        G          \x05 (G)
                111     7        T          \x01 (N) since we do not know whether we have a T or a bisulfite-converted C
    */
    i += runLength;

    while( runLength-- )
    {
        UINT8 s = (this->*getNextQi)( Qi, pQi );

#if TODO_CHOP_WHEN_DEBUGGED
    //if( isDebug )
        CDPrint( cdpCD0, "%s: Q=%c", __FUNCTION__, "..NNACGT"[s] );
#endif

        *(p++) = "\x00\x00\x01\x01\x02\x04\x05\x01"[s];
    }
}

/// [private] method emitXMgapQ
void SAMFormatBase::emitXMgapQ( char*& p, INT32 runLength, UINT64& R, const UINT64*& pR, UINT64& Qi, const UINT64*& pQi, INT32& i, mfnGetNextQi getNextQi )
{
    /* The current BRLEA run represents a gap in the R sequence. */
    while( runLength-- )
        getNextR( R, pR );
}

/// [private] method convertXM
void SAMFormatBase::convertXM( char* p, UINT32 cb, QAI* pQAI )
{
    /* The raw XM info emitted by parsing the BRLEA is encoded as follows (see above):
    
            0x00    (error)
            0x01    N
            0x02    A
            0x03    C
            0x04    Cm (methylcytosine)
            0x05    G
            0x06    T
    
       Here we replace the raw XM info with methylation-context calls in Bismark-compatible format.
    */

    // append two Ns so that the final two base positions can be converted
    *reinterpret_cast<UINT16*>(p+cb) = 0x0101;

    // encode
    for( UINT32 i=0; i<cb; ++i )
    {
        *p = m_ctxXM[p[0] | (p[1] << 3) | (p[2] << 6)];
        ++p;
    }

    // if the read maps to the reverse complement of R, reverse the XM (i.e., do what Bismark does)
    if( pQAI->flags & qaiRCr )
    {
        *p = 0;
        _strrev( p-cb );
    }
}
#pragma endregion

#pragma region protected methods
/// [protected] method emitCIGARf
UINT32 SAMFormatBase::emitCIGARf( char* pbuf, QAI* pQAI )
{
    char* p = pbuf;

    // look for "soft clipping" at the start of the alignment
    if( pQAI->pBH->Il )
    {
        p += u32ToString( p, pQAI->pBH->Il );
        *(p++) = m_CIGARsymbolS;
    }

    // emit the CIGAR symbols using the specified BRLEA
    emitCIGARsymbols( p, reinterpret_cast<BRLEAbyte*>(pQAI->pBH+1), pQAI->pBH->cb );

    // look for "soft clipping" at the end of the alignment
    INT16 nClipped = pQAI->N - (pQAI->pBH->Ir + 1);
    if( nClipped )
    {
        p += u32ToString( p, nClipped );
        *(p++) = m_CIGARsymbolS;
    }

    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitCIGARr
UINT32 SAMFormatBase::emitCIGARr( char* pbuf, QAI* pQAI )
{
    char* p = pbuf;

    // look for "soft clipping" at the start of the alignment
    INT16 nClipped = pQAI->N - (pQAI->pBH->Ir + 1);
    if( nClipped )
    {
        p += u32ToString( p, nClipped );
        *(p++) = m_CIGARsymbolS;
    }

    // emit the CIGAR symbols using the reverse BRLEA
    emitCIGARsymbols( p, m_revBRLEA.p, pQAI->pBH->cb );

    // look for "soft clipping" at the end of the alignment
    if( pQAI->pBH->Il )
    {
        p += u32ToString( p, pQAI->pBH->Il );
        *(p++) = m_CIGARsymbolS;
    }

    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitQUALf
void SAMFormatBase::emitQUALf( char* pbuf, QAI* pQAI, char* pMbuf )
{
    /* Copy the metadata into the output buffer:
        - we assume that the number of bytes of quality score data is the same as the Q sequence length
        - the quality scores are recorded as binary values, so we need to adjust by the bias specifed in the SAM specification,
           which is 33 (i.e., Sanger FASTQ format)
        - to minimize the number of memory accesses, we traverse the quality score bytes in two loops, the first of which processes 8 bytes per iteration
    */
    INT16 iCut = pQAI->N & (-static_cast<INT16>(sizeof(INT64)));
    for( INT16 i=0; i<iCut; i+=sizeof(INT64) )
    {
        // emit eight (i.e. sizeof(INT64)) quality scores at a time
        INT64 q8 = *reinterpret_cast<INT64*>(pMbuf);    // copy 8 quality scores
        q8 += 0x2121212121212121;                       // adjust all 8 quality scores
        *reinterpret_cast<INT64*>(pbuf) = q8;           // save 8 quality scores as ASCII symbols

        // update the buffer pointers
        pMbuf += sizeof(INT64);
        pbuf += sizeof(INT64);
    }

    // copy the remaining scores
    for( INT16 i=iCut; i<pQAI->N; ++i )
        *(pbuf++) = *(pMbuf++) + 0x21;
}

/// [protected] method emitQUALr
void SAMFormatBase::emitQUALr( char* pbuf, QAI* pQAI, char* pMbuf )
{
    // point to the byte in the output buffer at the tail of the quality scores
    pbuf += pQAI->N;

    /* Copy the metadata into the output buffer:
        - we assume that the number of bytes of quality score data is the same as the Q sequence length
        - since the quality scores are recorded as binary values, we need to adjust by the bias specifed in the SAM specification,
           which is 33 (i.e., Sanger FASTQ format)
        - to minimize the number of memory accesses, we traverse the quality score bytes in two loops, the first of which processes 8 bytes per iteration
    */
    INT16 iCut = pQAI->N & (-static_cast<INT16>(sizeof(UINT64)));
    for( INT16 i=0; i<iCut; i+=sizeof(UINT64) )
    {
        // emit eight (i.e. sizeof(INT64)) quality scores at a time
        UINT64 q8 = *reinterpret_cast<UINT64*>(pMbuf);  // copy 8 quality scores
        q8 += 0x2121212121212121;                       // adjust all 8 quality scores
        q8 = _byteswap_uint64( q8 );                    // reverse the byte order

        // update the buffer pointers
        pMbuf += sizeof(UINT64);
        pbuf -= sizeof(UINT64);

        // write the 8-byte value; since this is Intel hardware, we don't worry about the target address being 8-byte aligned
        *reinterpret_cast<UINT64*>(pbuf) = q8;           // save 8 quality scores as ASCII symbols
    }

    // copy the remaining scores
    for( INT16 i=iCut; i<pQAI->N; ++i )
        *(--pbuf) = *(pMbuf++) + 0x21;
}

/// [protected] method emitQUALfr
UINT32 SAMFormatBase::emitQUALfr( char* pbuf, QAI* pQAI, UINT32 iMate, UINT32 iMetadata )
{
    // if there are no quality scores, emit only a star
    if( !m_pqb->MFBq[0].buf.Count )
    {
        *pbuf = '*';
        return 1;
    }

    // point to the quality score metadata for the current Q sequence
    char* pMbuf = m_pqb->MFBq[iMate].buf.p + m_pqb->MFBq[iMate].ofs.p[iMetadata] + sizeof(MetadataRowHeader);

    // emit either forward or reverse quality scores
    switch( pQAI->flags & (qaiRCr|qaiRCq) )
    {
        case qaiRCr:        // Qf mapped to Rrc
        case qaiRCq:        // Qrc mapped to Rf
            emitQUALr( pbuf, pQAI, pMbuf );
            break;

        default:
            emitQUALf( pbuf, pQAI, pMbuf );
            break;    
    }

    // (there is one quality score for each symbol in the sequence)
    return pQAI->N;
}

/// [protected] method emitMDfr
UINT32 SAMFormatBase::emitMDfr( char* pbuf, QAI* pQAI )
{
    char* p = pbuf;

    // initialize a pointer to the forward reference sequence for the specified read
    const UINT64* pR0 = const_cast<UINT64*>(m_pab->R);
    pR0 += m_pab->ofsRplus.p[pQAI->subId];

    // if the read aligned to the reverse complement strand, use the reverse BRLEA
    BRLEAbyte* pB = (pQAI->flags & qaiRCr) ? m_revBRLEA.p : reinterpret_cast<BRLEAbyte*>(pQAI->pBH+1);

    // transform the BRLEA into an MD string
    BRLEAbyte* pBlimit = pB + pQAI->pBH->cb;
    BRLEAbyteType bbTypePrev = bbMatch;
    UINT32 j = pQAI->Jf;

    do
    {
        BRLEAbyteType bbType = static_cast<BRLEAbyteType>(pB->bbType);
        INT16 cchRun = getRunLengthBRLEA( pB, pBlimit );

        // special handling for a gap in the R sequence
        while( (pB < pBlimit) && (pB->bbType == bbGapR) )
        {
            getRunLengthBRLEA( pB, pBlimit );       // ignore the gap in R

            if( bbType != pB->bbType )              // fall out of the loop if the current run is complete
                break;

            // at this point we have two of the same type of run (bbMatch, bbMismatch, bbGapQ) separated by the gap in R
            cchRun += getRunLengthBRLEA( pB, pBlimit );     // consolidate the run lengths
        }

        switch( bbType )
        {
            case bbMatch:
                p += u32ToString( p, cchRun );      // emit the run length
                j += cchRun;                        // track the number of symbols represented in the MD string
                break;

            case bbMismatch:
                if( bbTypePrev == bbGapQ )          // if the mismatch is preceded by a gap in Q ...
                    *(p++) = '0';                   // ... use a zero-length "run" to separate the gap symbols from the mismatch symbols

                (this->*m_emitMDsymbols)( p, j, cchRun, pR0 );
                j += cchRun;
                break;

            case bbGapQ:
                if( (bbTypePrev == bbMismatch) && m_emitMDstandard )    // if the gap is preceded by a mismatch
                {                                                       //  and we need to conform to the SAM format spec,
                    *(p++) = '0';                                       //  use a zero-length "run" to separate the mismatch
                }                                                       //  from the gap symbols

                *(p++) = '^';                       // emit the start-of-gap symbol

                emitMDsymbolsCompact( p, j, cchRun, pR0 );
                j += cchRun;
                break;

            default:
                break;
        }

        bbTypePrev = bbType;
    }
    while( pB < pBlimit );

    return static_cast<UINT32>(p - pbuf);
}

/// [protected] method emitQNAMEpu
INT16 SAMFormatBase::emitQNAMEpu( char* pbuf, QAI* pQAI, UINT32 iMate, UINT32 iMetadata )
{
    // point to the Q sequence metadata
    char* pMetadataBytes = m_pqb->MFBm[iMate].buf.p;
    pMetadataBytes += m_pqb->MFBm[iMate].ofs.p[iMetadata];
    MetadataRowHeader* pmrh = reinterpret_cast<MetadataRowHeader*>(pMetadataBytes);


#if TODO_CHOP_WHEN_DEBUGGED
    isDebug = (pmrh->sqId == 0x0000040800010000);
    if( isDebug )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif


    // copy the metadata into the output buffer
    memcpy( pbuf, pmrh+1, pmrh->cb );

    return pmrh->cb;
}

#pragma warning ( push )
#pragma warning( disable:4996 )     // (don't nag us about strcpy being "unsafe")

/// [protected] method emitSubIdAsRNAME
UINT32 SAMFormatBase::emitSubIdAsRNAME( char* pbuf, INT16 subId )
{
    /* emit the subunit ID as 3 ASCII digits, padded left with zero (e.g. formatted as "%03d") */

    // initialize the buffer with two ASCII 0 characters
    strcpy( pbuf, "00" );

    // build the ASCII representation of the subunit ID at offset 2
    INT16 nDigits = u32ToString( pbuf+2, subId );

    // if the subunit ID is more than a single digit, we need to lose one or more of the preceding ASCII zeroes
    if( nDigits > 1 )
        *reinterpret_cast<UINT32*>(pbuf) = *reinterpret_cast<UINT32*>(pbuf+nDigits-1);

    // emit a tab character and return the total number of characters emitted
    pbuf[3] = '\t';
    return 4;           // 4: three-digit subId plus one trailing tab
}

/// [protected] method emitRGID
INT16 SAMFormatBase::emitRGID( char* pbuf, QAI* pQAI, UINT32 iMate, UINT32 iMetadata )
{
    // point to the Q sequence metadata
    char* pMetadataBytes = m_pqb->MFBm[iMate].buf.p;
    pMetadataBytes += m_pqb->MFBm[iMate].ofs.p[iMetadata];
    MetadataRowHeader* pmrh = reinterpret_cast<MetadataRowHeader*>(pMetadataBytes);

    // point to the read-group ID string
    char* pRGID = m_pab->paamb->RGMgr.RG.p + m_pab->paamb->RGMgr.OfsRG.p[pmrh->byteVal];

    // copy the read-group ID into the output buffer; the read-group ID string is preceded by its length (see SAMHDBuilder::saveRGIDs)
    INT16 cb = *reinterpret_cast<INT16*>(pRGID);
    memcpy( pbuf, pRGID+sizeof(INT16), cb );

    return cb;
}
#pragma warning( pop )

/// [protected] method emitXMfr
UINT32 SAMFormatBase::emitXMfr( char* pbuf, QAI* pQAI )
{
    /* Although it is possible to peek at the R sequence to determine methylation context, we rely exclusively on
        the Q sequence for this.  We do, of course, use R to determine whether T in Q represents an unmethylated
        cytosine (C); in contrast, any C in Q is assumed to represent methylcytosine (#):

        R:    C  H  .  C  H  .
        Q:    C  C  C  T  T  T
        base: #  #  #  C  T  T

       An unpaired T in Q is assumed to be a T (because we have no evidence that it represents a converted C).

       Computing a methylation context is thus a two-pass procedure:
        - determine methylation state for all C in Q
        - use Q to determine methylation context for all C, #, and ?.
    */

    // point to the output buffer
    char* p = pbuf;

    // point to the BRLEA
    BRLEAheader* const pBH = pQAI->pBH;
    BRLEAbyte* pBB = reinterpret_cast<BRLEAbyte*>(pBH+1);
    BRLEAbyte* pBBlimit = pBB + pBH->cb;

    // point to the encoded interleaved sequence (Qi) data
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(pQAI->qid);
    const INT16 iq = QID_IQ(pQAI->qid);
    const UINT64* pQi = m_pqb->QiBuffer.p + pQw->ofsQi + iq;

    // set up for forward or reverse-complement Q sequence
    mfnGetNextQi getNextQi;

    if( pQAI->flags & qaiRCq )
    {
        pQi += (blockdiv(pQw->N[iq],21) - 1) * CUDATHREADSPERWARP;
        getNextQi = &SAMFormatBase::getNextQiRC;
    }
    else
        getNextQi = &SAMFormatBase::getNextQiF;

    // point to the start of the reference sequence for the specified read
    const UINT64* pR = const_cast<UINT64*>(m_pab->R);
    pR += (pBH->J & 0x80000000) ? m_pab->ofsRminus.p[pQAI->subId] : m_pab->ofsRplus.p[pQAI->subId];
    INT32 j = static_cast<INT32>(pBH->J & 0x7FFFFFFF);  // isolate bits 0-30
    j = (j << 1) >> 1;                                  // sign-extend
    div_t qr = div( ((j >= 0) ? j : (j-20)), 21 );      // quot: index of A21-formatted R data element; rem: index of symbol within A21-formatted data
    pR += qr.quot;

    // prepare to iterate across the R symbols by positioning the "current" Rj prior to the start (see getNextR).
    UINT64 R = qr.rem ? (*(pR++) >> (3 * (qr.rem-1))) : 0;

#if TODO_CHOP_WHEN_DEBUGGED
    // debug the first instance of the following combination of flags and j%21
    if( (pQAI->flags & qaiRCq) && (pQAI->flags & qaiRCr) && !qr.rem )
    {
        if( nDebugHit == 0 )
        {
            CDPrint( cdpCD0, "%s: qaiRCq=%d qaiRCr=%d qr.rem=%d", __FUNCTION__, ((pQAI->flags & qaiRCq) ? 1 : 0), ((pQAI->flags & qaiRCr) ? 1 : 0), qr.rem );
            isDebug = true;
        }
        else
            isDebug = false;

        ++nDebugHit;
    }
    else
        isDebug = false;
#endif

    // prepare to iterate across the Qi symbols
    UINT64 Qi = 0;
    INT32 i = 0;

    // look for "soft clipping" at the start of the alignment
    INT32 nClipped = pBH->Il;
    if( nClipped )
        emitXMgapR( p, nClipped, R, pR, Qi, pQi, i, getNextQi );

    // traverse the BRLEA
    BRLEAbyteType bbPrev = bbMatch;     // (a BRLEA always starts with a match)
    INT32 runLength = 0;

    while( pBB < pBBlimit )
    {
        if( pBB->bbType != bbPrev )
        {
            // emit the previous BRLEA run
            (this->*(m_emitXM[bbPrev]))( p, runLength, R, pR, Qi, pQi, i, getNextQi );

            // reset the run length
            runLength = 0;
            bbPrev = static_cast<BRLEAbyteType>(pBB->bbType);
        }

        // accumulate the run length
        runLength = (runLength << 6) | pBB->bbRunLength;

        pBB++;
    }

    // emit the final BRLEA run
    (this->*(m_emitXM[(pBBlimit-1)->bbType]))( p, runLength, R, pR, Qi, pQi, i, getNextQi );

    // look for "soft clipping" at the end of the alignment
    nClipped = pQAI->N - (pBH->Ir + 1);
    if( nClipped )
        emitXMgapR( p, nClipped, R, pR, Qi, pQi, i, getNextQi );

    // convert raw XM context info to final Bismark-compatible form
    UINT32 cb = static_cast<UINT32>(p - pbuf);
    convertXM( pbuf, cb, pQAI );
    return cb;
}
#pragma endregion
