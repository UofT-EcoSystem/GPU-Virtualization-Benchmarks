/*
  baseARowBuilder.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
   This base class implements a few comparatively fast integer-to-string methods so that derived
    classes can replace code like this

        sprintf( buffer, "%d", integer_value );

   with

        u32ToString( buffer, integer_value );

   Avoiding the use of sprintf() makes the SAMbuilder* derived classes run about 50% faster.
*/
#include "stdafx.h"

#pragma region static member variables
INT16 baseARowBuilder::m_init = baseARowBuilder::initLUTs();
INT16 baseARowBuilder::m_ddc[64];                               // minimum decimal digit counts
UINT32 baseARowBuilder::m_ddcut32[32];                          // additional digit cutpoints
UINT64 baseARowBuilder::m_ddcut64[64];
UINT16 baseARowBuilder::m_hexCharPair[256];                     // hexadecimal character pairs

char baseARowBuilder::m_symbolDecode[] ={ '·', '·', 'N', 'N', 'A', 'C', 'G', 'T' };     // forward
char baseARowBuilder::m_symbolDecodeRC[] ={ '·', '·', 'N', 'N', 'T', 'G', 'C', 'A' };   // reverse complement
#pragma endregion

#pragma endregion

#pragma region constructors and destructor
/// [public] constructor (QBatch*, INT16)
baseARowBuilder::baseARowBuilder( QBatch* pqb, INT16 nEmitFields, bool emitA21TraceFields ) : m_pqb(pqb),
                                                                                              m_pab(pqb->pab),
                                                                                              m_cbRow(0),
                                                                                              m_cbInvariantFields1(0),
                                                                                              m_cbInvariantFields2(0),
                                                                                              m_nEmitFields(nEmitFields),
                                                                                              m_emitA21TraceFields(emitA21TraceFields),
                                                                                              m_flag(0)
{
}

/// [public] destructor
baseARowBuilder::~baseARowBuilder()
{
}
#pragma endregion

#pragma region private methods
/// [private, static] initLUTs
INT16 baseARowBuilder::initLUTs()
{
    /* This method is referenced by a static member variable so it is called by the C++ static initializer. */

    // initialize the minimum number of decimal digits required to represent each power of two
    for( INT16 n=0; n<64; ++n )
    {
        // compute the power of two
        UINT64 u64 = static_cast<UINT64>(1) << n;

        // count decimal digits
        do
        {
            m_ddc[n]++ ;
            u64 /= 10;
        }
        while( u64 );
    }

    // initialize the cutpoint lists with the largest possible unsigned integer values
    memset( m_ddcut32, 0xFF, sizeof m_ddcut32 );
    memset( m_ddcut64, 0xFF, sizeof m_ddcut64 );

    // for each power of 10, initialize the cutpoint at which the number of decimal digits increases
    UINT64 p10 = 0x0DE0B6B3A7640000;	// 10000000000000000000, the largest power of 10 that can be represented in 63 bits
    while( p10 )
    {
        // get the position of the high-order 1 bit of the power of 10
        DWORD pos;
        _BitScanReverse64( &pos, p10 );

        // save the cutpoint
        m_ddcut64[pos] = p10;
        if( pos < 32 )
            m_ddcut32[pos] = static_cast<UINT32>(p10);

        // compute the next lower power of 10
        p10 /= 10;
    }

    // initialize the list of hexadecimal character pairs
    for( INT16 u=0; u<256; ++u )
    {
        // build a little-endian 16-bit value that represents two consecutive hexadecimal characters
        UINT16 uh = (u < 0xA0) ? '0'+(u>>4) : ('A'-10)+(u>>4);
        UINT16 ul = ((u&0x0F) < 0x0A) ? '0'+(u&0x0F) : ('A'-10)+(u&0x0F);
        m_hexCharPair[u] = (ul << 8) | uh;
    }

    return 0;
}
#pragma endregion

#pragma region protected methods
/// [protected] method szToVarchar
INT16 baseARowBuilder::szToSBF( char* p, char* s )
{
    const INT16 cb = static_cast<INT16>(strlen(s));
    
    // string length
    *reinterpret_cast<INT16*>(p) = cb;
    p += sizeof(INT16);

    // string (without null terminator)
    memcpy_s( p, cb, s, cb );

    return cb+sizeof(INT16);
}

/// [protected] method u32ToString
INT16 baseARowBuilder::u32ToString( char* p, UINT32 u )
{
    /* This method builds a string representation of a non-negative decimal integer value. */

    // special handling for small values of u (i.e., u < 10)
    if( u < 10 )
    {
        *reinterpret_cast<UINT16*>(p) = '0' + u;    // save an ASCII decimal digit followed by a null byte
        return 1;
    }

    // use the high-order bit to look up the number of digits in the decimal value
    DWORD pos;
    _BitScanReverse( &pos, u );
    INT16 nDigits = m_ddc[pos] + ((u >= m_ddcut32[pos]) ? 1 : 0);

    // start with a null byte at the end of the string
    p += nDigits;
    *p = 0;

    // prepend the decimal digits
    div_t v = div( u, 10 );
    while( v.quot )
    {
        *(--p) = '0' + v.rem;
        v = div( v.quot, 10 );
    }
    *(--p) = '0' + v.rem;

    return nDigits;
}

/// [protected] method i32ToString
INT16 baseARowBuilder::i32ToString( char* p, INT32 n )
{
    /* This method builds a string representation of a signed decimal integer value. */

    // if the specified value is non-negative, treat it as an unsigned value
    if( n >= 0 )
        return u32ToString( p, n );

    // if the specified value is negative...
    *(p++) = '-';                       // precede the result with a minus sign
    n = -n;                             // make the value positive
    return u32ToString( p, n ) + 1;     // return the number of digits plus the minus sign
}

/// [protected] method u64ToString
INT16 baseARowBuilder::u64ToString( char* p, UINT64 u )
{
    /* This method builds a string representation of a non-negative decimal integer value. */

    // special handling for small values of u (i.e., u < 10)
    if( u < 10 )
    {
        *reinterpret_cast<UINT16*>(p) = static_cast<UINT16>('0' + u);    // save an ASCII decimal digit followed by a null byte
        return 1;
    }

    // use the high-order bit to look up the number of digits in the decimal value
    DWORD pos;
    _BitScanReverse64( &pos, u );
    INT16 nDigits = m_ddc[pos] + (u >= m_ddcut64[pos]);

    // start with a null byte at the end of the string
    p += nDigits;
    *p = 0;

    // prepend the decimal digits
    lldiv_t v = lldiv( u, 10 );
    while( v.quot )
    {
        *(--p) = static_cast<char>('0' + v.rem);
        v = lldiv( v.quot, 10 );
    }
    *(--p) = static_cast<char>('0' + v.rem);

    return nDigits;
}

/// [protected] method reverseBRLEA
void baseARowBuilder::reverseBRLEA( BRLEAheader* pBH )
{
    // point to the start and end of the forward BRLEA bytes
    BRLEAbyte* pBf = reinterpret_cast<BRLEAbyte*>(pBH+1);
    BRLEAbyte* pBlimit = pBf + pBH->cb;

    // point past the end of the reverse BRLEA buffer
    BRLEAbyte* pBr = m_revBRLEA.p + pBH->cb;

    // traverse the specified BRLEA from back to front
    do
    {
        // accumulate the number of bytes in the current run
        INT16 cchRun = 1;
        while( (++pBf < pBlimit) && (pBf[0].bbType == pBf[-1].bbType) )
            ++cchRun;

        // copy the run
        for( INT16 n=-1; n>=(-cchRun); --n )
            *(--pBr) = pBf[n];
    }
    while( pBf < pBlimit );
}

/// [protected] method setSqIdPairBit
UINT64 baseARowBuilder::setSqIdPairBit( const UINT64 sqId, const QAI* pQAI )
{
    return (pQAI->flags & qaiParity) ? (sqId | AriocDS::SqId::MaskMateId) : (sqId & (~AriocDS::SqId::MaskMateId));
}

/// [protected] method setSqIdSecBit(UINT64, PAI*)
UINT64 baseARowBuilder::setSqIdSecBit( const UINT64 sqId, const PAI* pPAI )
{
    return (pPAI->arf & arfPrimary) ? (sqId & ~AriocDS::SqId::MaskSecBit) : (sqId | AriocDS::SqId::MaskSecBit);
}

/// [protected] method setSqIdSecBit(UINT64, QAI*)
UINT64 baseARowBuilder::setSqIdSecBit( const UINT64 sqId, const QAI* pQAI )
{
    return (pQAI->flags & qaiBest) ? (sqId & ~AriocDS::SqId::MaskSecBit) : (sqId | AriocDS::SqId::MaskSecBit);
}

/// [protected] method stringizeSEQf
UINT32 baseARowBuilder::stringizeSEQf( char* pbuf, QAI* pQAI )
{
    // point to the output buffer
    char* p = pbuf;

    // point to the encoded interleaved sequence data
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(pQAI->qid);
    UINT64* pQ = m_pqb->QiBuffer.p + pQw->ofsQi + QID_IQ(pQAI->qid);

    UINT64 b21 = 0;
    for( INT16 i=0; i<pQAI->N; ++i )
    {
        // if necessary, get the next 21 symbols
        if( b21 == 0 )
        {
            b21 = *pQ;
            pQ += CUDATHREADSPERWARP;
        }

        // emit the ith symbol
        *(p++) = m_symbolDecode[b21 & 7];

        // shift the next symbol into the low-order bits of b21
        b21 >>= 3;
    }

    return pQAI->N;
}

/// [protected] method stringizeSEQr
UINT32 baseARowBuilder::stringizeSEQr( char* pbuf, QAI* pQAI )
{
    // point to the output buffer
    char* p = pbuf;

    // point to the first 64-bit value in the encoded interleaved sequence data
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(pQAI->qid);
    UINT64* pQ = m_pqb->QiBuffer.p + pQw->ofsQi + QID_IQ(pQAI->qid);

    // get the final 64-bit value
    div_t qr = div( pQAI->N, 21 );
    pQ += CUDATHREADSPERWARP * qr.quot;
    UINT64 b21 = *pQ;

    // compute the number of symbols to shift
    INT16 shl = (qr.rem ? 21-qr.rem : 0);

    // shift the high-order symbol into the high-order bits of the 64-bit value
    b21 <<= (3*shl + 1);

    for( INT16 i=pQAI->N-1; i>=0; --i )
    {
        // if necessary, get the next 21 symbols
        if( b21 == 0 )
        {
            pQ -= CUDATHREADSPERWARP;
            b21 = *pQ << 1;     // shift the high-order symbol into the high-order bits of the 64-bit value
        }

        // rotate the next symbol into the low-order bits of b21
        b21 = _rotl64( b21, 3 );

        // emit the ith symbol
        *(p++) = m_symbolDecodeRC[b21 & 7];

        // zero the ith symbol
        b21 &= ~7;
    }

    return pQAI->N;
}

/// [protected] method computeA21Hash32
UINT32 baseARowBuilder::computeA21Hash32( QAI* pQAI )
{
    // point to the encoded interleaved sequence data
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(pQAI->qid);
    UINT64* pQ = m_pqb->QiBuffer.p + pQw->ofsQi + QID_IQ(pQAI->qid);

    // compute the number of 64-bit values in the encoded Q sequence representation
    INT32 cel = blockdiv( pQAI->N, 21 );

    // compute a 32-bit hash
    return Hash::ComputeH32( pQ, cel, CUDATHREADSPERWARP );
}

/// [protected] method computeA21Hash64
UINT64 baseARowBuilder::computeA21Hash64( QAI* pQAI )
{
    // point to the encoded interleaved sequence data
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(pQAI->qid);
    UINT64* pQ = m_pqb->QiBuffer.p + pQw->ofsQi + QID_IQ(pQAI->qid);

    // compute the number of 64-bit values in the encoded Q sequence representation
    INT32 cel = blockdiv( pQAI->N, 21 );

    // compute a 64-bit hash
    return Hash::ComputeH64( pQ, cel, CUDATHREADSPERWARP );
}
#pragma endregion

#pragma region virtual base implementations
/// [public] method WriteRowUm
INT64 baseARowBuilder::WriteRowUm( baseARowWriter* pw, INT64 sqId, QAI* pQAI )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public] method WriteRowUu
INT64 baseARowBuilder::WriteRowUu( baseARowWriter* pw, INT64 sqId, QAI* pQAI )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public] method WriteRowPc
INT64 baseARowBuilder::WriteRowPc( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public] method WriteRowPd
INT64 baseARowBuilder::WriteRowPd( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public] method WriteRowPr
INT64 baseARowBuilder::WriteRowPr( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

/// [public] method WriteRowPu
INT64 baseARowBuilder::WriteRowPu( baseARowWriter* pw, INT64 sqId, PAI* pPAI )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}
#pragma endregion
