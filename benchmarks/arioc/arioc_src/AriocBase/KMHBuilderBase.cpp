/*
  KMHBuilderBase.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static member variables
/* The following defines the fields in the KMH file; the field format is such that it can be used for
    bulk insert into a SQL table:
    
        sqId  bigint  not null,  -- bitmap contains srcId (see AriocDS.h)
        s64   bigint  not null   -- 64-bit kmer "sketch bits"
*/
KMHBuilderBase::mfnEmitField KMHBuilderBase::m_emitField[] ={ &KMHBuilderBase::emitS64,
                                                              &KMHBuilderBase::emitSqId };
#if TODO_CHOP_IF_UNUSED
                                                               &KMHBuilderBase::emitMapCat,
                                                               &KMHBuilderBase::emitRGID,
                                                               &KMHBuilderBase::emitSqLen,
                                                               &KMHBuilderBase::emitHash32,
                                                               &KMHBuilderBase::emitBRLEQ,   // minBQS, maxBQS, BRLEQ
                                                               &KMHBuilderBase::emitA21,
                                                               &KMHBuilderBase::emitSEQ };
#endif
#pragma endregion

#pragma region constructors and destructor
/// <summary>
/// Implements functionality for reporting kmer-hashed read sequences
///  for the Terabase Search Engine in SQL Server binary data import/export format
/// </summary>
KMHBuilderBase::KMHBuilderBase( QBatch* pqb ) : TSEFormatBase(pqb, sizeof(KMH_FIXED_FIELDS), arraysize(m_emitField)),
                                                m_kmerSize(pqb->pab->KmerSize),
                                                m_nSketchBits(34)
{
    /* Compute a bitmask for counting Ns in a given kmer:
        - we need a bitmask that covers the low-order symbols in an A21-formatted 64-bit value
        - MASK100 masks the high-order bit of each 3-bit symbol
    */
    INT32 shr = 63 - (m_kmerSize*3);
    m_maskNonN = MASK100 >> shr;

    // compute a bitmask for the low-order bits of an A21-formatted 64-bit value
    m_maskK = MASKA21 >> shr;

    // look for a configured override for m_nSketchBits
    INT32 i = pqb->pab->paamb->Xparam.IndexOf( "KMHsketchBits" );
    if( i >= 0 )
        m_nSketchBits = static_cast<INT16>(pqb->pab->paamb->Xparam.Value(i));
    if( (m_nSketchBits < 32) || (m_nSketchBits > 40) )
        throw new ApplicationException( __FILE__, __LINE__, "KMHsketchBits must be between 32 and 40" );

    // performance metrics
    AriocBase::aam.n.SketchBits = m_nSketchBits;
}

/// [public] destructor
KMHBuilderBase::~KMHBuilderBase()
{
}
#pragma endregion

#pragma region private methods
#if TODO_CHOP_WHEN_CUDA_EQUIVALENT_WORKS
/// [private static] method Xcomparer
int KMHBuilderBase::Xcomparer( const void* a, const void* b )
{
    return (*reinterpret_cast<const UINT32*>(a) < *reinterpret_cast<const UINT32*>(b)) ? -1 :
           (*reinterpret_cast<const UINT32*>(a) > *reinterpret_cast<const UINT32*>(b)) ?  1 :
                                                                                          0;
}

/// [private] method computeX
UINT32 KMHBuilderBase::computeX( UINT32* X, UINT32 qid )
{
    UINT32 nX = 0;

    // point to the encoded interleaved sequence data
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
    const INT32 iq = QID_IQ(qid);
    UINT64* pQi = m_pqb->QiBuffer.p + pQw->ofsQi + iq;

    // compute the maximum seed position within the Q sequence
    const INT32 N = pQw->N[iq];
    INT32 posMax = N - m_pab->KmerSize;

    // get the first set of encoded symbols
    UINT64 i64 = *pQi;
    UINT64 i64n = 0;

    for( INT32 pos=0; pos<=posMax; ++pos )
    {
        UINT64 k64 = i64 & m_maskK;

#if TODO_CHOP_WHEN_DEBUGGED
        AriocCommon::DumpB2164( k64, pos );
#endif

        // hash the kmer only if it contains no Ns
        if( (k64 & m_maskNonN) == m_maskNonN )
            X[nX++] = m_Hash.ComputeH32( k64 );

        // get the next set of encoded symbols
        if( (i64n == 0) && (pos < (N-21)) )
        {
            i64n = *(pQi += CUDATHREADSPERWARP);

#if TODO_CHOP_WHEN_DEBUGGED
            CDPrint( cdpCD0, "%s: fetched i64n=0x%016llx", __FUNCTION__, i64n );
#endif
        }

        // shift to the next symbol position
        i64 = (i64 >> 3) | ((i64n & 7) << 60);
        i64n >>= 3;
    }

    return nX;
}
#endif
#pragma endregion


#if TODO_CHOP_WHEN_DEBUGGED
static UINT64 totalBits = 0;
static UINT32 nQ = 0;
#endif


#pragma region protected methods
/// [protected] method emitS64
UINT32 KMHBuilderBase::emitS64( char* pbuf, INT64 sqId, QAI* pQAI )
{
#if TODO_CHOP_WHEN_CUDA_EQUIVALENT_WORKS
    // allocate an array to contain kmer hash values
    UINT32* X = reinterpret_cast<UINT32*>(alloca( m_maxNforTSE * sizeof(UINT32) )); // allocate on the stack (automatically freed)
    
    // compute kmer hash values
    UINT32 nX = computeX( X, pQAI->qid );

    /* Build a 64-bit "sketch" bitmap for the specified Q sequence:

       We want a bit pattern that can be efficiently indexed and searched in a database, e.g., a 64-bit
        value in which 32 bits are set.

       We generate that bit pattern here by choosing the smallest values from the "spectrum" of kmer
        hash values for the Q sequence.  We then use 6 bits out of each hash values to set the
        corresponding bit in a 64-bit value.
    */

    // sort the list of kmer hash values
    qsort( X, nX, sizeof X[0], &Xcomparer );

    // build the "sketch" bitmap from the smallest distinct (unduplicated) hash values
    UINT64 sketchBits = 0;
    for( UINT32 n=0; n<nX; ++n )
    {
        // set the bit whose offset is specified by bits 16..21 of the nth hash value (i.e., a value between 0 and 63)
        _bittestandset64( reinterpret_cast<INT64*>(&sketchBits), (X[n] >> 16) & 0x3F );

        // clamp the popcount
        if( __popcnt64( sketchBits ) == m_nSketchBits )
            break;
    }
    

    // TEST: did we do it right?
    if( sketchBits != m_pqb->S64.p[pQAI->qid] )
        CDPrint( cdpCD0, __FUNCTION__ );

    
    return intToSBF<UINT64>( pbuf, sketchBits );
#endif

    return intToSBF<UINT64>( pbuf, m_pqb->S64.p[pQAI->qid] );
}

/// [protected] method emitA21
UINT32 KMHBuilderBase::emitA21( char* pbuf, INT64 sqId, QAI* pQAI )
{
    // point to the encoded interleaved sequence data
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(pQAI->qid);
    UINT64* pQ = m_pqb->QiBuffer.p + pQw->ofsQi + QID_IQ(pQAI->qid);

    // compute the number of 64-bit values to copy
    INT16 cv = blockdiv( pQAI->N, 21 );

    // copy the encoded symbols in 64-bit chunks
    UINT64* pv = reinterpret_cast<UINT64*>(pbuf + sizeof(INT16));
    for( INT16 i=0; i<cv; ++i )
    {
        *(pv++) = *pQ;
        pQ += CUDATHREADSPERWARP;
    }

    // save the number of bytes in the binary A21 string
    INT16 cchTail = pQAI->N % 21;
    INT16 cb;
    if( cchTail == 0 )
        cb = cv * sizeof(UINT64);
    else
    {
        INT16 cbTail = blockdiv(cchTail*3, 8);      // 3 bits per encoded symbol / 8 bits per byte
        cb = (cv-1) * sizeof(UINT64) + cbTail;
    }

    *reinterpret_cast<INT16*>(pbuf) = cb;
    return cb + sizeof(INT16);
}
#pragma endregion
