/*
  tuFinalizeN1.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuFinalizeN1::tuFinalizeN1()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="ppi">partition info</param>
tuFinalizeN1::tuFinalizeN1( QBatch* pqb, pinfoN* ppi ) : m_pqb(pqb), m_pab(pqb->pab), m_ppi(ppi), m_baseConvCT(pqb->pab->a21ss.baseConvert == A21SpacedSeed::bcCT)
{
}

/// destructor
tuFinalizeN1::~tuFinalizeN1()
{
}
#pragma endregion

#pragma region private methods
/// [private] method tryTrimBRLEAend
bool tuFinalizeN1::tryTrimBRLEAend( BRLEAheader* pBH, BRLEAbyte*& pEnd, INT16 nMatches, INT16 nMismatches, INT16 cbTrim )
{
    // return false if trimming (soft clipping) would lower the alignment score
    INT16 Vdiff = nMismatches * m_pab->aas.ASP.Wx - nMatches * m_pab->aas.ASP.Wm;
    if( Vdiff < 0 )
        return false;
    
    // update the BRLEAheader
    pBH->V += Vdiff;                                            // adjust the V score
    pBH->cb -= cbTrim;
    INT16 nClipped = nMatches + nMismatches;
    pBH->Ir -= nClipped;                                        // adjust the offset of the righmost matched symbol
    pBH->Ma -= nClipped;                                        // adjust the number of R symbols spanned by the alignment

    // trim the BRLEA string
    pEnd -= cbTrim;                                             // point to the new end of the BRLEA

#if TODO_CHOP_WHEN_DEBUGGED
    *reinterpret_cast<UINT8*>(pEnd) = 0xFF;                     // append a null byte (helpful for troubleshooting)
#endif

    // return true to indicate that the BRLEA was trimmed
    return true;
}

/// [private] method tryTrimBRLEAstart
bool tuFinalizeN1::tryTrimBRLEAstart( BRLEAheader* pBH, BRLEAbyte* pStart, BRLEAbyte*& pEnd, INT16 nMatches, INT16 nMismatches, INT16 cbTrim )
{
    // return false if trimming (soft clipping) would lower the alignment score
    INT16 Vdiff = nMismatches * m_pab->aas.ASP.Wx - nMatches * m_pab->aas.ASP.Wm;
    if( Vdiff < 0 )
        return false;

    // update the BRLEAheader
    pBH->V += Vdiff;                                            // adjust the V score
    pBH->cb -= cbTrim;
    INT16 nClipped = nMatches + nMismatches;
    pBH->Ma -= nClipped;                                        // adjust the number of R symbols spanned by the alignment
    pBH->J = (pBH->J & 0x80000000) |                            // adjust the start of the alignment on the reference sequence
             ((pBH->J+nClipped) & 0x7FFFFFFF);
    pBH->Il += nClipped;                                        // adjust the offset of the leftmost matched symbol
 
    // trim the BRLEA string
    memmove( pStart, pStart+cbTrim, pBH->cb );
    pEnd -= cbTrim;                                             // point to the new end of the BRLEA
#if TODO_CHOP_WHEN_DEBUGGED
    *reinterpret_cast<UINT8*>(pEnd) = 0xFF;                     // append a null byte (helpful for troubleshooting)
#endif

    // return true to indicate that the BRLEA was trimmed
    return true;
}

/// [private] method appendRunLength
inline void tuFinalizeN1::appendRunLength( BRLEAbyte*& p, INT16& cchRun )
{
    /* Write the run length as big-endian BRLEA bytes (see also AriocKernelQDP.3.cu).  We're always dealing with a run of
        matching symbols, so we always set the BRLEA byte type to 0 (i.e., bbMatch). */
    if( cchRun >= (1 << 12) )
        *reinterpret_cast<UINT8*>(p++) = static_cast<UINT8>((cchRun >> 12) & 0x3F);
    if( cchRun >= (1 << 6) )
        *reinterpret_cast<UINT8*>(p++) = static_cast<UINT8>((cchRun >> 6) & 0x3F);
    *reinterpret_cast<UINT8*>(p++) = static_cast<UINT8>(cchRun & 0x3F);

    // reset the run length
    cchRun = 0;
}

/// [private] method flagMismatches
inline UINT64 tuFinalizeN1::flagMismatches( const UINT64 Rcurrent, const UINT64 Qcurrent, const UINT64 tailMask )
{
    UINT64 x = (Qcurrent ^ Rcurrent) & tailMask;        // each 3-bit symbol is nonzero where the symbols don't match
    UINT64 b = ((x >> 1) | x | (x << 1)) & MASK010;     // bit 1 of each 3-bit symbol is set where the symbols don't match

    /* For bsDNA alignments, C (101) in the reference matches T (111) in the query:
        - in this situation, Q^R = 010, but this is ambiguous:

            R      Q    
            C 101  T 111
            T 111  C 101
            A 100  G 110
            G 110  A 100

        - we disambiguate this from the three other possible ways that 010 can occur in the XOR string
            by zeroing bit 1 of each 3-bit symbol in the Q^R value wherever R contains C (101) and Q
            contains T (111)
    */
    if( m_baseConvCT )
    {
        UINT64 cr = ((Rcurrent >> 1) & (~Rcurrent) & (Rcurrent << 1));  // bit 1 of each 3-bit symbol is set where the symbol is C (101b)
        UINT64 tq = ((Qcurrent >> 1) & Qcurrent & (Qcurrent << 1));     // bit 1 of each 3-bit symbol is set where the symbol is T (111b)
        b &= ~((cr & tq) & MASK010);
    }

    // return the flag bits
    return b;
}

/// [private] method computeBRLEAforQ
void tuFinalizeN1::computeBRLEAforQ( BRLEAheader* pBH, UINT32 qid, UINT64* pQi, INT16 pincrQi, INT16 N, UINT8 subId, UINT32 J )
{
    // point to the start of the first byte of BRLEA data
    BRLEAbyte* p0 = reinterpret_cast<BRLEAbyte*>(pBH + 1);    // the first BRLEA byte immediately follows the BRLEAheader
    BRLEAbyte* p = p0;

    // point to the first 64-bit value in the Q sequence
    UINT64* pQcurrent = pQi;

    // point to the R sequence for the current alignment
    UINT64* pR = m_pab->R;

    // bit 31 of the J value indicates whether we are aligning to the R+ or the R- sequence
    pR += (J & 0x80000000) ? m_pab->ofsRminus.p[subId] : m_pab->ofsRplus.p[subId];

    /* Compute the offset of the 64-bit value in R that contains the start of the alignment.  The offset can be negative, but
        because nulls are stored in the R table at negative offsets, the alignment will start with one or more mismatches;
        these mismatches will get soft-clipped later on (see below).
    */
    INT32 j = static_cast<INT32>(J & 0x7FFFFFFF);   // isolate bits 0-30
    j = (j << 1) >> 1;                              // sign-extend


#if TODO_CHOP_WHEN_DEBUGGED
    if( j < 0 )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
        CDPrint( cdpCD0, "%s: j = %d for sqId = 0x%016llx", __FUNCTION__, j, pQw->sqId[QID_IQ(qid)] );
        isNegativeJ = true;
    }
#endif

#if TRACE_SQID
    // dump the Qwarp info
    Qwarp* pQwx = m_pqb->QwBuffer.p + QID_IW(qid);
    UINT64 sqIdx = pQwx->sqId[QID_IQ(qid)];
    if( (sqIdx | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        CDPrint( cdpCD0, "%s: sqId=0x%016llx qid=0x%08x N=%d)", __FUNCTION__, sqIdx, qid, pQwx->N[QID_IQ(qid)] );
#endif


    INT32 ofsR = ((j >= 0) ? j : (j-20)) / 21;      // compute the offset into the R sequence
    pR += ofsR;

    // copy the R bits from the two adjacent 64-bit values
    INT64 Rcurrent = *pR;
    INT64 Rnext = *(++pR);

    /* Compute the number of bits to shift each 64-bit value so that the first R symbol is in the low-order position:
        - if j >= 0, the computation is straightforward
        - if j < 0, the number of symbols to shift right is ((j+1) % 21) + 20
    */
    INT16 posToShift = (j >= 0) ? (j%21) : ((j+1)%21)+20;
    const INT16 shr = 3 * posToShift;
    const INT16 shl = 63 - shr;
    if( shr )
    {
        // shift bits into position to give 21 adjacent 3-bit symbols
        Rcurrent = ((Rcurrent >> shr) | (Rnext << shl)) & MASKA21;
    }

    /* Score the alignment between the Q sequence and the R sequence:
        - loop over the 64-bit (21-symbol) values in the encoded Q sequence
        - process any remaining 64-bit value (containing fewer than 21 symbols) at the end of the Q sequence
        - we assume that there are at least 21 symbols in the Q sequence
    */

    INT32 nAx = 0;  // number of mismatches in the alignment
    INT16 cchRun = 0;
    INT16 tailLength = 21;

    /* align the first 64-bit (21-symbol) value */

    // look for mismatches in the current 21 symbols
    UINT64 x = flagMismatches( Rcurrent, *pQcurrent );

    DWORD bitPos;
    while( _BitScanForward64( &bitPos, x ) )
    {
        // count mismatches
        ++nAx;

        // compute the symbol position in x
        INT16 symbolPos = static_cast<INT16>(bitPos / 3);

        // update the length of the run of matching symbols that precedes the mismatch
        cchRun = symbolPos;
        if( cchRun )
        {
            // write the run length of matches and reset the run length
            appendRunLength( p, cchRun );

            // write the mismatch
            *reinterpret_cast<UINT8*>(p++) = packBRLEAbyte(bbMismatch, 1);
        }

        else
        {
            /* The run length (i.e. symbol position) can be zero in either of the following cases:
                - the first symbol in the alignment is a mismatch
                - the previous symbol was also a mismatch
               In the first case, the pointer to the next byte to be written will have its initial value.  Otherwise, we increment the previously-
                written mismatch run length (with the assumption that the mismatch run length will never exceed 0x3F = 63 mismatches).
            */
            if( p == p0 )
                *reinterpret_cast<UINT8*>(p++) = packBRLEAbyte(bbMismatch, 1);
            else
                reinterpret_cast<UINT8*>(p)[-1]++ ;
        }

        // track the number of symbol positions remaining in x
        tailLength -= (symbolPos + 1);

        // right-shift the mismatched symbol out of x
        x >>= 3*(symbolPos+1);
    }

    // accumulate the tail length
    cchRun += tailLength;

    /* align the remaining 64-bit (21-symbol) values */
    div_t qr = div( N, 21 );
    for( INT32 s=1; s<qr.quot; ++s )
    {
        // advance to the next 64-bit values in the Q and R sequences
        pQcurrent += pincrQi;

        Rcurrent = Rnext;
        Rnext = *(++pR);
        if( shr )
            Rcurrent = ((Rcurrent >> shr) | (Rnext << (63 - shr))) & MASKA21;

        // look for mismatches in the current 21 symbols
        x = flagMismatches( Rcurrent, *pQcurrent );

        tailLength = 21;
        DWORD bitPos;
        while( _BitScanForward64( &bitPos, x ) )
        {
            // count mismatches
            ++nAx;

            // compute the symbol position in x
            INT16 symbolPos = static_cast<INT16>(bitPos / 3);

            // update the length of the run of matching symbols that precedes the mismatch
            cchRun += symbolPos;
            if( cchRun )
            {
                // write the run length of matches and reset the run length
                appendRunLength( p, cchRun );

                // write the mismatch
                *reinterpret_cast<UINT8*>(p++) = packBRLEAbyte(bbMismatch, 1);
            }
            else
            {
                // the run length can be zero only when the previous symbol was also a mismatch
                reinterpret_cast<UINT8*>(p)[-1]++ ;
            }

            // track the number of symbol positions remaining in x
            tailLength -= (symbolPos + 1);

            // right-shift the mismatched symbol out of x
            x >>= 3*(symbolPos+1);
        }

        // accumulate the current "match" run length
        cchRun += tailLength;
    }

    /* align any remaining symbols in the Q sequence */
    if( qr.rem )
    {
        // advance to the next 64-bit values in the Q and R sequences
        pQcurrent += pincrQi;

        Rcurrent = Rnext;
        Rnext = *(++pR);
        if( shr )
            Rcurrent = ((Rcurrent >> shr) | (Rnext << (63 - shr))) & MASKA21;

        // create a mask to zero the bit positions that do not contain alignable symbols
        INT64 tailMask = (static_cast<INT64>(1) << (3*qr.rem)) - 1;

        // look for mismatches in the current remaining symbols
        x = flagMismatches( Rcurrent, *pQcurrent, tailMask );

        tailLength = qr.rem;
        DWORD bitPos;
        while( _BitScanForward64( &bitPos, x ) )
        {
            // count mismatches
            ++nAx;

            // compute the symbol position in x
            INT16 symbolPos = static_cast<INT16>(bitPos / 3);

            // update the length of the run of matching symbols that precedes the mismatch
            cchRun += symbolPos;
            if( cchRun )
            {
                // write the run length of matches and reset the run length
                appendRunLength( p, cchRun );

                // write the mismatch
                *reinterpret_cast<UINT8*>(p++) = packBRLEAbyte(bbMismatch, 1);
            }
            else
            {
                // the run length can be zero only when the previous symbol was also a mismatch
                reinterpret_cast<UINT8*>(p)[-1]++ ;
            }

            // track the number of symbol positions remaining in x
            tailLength -= (symbolPos + 1);

            // right-shift the mismatched symbol out of x
            x >>= 3*(symbolPos+1);
        }
    }
    else
        tailLength = 0;
    
    // write the "tail" run length
    cchRun += tailLength;
    if( cchRun )
    {
        // write the run length of matches
        appendRunLength( p, cchRun );
    }

    /* at this point we have written the BRLEA bytes for the current D value */

    // update the BRLEAheader for the current alignment
    pBH->qid = qid;
    pBH->V = (m_pab->aas.ASP.Wm * (N-nAx)) - (m_pab->aas.ASP.Wx * nAx);
    pBH->nTBO = 1;
    pBH->Il = 0;        // (the nongapped aligner generates end-to-end alignments)
    pBH->Ir = N - 1;
    pBH->Ma = N;
    pBH->J = J;         // 0-based offset from the start of the R sequence
    pBH->subId = subId;
    pBH->nTBO = 1;
    pBH->cb = static_cast<INT16>(p - p0);

    /* Handle the unusual cases where an alignment has one or more mismatches near one of the ends; it's cheaper to
        fix things up here than to include more complex logic above. */

    // if the alignment ends with a mismatch ...
    if( p[-1].bbType == bbMismatch )
        tryTrimBRLEAend( pBH, p, 0, p[-1].bbRunLength, 1 );

    // if the alignment starts with a mismatch ...
    if( p0->bbType == bbMismatch )
        tryTrimBRLEAstart( pBH, p0, p, 0, p0->bbRunLength, 1 );

    /* At this point the alignment starts and ends with a match.  It is nevertheless possible that a higher-scoring
        local alignment can be found by trimming ("soft clipping") one or both ends of the alignment.

       For two mismatches, there are only a few BRLEA byte patterns that are candidates for soft clipping:

            Mx...
            MxMx...
            ... xM
            ... xMxM
       
       Because there are occurrences where clipping MxMx... leads to a higher score even when clipping Mx... does not,
        we always look for the MxMx... pattern whenever the Mx... pattern is not clipped.

       By assuming that the run lengths fit into only one byte, we limit soft clipping to 63 consecutive matches.
    */
    bool tryTrim;
    do
    {
        if( (pBH->cb > 2) && (p[-2].bbType == bbMismatch) )
        {
            // ...xM
            INT16 nMatches = p[-1].bbRunLength;
            INT16 nMismatches = p[-2].bbRunLength;
            tryTrim = tryTrimBRLEAend( pBH, p, nMatches, nMismatches, 2 );

            if( !tryTrim && (pBH->cb > 4) && (p[-4].bbType == bbMismatch) )
            {
                // ...xMxM
                INT16 nMatches = p[-1].bbRunLength + p[-3].bbRunLength;
                INT16 nMismatches = p[-2].bbRunLength + p[-4].bbRunLength;
                tryTrim = tryTrimBRLEAend( pBH, p, nMatches, nMismatches, 4 );
            }
        }
        else
            tryTrim = false;
    }
    while( tryTrim );

    do
    {
        if( (pBH->cb > 2) && (p0[1].bbType == bbMismatch) )
        {
            // Mx...
            INT16 nMatches = p0[0].bbRunLength;
            INT16 nMismatches = p0[1].bbRunLength;
            tryTrim = tryTrimBRLEAstart( pBH, p0, p, nMatches, nMismatches, 2 );

            if( !tryTrim && (pBH->cb > 4) && (p0[3].bbType == bbMismatch) )
            {
                // MxMx...
                INT16 nMatches = p0[0].bbRunLength + p0[2].bbRunLength;
                INT16 nMismatches = p0[1].bbRunLength + p0[3].bbRunLength;
                tryTrim = tryTrimBRLEAstart( pBH, p0, p, nMatches, nMismatches, 4 );
            }
        }
        else
            tryTrim = false;
    }
    while( tryTrim );

#if TRACE_SQID
    // dump the BRLEAHeader info
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
    UINT64 sqId = pQw->sqId[QID_IQ(qid)];
    if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        CDPrint( cdpCD0, "%s: sqId=0x%016llx qid=0x%08x V=%d subId=%d J=0x%08x (%u)", __FUNCTION__, sqId, pBH->qid, pBH->V, pBH->subId, pBH->J, pBH->J&0x7FFFFFFF );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCDb, "%s: sqId 0x%016llx: V=%d I=%d J=%d cb=%d", __FUNCTION__, sqId, pBH->V, pBH->I, pBH->J, pBH->cb );
#endif
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Builds BRLEAs for reportable nongapped alignments
/// </summary>
void tuFinalizeN1::main()
{
    CRVALIDATOR;

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s starts (iDm=%u nDm=%u)", m_pqb->pgi->deviceId, __FUNCTION__, m_ppi->iDm, m_ppi->nDm );
#endif

    // allocate a buffer for reverse complements of Q sequences
    WinGlobalPtr<UINT64> rcQi( blockdiv(m_pqb->Nmax,21), false );

    // point to the start of the BRLEA buffer for the current partition (see tuAlignN30::copyKernelResults for BRLEA buffer initialization)
    UINT32* pBn = m_pqb->HBn.BRLEA.p + (m_ppi->iDm * m_pqb->celBRLEAperQ);

    // traverse the list of mapped D values in the current partition and compute a BRLEA for each
    UINT64* pDm = m_pqb->HBn.Dm.p + m_ppi->iDm;
    UINT64* pDmLimit = pDm + m_ppi->nDm;

    while( pDm < pDmLimit )
    {
        /* The Dm value is bitmapped like this:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
            UINT64  flags :  3;     // bits 61..63: flags
        */
        UINT64 Dm = *pDm;

        // point to the Qwarp for the current D value
        UINT32 qid = AriocDS::D::GetQid( Dm );
        UINT32 iq = AriocDS::QID::GetQ( qid );
        UINT16 iw = AriocDS::QID::GetW( qid );
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;

        // point to the interleaved Q sequence data for the current Q sequence
        UINT64* pQi = m_pqb->QiBuffer.p + pQw->ofsQi + iq;

        // get the subId, J value, and rc flag from the list
        UINT8 subId = static_cast<UINT8>(Dm >> 32) & 0x7F;
        UINT32 J = static_cast<UINT32>(Dm);                 // all alignments start at the beginning of the Q sequence, so the R-diagonal D value is the same as J
        
        // point to the start of the next available space in the BRLEA buffer
        BRLEAheader* pBH = reinterpret_cast<BRLEAheader*>(pBn);

        // set bit 31 of the QID if the alignment uses the reverse complement of the Q sequence (bsDNA alignments only)
        if( Dm & AriocDS::D::flagRCq )
        {
            // set a bit in the QID to indicate that the reverse complement of the Q sequence is mapped
            qid |= AriocDS::QID::maskRC;

            // compute a BRLEA using the reverse complement of the Q sequence
            AriocCommon::A21ReverseComplement( rcQi.p, pQi, CUDATHREADSPERWARP, pQw->N[iq] );
            computeBRLEAforQ( pBH, qid, rcQi.p, 1, pQw->N[iq], subId, J );
        }

        else
            computeBRLEAforQ( pBH, qid, pQi, CUDATHREADSPERWARP, pQw->N[iq], subId, J );

#if TRACE_SQID
        // dump mappings for a specified sqId
        if( (pQw->sqId[iq] | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT32 j = J & 0x7FFFFFFF;
            j = (j << 1) >> 1;
            if( J & 0x80000000 )
            {
                INT32 M = m_pab->M.p[subId];
                j = (M-1) - j;
            }
            CDPrint( cdpCD0, "%s: qid=0x%08x sqId=0x%016llx subId=%d J=0x%08x Jf=%d", __FUNCTION__, qid, pQw->sqId[iq], subId, J, j );
        }
#endif



        // update the buffer pointers
        pBn += m_pqb->celBRLEAperQ;                 
        pDm++ ;
    }

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s: completed", m_pqb->pgi->deviceId, __FUNCTION__ );
#endif
}
