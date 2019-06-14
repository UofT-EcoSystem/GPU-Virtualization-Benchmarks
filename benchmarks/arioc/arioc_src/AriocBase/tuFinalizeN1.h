/*
  tuFinalizeN1.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuFinalizeN1__

struct pinfoN       // info for one partition of Q sequences mapped by the nongapped aligner
{
    UINT32  iDm;    // offset of the first D value
    UINT32  nDm;    // number of D values in the partition
};



/// <summary>
/// Class <c>tuFinalizeN1</c> builds a BRLEA for each reportable nongapped mapping
/// </summary>
class tuFinalizeN1 : public tuBaseA
{
    private:
        QBatch*     m_pqb;
        AriocBase*  m_pab;
        pinfoN*     m_ppi;
        bool        m_baseConvCT;

    protected:
        void main( void );

    private:
        tuFinalizeN1( void );
        bool tryTrimBRLEAend( BRLEAheader* pBH, BRLEAbyte*& pEnd, INT16 nMatches, INT16 nMismatches, INT16 cbTrim );
        bool tryTrimBRLEAstart( BRLEAheader* pBH, BRLEAbyte* pStart, BRLEAbyte*& pEnd, INT16 nMatches, INT16 nMismatches, INT16 cbTrim );
        void appendRunLength( BRLEAbyte*& p, INT16& cchRun );
        UINT64 flagMismatches( const UINT64 Rcurrent, const UINT64 Qcurrent, const UINT64 tailMask = _I64_MAX );
        void computeBRLEAforQ( BRLEAheader* pBH, UINT32 qid, UINT64* pQi, INT16 pincrQi, INT16 N, UINT8 subId, UINT32 J );

    public:
        tuFinalizeN1( QBatch* pqb, pinfoN* ppi );
        virtual ~tuFinalizeN1( void );
};

