/*
  tuClassifyP2.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuClassifyP2__

#ifndef __baseMAPQ__
#include "baseMAPQ.h"
#endif

/// <summary>
/// Class <c>tuClassifyP2</c> finds pairs
/// </summary>
class tuClassifyP2 : public tuBaseA, private baseMAPQ
{
    typedef bool (tuClassifyP2::*mfnEvaluateCandidatePair)( PAI& pai );

    private:
        struct pairTypeCounts
        {
            INT32 nConcordant1;
            INT32 nConcordant2;
            INT32 nDiscordant;
            INT32 nPairsRejected;   // (unmapped pairs with both ends mapped)
            INT32 nPairsUnmapped;   // (unmapped pairs with one or both ends unmapped)
            INT32 nMatesUnmapped;
            INT32 nMatesMapped1;
            INT32 nMatesMappedU1;
            INT32 nMatesMapped2;
            INT32 nMatesMappedU2;
        };

    private:
        QBatch*                     m_pqb;
        AriocBase*                  m_pab;
        AlignmentResultFlags        m_arfMaskCollision;
        mfnEvaluateCandidatePair    m_evaluateCandidatePair;
        volatile UINT32*            m_pqid;
        WinGlobalPtr<PAI>           m_paiq;

    protected:
        void main( void );

    private:
        tuClassifyP2( void );
        static int PAIComparer( const void*, const void* );
        bool verifyCandidatePairFragmentLength( PAI& pai );
        bool evaluateCandidatePairCollision( PAI& pai, QAI* pQAIu, QAI* pQAId );
        bool evaluateCandidatePair_same( PAI& pai );
        bool evaluateCandidatePair_convergent( PAI& pai);
        bool evaluateCandidatePair_divergent( PAI& pai );
        INT16 computeVpair( PAI& pai );
        void appendPAI( PAI& pai );
        void appendPAIforQ( void );
        INT32 findPairUnique( UINT32 qid, QAI* pQAI1, QAI* pQAI2, pairTypeCounts* pptc );
        void trackMappedV( INT16* pV, INT16 nV, INT16 v );
        INT32 findPairMultiple( UINT32 qid, QAI* pQAI1, QAI* pQAI2, QAIReference* pqair0, QAIReference* pqair1, pairTypeCounts* pptc );
        void findUnpairedMappings( UINT32 qid, QAIReference* pqair, AlignmentResultFlags arf0, INT32 nPairedMappings, pairTypeCounts* pptc );
        void findUnpairedMates( UINT32 qid, QAIReference* pqair0, QAIReference* pqair1, INT32 nPairedMappings, pairTypeCounts* pptc );
// TODO: CHOP        void filterMappings( UINT32 iPAIfrom, QAIReference* pqair0, QAIReference* pqair1 );
        void filterMappings( QAIReference* pqair0, QAIReference* pqair1 );
        void computeMappingQualities( void );
        void findPairs( void );

    public:
        tuClassifyP2( QBatch* pqb, volatile UINT32* pqid );
        virtual ~tuClassifyP2( void );
};

