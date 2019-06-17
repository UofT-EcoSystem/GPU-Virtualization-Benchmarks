/*
  baseMAPQ.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseMAPQ__

/// <summary>
/// Class <c>baseMAPQ</c> computes mapping quality scores
/// </summary>
class baseMAPQ
{
    public:
        /// <summary>
        /// Class <c>baseMAPQ::Vinfo</c> tracks alignment scores for computing mapping quality scores
        /// </summary>
        class Vinfo
        {
            private:
                AriocBase*    m_pab;

            public:
                INT16   N1;         // number of symbols in a specified read (unpaired) or mate 1 (paired)
                INT16   N2;         // number of symbols in mate 2
                INT32   Vmax1;      // highest (maximum) V score
                INT32   Vmax2;      // second-highest V score
                INT32   Vp;         // perfect score for a specified read
                INT32   Vt;         // threshold score for a specified read
                INT32   nAu;        // total number of unduplicated mappings found by the aligner
                INT32   nVmax1;     // total number of distinct mappings with the highest V score
                INT32   nVmax2;     // number of mappings with the second-highest V score

            private:
                Vinfo( void );

            public:
                Vinfo( AriocBase* pab );
                ~Vinfo( void );
                void TrackV( QAI* pQAI );
                void TrackV( QAI* pQAI1, QAI* pQAI2 );
                void FinalizeV( void );
        };

    private:
        typedef void (baseMAPQ::*mfnComputeMAPQ)( QAI* pQAI, Vinfo& mvi );                                      // unpaired
        typedef void (baseMAPQ::*mfnComputeMAPQp)( PAI* pPAI, Vinfo& mviCombined, Vinfo& mvi1, Vinfo& mvi2 );   // paired

        AriocBase*  m_pab;
        INT16       m_altVsec;
        INT16       m_mismatchPenalty;
        
        static const int m_mcl;
        static const double m_mcf;
    
        INT32       computeSeedCoverage( BRLEAheader* pBH );
        UINT8       computeMAPQ_Ariocx( INT16 V, Vinfo& mvi );
        void        computeMAPQ_Ariocx( QAI* pQAI, Vinfo& mvi );
        void        computeMAPQ_Ariocxp( PAI* pPAI, Vinfo& mviCombined, Vinfo& mvi1, Vinfo& mvi2 );
        UINT8       computeMAPQ_BT2( INT16 V, Vinfo& mvi );
        void        computeMAPQ_BT2( QAI* pQAI, Vinfo& mvi );
        void        computeMAPQ_BT2p( PAI* pPAI, Vinfo& mviCombined, Vinfo& mvi1, Vinfo& mvi2 );
        UINT8       computeMAPQ_BWA( BRLEAheader* pBH, Vinfo& mvi );
        void        computeMAPQ_BWA( QAI* pQAI, Vinfo& mvi );
        void        computeMAPQ_BWAp( PAI* pPAI, Vinfo& mviCombined, Vinfo& mvi1, Vinfo& mvi2 );

    protected:
        baseMAPQ( void );
        mfnComputeMAPQ  m_computeMAPQ;
        mfnComputeMAPQp m_computeMAPQp;

    public:
        baseMAPQ( AriocBase* pab );
        virtual ~baseMAPQ( void );
};

