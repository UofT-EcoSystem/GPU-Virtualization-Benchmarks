/*
  tuWriteA.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuWriteA__

/// <summary>
/// Class <c>tuWriteA</c> writes paired alignment results
/// </summary>
class tuWriteA : public tuBaseA
{
    private:
        QBatch*                         m_pqb;
        baseARowBuilder*                m_parb;
        WinGlobalPtr<baseARowWriter*>*  m_parw;
        PAI                             m_paiU;
        QAI                             m_qaiU0;
        QAI                             m_qaiU1;
        bool                            m_wantCounts;
        AriocTaskUnitMetrics*           m_ptum;
        HiResTimer                      m_hrt;

    protected:
        void main( void );

    private:
        tuWriteA( void );
        PAI* buildTempQAI( PAI* pPAI, Qwarp* pQw );
        void emitPaired( PairWriteCounts& pwc );

    public:
        tuWriteA( QBatch* pqb, baseARowBuilder* parb, WinGlobalPtr<baseARowWriter*>* parw, const char* tumKey, bool wantCounts );
        virtual ~tuWriteA( void );
};

/// <summary>
/// Class <c>tuWriteSAM</c> writes paired alignment results to SAM-formatted files
/// </summary>
class tuWriteSAM : public tuWriteA
{
    private:
        SAMBuilderPaired    m_sbp;

    public:
        tuWriteSAM( QBatch* pqb, bool wantCounts ) : tuWriteA( pqb, &m_sbp, &pqb->pab->SAMwriter, "tuWriteSAM", wantCounts ),
                                                     m_sbp( pqb )
        {
        }

        virtual ~tuWriteSAM( void )
        {
        }
};

/// <summary>
/// Class <c>tuWriteSBF</c> writes paired alignment results to SBF-formatted files (SAM-like fields)
/// </summary>
class tuWriteSBF : public tuWriteA
{
    private:
        SBFBuilderPaired    m_sbp;

    public:
        tuWriteSBF( QBatch* pqb, bool wantCounts ) : tuWriteA( pqb, &m_sbp, &pqb->pab->SBFwriter, "tuWriteSBF", wantCounts ),
                                                     m_sbp( pqb )
        {
        }

        virtual ~tuWriteSBF( void )
        {
        }
};

/// <summary>
/// Class <c>tuWriteTSE</c> writes paired alignment results to SBF-formatted files (Terabase Search Engine fields)
/// </summary>
class tuWriteTSE : public tuWriteA
{
private:
    TSEBuilderPaired    m_tbp;

public:
    tuWriteTSE( QBatch* pqb, bool wantCounts ) : tuWriteA( pqb, &m_tbp, &pqb->pab->TSEwriter, "tuWriteTSE", wantCounts ),
                                                 m_tbp( pqb )
    {
    }

    virtual ~tuWriteTSE( void )
    {
    }
};

/// <summary>
/// Class <c>tuWriteKMH</c> writes kmer-hashed paired read sequences
/// </summary>
class tuWriteKMH : public tuWriteA
{
private:
    KMHBuilderPaired    m_kbp;

public:
    tuWriteKMH( QBatch* pqb, bool wantCounts ) : tuWriteA( pqb, &m_kbp, &pqb->pab->KMHwriter, "tuWriteKMH", wantCounts ),
                                                 m_kbp( pqb )
    {
    }

    virtual ~tuWriteKMH( void )
    {
    }
};
