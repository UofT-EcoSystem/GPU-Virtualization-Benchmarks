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
/// Class <c>tuWriteA</c> writes unpaired alignment results
/// </summary>
class tuWriteA : public tuBaseA
{
    private:
        QBatch*                         m_pqb;
        baseARowBuilder*                m_parb;
        WinGlobalPtr<baseARowWriter*>*  m_parw;
        QAI                             m_qaiU;
        bool                            m_wantCounts;
        AriocTaskUnitMetrics*           m_ptum;
        HiResTimer                      m_hrt;

    protected:
        void main( void );

    private:
        tuWriteA( void );
        QAI* buildTempQAI( Qwarp* pQw, UINT32 iw, INT16 iq );
        void emitUnpaired( RowWriteCounts& rwc );

    public:
        tuWriteA( QBatch* pqb, baseARowBuilder* parb, WinGlobalPtr<baseARowWriter*>* parw, const char* tumKey, bool wantCounts );
        virtual ~tuWriteA( void );
};

/// <summary>
/// Class <c>tuWriteSAM</c> writes unpaired alignment results to SAM-formatted files
/// </summary>
class tuWriteSAM : public tuWriteA
{
    private:
        SAMBuilderUnpaired  m_sbu;

    public:
        tuWriteSAM( QBatch* pqb, bool wantCounts ) : tuWriteA( pqb, &m_sbu, &pqb->pab->SAMwriter, "tuWriteSAM", wantCounts ),
                                                     m_sbu( pqb )
        {
        }

        virtual ~tuWriteSAM( void )
        {
        }
};

/// <summary>
/// Class <c>tuWriteSBF</c> writes unpaired alignment results to SBF-formatted files
/// </summary>
class tuWriteSBF : public tuWriteA
{
    private:
        SBFBuilderUnpaired  m_sbu;

    public:
        tuWriteSBF( QBatch* pqb, bool wantCounts ) : tuWriteA( pqb, &m_sbu, &pqb->pab->SBFwriter, "tuWriteSBF", wantCounts ),
                                                     m_sbu( pqb )
        {
        }

        virtual ~tuWriteSBF( void )
        {
        }
};

/// <summary>
/// Class <c>tuWriteTSE</c> writes unpaired alignment results to SBF-formatted files (Terabase Search Engine fields)
/// </summary>
class tuWriteTSE : public tuWriteA
{
private:
    TSEBuilderUnpaired    m_tbu;

public:
    tuWriteTSE( QBatch* pqb, bool wantCounts ) : tuWriteA( pqb, &m_tbu, &pqb->pab->TSEwriter, "tuWriteTSE", wantCounts ),
                                                 m_tbu( pqb )
    {
    }

    virtual ~tuWriteTSE( void )
    {
    }
};

/// <summary>
/// Class <c>tuWriteKMH</c> writes kmer-hashed unpaired paired read sequences
/// </summary>
class tuWriteKMH : public tuWriteA
{
private:
    KMHBuilderUnpaired    m_kbu;

public:
    tuWriteKMH( QBatch* pqb, bool wantCounts ) : tuWriteA( pqb, &m_kbu, &pqb->pab->KMHwriter, "tuWriteKMH", wantCounts ),
                                                 m_kbu( pqb )
    {
    }

    virtual ~tuWriteKMH( void )
    {
    }
};
