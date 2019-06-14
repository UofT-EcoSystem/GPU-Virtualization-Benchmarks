/*
  QReaderU.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __QReaderU__


// forward declaration
class QBatch;

/// <summary>
/// Loads encoded short-read data from disk files
/// </summary>
class QReaderU : public baseQReader
{
    private:
        RaiiFile    m_fileQ;    // Q-sequence files ($a21)
        MfileInfo*  m_pMFIm;    // read metadata file info (sqm)
        MfileInfo*  m_pMFIq;    // base quality score file info (sqq)
        HiResTimer  m_hrt;

public:
        RaiiFile    fileMm;     // metadata file ($sqm)
        RaiiFile    fileMq;     // metadata file ($sqq)

    private:
        QReaderU( void );

    public:
        QReaderU( QBatch* pqb, INT16 iPart );
        virtual ~QReaderU( void );
        virtual bool LoadQ( QBatch* pqb );
};
