/*
  QReaderP.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __QReaderP__

// forward declaration
class QBatch;

/// <summary>
/// Loads encoded short-read data from disk files
/// </summary>
class QReaderP : public baseQReader
{
    private:
        RaiiFile    m_fileQ[2];     // paired Q-sequence files ($a21)
        MfileInfo*  m_pMFIm[2];     // paired metadata file info
        MfileInfo*  m_pMFIq[2];
        HiResTimer  m_hrt;

    public:
        RaiiFile    fileMm[2];      // paired metadata files ($sqm)
        RaiiFile    fileMq[2];      // paired metadata files ($sqq)

    private:
        QReaderP( void );

    public:
        QReaderP( QBatch* pqb, INT16 iPart );
        virtual ~QReaderP( void );
        virtual bool LoadQ( QBatch* pqb );
};
