/*
  tuLoadM.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuLoadM__

/// <summary>
/// Class <c>tuLoadM</c> loads Q-sequence metadata for the current batch
/// </summary>
class tuLoadM : public tuBaseA
{
    protected:
        QBatch*             m_pqb;
        QfileInfo*          m_pQFI[2];
        RaiiFile*           m_pFileM[2];
        MfileInfo*          m_pMFI[2];
        MfileBuf*           m_pMFB[2];
        AriocTaskUnitMetrics* m_ptum;
        HiResTimer          m_hrt;

    protected:
        tuLoadM( void );
        void readMetadata( RaiiFile* pFileM, QfileInfo* pqfi, MfileInfo* pmfi, MfileBuf* pmfb, UINT64 sqIdInitial, UINT64 sqIdFinal );
        virtual void main( void ) = 0;

    public:
        tuLoadM( QBatch* pqb );
        virtual ~tuLoadM( void );
}
;
