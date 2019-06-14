/*
  tuClassify.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuClassify__

#if TODO_CHOP_IF_THE_NEW_STUFF_WORKS
struct pinfoC       // info for one partition of Qwarps
{
    UINT32  iQw;                // offset of the first Qwarp
    UINT32  nQw;                // number of Qwarps

    WinGlobalPtr<PAI>   pai;    // pair alignment info
    UINT32  iPAI;               // cumulative offset into coalesced PAI buffer
};
#endif


/// <summary>
/// Class <c>tuClassify</c> implements common functionality for derived classes <c>tuClassifyP</c> and <c>tuClassifyU</c>.
/// </summary>
class tuClassify : public tuBaseS
{
    protected:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        AriocTaskUnitMetrics*   m_ptum;
#if TODO_CHOP_IF_THE_NEW_WAY_WORKS
        WinGlobalPtr<pinfoC>    m_pi;
#endif
        volatile UINT32         m_qid;

        HiResTimer              m_hrt;

    protected:
        tuClassify( void );
        tuClassify( QBatch* pqb );
        virtual ~tuClassify( void );
        INT16 computeEditDistance( BRLEAheader* pBH, INT16 N );
        QAI* appendQAI( BRLEAheader* pBH, INT16 N, QAIflags qaiFlags );
};

