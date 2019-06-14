/*
  tuClassify1.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuClassify1__

/// <summary>
/// Class <c>tuClassify1</c> implements a base class for sorting and unduplicating mappings.
/// </summary>
class tuClassify1 : public tuBaseA
{
    typedef int (*pQAIComparer)( const void* pa, const void* pb);

    private:
        QBatch*             m_pqb;
        AriocBase*          m_pab;
#if TODO_CHOP_IF_THE_NEW_STUFF_WORKS
        pinfoC*             m_ppi;
#endif
        pQAIComparer        m_pQAIComparer;
        volatile UINT32*    m_pqid;

    protected:
        void main( void );
        tuClassify1( void );

    private:
        UINT32 exciseDuplicateMappings( QAIReference* pqair, QAI* pqai );

    public:
        tuClassify1( QBatch* pqb, pQAIComparer pfn, volatile UINT32* pqid );
        virtual ~tuClassify1( void );
        static void computeMaximumContiguousMatches( QAI* pQAI );
};
