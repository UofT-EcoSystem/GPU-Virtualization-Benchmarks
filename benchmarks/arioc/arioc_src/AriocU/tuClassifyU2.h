/*
  tuClassifyU2.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuClassifyU2__

#ifndef __baseMAPQ__
#include "baseMAPQ.h"
#endif

/// <summary>
/// Class <c>tuClassifyU2</c> filters mappings for reporting and computes mapping qualities (MAPQ)
/// </summary>
class tuClassifyU2 : public tuBaseA, private baseMAPQ
{

    private:
        QBatch*             m_pqb;
        AriocBase*          m_pab;
        volatile UINT32*    m_pqid;

    protected:
        void main( void );

    private:
        tuClassifyU2( void );
        void computeMappingQualities( QAIReference* pqair, UINT32 nLimit );
        void prioritizeMappings( void );
        void filterMappings( void );
        static int QAIComparer( const void* a, const void* b );

    public:
        tuClassifyU2( QBatch* pqb, volatile UINT32* pqid );
        virtual ~tuClassifyU2( void );
};

