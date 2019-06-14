/*
  tuClassifyU1.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuClassifyU1::tuClassifyU1()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuClassifyU1::tuClassifyU1( QBatch* pqb, volatile UINT32* pqid ) : tuClassify1( pqb, &tuClassifyU1::QAIComparer, pqid )
{
}

/// destructor
tuClassifyU1::~tuClassifyU1()
{
}
#pragma endregion

#pragma region private methods
/// [private static] method QAIComparer
int tuClassifyU1::QAIComparer( const void* a, const void* b )
{
    const QAI* pa = reinterpret_cast<const QAI*>(a);
    const QAI* pb = reinterpret_cast<const QAI*>(b);

    // order by V descending, then subId, then Jf
    int rval = pb->pBH->V - pa->pBH->V;
    if( rval == 0 )
    {
        rval = static_cast<int>(pa->subId) - static_cast<int>(pb->subId);
        if( rval == 0 )
            rval = static_cast<int>(pa->Jf) - static_cast<int>(pb->Jf);
    }

    return rval;
}
#pragma endregion
