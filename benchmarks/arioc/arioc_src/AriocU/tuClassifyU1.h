/*
  tuClassifyU1.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuClassifyU1__

/// <summary>
/// Class <c>tuClassifyU1</c> sorts and unduplicates mappings.
/// </summary>
class tuClassifyU1 : public tuClassify1
{
    private:
        tuClassifyU1( void );
        static int QAIComparer( const void*, const void* );

    public:
        tuClassifyU1( QBatch* pqb, volatile UINT32* pqid );
        virtual ~tuClassifyU1( void );
};

