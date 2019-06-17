/*
  tuAlignGwn.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuAlignGwn__

#ifndef __tuAlignGw10__
#include "tuAlignGw10.h"
#endif


/// <summary>
/// Class <c>tuAlignGwn</c> computes gapped short-read alignments for pairs with one unmapped mate, using the mapped mate as an anchor
/// </summary>
class tuAlignGwn : public tuBaseS
{
    private:
        QBatch*     m_pqb;
        AriocBase*  m_pab;
        HiResTimer  m_hrt;

    protected:
        void main( void );

    private:
        tuAlignGwn( void );

        void initGlobalMemory00( void );
        void launchKernel00( void );
        void resetGlobalMemory00( void );
        void alignGwn00( void );

        void alignGwn10( void );
        void launchKernel11( void );
        void resetGlobalMemory19( void );

    public:
        tuAlignGwn( QBatch* pqb );
        virtual ~tuAlignGwn( void );
};

