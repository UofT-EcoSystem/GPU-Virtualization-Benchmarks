/*
  tuAlignGs.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuAlignGs__

#ifndef __tuAlignGs10__
#include "tuAlignGs10.h"
#endif


/// <summary>
/// Class <c>tuAlignGs</c> computes gapped short-read alignments for paired-end reads
/// </summary>
class tuAlignGs : public tuBaseS
{
    private:
        QBatch*     m_pqb;
        AriocBase*  m_pab;
        HiResTimer  m_hrt;

    protected:
        void main( void );

    private:
        tuAlignGs( void );

        void alignGs10( void );
        void launchKernel11( void );

        void alignGs20( void );
        void initGlobalMemory20( void );

        void alignGs30( void );
        UINT32 initGlobalMemory30( void );
        void launchKernel34( void );
        void initGlobalMemory36( void );
        void launchKernel36( void );
        void resetGlobalMemory36( void );

        void alignGs40( void );
        void launchKernel49( void );
        void resetGlobalMemory49( void );

    public:
        tuAlignGs( QBatch* pqb );
        virtual ~tuAlignGs( void );
};
