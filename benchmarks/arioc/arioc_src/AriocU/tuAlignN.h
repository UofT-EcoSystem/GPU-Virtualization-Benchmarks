/*
  tuAlignN.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuAlignN__

#if !defined(__tuAlignN32__)
#include "tuAlignN32.h"
#endif

/// <summary>
/// Class <c>tuAlignN</c> does nongapped short-read alignment
/// </summary>
class tuAlignN : public tuBaseS
{
    private:
        QBatch*         m_pqb;
        AriocBase*      m_pab;
        HiResTimer      m_hrt;

    protected:
        void main( void );

    private:
        tuAlignN( void );

        void alignN10( void );

        void alignN20( void );
        void initGlobalMemory22( void );
        void launchKernel22( void );
        void copyKernelResults22( void );
        void resetGlobalMemory22( void );

        void alignN30( void );
        void initGlobalMemory34( void );
        void launchKernel34( void );
        void resetGlobalMemory34( void );

public:
        tuAlignN( QBatch* pqb );
        virtual ~tuAlignN( void );
};
