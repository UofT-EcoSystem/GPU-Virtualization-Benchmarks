/*
  tuFinalizeN.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuFinalizeN__

#ifndef __tuFinalizeN1__
#include "tuFinalizeN1.h"
#endif

/// <summary>
/// Class <c>tuFinalizeN</c> counts and computes a BRLEA for each successful nongapped alignment
/// </summary>
class tuFinalizeN : public tuBaseA
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        WinGlobalPtr<pinfoN>    m_pi;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    protected:
        void main( void );

    private:
        tuFinalizeN( void );

    public:
        tuFinalizeN( QBatch* pqb );
        virtual ~tuFinalizeN( void );
};

