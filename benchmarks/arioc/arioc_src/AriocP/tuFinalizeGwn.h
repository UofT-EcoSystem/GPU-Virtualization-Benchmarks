/*
  tuFinalizeGwn.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuFinalizeGwn__


/// <summary>
/// Class <c>tuFinalizeGwn</c> reformats and consolidates the BRLEAs generated by windowed gapped-alignment traceback.
/// </summary>
class tuFinalizeGwn : public tuFinalizeG
{
    private:
        tuFinalizeGwn( void );

    public:
        tuFinalizeGwn( QBatch* pqb );
        virtual ~tuFinalizeGwn( void );
};

