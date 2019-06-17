/*
  tuLoadMU.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuLoadMU__

/// <summary>
/// Class <c>tuLoadMU</c> loads Q-sequence metadata for the current batch
/// </summary>
class tuLoadMU : public tuLoadM
{
    protected:
        void main( void );

    private:
        tuLoadMU( void );

    public:
        tuLoadMU( QBatch* pqb, QfileInfo* pQFI, RaiiFile* pFileM, MfileInfo* pMFI, MfileBuf* pMFB );
        virtual ~tuLoadMU( void );
};

