/*
  tuLoadMP.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuLoadMP__

/// <summary>
/// Class <c>tuLoadMP</c> loads Q-sequence metadata for the current batch
/// </summary>
class tuLoadMP : public tuLoadM
{
    protected:
        void main( void );

    private:
        tuLoadMP( void );

    public:
        tuLoadMP( QBatch* pqb, QfileInfo QFI[2], RaiiFile fileM[2], MfileInfo MFI[2], MfileBuf MFB[2] );
        virtual ~tuLoadMP( void );
};

