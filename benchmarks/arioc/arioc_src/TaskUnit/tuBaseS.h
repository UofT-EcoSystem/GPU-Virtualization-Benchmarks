/*
  tuBaseS.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuBaseS__

/// <summary>
/// Task unit base class implementation (synchronous)
/// </summary>
class tuBaseS
{
    protected:
        virtual void main( void ) = 0;

    public:
        tuBaseS( void );
        virtual ~tuBaseS( void );
        void Start( void );
        void Wait( UINT32 msTimeout = INFINITE );
};
