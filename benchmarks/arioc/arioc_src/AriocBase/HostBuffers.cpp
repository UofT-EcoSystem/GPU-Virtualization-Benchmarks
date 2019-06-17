/*
  HostBuffers.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region HostBuffers
/// constructor
HostBuffers::HostBuffers()                                          
{
}

/// destructor
HostBuffers::~HostBuffers()
{
}
#pragma endregion

#pragma region public methods
/// [public] method Reset
void HostBuffers::Reset()
{
    this->Dm.n = 0;
    this->nMapped = 0;
    this->BRLEA.n = 0;
    if( this->BRLEA.p )
        memset( this->BRLEA.p, 0, this->BRLEA.cb );
}
#pragma endregion
