/*
  HostBuffers.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __HostBuffers__

/// <summary>
/// Class <c>HostBuffers</c> defines buffers common to the CUDA kernels that join J lists (Df lists) for paired-end alignment
/// </summary>
class HostBuffers
{
    public:
        WinGlobalPtr<UINT64>    Dm;         // D values for mapped Q sequences
        WinGlobalPtr<UINT32>    BRLEA;      // BRLEAs for mapped Q sequences
        UINT32                  nMapped;    // number of mapped Q sequences (i.e. number of BRLEAs)

    public:
        HostBuffers();
        virtual ~HostBuffers();
        void Reset( void );
};
