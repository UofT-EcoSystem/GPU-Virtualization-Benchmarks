/*
  tuGpuU.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuGpuU__

/// <summary>
/// Class <c>tuGpuU</c> implements a sequence-alignment pipeline.
/// </summary>
/// <remarks>There is one instance of this class for each GPU.</remarks>
class tuGpuU : public tuGpu
{
    protected:
        void main( void );

    private:
        tuGpuU( void );

    public:
        tuGpuU( INT16 _gpuDeviceOrdinal, AriocBase* _pab );
        virtual ~tuGpuU( void );
};

