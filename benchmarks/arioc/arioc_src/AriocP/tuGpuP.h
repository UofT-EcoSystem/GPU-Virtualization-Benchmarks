/*
  tuGpuP.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuGpuP__

/// <summary>
/// Class <c>tuGpuP</c> implements a sequence-alignment pipeline.
/// </summary>
/// <remarks>There is one instance of this class for each GPU.</remarks>
class tuGpuP : public tuGpu
{
    protected:
        void main( void );

    private:
        tuGpuP( void );

    public:
        tuGpuP( INT16 gpuDeviceOrdinal, AriocBase* pab );
        virtual ~tuGpuP( void );
};

