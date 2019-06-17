/*
  tuGpu.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuGpu__

/// <summary>
/// Class <c>tuGpu</c> is the base implementation for the sequence-alignment pipeline.
/// </summary>
/// <remarks>There is one instance of this class for each GPU.</remarks>
class tuGpu : public tuBaseA
{
    protected:
        static const INT32 MAINLOOPJOIN_TIMEOUT = 2 * 60000;    // timeout for main-loop thread join

        INT16       m_gpuDeviceOrdinal;     // 0-based GPU ordinal
        AriocBase*  m_pab;
        INT64       m_cbCgaReserved;
        HiResTimer  m_hrt;

    protected:
        tuGpu( void );
        void loadR( GpuInfo* pgi );
        void loadH( GpuInfo* _pgi );
        void loadQ( QBatch* pqb );
        void unloadQ( QBatch* pqb );
        virtual void main( void ) = 0;      // implemented in tuGpuP and tuGpuU

    public:
        tuGpu( INT16 gpuDeviceOrdinal, AriocBase* pab );
        virtual ~tuGpu( void );
};

