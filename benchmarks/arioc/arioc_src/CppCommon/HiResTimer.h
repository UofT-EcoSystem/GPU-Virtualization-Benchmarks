/*
  HiResTimer.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __HiResTimer__

enum HiResTimerGranularity
{
    ms = 0,     // millisecond
    us = 1      // microsecond
};


/// <summary>
/// Class <c>HiResTimer</c> implements simple elapsed-time measurement with millisecond or microsecond granularity (although the actual granularity depends on the OS).
/// </summary>
class HiResTimer
{
    private:
        INT32   m_granularity;
        INT64   m_freq;
        INT64   m_startAt;
        bool    m_usePerformanceCounter;
        INT64   getCurrent( void );

    public:
        HiResTimer( const HiResTimerGranularity granularity = ms );
        virtual ~HiResTimer( void );
        void Restart( void );
        int GetElapsed( const bool restart );

#ifdef __GNUC__
        static void IntervalToAbsoluteTime( timespec& ts, const UINT32 msInterval );
#endif
};

