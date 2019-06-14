/*
  HiResTimer.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

   Notes:
    Implements a simple hi-resolution timer.

    The start of the interval begins when a HiResTimer object is constructed.
    The interval can be restarted explicitly (Restart method) or when the interval is measured (GetElapsed method).

    Example 1:
        HiResTimer sw(us);       // (microsecond granularity)
        ...
        CDPrint( cdpCD4, "elapsed: %dus", sw.GetElapsed(false) );  // interval since HiResTimer object was constructed

    Example 2:
        HiResTimer sw(us);       // (microsecond granularity)
        ...
        sw.Restart();
        ...
        CDPrint( cdpCD4, "elapsed: %dus", sw.GetElapsed(false) );  // interval since Restart()

    Example 3:
        HiResTimer sw(us);       // (microsecond granularity)
        do
        {
            ...
            CDPrint( cdpCD4, "elapsed: %dus", sw.GetElapsed(true) );  // interval since start of current iteration
        }
        while( ... );

    Note: There is a class named Stopwatch in the CUDA source-code distribution that has similar functionality.
*/
#include "stdafx.h"

#ifdef _WIN32
// default constructor (HiResTimerGranularity = ms)
HiResTimer::HiResTimer( HiResTimerGranularity granularity )
{
    // save the specified granularity
    switch( granularity )
    {
        case us:    // microsecond
            m_granularity = 1000000;
            break;

        default:    // millisecond
            m_granularity = 1000;
            break;
    }

    // if QueryPerformanceFrequency returns true and its frequency is not 1000 (emulation mode)...
    if( QueryPerformanceFrequency( reinterpret_cast<LARGE_INTEGER*>(&m_freq) ) && (m_freq != 1000) )
    {
        // ... use the system hi-res timer
        m_usePerformanceCounter = true;
    }
    else
    {
        // ... use the system tick count
        m_usePerformanceCounter = false;
        m_freq = 1000;
    }

    // get the current tick count
    m_startAt = getCurrent();
}

// destructor
HiResTimer::~HiResTimer()
{
}

// method Restart
void HiResTimer::Restart()
{
    // reset the start time
    m_startAt = getCurrent();
}

// method GetElapsed
int HiResTimer::GetElapsed( const bool restart )
{
    // get the current tick count
    INT64 stopAt = getCurrent();

    // compute the interval in milliseconds or microseconds
    INT32 s = static_cast<INT32>(((stopAt - m_startAt) * m_granularity) / m_freq);

    // optionally save the current tick count as the start of the next interval
    if( restart )
        m_startAt = stopAt;

    // return the interval in ms or us
    return s;
}

// private method getCurrent
INT64 HiResTimer::getCurrent()
{
    INT64 t;

    if( m_usePerformanceCounter )
        QueryPerformanceCounter( reinterpret_cast<LARGE_INTEGER*>(&t) );
    else
        t = GetTickCount();

    return t;
}
#endif

#ifdef __GNUC__
// default constructor (HiResTimerGranularity = ms)
HiResTimer::HiResTimer( HiResTimerGranularity granularity )
{
    // save the specified granularity
    switch( granularity )
    {
        case us:    // microsecond
            m_granularity = 1;
            break;

        default:    // millisecond
            m_granularity = 1000;
            break;
    }

    // get the current time of day in microseconds since the Linux epoch
    m_startAt = getCurrent();
}

// destructor
HiResTimer::~HiResTimer()
{
}

// method Restart
void HiResTimer::Restart()
{
    // reset the start time
    m_startAt = getCurrent();
}

// method GetElapsed
int HiResTimer::GetElapsed( const bool restart )
{
    // get the current time of day in microseconds since the Linux epoch
    INT64 stopAt = getCurrent();

    // compute the interval in microseconds
    INT64 us = stopAt - m_startAt;

    // optionally save the current tick count as the start of the next interval
    if( restart )
        m_startAt = stopAt;

    // return the interval in ms (m_granularity==1000) or us (m_granularity==1)
    return us / m_granularity;
}

// private method getCurrent
INT64 HiResTimer::getCurrent()
{
	timeval tv;
	gettimeofday( &tv, NULL );

    return static_cast<INT64>(tv.tv_sec) * 1000000 +    // current time of day (seconds part)
           tv.tv_usec;                                  // current time of day (microseconds part)
}

/// <summary>
/// Computes the absolute time a specified number of milliseconds from now.
/// </summary>
void HiResTimer::IntervalToAbsoluteTime( timespec& ts, const UINT32 msInterval )
{
    // compute the absolute system time at which the timeout will occur if the condition variable isn't signalled
	timeval tv;
	gettimeofday( &tv, NULL );

    // compute the time of day at which the specified interval ends, in nanoseconds
	UINT64 ns = static_cast<UINT64>(tv.tv_sec) * 1000000000 +   // current time of day (seconds part)
                static_cast<UINT64>(tv.tv_usec) * 1000 +        // current time of day (microseconds part)
                static_cast<UINT64>(msInterval) * 1000000;      // interval (specified in milliseconds)

	ts.tv_sec = ns / 1000000000;
	ts.tv_nsec = ns - (static_cast<UINT64>(ts.tv_sec) * 1000000000);
}
#endif