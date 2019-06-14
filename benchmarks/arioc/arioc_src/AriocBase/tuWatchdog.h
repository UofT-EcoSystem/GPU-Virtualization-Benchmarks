/*
  tuWatchdog.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuWatchdog__

// forward definition
class AriocBase;

/// <summary>
/// Class <c>tuWatchdog</c> runs a "watchdog" thread.
/// </summary>
class tuWatchdog : public tuBaseA
{
    private:
        const UINT32 MININTERVAL = 1;       // 1 second
        const UINT32 MAXINTERVAL = 10 * 60; // 10 minutes

        AriocBase*  m_pab;
        UINT32      m_msInterval;

        static UINT32   m_threadId[32];
        static UINT32   m_watchdogState[32];
        static UINT32   m_watchdogStateChanged;
        static UINT32   m_isWatchdogDevice;

    private:
        tuWatchdog( void );

    protected:
        void main( void );

    public:
        tuWatchdog( AriocBase* _pab );
        virtual ~tuWatchdog( void );
        void Watch( INT32 _deviceId );
        void SetWatchdogState( INT32 _deviceId, UINT32 _state, bool _verifyDeviceId=true );
        void Halt( INT32 _deviceId );
};
