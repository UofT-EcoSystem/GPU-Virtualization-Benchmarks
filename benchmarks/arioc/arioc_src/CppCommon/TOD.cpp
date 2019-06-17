/*
  TOD.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// default constructor
TOD::TOD()
{
    Refresh();
}

TOD::~TOD()
{
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Get the current time of day.
/// </summary>
void TOD::Refresh()
{
#ifdef _WIN32
    SYSTEMTIME st;
    GetLocalTime( &st );
    this->yr = st.wYear;
    this->mo = st.wMonth;
    this->da = st.wDay;
    this->hr = st.wHour;
    this->mi = st.wMinute;
    this->se = st.wSecond;
    this->ms = st.wMilliseconds;
#endif

#ifdef __GNUC__
    timeval tv;
    gettimeofday( &tv, NULL );
    struct tm* ptm = localtime( &tv.tv_sec );
    this->yr = 1900 + ptm->tm_year;
    this->mo = 1 + ptm->tm_mon;
    this->da = ptm->tm_mday;
    this->hr = ptm->tm_hour;
    this->mi = ptm->tm_min;
    this->se = ptm->tm_sec;
    this->ms = static_cast<UINT32>(tv.tv_usec/1000);
#endif
}
#pragma endregion