/*
  AppGlobal.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

/// <summary>
/// This is the main program entry point.
/// </summary>
int main( int argc, char* argv[] )
{
    AppMain appMain;
    AppGlobalCommon appGlobalCommon( &appMain );
    return appGlobalCommon.Run( argc, argv, "AriocU" );
}
