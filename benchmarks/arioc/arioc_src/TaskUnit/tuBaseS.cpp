/*
  tuBaseS.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// default constructor
tuBaseS::tuBaseS()
{
}

/// destructor
tuBaseS::~tuBaseS()
{
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Runs the virtual implementation of main() on a worker thread
/// </summary>
/// <remarks>This is implemented as a no-op.</remarks>
void tuBaseS::Start()
{
}

/// <summary>
/// Waits for the virtual implementation of main() to terminate
/// </summary>
/// <remarks>If any worker thread has thrown an exception, it is rethrown here on the caller thread.</remarks>
void tuBaseS::Wait( UINT32 msTimeout )
{
    // execute the virtual method implementation on the caller's thread
    this->main();
}
#pragma endregion
