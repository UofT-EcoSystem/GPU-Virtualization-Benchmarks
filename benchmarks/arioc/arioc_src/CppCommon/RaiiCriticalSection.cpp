/*
  RaiiCriticalSection.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#ifdef _WIN32
#pragma region static member variable definitions
CRITICAL_SECTION RcsFactory::m_CS[MAX_CS_COUNT];
bool RcsFactory::m_isCS[MAX_CS_COUNT];

/* this static instance of the RcsFactory class is declared to cause the destructor to be called when the application terminates */
static RcsFactory RCSF;
#pragma endregion

#pragma region class RcsFactory
/// default constructor
RcsFactory::RcsFactory()
{
}

/// destructor
RcsFactory::~RcsFactory()
{
    // release OS resources for each initialized critical section
    for( INT16 n=0; m_isCS[n] && (n<MAX_CS_COUNT); ++n )
        DeleteCriticalSection( m_CS+n );
}

/// [private, static] method getNextCS
CRITICAL_SECTION* RcsFactory::getNextCS()
{
    static UINT32 nCS( 0 );             // (there is one instance of this static variable)

    /* Increment the count of critical section instances that have been initialized.

        - This method should be called by GetCriticalSection<T>, which is called when a specialization of RaiiCriticalSection<T>
            is instantiated; the result goes into a static variable, so this method should be called once by the C++ runtime's
            static initializer for each specialization of the RaiiCriticalSection<T> class template.
        - We have no guarantee that the runtime static initializer is single-threaded, so we use a thread-safe API to increment
            the count.
    */
    UINT32 iCS = InterlockedExchangeAdd( &nCS, 1 );

    // sanity check
    if( iCS >= MAX_CS_COUNT )
        throw new ApplicationException( __FILE__, __LINE__, "%s: the maximum number of critical sections is limited to %u", __FUNCTION__, MAX_CS_COUNT );

    // point to the critical-section instance
    CRITICAL_SECTION* pCS = m_CS + iCS;

    // initialize the critical section and set a flag to indicate that this has been done
    InitializeCriticalSection( pCS );
    m_isCS[iCS] = true;

    // return the pointer to the initialized critical-section instance
    return pCS;
}
#pragma endregion
#endif

#ifdef __GNUC__
#pragma region static member variable definitions
pthread_mutex_t RcsFactory::m_CS[MAX_CS_COUNT];
bool RcsFactory::m_isCS[MAX_CS_COUNT];

/* this static instance of the RcsFactory class is declared to cause the destructor to be called when the application terminates */
static RcsFactory RCSF;
#pragma endregion

#pragma region class RcsFactory
/// default constructor
RcsFactory::RcsFactory()
{
}

/// destructor
RcsFactory::~RcsFactory()
{
    // release OS resources for each initialized critical section
    for( UINT32 n=0; m_isCS[n] && (n<MAX_CS_COUNT); ++n )
        pthread_mutex_destroy( m_CS+n );
}

/// [private, static] method getNextCS
pthread_mutex_t* RcsFactory::getNextCS()
{
    static UINT32 nCS( 0 );             // (there is one instance of this static variable)

    /* Increment the count of critical section instances that have been initialized.

        - This method should be called by GetCriticalSection<T>, which is called when a specialization of RaiiCriticalSection<T>
            is instantiated; the result goes into a static variable, so this method should be called once by the C++ runtime's
            static initializer for each specialization of the RaiiCriticalSection<T> class template.
        - We have no guarantee that the runtime static initializer is single-threaded, so we use a thread-safe API to increment
            the count.
    */
    UINT32 iCS = InterlockedExchangeAdd( &nCS, 1 );

    // sanity check
    if( iCS >= MAX_CS_COUNT )
        throw new ApplicationException( __FILE__, __LINE__, "%s: the maximum number of critical sections is limited to %u", __FUNCTION__, MAX_CS_COUNT );

    // point to the critical-section instance
    pthread_mutex_t* pCS = m_CS + iCS;

    // initialize the critical section and set a flag to indicate that this has been done
    int rval = pthread_mutex_init( pCS, NULL );
    if( rval )
        throw new ApplicationException( __FILE__, __LINE__, "%s:, pthread_mutex_init returned %d", __FUNCTION__, rval );
    m_isCS[iCS] = true;

    // return the pointer to the initialized critical-section instance
    return pCS;
}
#pragma endregion
#endif
