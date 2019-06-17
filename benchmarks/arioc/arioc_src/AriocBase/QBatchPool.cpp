/*
  QBatchPool.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
QBatchPool::QBatchPool() : m_semPool(0,0), m_evPoolFull(true,false)
{
}

/// constructor
QBatchPool::QBatchPool( INT32 _maxPool, AriocBase* _pab, GpuInfo* _pgi ) : m_semPool(0,_maxPool),     // initial count: 0; maximum count: (as specified)
                                                                           m_evPoolFull(true,false)   // (manual reset; initially nonsignalled)
{
    // allocate an empty list of QBatch objects
    m_qb.Realloc( _maxPool, true );

    // create QBatch objects
    for( INT32 n=0; n<_maxPool; ++n )
        this->Enpool( new QBatch( this, n, _pab, _pgi ) );
}

/// destructor
QBatchPool::~QBatchPool()
{
#if TODO_CHOP_WHEN_DEBUGGED
    // sanity check
    if( m_qb.n != m_qb.Count )
        throw new ApplicationException( __FILE__, __LINE__, "non-empty QBatchPool" );   // (yes, we know it's in a destructor, but it's going away once all is debugged)
#endif

    for( UINT32 n=0; n<static_cast<UINT32>(m_qb.Count); ++n )
        delete m_qb.p[n];
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Pushes a QBatch into the pool
/// </summary>
void QBatchPool::Enpool( QBatch* pqb )
{
#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s: QBatch instance at 0x%016llx", pqb->pgi->deviceId, __FUNCTION__, pqb );
#endif

    RaiiCriticalSection<QBatchPool> rcs;

    // add to the list of allocated QBatch objects
    m_qb.p[m_qb.n++] = pqb;

    // release the semaphore
    m_semPool.Release( 1 );

    // conditionally signal that the pool is full
    if( m_qb.n == m_qb.Count )
    {
#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCDb, "QBatchPool::Enpool signals that the pool is full" );
#endif
        m_evPoolFull.Set();
    }

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s: QBatch instance at 0x%016llx is back in the pool", pqb->pgi->deviceId, __FUNCTION__, pqb );
#endif
}

/// <summary>
/// Pops a QBatch out of the pool
/// </summary>
QBatch* QBatchPool::Depool( INT16 estimatedNmax, InputFileGroup::FileInfo* pfi )
{
#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: waiting for a QBatch instance...", __FUNCTION__ );
#endif

    RaiiCriticalSection<QBatchPool> rcs;

    // wait for a QBatch instance to become available
    m_semPool.Wait( POOL_TIMEOUT );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: got a QBatch instance", __FUNCTION__ );
#endif

    // reset the signal
    m_evPoolFull.Reset();

    // extract the next available QBatch instance from the list
    QBatch* pqb = m_qb.p[--m_qb.n];
    m_qb.p[m_qb.n] = NULL;

    // return the empty QBatch instance
    pqb->Initialize( estimatedNmax, pfi );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: QBatch instance at 0x%016llx", __FUNCTION__, pqb );
#endif

    return pqb;
}

/// <summary>
/// Blocks until the pool is full (i.e. until all QBatch instances have been released)
/// </summary>
bool QBatchPool::Wait( UINT32 msTimeout )
{
    CDPrint( cdpCDb, "QBatchPool::Wait waits until the pool is full" );
    return m_evPoolFull.Wait( msTimeout );
}
#pragma endregion
