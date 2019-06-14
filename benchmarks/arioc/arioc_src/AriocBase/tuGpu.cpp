/*
  tuGpu.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] default constructor
tuGpu::tuGpu()
{
}

/// <param name="iPart">a 0-based ordinal (assigned by the caller) (one per GPU)</param>
/// <param name="pab">a reference to the application's <c>AriocP</c> or <c>AriocU</c> instance</param>
tuGpu::tuGpu( INT16 gpuDeviceOrdinal, AriocBase* pab ) : m_gpuDeviceOrdinal(gpuDeviceOrdinal),
                                                         m_pab(pab),
                                                         m_cbCgaReserved(CUDAMINRESERVEDGLOBALMEMORY)
{
    // look for the optional "cgaReserved" Xparam
    INT32 i = pab->paamb->Xparam.IndexOf( "cgaReserved" );
    if( i >= 0 )
        m_cbCgaReserved = max2( CUDAMINRESERVEDGLOBALMEMORY, pab->paamb->Xparam.Value(i) );
}

/// destructor
tuGpu::~tuGpu()
{
}
#pragma endregion

#pragma region protected methods
/// [protected] method loadR
void tuGpu::loadR( GpuInfo* pgi )
{
    if( m_pab->usePinnedR )
        pgi->pR = m_pab->R;
    else
    {
        // copy R table to CUDA global memory
        CRVALIDATOR;
        SET_CUDAGLOBALPTR_TAG( pgi->bufR, "R" );
        CREXEC( pgi->bufR.Realloc( m_pab->celR, false ) );
        CREXEC( pgi->bufR.CopyToDevice( m_pab->R, pgi->bufR.Count ) );
        pgi->pR = pgi->bufR.p;

        CDPrint( cdpCD2, "%s: initialized R buffer (%llu bytes) in CUDA global memory", __FUNCTION__, pgi->bufR.cb );
    }
}

/// [protected] method loadH
void tuGpu::loadH( GpuInfo* pgi )
{
    try
    {
        if( m_pab->Hn.p )
            pgi->pHn = m_pab->Hn.p;     // nongapped H table in page-locked ("pinned") memory
        else
        {
            // copy nongapped H table to CUDA global memory
            CRVALIDATOR;
            CREXEC( pgi->bufHn.Realloc( m_pab->Hng.Count, false ) );
            CREXEC( pgi->bufHn.CopyToDevice( m_pab->Hng.p, pgi->bufHn.Count ) );
            pgi->pHn = pgi->bufHn.p;

            CDPrint( cdpCD2, "%s: initialized Hn buffer (%llu bytes) in CUDA global memory", __FUNCTION__, pgi->bufHn.cb );

            // discard the host buffer if it is no longer needed
            if( InterlockedIncrement( &m_pab->Hng.n ) == static_cast<UINT32>(m_pab->nGPUs) )
            {
                m_pab->Hng.Free();
                CDPrint( cdpCD2, "%s: freed Hn staging buffer", __FUNCTION__ );
            }
        }

        if( m_pab->Hg.p )
            pgi->pHg = m_pab->Hg.p;     // gapped H table in page-locked ("pinned") memory
        else
        {
            // copy gapped H table to CUDA global memory
            CRVALIDATOR;
            CREXEC( pgi->bufHg.Realloc( m_pab->Hgg.Count, false ) );
            CREXEC( pgi->bufHg.CopyToDevice( m_pab->Hgg.p, pgi->bufHg.Count ) );
            pgi->pHg = pgi->bufHg.p;

            CDPrint( cdpCD2, "%s: initialized Hg buffer (%llu bytes) in CUDA global memory", __FUNCTION__, pgi->bufHg.cb );

            // discard the host buffer if it is no longer needed
            if( InterlockedIncrement( &m_pab->Hgg.n ) == static_cast<UINT32>(m_pab->nGPUs) )
            {
                m_pab->Hgg.Free();
                CDPrint( cdpCD2, "%s: freed Hg staging buffer", __FUNCTION__ );
            }
        }
    }
    catch( ApplicationException* _pex )
    {
        _pex->SetCallerExceptionInfo( __FILE__, __LINE__, "unable to allocate CUDA global memory for H table" );
        throw _pex;
    }
}

/// [protected] method loadQ
void tuGpu::loadQ( QBatch* pqb )
{
    CRVALIDATOR;

    HiResTimer hrt(us);

    try
    {
        /* CUDA global memory layout for all kernels:

             low:   Qw      Qwarps
                    Qi      interleaved Q sequence data
                    ...     ...

             high:  ...     ...
        */

#if TODO_CHOP_WHEN_DEBUGGED
        pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
#endif


        /* Load the Qwarp and Q sequence data into global memory on the current GPU; these allocations are freed just before the
            current QBatch instance is released (see tuTailP::main) */

        // Qwarp data
        CREXEC( pqb->DB.Qw.Alloc( cgaLow, pqb->QwBuffer.n, false ) );
        CREXEC( pqb->DB.Qw.CopyToDevice( pqb->QwBuffer.p, pqb->DB.Qw.Count ) );
        pqb->DB.Qw.n = pqb->QwBuffer.n;
        SET_CUDAGLOBALPTR_TAG( pqb->DB.Qw, "DB.Qw" );

        // interleaved Q sequence data
        CREXEC( pqb->DB.Qi.Alloc( cgaLow, pqb->QiBuffer.n, false ) );
        CREXEC( pqb->DB.Qi.CopyToDevice( pqb->QiBuffer.p, pqb->DB.Qi.Count ) );
        pqb->DB.Qi.n = pqb->QiBuffer.n;
        SET_CUDAGLOBALPTR_TAG( pqb->DB.Qi, "DB.Qi" );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }

    // performance metrics
    InterlockedExchangeAdd( &AriocBase::aam.us.XferQ, hrt.GetElapsed(false) );
}

/// [protected] method unloadQ
void tuGpu::unloadQ( QBatch* pqb )
{
    // free the GPU buffers that contain the Qwarp and interleaved Q sequence data   
    pqb->DB.Qi.Free();
    pqb->DB.Qw.Free();

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "tuGpu::unloadQ dumps the global allocation for batch 0x%016llx...", pqb );
    pqb->pgi->pCGA->DumpUsage();
#endif
}
#pragma endregion
