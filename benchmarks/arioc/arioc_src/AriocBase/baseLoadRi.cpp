/*
  baseLoadRi.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseLoadRi::baseLoadRi()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
baseLoadRi::baseLoadRi( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, RiFlags flagRi ) : CudaLaunchCommon(4*CUDATHREADSPERWARP),
                                                                                                      m_pqb(pqb),
                                                                                                      m_pab(pqb->pab),
                                                                                                      m_pdbb(pdbb), m_pD(NULL), m_nD(0),
                                                                                                      m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseLoadRi")),
                                                                                                      m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseLoadRi", m_cudaThreadsPerBlock );

    /* Set up the constant parameter values for CUDA kernels; the DeviceBuffersBase pointer references an object defined in
        the current QBatch instance so these values will remain available for subsequent CUDA kernels to use ... */
    pdbb->InitKernelParameters( pqb );

    // use a flag specified by the caller to determine which buffer to use
    m_pdbb->flagRi = flagRi;
    switch( flagRi )
    {
        case riDc:
            m_pD = pdbb->Dc.p;
            m_nD = pdbb->Dc.n;
            break;

        case riDm:
            m_pD = pdbb->Dm.p;
            m_nD = pdbb->Dm.n;
            break;

        case riDx:
            m_pD = pdbb->Dx.p;
            m_nD = pdbb->Dx.n;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for flagRi: %d", flagRi );
            break;
    }


#if TODO_CHOP_WHEN_DEBUGGED
    if( !m_pD || !m_nD )
        DebugBreak();
#endif
}

/// destructor
baseLoadRi::~baseLoadRi()
{
}
#pragma endregion

#pragma region protected methods
/// [protected] method initGlobalMemory
void baseLoadRi::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

            low:    Qw      Qwarps
                    Qi      interleaved Q sequence data
                    ...     ...
                    Ri      interleaved R sequence data

            high:   ...     ...
        */

        // allocate a buffer to contain interleaved R sequence data
        INT64 celRi = ComputeRiBufsize( m_pdbb->AKP.Mr, m_nD );     // total number of 64-bit values in the Ri buffer
        CREXEC( m_pqb->DB.Ri.Alloc( cgaLow, celRi, true ) );
        SET_CUDAGLOBALPTR_TAG( m_pqb->DB.Ri, "DB.Ri" );
    }
    catch( ApplicationException* pex )
    {
        const char details[] = "Unable to allocate GPU memory for interleaved R sequence data. "
                               "Try a smaller maximum J-list size (maxJ) for gapped alignments, "
                               "a smaller maximum batch size (batchSize), or "
                               "a smaller maximum seed depth.";
        pex->SetCallerExceptionInfo( __FILE__, __LINE__, details );
        throw pex;
    }
}

/// [protected] method computeGridDimensions
void baseLoadRi::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nD );
}

/// [protected] method copyKernelResults
void baseLoadRi::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // performance metrics
        InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_usXferR );     // (at this point, XferR contains only the prelaunch timing)

        // block until the CUDA kernel completes
        CREXEC( waitForKernel() );


#if TODO_CHOP_WHEN_DEBUGGED
        if( m_pqb->DB.Ri.Count )
        {
            WinGlobalPtr<UINT64> Rixxx( m_pqb->DB.Ri.Count, true );
            m_pqb->DB.Ri.CopyToHost( Rixxx.p, Rixxx.Count );

            // dump the first R sequence
            CDPrint( cdpCD0, "%s: first R in DB.Ri at 0x%016llx...", __FUNCTION__, m_pqb->DB.Ri.p );
            AriocCommon::DumpA21( Rixxx.p, 151, CUDATHREADSPERWARP );
        }
        else
            CDPrint( cdpCD0, "%s: m_pqb->DB.Ri.Count=0", __FUNCTION__ );
#endif

        // performance metrics
        UINT64 usLaunch = m_hrt.GetElapsed( true );
        m_usXferR += usLaunch;
        InterlockedExchangeAdd( &m_ptum->us.Launch, usLaunch );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [protected] method resetGlobalMemory
void baseLoadRi::resetGlobalMemory( void )
{
    /* CUDA global memory layout after reset:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                ...     ...
                Ri      interleaved R sequence data

        high:   ...     ...
    */
}
#pragma endregion

#pragma region public methods
/// [public static] method ComputeRiBufsize
INT64 baseLoadRi::ComputeRiBufsize( INT32 _Mr, UINT32 _nD )
{
    /* To compute the number of elements in the buffer that will contain interleaved R sequence data, we lay out the
        64-bit values in a set of 2-dimensional tiles, each of which contains the interleaved 64-bit values for one
        CUDA warp.  The size of each tile is thus
            celMr * CUDATHREADSPERWARP
        and the number of 64-bit elements in the buffer is
            nWarps * (celMr * CUDATHREADSPERWARP)
    */
    INT64 celMr = blockdiv(_Mr, 21);                    // number of 64-bit values required to represent Mr symbols
    INT64 nWarps = blockdiv( _nD, CUDATHREADSPERWARP ); // number of warps required to load the R values
    return nWarps * (celMr * CUDATHREADSPERWARP);       // total number of 64-bit values in the Ri buffer
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Loads interleaved R sequence data.
/// </summary>
void baseLoadRi::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s ...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    launchKernel( d3g, d3b, initSharedMemory() );

    copyKernelResults();
    resetGlobalMemory();
    
    // performance metrics
    INT64 usPostLaunch = m_hrt.GetElapsed( false );
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, usPostLaunch );
    m_usXferR += usPostLaunch;
    InterlockedExchangeAdd( &AriocBase::aam.us.XferR, m_usXferR );

    CDPrint( cdpCD3, "[%d] %s::%s completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
