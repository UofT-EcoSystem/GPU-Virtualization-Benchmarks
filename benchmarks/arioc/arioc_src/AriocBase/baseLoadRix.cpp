/*
  baseLoadRix.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseLoadRix::baseLoadRix()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
baseLoadRix::baseLoadRix( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, RiFlags flagRi ) : CudaLaunchCommon(4*CUDATHREADSPERWARP),
                                                                                                        m_pqb(pqb),
                                                                                                        m_pab(pqb->pab),
                                                                                                        m_pdbb(pdbb), m_pD(NULL), m_nD(0), m_nRperBlock(0),
                                                                                                        m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"baseLoadRi")),
                                                                                                        m_hrt(us)
{
    pqb->GetGpuConfig( "ThreadsPerBlock_baseLoadRix", m_cudaThreadsPerBlock );

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
baseLoadRix::~baseLoadRix()
{
}
#pragma endregion

#pragma region protected methods
/// [protected] method initGlobalMemory
void baseLoadRix::initGlobalMemory()
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
        INT64 celRi = baseLoadRi::ComputeRiBufsize( m_pdbb->AKP.Mr, m_nD );     // total number of 64-bit values in the Ri buffer
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
UINT32 baseLoadRix::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    /* This implementation (baseLoadRix) differs from the original Arioc implementation (baseLoadRi) in that
        here we are using cooperative threading to get coalesced reads and writes.
        
       The basic strategy is:
        - For each D value, we read all of the needed R-sequence values at once into a set of "adjacent" threads
           in a warp.  These are A21-encoded 8-byte (64-bit) values.  Assuming we provide for the maximum number
           of symbols (Mr) per R sequence, we compute the number of UINT64s per R sequence as:

                (Mr+40) / 21
                    
          This allows for R sequences that straddle UINT64 boundaries because the first symbol is generally not in
           the low-order position in the first UINT64 value.  This gives us:

                Mr        celPerR    nRperWarp (32 threads)
                 86-106  	6	     5
                107-148	    7- 8     4
                149-190     9-10     3
                191-316    11-16     2
                317-652    17-32     1

          More than 652 symbols and we have to fall back on the original implementation (baseLoadRi).
           
          (We still only need (Mr+20)/21 UINT64 values to contain the interleaved R sequences, because the start of
           each Ri is shifted into the low-order position in its first UINT64 value.)
          
          We are trying to do coalesced reads here.  The hardware may have to do an additional read transaction if
           the R sequence straddles its transaction-size boundary (which, as of 2018, is still 128 bytes).  And since
           many R sequences contain fewer than 128 bytes' worth of symbols, we obvoiusly can never be 100% efficient,
           but it's a lot better than reading one 8-byte value at a time from a different address into each thread.
        
        - We do the necessary bit shifting in adjacent threads.  We use PTX warp shuffle instructions to make
           this happen without additional memory accesses.

        - Adjacent threads write their 8-byte results into shared memory.  This involves a transposition:  a "row"
           of original R sequence becomes a "column" of interleaved R sequence).

        - The entire CUDA block then copies the data from shared memory to global memory.

       Each CUDA thread can handle one 8-byte value at a time.  The number of CUDA threads per block and the amount
        of shared memory are both limiting here, but in practice the maximum number of threads per block is only
        1024, which implies only 8Kb or so of shared-memory needed (which is much less than NVidia devices provide).
        Our job here is to compute this so that we can determine the CUDA block and grid dimensions.
    */

    // number of R sequences that can be handled in each warp
    INT32 celPerR = (m_pdbb->AKP.Mr + 40) / 21;
    INT32 nRperWarp = CUDATHREADSPERWARP / celPerR;

    // number of warps per block
    m_cudaThreadsPerBlock = m_pqb->pgi->pCDB->GetDeviceProperties()->maxThreadsPerBlock;
    INT32 nWarpsPerBlock = m_cudaThreadsPerBlock / CUDATHREADSPERWARP;

    // number of R sequences per block
    m_nRperBlock = nRperWarp * nWarpsPerBlock;

    // each block (except the last) computes an even multiple of CUDATHREADSPERWARP Ri sequences
    m_nRperBlock &= -CUDATHREADSPERWARP;      // round down to nearest multiple of CUDATHREADSPERWARP

    // number of 8-byte values per block
    INT32 celPerRi = blockdiv(m_pdbb->AKP.Mr,21);

    /* Here is where we worry about shared-memory bank conflicts.  The data is laid out in shared memory in two dimensions;
        if R00, R01, R02... are successive R sequences, the ith row contains the ith element in each R sequence:

            +-------+-------+-------+-------+
            | R00 0 | R01 0 | R02 0 | R03 0 | ...
            +-------+-------+-------+-------+
            | R00 1 | R01 1 | R02 1 | R03 1 | ...
            +-------+-------+-------+-------+
            | R00 2 | R01 2 | R02 2 | R03 2 | ...
            +-------+-------+-------+-------+
            | R00 3 | R01 3 | R02 3 | R03 3 | ...
            +-------+-------+-------+-------+
                .       .       .       .
                .       .       .       .
                .       .       .       .
           
      Since adjacent threads write to "vertically-adjacent" locations in shared memory, we must avoid shared-memory bank
       conflicts by ensuring that these elements are not an even multiple of 32 elements apart.

      After the data is written to shared memory, it is copied to the Ri buffer in global memory (one thread per
       "column" of shared-memory data), so these writes are coalesced and obviously involve no shared-memory bank
       conflicts.
    */

    // compute the shared-memory "stride" (number of 64-bit elements per row)
    INT32 sharedMemStride = (m_nRperBlock & 0x1F) ? m_nRperBlock : m_nRperBlock+1;

    // compute the amount of "dynamic" shared memory per block (to specify as a kernel launch parameter)
    INT32 cbSharedMem = sharedMemStride * celPerRi * sizeof(UINT64);

    // sanity check
    if( cbSharedMem > static_cast<INT32>(m_pqb->pgi->pCDB->GetDeviceProperties()->sharedMemPerBlock) )
    {
        /* This shouldn't happen unless NVidia increases the maximum number of threads per block to 4K or more,
            at which point we ought to be rethinking our approach here. */
        throw new ApplicationException( __FILE__, __LINE__, "unable to allocate shared memory" );
    }

    /* At this point we are prepared for a sharedMemStride x celPerRi array in shared memory. */

    // set block and grid dimensions
    UINT64 nBlocks = blockdiv( m_nD, static_cast<UINT32>(m_nRperBlock));
    UINT64 nKernelThreads = nBlocks * m_cudaThreadsPerBlock;
    computeKernelGridDimensions( d3g, d3b, nKernelThreads );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: m_nD=%u Mr=%d celPerR=%d celPerRi=%d m_nRperBlock=%d sharedMemStride=%d cbShared=%d d3g=[%u,%u,%u] d3b=[%u,%u,%u]",
                        __FUNCTION__,
                        m_nD, m_pdbb->AKP.Mr, celPerR, celPerRi, m_nRperBlock, sharedMemStride, cbSharedMem,
                        d3g.x, d3g.y, d3g.z, d3b.x, d3b.y, d3b.z );
#endif

    // return the number of bytes of shared memory
    return cbSharedMem;
}

/// [protected] method copyKernelResults
void baseLoadRix::copyKernelResults()
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


            WinGlobalPtr<UINT64> Dcxxx( m_pdbb->Dc.Count, true );
            m_pdbb->Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );



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
void baseLoadRix::resetGlobalMemory( void )
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
/// [public] method GetInstance
tuBaseS* baseLoadRix::GetInstance( const char* _ptumKey, QBatch* _pqb, DeviceBuffersBase* _pdbb, RiFlags _flagRi )
{
    /* Return an instance of baseLoadRix only if both of the following conditions are met:
        - the optional "loadRix" XParam is true (or not specified)
        - the maximum number of Ri sequences is not exceeded (see comments in computeGridDimensions())

       The caller is responsible for freeing the returned instance, e.g.,
            RaiiPtr<tuBaseS> pBaseLoadRi = baseLoadRix::GetInstance( ... );
    */
    if( _pqb->pab->preferLoadRix && (_pdbb->AKP.Mr <= baseLoadRix::MaxMr) )
        return reinterpret_cast<tuBaseS*>(new baseLoadRix( _ptumKey, _pqb, _pdbb, _flagRi ));
    
    return reinterpret_cast<tuBaseS*>(new baseLoadRi( _ptumKey, _pqb, _pdbb, _flagRi ));
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Loads interleaved R sequence data.
/// </summary>
void baseLoadRix::main()
{
    CDPrint( cdpCD3, "[%d] %s::%s ...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );

    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    UINT32 cbSharedMem = computeGridDimensions( d3g, d3b );

    launchKernel( d3g, d3b, initSharedMemory(cbSharedMem) );

    copyKernelResults();
    resetGlobalMemory();
    
    // performance metrics
    INT64 usPostLaunch = m_hrt.GetElapsed( false );
    InterlockedExchangeAdd( &m_ptum->us.PostLaunch, usPostLaunch );
    m_usXferR += usPostLaunch;
    InterlockedExchangeAdd( &AriocBase::aam.us.XferR, m_usXferR );

    CDPrint( cdpCD3, "[%d] %s::%s completed", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
}
#pragma endregion
