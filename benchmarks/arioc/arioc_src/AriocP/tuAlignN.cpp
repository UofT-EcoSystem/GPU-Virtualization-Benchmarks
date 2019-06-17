/*
  tuAlignN.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"
#include <thrust/system_error.h>

#pragma region constructor/destructor
/// [private] constructor
tuAlignN::tuAlignN()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuAlignN::tuAlignN( QBatch* pqb ) : m_pqb(pqb), m_pab(pqb->pab)
{
}

/// destructor
tuAlignN::~tuAlignN()
{
}
#pragma endregion

#pragma region private methods
/// [private] method countDxForIteration
UINT32 tuAlignN::countDxForIteration()
{
    // start by assuming that we can handle all of the values in the Dx buffer in one iteration
    UINT32 n = m_pqb->DBn.Dx.n;
    INT32 nIterations = 1;

    // estimate the total amount of GPU memory required for Ri (interleaved R sequence) data
    INT64 cb = baseLoadRi::ComputeRiBufsize( m_pqb->DBn.AKP.Mr, m_pqb->DBn.Dx.n ) * sizeof(UINT64);

    // get the amount of available GPU memory
    INT64 cbAvailable = m_pqb->pgi->pCGA->cbFree;

    if( cb > cbAvailable )
    {
        // with two or more iterations, we need space for a copy of the Dx buffer
        INT64 cbDxCopy = round2power(m_pqb->DBn.Dx.n*sizeof(UINT64), CudaGlobalAllocator::Granularity);
        cbAvailable = m_pqb->pgi->pCGA->cbFree - cbDxCopy;

        do
        {
            /* Recompute Ri memory usage for a smaller number of Dx values.

               We don't expect more than a few iterations, so we simply try increasingly large denominators
                until Ri memory usage fits into what's available.
            */
            ++nIterations;
            n = blockdiv( m_pqb->DBn.Dx.n, nIterations );
            cb = baseLoadRi::ComputeRiBufsize( m_pqb->DBn.AKP.Mr, n ) * sizeof( UINT64 );
        }
        while( cb > cbAvailable );
    }
    
    if( nIterations > 1 )
        CDPrint( cdpCD3, "[%d] %s: Dx.n=%u nIterations=%d n=%u Ri bytes needed/available = %llu/%llu",
                         m_pqb->pgi->deviceId, __FUNCTION__, m_pqb->DBn.Dx.n, nIterations, n, cb, cbAvailable );

    // return the number of Dx values for which the Ri buffer will fit into available GPU memory
    return n;
}

/// [private] method alignN10
void tuAlignN::alignN10()
{
    /* CUDA global memory here:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBn.Dc  paired candidates for nongapped alignment

        high:   DBn.Dx  unpaired candidates for nongapped alignment
    */

    if( m_pqb->DBn.Dc.n  == 0 )
        return;

    /* N10: load interleaved R sequence data */
    RaiiPtr<tuBaseS> pBaseLoadRi = baseLoadRix::GetInstance( "tuAlignN10", m_pqb, &m_pqb->DBn, riDc );
    pBaseLoadRi->Start();
    pBaseLoadRi->Wait();


#if TODO_GET_RID_OF_THIS_ASAP
    if( m_loadRix )
    {
        baseLoadRix k10( "tuAlignN10", m_pqb, &m_pqb->DBn, riDc );
        k10.Start();
        k10.Wait();


        //WinGlobalPtr<UINT64> Rixxx( m_pqb->DB.Ri.Count, true );
        //m_pqb->DB.Ri.CopyToHost( Rixxx.p, Rixxx.Count );

        //m_pqb->DB.Ri.Free();

        //baseLoadRi k10o( "tuAlignN10", m_pqb, &m_pqb->DBn, riDc );
        //k10o.Start();
        //k10o.Wait();

        //WinGlobalPtr<UINT64> Riooo( m_pqb->DB.Ri.Count, true );
        //m_pqb->DB.Ri.CopyToHost( Riooo.p, Riooo.Count );

        //if( Rixxx.Count != Riooo.Count )
        //    CDPrint( cdpCD0, __FUNCTION__ );

        //for( UINT32 i=0; i<m_pqb->DBn.Dc.n; ++i )
        //{
        //    for( UINT32 ii=0; ii<8; ++ii )
        //    {
        //        if( Rixxx.p[i+ii*CUDATHREADSPERWARP] != Riooo.p[i+ii*CUDATHREADSPERWARP] )
        //            CDPrint( cdpCD0, __FUNCTION__ );            
        //    }
        //}

        //CDPrint( cdpCD0, "%s: Mr=%d nDc=%u", __FUNCTION__, m_pqb->DBn.AKP.Mr, m_pqb->DBn.Dc.n ); 
    }
    else
    {
        baseLoadRi k10( "tuAlignN10", m_pqb, &m_pqb->DBn, riDc );
        k10.Start();
        k10.Wait();
    }
#endif
}

/// [private] method alignN20
void tuAlignN::alignN20()
{
    /* N20: do nongapped alignment (Dc list) */
    baseAlignN k20( "tuAlignN20", m_pqb, &m_pqb->DBn, riDc, true );
    k20.Start();
    k20.Wait();

    /* N21: count the number of mapped positions (D values) */
    launchKernel21();

    /* N22: build a list (Dm) that contains only the mapped D values */
    initGlobalMemory22();
    launchKernel22();
    copyKernelResults22();
    resetGlobalMemory22();
}

/// [private] method alignN30
void tuAlignN::alignN30()
{
    /* N30: count per-Q mappings (update the Qwarp buffer in GPU global memory) */
    baseCountA k30( "tuAlignN30", m_pqb, &m_pqb->DBn, riDm );
    k30.Start();
    k30.Wait();

    // N31: flag candidates for windowed gapped alignment
    launchKernel31();

#if TODO_MAYBE
    /* N34: build "candidate" (Du) and "leftover" (Dl) lists for subsequent (gapped) alignment */
    initGlobalMemory34();
    launchKernel34();
    resetGlobalMemory34();
#endif


}

/// [private] method alignN40
void tuAlignN::alignN40()
{

#if TRACE_SQID
    // what's the concordant-mapping status for the sqId?
    WinGlobalPtr<Qwarp> Qwxxx( m_pqb->DB.Qw.n, false );
    m_pqb->DB.Qw.CopyToHost( Qwxxx.p, Qwxxx.Count );
    for( UINT32 iw=0; iw<m_pqb->DB.Qw.n; ++iw )
    {
        Qwarp* pQw = Qwxxx.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            UINT64 sqId = pQw->sqId[iq];
            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                CDPrint( cdpCD0, "%s: 0x%016llx iw=%u iq=%u nAc=%d", __FUNCTION__, sqId, iw, iq, pQw->nAc[iq] );
            }
        }
    }
#endif


    // N40: identify concordantly-mapped pairs
    baseCountC k40( "tuAlignN40", m_pqb, &m_pqb->DBn, riDm, false );
    k40.Start();
    k40.Wait();

#if TRACE_SQID
    // what's the concordant-mapping status for the sqId?
    WinGlobalPtr<Qwarp> Qwyyy( m_pqb->DB.Qw.n, false );
    m_pqb->DB.Qw.CopyToHost( Qwyyy.p, Qwyyy.Count );
    for( UINT32 iw=0; iw<m_pqb->DB.Qw.n; ++iw )
    {
        Qwarp* pQw = Qwyyy.p + iw;

        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            UINT64 sqId = pQw->sqId[iq];
            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                CDPrint( cdpCD0, "%s: 0x%016llx iw=%u iq=%u nAc=%d", __FUNCTION__, sqId, iw, iq, pQw->nAc[iq] );
            }
        }
    }

#endif


    /* N45: eliminate concordantly-mapped pairs as candidates for subsequent alignment */
    baseFilterD k45m( "tuAlignN45m", m_pqb, &m_pqb->DBn, riDm, m_pab->aas.ACP.AtN );
    k45m.Start();
    k45m.Wait();

    baseFilterD k45x( "tuAlignN45x", m_pqb, &m_pqb->DBn, riDx, m_pab->aas.ACP.AtN );
    k45x.Start();
    k45x.Wait();

    // N47: count the number of candidates for subsequent alignment
    launchKernel47();

    // N48: build the list of candidates for subsequent alignment
    initGlobalMemory48();
    launchKernel48();
    copyKernelResults48();
    resetGlobalMemory48();

    // N49: build a list of all D values that were not flagged for potential nongapped alignment
    launchKernel49();
}

/// [private] method alignN50
void tuAlignN::alignN50()
{
    /* CUDA global memory here:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                DBgw.Dc Dc list (paired candidates for windowed gapped alignment)

        high:   DBn.Dx  unpaired candidates for nongapped alignment
    */

    if( m_pqb->DBn.Dx.n )
    {
        CRVALIDATOR;

        /* Iterate to ensure that the memory needed for the interleaved R sequence data (Ri) does not exceed what is available.

           If two or more iterations are required, we use an additional buffer to aggregate the results.
        */
        UINT32 nTotal = m_pqb->DBn.Dx.n;
        UINT32 nRemaining = nTotal;
        UINT32 nPerIteration = countDxForIteration();
        CudaGlobalPtr<UINT64>* pDxAgg = NULL;
        if( nRemaining > nPerIteration )
        {
            pDxAgg = new CudaGlobalPtr<UINT64>( m_pqb->pgi->pCGA );
            pDxAgg->Alloc( cgaHigh, nTotal, false );
        }

        m_pqb->DBn.Dx.n = nPerIteration;
        while( nRemaining )
        {
            /* N50: load interleaved R sequence data */
            RaiiPtr<tuBaseS> k50 = baseLoadRix::GetInstance( "tuAlignN50", m_pqb, &m_pqb->DBn, riDx );
            k50->Start();
            k50->Wait();

            /* N52: do nongapped alignment (Dx list) */
            baseAlignN k52( "tuAlignN52", m_pqb, &m_pqb->DBn, riDx, true );
            k52.Start();
            k52.Wait();

            if( pDxAgg )
            {
                // save Dx values for the current iteration
                CREXEC( m_pqb->DBn.Dx.CopyInDevice( pDxAgg->p+pDxAgg->n, m_pqb->DBn.Dx.n ) );
                pDxAgg->n += m_pqb->DBn.Dx.n;
            }

            // iterate
            nRemaining -= m_pqb->DBn.Dx.n;
            m_pqb->DBn.Dx.n = min2(nRemaining, nPerIteration);
            if( m_pqb->DBn.Dx.n )
            {
                // copy the next iteration's Dx values to the start of the Dx buffer
                CREXEC( cudaMemcpy( m_pqb->DBn.Dx.p,
                                    m_pqb->DBn.Dx.p+(nTotal-nRemaining),
                                    m_pqb->DBn.Dx.n*sizeof(UINT64),
                                    cudaMemcpyDeviceToDevice ) );
            }
        }

        // if multiple iterations were necessary, flip the updated Dx values back into the buffer
        if( pDxAgg )
        {
            CREXEC( pDxAgg->CopyInDevice( m_pqb->DBn.Dx.p, pDxAgg->Count ) );
            pDxAgg->Free();
        }

        // restore the original number of values in the Dx buffer
        m_pqb->DBn.Dx.n = nTotal;

        // accumulate nongapped mapping counts in the Qwarps
        baseCountA bCA( "tuAlignN52", m_pqb, &m_pqb->DBn, riDx );
        bCA.Start();
        bCA.Wait();
    }
}

/// [private] method alignN60
void tuAlignN::alignN60()
{
    // N66: merge mapped unpaired D values from the Dx list into the Dc list */
    m_hrt.Restart();

    launchKernel66();
    copyKernelResults66();
    resetGlobalMemory66();

    // performance metrics
    AriocTaskUnitMetrics* ptum = AriocBase::GetTaskUnitMetrics( "tuAlignN66" );
    InterlockedExchangeAdd( &ptum->ms.Elapsed, m_hrt.GetElapsed(false) );

#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dcxxx( m_pqb->DBgw.Dc.n, false );
    m_pqb->DBgw.Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );

    for( UINT32 n=0; n<m_pqb->DBgw.Dc.n; ++n )
    {
        UINT64 Dc = Dcxxx.p[n];
        INT16 subId = (Dc >> 32) & 0x007f;
        UINT32 qid = static_cast<UINT32>(Dc >> 39) & AriocDS::QID::maskQID;
        if( (qid == 0xed4) || (subId > 25) )
        {
            CDPrint( cdpCD0, "tuAlignN::before alignN65: DBgw.Dc: n=%u qid=0x%08x Dc=0x%016llx subId=%d ...", n, qid, Dc, subId );
        }
    }
#endif


#if TRACE_SQID
    WinGlobalPtr<UINT64> Dcxxx( m_pqb->DBgw.Dc.n, false );
    m_pqb->DBgw.Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );

    for( UINT32 n=0; n<m_pqb->DBgw.Dc.n; ++n )
    {
        UINT64 Dc = Dcxxx.p[n];
        UINT32 qid = static_cast<UINT32>(Dc >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = (Dc >> 32) & 0x007f;
            INT32 pos = static_cast<INT32>(Dc & 0x7FFFFFFF);
            if( Dc & AriocDS::D::maskStrand )
                pos = (m_pab->M.p[subId] - 1) - pos;

            CDPrint( cdpCD0, "%s::after alignN61: DBgw.Dc: n=%u sqId=0x%016llx qid=0x%08x Dc=0x%016llx subId=%d Jf=%d...",
                                __FUNCTION__, n, sqId, qid, Dc, subId, pos );
        }
    }
//    CDPrint( cdpCD0, "%s: after alignN65", __FUNCTION__ );

#endif


}


#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Uses a CUDA kernel to do nongapped alignments for paired-end reads
/// </summary>
void tuAlignN::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    try
    {
        // if the user has configured the nongapped aligner ...
        if( !m_pab->a21ss.IsNull() )
        {
            alignN10();
            alignN20();
            alignN30();
            alignN40();
            alignN50();
            alignN60();
        }
    }
    catch( thrust::system_error& ex )
    {
        int cudaErrno = ex.code().value();
        throw new ApplicationException( __FILE__, __LINE__,
                                        "CUDA error %u (0x%08x): %s\r\nCUDA Thrust says: %s",
                                        cudaErrno, cudaErrno, ex.code().message().c_str(), ex.what() );
    }

    CDPrint( cdpCD3, "[%d] %s completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
#pragma endregion
