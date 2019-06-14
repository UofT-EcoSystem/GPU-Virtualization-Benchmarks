/*
  baseMapGc.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/set_operations.h>
#include <thrust/merge.h>

#pragma region CUDA device code and data
/* CUDA constant memory
*/
static __device__ __constant__ __align__(4) UINT32  ccM[AriocDS::SqId::MaxSubId+1];

/// [kernel] baseMapGc_Kernel1
static __global__ void baseMapGc_Kernel1(       UINT64* const pDbuffer,     // in,out: pointer to D values for candidates
                                          const UINT32        nD,           // in: number of D values for candidates
                                          const UINT32        wcgsc         // in: worst-case gap space count
                                        )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( (tid == 0) || (tid >= nD) )
        return;

    /* Each D value represents the location of one seed location on the reference:
    
        bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
        bits 31..31: strand (0: forward; 1: reverse complement)
        bits 32..38: subId
        bits 39..59: QID
        bits 60..60: seed from reverse complement
        bits 61..63: flags (flagX: 0; flagCandidate: 1; flagMapped: 0)
    
       The goal of this kernel is to identify sets of two or more D values whose locations represent nearby
        locations on the reference sequence.  For our purposes, "nearby" means "close enough to be considered
        together in a single gapped-alignment computation".

       We start simply by computing the difference between adjacent D values:
        - Because of the bitfield ordering in the D values, two adjacent D values that differ in QID, subId, or strand
           always have a difference that is greater than the worst-case gap space count.
        - If the difference is no greater than the worst-case gap space width, the D values can be covered in a single
           dynamic programming (gapped alignment) computation, so the second one can be removed from the list.  In this
           case, we reset the "candidate" flag.
        - Otherwise, we assume that the neighboring D values are too far apart to be covered in a single DP computation.
           In this case, we leave the "candidate" flag set.
            
          We will merge these D values with the D values from subsequent seed iterations.
    */

    // load a pair of adjacent D values
    UINT64 D0 = pDbuffer[tid-1];
    UINT64 D1 = pDbuffer[tid];

    // if the adjacent D values are within the worst-case gap space count of each other...
    INT64 iD0 = static_cast<INT64>(D0 & ~AriocDS::D::maskFlags);
    INT64 iD1 = static_cast<INT64>(D1 & ~AriocDS::D::maskFlags);
    if( (iD1-iD0) <= static_cast<UINT64>(wcgsc) )
    {
        // reset the "candidate" flag on both D values
        pDbuffer[tid-1] = D0 & ~AriocDS::D::flagCandidate;
        pDbuffer[tid]   = D1 & ~AriocDS::D::flagCandidate;
    }
}

/// [kernel] baseMapGc_Kernel2
static __global__ void baseMapGc_Kernel2(       UINT64* const pDbuffer,     // in,out: pointer to D values for candidates
                                          const UINT32        nD,           // in: number of D values for candidates
                                          const UINT32        wcgsc         // in: worst-case gap space count
                                        )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if( tid >= nD )
        return;

    // get the D value for the current CUDA thread
    UINT64 D0 = pDbuffer[tid];

    // do nothing if the D value is a candidate for removal from the list
    if( D0 & AriocDS::D::flagCandidate )
        return;

    // do nothing if the preceding D value is not a candidate for removal from the list
    if( tid && ((pDbuffer[tid-1] & AriocDS::D::flagCandidate) == 0) )
        return;

    /* At this point the current thread's D value is the first in a group of of one or more adjacent D values
        that can be covered by a single dynamic programming (gapped alignment) computation.
        
       Example:
            566 0x40000110003fc77d ...
        --> 567 0x0000011000c17341 qid=0x00000002 subId=16 pos=12677953 Jf=12677953
            568 0x0000011000c17341 qid=0x00000002 subId=16 pos=12677953 Jf=12677953
            569 0x4000011000c17342 qid=0x00000002 subId=16 pos=12677954 Jf=12677954
            570 0x4000011001224bab ...

       We want to remove the D values at 568 and 569 from the list.  (The isolated D values at 566 and 570
        will participate in subsequent seed iterations; the D value at 567, because it is close to two
        other D values, will be aligned in the current seed iteration.)
    */

    // initialize the loop
    UINT32 ofs = tid + 1;
    UINT64 D1 = (ofs < nD) ? pDbuffer[ofs] : _UI64_MAX;

    /* Loop until the difference in seed positions exceeds the worst-case gap space count:
        - The end of the D list is expected to contain an "infinite" high value with its "candidate" bit set
           so as to delimit any group at the end of the list.
        - The high-order bits of the D values are always zero, so we can safely do 64-bit arithmetic on them.
    */
    while( (D1 & AriocDS::D::flagCandidate) == 0 )
    {
        if( (D1-D0) <= static_cast<UINT64>(wcgsc) )
        {
            // set the "X" flag to indicate that the D value can be removed from the list
            pDbuffer[ofs] = D1 | AriocDS::D::flagX;
        }
        else
        {
            /* D1 is too far from D0 to be covered in a single DP computation:
                - we leave its "candidate" flag unset so that it will not be removed from the list
                - we need to start a new group of D values at D1
            */
            D0 = D1;
        }

        // advance to the next D value in the list (or an all-bits-set null if the end of the list has been reached)
        D1 = (++ofs < nD) ? pDbuffer[ofs] : _UI64_MAX;
    }
}

#pragma endregion

#pragma region private methods
/// [private] method initSharedMemory
UINT32 baseMapGc::initSharedMemory()
{
    CRVALIDATOR;
    
    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseMapGc_Kernel1, cudaFuncCachePreferL1 );
    CRVALIDATE = cudaFuncSetCacheConfig( baseMapGc_Kernel2, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method initConstantMemory5
void baseMapGc::initConstantMemory5()
{
    CRVALIDATOR;

    // copy parameters into CUDA constant memory
    CRVALIDATE = cudaMemcpyToSymbol( ccM, m_pab->M.p, m_pab->M.cb );
}

/// [private] method accumulateJcounts
void baseMapGc::accumulateJcounts()
{
    // use a thrust "reduction" API to build a list of cumulative J-list sizes
    thrust::device_ptr<UINT32> tpnJ( m_pqb->DBj.nJ.p );
    thrust::device_ptr<UINT32> tpcnJ( m_pqb->DBj.cnJ.p );

    /* The range of values to scan includes a zero following the last element in the list, so we can also obtain
        the sum of all the values:
        
        - DBj.cnJ.Count includes the trailing 0 element
        - DBj.celJ = nQ*seedsPerQ, i.e. it excludes the trailing zero
    */
    thrust::exclusive_scan( epCGA, tpnJ, tpnJ+m_pqb->DBj.celJ+1, tpcnJ );

    HiResTimer hrt(us);

    // get a copy of the cumulative J-list counts into a host buffer
    if( m_cnJ.Count < m_pqb->DBj.cnJ.Count )
        m_cnJ.Realloc( m_pqb->DBj.cnJ.Count, false );
    m_pqb->DBj.cnJ.CopyToHost( m_cnJ.p, m_pqb->DBj.cnJ.Count );

    // save the total number of J values in all J lists
    m_pqb->DBj.totalD = m_cnJ.p[m_pqb->DBj.celJ];

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: DBj.totalD = %u", __FUNCTION__, m_pqb->DBj.totalD );
#endif

    // performance metrics
    InterlockedExchangeAdd( &AriocBase::aam.us.XferJ, hrt.GetElapsed(false) );
}

/// [private] method loadJforQ
void baseMapGc::loadJforQ( UINT32 iQ, UINT32 nQ, UINT32 nJ )
{
    // build a list of QID and pos for each J value in the current iteration
    baseSetupJs setupJs( m_ptum->Key, m_pqb, m_pdbg, m_isi, iQ, nQ );
    setupJs.Start();
    setupJs.Wait();

    // load the D values (J values that are NOT adjusted for seed position) for the current iteration
    baseLoadJs loadJs( m_ptum->Key, m_pqb, m_pdbg, m_isi, nJ ); 
    loadJs.Start();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->n.CandidateD, nJ );

    loadJs.Wait();

    // use a thrust "stream compaction" API to remove null values from the list of candidates
    thrust::device_ptr<UINT64> tpD( m_pdbg->Diter.p );
    thrust::device_ptr<UINT64> tpEolD = thrust::remove( epCGA, tpD, tpD+nJ, _UI64_MAX );
    m_pdbg->Diter.n = static_cast<UINT32>(tpEolD.get() - tpD.get());


#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: looking for sqId 0x%016llx in D list...", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    
    UINT32 qidxxx = _UI32_MAX;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            if( pQw->sqId[iq] == TRACE_SQID )
            {
                qidxxx = PACK_QID(iw,iq);
                break;
            }
        }
    }

    if( qidxxx == _UI32_MAX )
        CDPrint( cdpCD0, "[%d] %s::%s: sqId 0x%016llx not in current batch", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    else
    {
        WinGlobalPtr<UINT64> Djyyy( m_pdbg->Diter.n, false );
        m_pdbg->Diter.CopyToHost( Djyyy.p, Djyyy.Count );

        for( UINT32 n=0; n<m_pdbg->Diter.n; ++n )
        {
            UINT64 Dj = Djyyy.p[n];
            UINT32 qid = AriocDS::D::GetQid(Dj);
            if( qid == qidxxx )
            {
                INT8 subId = static_cast<INT8>(Dj >> 32) & 0x007F;
                INT32 J = Dj & 0x7FFFFFFF;
                INT32 Jf = (Dj & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "[%d] %s::%s: %6u qid=0x%08X Dj=0x%016llx subId=%d J=%d Jf=%d",
                                    m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                    n, qid, Dj, subId, J, Jf );

            }
        }
        
        // show Dc values for the opposite mate
        qidxxx ^= 1;
        WinGlobalPtr<UINT64> Dcyyy( m_pdbg->Dc.n, false );
        m_pdbg->Dc.CopyToHost( Dcyyy.p, Dcyyy.Count );

        for( UINT32 n=0; n< m_pdbg->Dc.n; ++n )
        {
            UINT64 Dc = Dcyyy.p[n];
            UINT32 qid = AriocDS::D::GetQid(Dc);
            if( qid == qidxxx )
            {
                INT8 subId = static_cast<INT8>(Dc >> 32) & 0x007F;
                INT32 J = Dc & 0x7FFFFFFF;
                INT32 Jf = (Dc & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
                CDPrint( cdpCD0, "[%d] %s::%s: %6u qid=0x%08X Dc=0x%016llx subId=%d J=%d Jf=%d",
                                    m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                                    n, qid, Dc, subId, J, Jf );
            }
        }
    }
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: loaded %u/%u Dj values", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_pdbg->Diter.n, nJ );
#endif
}

/// [private] method loadJforSeedInterval
void baseMapGc::loadJforSeedInterval()
{
    // accumulate the J-list counts
    accumulateJcounts();

    // performance metrics
    InterlockedIncrement( &m_ptum->n.Instances );

    // abort the current iteration if there are no J values; leaves DBj.cnJ allocated
    if( m_pqb->DBj.totalD == 0 )
    {
        DebugBreak();   // TODO: verify that the CUDA memory allocation strategy is the same as in AriocU
        return;
    }

    /* Iterate across the list of QIDs for Q sequences whose J lists need to be loaded. */
    UINT32 nIterations = 0;
    UINT32 iQ = 0;          // index of first QID (Q sequence) for the current iteration
    UINT32 nQ = 0;          // number of QIDs for the current iteration
    UINT32 nJ = 0;          // number of J values to load for the current iteration
    INT64 nJremaining = m_pqb->DBj.totalD;
    UINT32 nQremaining = m_pdbg->Qu.n;
    while( nQremaining && nJremaining )
    {
        // compute the number of QIDs (Q sequences) and J values to process in the current iteration
        prepareIteration( nQremaining, nJremaining, iQ, nQ, nJ );

#if TODO_CHOP_WHEN_DEBUGGED
        if( nQ == 0 )
            DebugBreak();   // shouldn't happen, right?
#endif

      
        
        // set up buffers in CUDA global memory
        initGlobalMemory_LJSIIteration( nJ );

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: calling loadJforQ (inner-loop iteration %u: nJ=%u)...",
                            m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                            nIterations, nJ );
#endif

        // load D values for the Q sequences
        loadJforQ( iQ, nQ, nJ );

        // consolidate the list of D values and the corresponding list of seed coverage
        resetGlobalMemory_LJSIIteration();

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s::%s: after resetting global memory for iteration", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__ );
#endif

        // iterate
        nQremaining -= nQ;
        nJremaining -= nJ;
        iQ += nQ;
        nIterations++ ;
    }

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->n.Iterations, nIterations );

    /* Sort the D values by position.  Verify that there is enough memory for Thrust to do the sort; empirically,
        Thrust seems to need a bit more than the amount of free space that would be required to make a copy of all
        of the data being sorted. */
    INT64 cbFree = m_pqb->pgi->pCGA->GetAvailableByteCount();
    INT64 cbNeeded = (m_pqb->DBj.D.n * sizeof(UINT64)) + (16*1024);
    if( cbFree < cbNeeded )
        throw new ApplicationException( __FILE__, __LINE__, "%s: insufficient GPU global memory for batch size %u: cbNeeded=%lld cbFree=%lld", __FUNCTION__, m_pab->BatchSize, cbNeeded, cbFree );
        
    thrust::device_ptr<UINT64> tpD( m_pqb->DBj.D.p );
    thrust::stable_sort( epCGA, tpD, tpD+m_pqb->DBj.D.n, TSX::isLessD() );


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s: stable_sort 1 in %dms", m_pqb->pgi->deviceId, __FUNCTION__, hrt.GetElapsed(false) );
    sprintf_s( buf, 256, "[%d] %s: after stable_sort 1", m_pqb->pgi->deviceId, __FUNCTION__ );
    nvtxMark( buf );
#endif


#if TRACE_SQID
    WinGlobalPtr<UINT64> Dxxx( m_pqb->DBj.D.n, false );
    m_pqb->DBj.D.CopyToHost( Dxxx.p, Dxxx.Count );

    for( UINT32 n=0; n<m_pqb->DBj.D.n; ++n )
    {
        UINT64 D = Dxxx.p[n];
        UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 pos = static_cast<INT32>(D & 0x7FFFFFFF);
            if( D & AriocDS::D::maskStrand )
                pos = (m_pab->M.p[subId] - 1) - pos;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "[%d] %s::%s: before kernel1 (m_isi=%u): %4u 0x%016llx qid=0x%08x subId=%d Jf=%d",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_isi,
                        n, D, qid, subId, pos );
        }
    }
    CDPrint( cdpCD0, __FUNCTION__ );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s complete", m_pqb->pgi->deviceId, __FUNCTION__ );
#endif

}

/// [private] method filterJforSeedInterval
void baseMapGc::filterJforSeedInterval( UINT32 cbSharedPerBlock )
{
    /* CUDA global memory here:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs of unmapped mates
                Ru      subId bits for QIDs of unmapped mates (null)
                Dm      QIDs of mapped mates
                DBj.D   consolidated D list
                
        high:   Di      isolated D values
                Du      unmapped D values
    */
    CRVALIDATOR;
    dim3 d3g;
    dim3 d3b;

    if( m_pdbg->Di.n )
    {
        /* Merge previously-isolated D values with the just-loaded D values in hopes of finding multiple-seed coverage
            for individual reads:
            - The merge operation is stable, so we do not need to re-sort the result.
            - The Thrust merge implementation is like UNION ALL in SQL, that is, duplicate values are preserved.
               This is important because duplicate values represent different seeds mapping to the same "diagonal" in
               a dynamic programming (gapped alignment) computation, i.e., duplicate values indicate that a read
               has two or more seeds' worth of coverage.  The minimum coverage is therefore one greater than the
               seed size (i.e., two seeds that overlap in all but one position).
        */

#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: about to merge the previous D list (Di.n=%u) with the current D list (DBj.D.n=%u) (m_isi=%u)",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        m_pdbg->Di.n, m_pqb->DBj.D.n, m_isi );

    WinGlobalPtr<UINT64> Dixxx( m_pdbg->Di.n, false );
    m_pdbg->Di.CopyToHost( Dixxx.p, Dixxx.Count );

    for( UINT32 n=0; n<m_pdbg->Di.n; ++n )
    {
        UINT64 D = Dixxx.p[n];
        UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 pos = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - pos) : pos;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "%s::%s: (m_isi=%u): Di.p[%4u] 0x%016llx qid=0x%08x subId=%d pos=%d Jf=%d",
                        m_ptum->Key, __FUNCTION__,
                        m_isi, n, D, qid, subId, pos, Jf );
        }
    }

    WinGlobalPtr<UINT64> Djxxx( m_pqb->DBj.D.n, false );
    m_pqb->DBj.D.CopyToHost( Djxxx.p, Djxxx.Count );

    for( UINT32 n=0; n<static_cast<UINT32>(Djxxx.Count); ++n )
    {
        UINT64 D = Djxxx.p[n];
        UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 pos = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - pos) : pos;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "%s::%s: (m_isi=%u): DBj.D.p[%4u] 0x%016llx qid=0x%08x subId=%d pos=%d Jf=%d",
                        m_ptum->Key, __FUNCTION__,
                        m_isi, n, D, qid, subId, pos, Jf );
        }
    }
#endif

        // append the Di list to the D list
        UINT32 cel = m_pqb->DBj.D.n + m_pdbg->Di.n;
        m_pqb->DBj.D.Resize( cel );
        m_pdbg->Di.CopyInDevice( m_pqb->DBj.D.p+m_pqb->DBj.D.n, m_pdbg->Di.n );
        m_pqb->DBj.D.n = cel;

        // sort
        thrust::device_ptr<UINT64> tpD(m_pqb->DBj.D.p);
        thrust::stable_sort( epCGA, tpD, tpD+m_pqb->DBj.D.n, TSX::isLessD() );
    }

    // look for reads with two or more nearby D values
    computeKernelGridDimensions( d3g, d3b, m_pqb->DBj.D.n );
    baseMapGc_Kernel1<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DBj.D.p,    // in,out: D values for candidates
                                                         m_pqb->DBj.D.n,    // in: number of candidates for the current seed interval
                                                         m_pdbg->AKP.wcgsc  // in: width of horizontal dynamic-programming band (for gapped aligner)
                                                       );
    // wait for the kernel to complete
    CREXEC( waitForKernel() );

#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: back from Kernel1 (m_isi=%u)", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_isi );

    WinGlobalPtr<UINT64> Dyyy( m_pqb->DBj.D.n, false );
    m_pqb->DBj.D.CopyToHost( Dyyy.p, Dyyy.Count );

    for( UINT32 n=0; n<m_pqb->DBj.D.n; ++n )
    {
        UINT64 D = Dyyy.p[n];
        UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 pos = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - pos) : pos;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "%s::%s: after kernel1 (m_isi=%u): %4u 0x%016llx qid=0x%08x subId=%d pos=%d Jf=%d",
                        m_ptum->Key, __FUNCTION__,
                        m_isi, n, D, qid, subId, pos, Jf );
        }
    }
#endif

    // flag redundant D values
    baseMapGc_Kernel2<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->DBj.D.p,    // in,out: D values for candidates
                                                         m_pqb->DBj.D.n,    // in: number of candidates for the current seed interval
                                                         m_pdbg->AKP.wcgsc  // in: width of horizontal dynamic-programming band (for gapped aligner)
                                                       );

    // wait for the kernel to complete
    CREXEC( waitForKernel() );

#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: back from Kernel2 (m_isi=%u)", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_isi );

    m_pqb->DBj.D.CopyToHost( Dyyy.p, Dyyy.Count );

    for( UINT32 n=0; n<m_pqb->DBj.D.n; ++n )
    {
        UINT64 D = Dyyy.p[n];
        UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 pos = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - pos) : pos;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "%s::%s: after kernel2 (m_isi=%u): %4u 0x%016llx qid=0x%08x subId=%d pos=%d Jf=%d",
                        m_ptum->Key, __FUNCTION__,
                        m_isi, n, D, qid, subId, pos, Jf );
        }
    }
#endif


#if TODO_CHOP_WHEN_DEBUGGED
    UINT32 nBefore = m_pqb->DBj.D.n;
#endif


    // zap the redundant D values
    thrust::device_ptr<UINT64> tpD( m_pqb->DBj.D.p );
    thrust::device_ptr<UINT64> eolD = thrust::remove_if( epCGA, tpD, tpD+m_pqb->DBj.D.n, TSX::testMask<UINT64>(AriocDS::D::flagX) );
    m_pqb->DBj.D.n = static_cast<UINT32>(eolD.get() - tpD.get());
    m_pqb->DBj.D.Resize( m_pqb->DBj.D.n );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s::%s: after removing redundant D values (flagX set): %u/%u in list",
                        m_ptum->Key, __FUNCTION__,
                        m_pqb->DBj.D.n, nBefore );
    CDPrint( cdpCD0, "%s: m_pqb->DBj.D at 0x%016llx", __FUNCTION__, m_pqb->DBj.D.p );
#endif


#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: after zapping redundant D values from Kernel2 (m_isi=%u)", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_isi );

    m_pqb->DBj.D.CopyToHost( Dyyy.p, m_pqb->DBj.D.Count );

    for( UINT32 n=0; n<m_pqb->DBj.D.n; ++n )
    {
        UINT64 D = Dyyy.p[n];
        UINT32 qid = static_cast<UINT32>(D >> 39) & AriocDS::QID::maskQID;
        UINT32 iw = QID_IW(qid);
        UINT32 iq = QID_IQ(qid);
        UINT64 sqId = m_pqb->QwBuffer.p[iw].sqId[iq];

        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 pos = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & AriocDS::D::maskStrand) ? ((m_pab->M.p[subId] - 1) - pos) : pos;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "%s::%s: after kernel2 (m_isi=%u): %4u 0x%016llx qid=0x%08x subId=%d pos=%d Jf=%d",
                        m_ptum->Key, __FUNCTION__,
                        m_isi, n, D, qid, subId, pos, Jf );
        }
    }
#endif

    if( m_isi < m_isiLimit )
    {
        // build a new Di list for the subsequent iteration
        CudaGlobalPtr<UINT64> newDi(m_pqb->pgi->pCGA);
        newDi.Alloc( cgaLow, m_pqb->DBj.D.n, false );
        thrust::device_ptr<UINT64> tpD(m_pqb->DBj.D.p);
        thrust::device_ptr<UINT64> tpNewDi(newDi.p);
        thrust::device_ptr<UINT64> eolNewDi = thrust::copy_if( epCGA, tpD, tpD+m_pqb->DBj.D.n, tpNewDi, TSX::isCandidateDvalue() );
        newDi.n = static_cast<UINT32>(eolNewDi.get()-tpNewDi.get());

        // merge the new Di list with the previous one
        UINT32 cel = newDi.n + m_pdbg->Di.n;
        CudaGlobalPtr<UINT64> mergedDi(m_pqb->pgi->pCGA);
        mergedDi.Alloc( cgaLow, cel, false );
        thrust::device_ptr<UINT64> tpMergedDi(mergedDi.p);
        thrust::device_ptr<UINT64> tpPrevDi(m_pdbg->Di.p);
        thrust::device_ptr<UINT64> eolMergedDi = thrust::merge( epCGA, tpPrevDi, tpPrevDi+m_pdbg->Di.n, tpNewDi, tpNewDi+newDi.n, tpMergedDi, TSX::isLessD() );
        mergedDi.n = static_cast<UINT32>(eolMergedDi.get()-tpMergedDi.get());
        if( mergedDi.n != cel )
            throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: mergedDi.n=%u expected=%u", mergedDi.n, cel );

        // reset the flags in the merged Di list
        thrust::for_each( epCGA, mergedDi.p, mergedDi.p+mergedDi.n, TSX::initializeDflags(0) );

        // unduplicate the merged list
        eolMergedDi = thrust::unique( epCGA, tpMergedDi, tpMergedDi+mergedDi.n );
        mergedDi.n = static_cast<UINT32>(eolMergedDi.get()-tpMergedDi.get());

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "[%d] %s: mergedDi.n=%u (Di.n=%u)", m_pqb->pgi->deviceId, __FUNCTION__, mergedDi.n, m_pdbg->Di.n );
#endif

        // copy the merged Di list into the high memory region
        m_pdbg->Di.Free();
        m_pdbg->Di.Alloc( cgaHigh, mergedDi.n, false );
        mergedDi.CopyInDevice( m_pdbg->Di.p, mergedDi.n );
        m_pdbg->Di.n = mergedDi.n;
        SET_CUDAGLOBALPTR_TAG(m_pdbg->Di,"Di");
        mergedDi.Free();

        // discard the current iteration's list of isolated D values
        newDi.Free();
    }
    else
    {
        // discard the remaining isolated D values
        m_pdbg->Di.Free();
    }

    if( m_isi < m_isiLimit )
    {
        // remove isolated D values from the current D list
        thrust::device_ptr<UINT64> tpEolD = thrust::remove_if( epCGA, tpD, tpD+m_pqb->DBj.D.n, TSX::isCandidateDvalue() );
        m_pqb->DBj.D.n = static_cast<UINT32>(tpEolD.get() - tpD.get() );
        m_pqb->DBj.D.Resize( m_pqb->DBj.D.n );
    }

    /* At this point the D list contains the D values at which we need to perform gapped alignment. */

    // use Dx to refer to the list of D values
    m_pdbg->Dx.Swap( &m_pqb->DBj.D );

    // reset the flags in the D list
    thrust::for_each( epCGA, tpD, tpD+m_pdbg->Dx.n, TSX::initializeDflags(AriocDS::D::flagCandidate) );

    if( m_pdbg->Du.n )
    {
        // avoid redoing unsuccessful alignments by removing values in the D list that are close enough to previously-evaluated D values
        exciseRedundantDvalues( m_pdbg );
    }
}

/// [private] method mergeDu
void baseMapGc::mergeDu()
{
    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs of reads that need gapped alignment
                Ru      subId bits for QIDs of unmapped mates
                Dc      D values for mapped mates
                Dx      D list (flagged if mapped)
                Ri      interleaved R sequence data
                VmaxDx  Vmax values for Dx list

        high:   Di      isolated D values
                Du      previously-processed D values
    */

    /* build a consolidated Du list that includes previously-aligned D values (regardless of whether or not they mapped)
        and D values that were aligned in the current iteration */
    UINT32 cel = m_pdbg->Du.n + m_pdbg->Dx.n + 1;
    CudaGlobalPtr<UINT64> tempDu( m_pqb->pgi->pCGA );
    tempDu.Alloc( cgaLow, cel, false );

    // make a sorted copy of the values in the Dx list
    CudaGlobalPtr<UINT64> tempDx( m_pqb->pgi->pCGA );
    tempDx.Alloc( cgaLow, m_pdbg->Dx.n, false );
    m_pdbg->Dx.CopyInDevice( tempDx.p, m_pdbg->Dx.n );
    thrust::device_ptr<UINT64> ttpDx( tempDx.p );
    thrust::stable_sort( epCGA, ttpDx, ttpDx+m_pdbg->Dx.n, TSX::isLessD() );

    thrust::device_ptr<UINT64> tpDu( m_pdbg->Du.p );
    thrust::device_ptr<UINT64> ttpDu( tempDu.p );

    
#if TODO_CHOP_WHEN_DEBUGGED
        bool isSorted = thrust::is_sorted( epCGA, tpDu, tpDu+m_pdbg->Du.n, TSX::isLessD() );
        if( !isSorted ) DebugBreak();
        isSorted = thrust::is_sorted( epCGA, ttpDx, ttpDx+m_pdbg->Dx.n, TSX::isLessD() );
        if( !isSorted ) DebugBreak();
#endif

    // merge the current set of D values (in the sorted copy of Dx) with the previous list of D values (in Du)
    thrust::device_ptr<UINT64> tpEolDu = thrust::merge( epCGA, ttpDx, ttpDx+m_pdbg->Dx.n, tpDu, tpDu+m_pdbg->Du.n, ttpDu, TSX::isLessD() );
    tempDu.n = static_cast<UINT32>(tpEolDu.get() - ttpDu.get());

    // zap the temporary copy of the Dx list
    tempDx.Free();

    // reset the flags in the Du buffer
    thrust::for_each( epCGA, ttpDu, ttpDu+tempDu.n, TSX::initializeDflags(0) );

#if TODO_CHOP_IF_UNUSED
    // unduplicate
    UINT32 nBefore = tempDu.n;
    tpEolDu = thrust::unique( epCGA, ttpDu, ttpDu+tempDu.n );
    tempDu.n = static_cast<UINT32>(tpEolDu.get() - ttpDu.get());

    // is the above call to thrust::unique really needed?
    if( tempDu.n != nBefore ) DebugBreak();
#endif

    // shuffle the Di list out of the way
#if TODO_CHOP_WHEN_SHUFFLEHILO_WORKS
    CudaGlobalPtr<UINT64> tempDi(m_pqb->pgi->pCGA);
    tempDi.Alloc( cgaLow, m_pdbg->Di.n, false );
    m_pdbg->Di.CopyInDevice( tempDi.p, tempDi.Count );
    tempDi.n = m_pdbg->Di.n;
    m_pdbg->Di.Free();
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dixxxx( m_pdbg->Di.Count, false );
    m_pdbg->Di.CopyToHost( Dixxxx.p, Dixxxx.Count );
#endif

    m_pdbg->Di.ShuffleHiLo();

#if TODO_CHOP_WHEN_DEBUGGED
    m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
    WinGlobalPtr<UINT64> Diyyyy( m_pdbg->Di.Count, false);
    m_pdbg->Di.CopyToHost(Diyyyy.p,Diyyyy.Count);
    if( memcmp( Dixxxx.p, Diyyyy.p, Dixxxx.cb ) )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif

    // discard the old Du list
    m_pdbg->Du.Free();

    // move the Du list to the high memory region
    m_pdbg->Du.Alloc( cgaHigh, tempDu.n, false );
    tempDu.CopyInDevice( m_pdbg->Du.p, m_pdbg->Du.Count );
    m_pdbg->Du.n = tempDu.n;
    SET_CUDAGLOBALPTR_TAG(m_pdbg->Du,"Du");
    tempDu.Free();

    // move the Di list back into the high memory region
#if TODO_CHOP_WHEN_SHUFFLELOHI_WORKS
    m_pdbg->Di.Alloc( cgaHigh, tempDi.Count, false );
    tempDi.CopyInDevice( m_pdbg->Di.p, tempDi.Count );
    m_pdbg->Di.n = tempDi.n;
    SET_CUDAGLOBALPTR_TAG(m_pdbg->Di,"Di");
    tempDi.Free();
#endif

    m_pdbg->Di.ShuffleLoHi();

#if TODO_CHOP_WHEN_DEBUGGED
    m_pqb->pgi->pCGA->DumpUsage( __FUNCTION__ );
    m_pdbg->Di.CopyToHost(Diyyyy.p,Diyyyy.Count);
    if( memcmp( Dixxxx.p, Diyyyy.p, Dixxxx.cb ) )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif
    
#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: m_isi=%d Du.n=%u", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_isi, m_pdbg->Du.n );
#endif
}

/// [private] method mapJforSeedInterval
void baseMapGc::mapJforSeedInterval()
{
    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs of unmapped mates
                Ru      subId bits for QIDs of unmapped mates (null)
                Dc      D values of mapped mates
                Dx      D list (prioritized by coverage)

        high:   Di      isolated D values
                Du      unmapped D values
    */


#if TODO_CHOP_WHEN_DEBUGGED
    // copy the D values to a host buffer
    WinGlobalPtr<UINT64> Dx3( m_pdbg->Dx.n, false );
    cudaMemcpy( Dx3.p, m_pdbg->Dx.p, m_pdbg->Dx.n*sizeof(UINT64), cudaMemcpyDeviceToHost );
    bool isBadDx3 = false;
    for( UINT32 n=0; n<m_pdbg->Dx.n; ++n )
    {
        UINT64 Dx = Dx3.p[n];
        INT16 subId = static_cast<INT16>((Dx >> 32) & 0x7F);
        if( subId >= 0x20 )
        {
            CDPrint( cdpCD0, "%s: n=%u Dx=0x%016llx subId=%d", __FUNCTION__, n, Dx, subId );
            isBadDx3 = true;
        }

        UINT32 pos = static_cast<UINT32>(Dx & 0x7FFFFFFF);
        if( m_pab->M.p[subId] < pos )
        {
            CDPrint( cdpCD0, "%s: n=%u Dx=0x%016llx subId=%d pos=%u M=%u", __FUNCTION__, n, Dx, subId, pos, m_pab->M.p[subId] );
            isBadDx3 = true;
        }
    }

    if( isBadDx3 )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif



    // load interleaved R sequence data for the Dx list
    RaiiPtr<tuBaseS> kLoadRi1 = baseLoadRix::GetInstance( m_ptum->Key, m_pqb, m_pdbg, riDx );
    kLoadRi1->Start();
    kLoadRi1->Wait();

    // find maximum alignment scores for seed-and-extend gapped alignment
    baseMaxV kMaxV( m_ptum->Key, m_pqb, m_pdbg, riDx );
    kMaxV.Start();
    kMaxV.Wait();


#if TODO_CHOP_WHEN_DEBUGGED
    thrust::device_ptr<UINT64> tpDx0( m_pdbg->Dx.p );
    bool isSorted = thrust::is_sorted( epCGA, tpDx0, tpDx0+m_pdbg->Dx.n, TSX::isLessD() );
    if( !isSorted ) DebugBreak();
#endif

#if TRACE_SQID
    CDPrint( cdpCD0, "[%d] %s::%s: looking for sqId 0x%016llx in Dx list...",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    
    UINT32 qidxxx = _UI32_MAX;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            if( pQw->sqId[iq] == TRACE_SQID )
            {
                qidxxx = PACK_QID(iw,iq);
                break;
            }
        }
    }

    if( qidxxx == _UI32_MAX )
        CDPrint( cdpCD0, "[%d] %s::%s: sqId 0x%016llx not in current batch", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, TRACE_SQID );
    else
    {
        // dump the D values for the QID
        WinGlobalPtr<UINT64> Dxxx( m_pdbg->Dx.n, false );
        m_pdbg->Dx.CopyToHost( Dxxx.p, Dxxx.Count );
        WinGlobalPtr<INT16> VmaxDxxx( m_pdbg->VmaxDx.n, false );
        m_pdbg->VmaxDx.CopyToHost( VmaxDxxx.p, VmaxDxxx.Count );
        for( UINT32 n=0; n<m_pdbg->Dx.n; ++n )
        {
            UINT64 D = Dxxx.p[n];
            INT16 Vmax = VmaxDxxx.p[n];
            UINT32 qid = AriocDS::D::GetQid(D);
            if( (qid ^ qidxxx) > 1 )
                continue;

            Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
            UINT64 sqId = pQw->sqId[QID_IQ(qid)];
            INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
            INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & 0x80000000) ? (m_pab->M.p[subId] - 1) - J : J;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "[%d] %s::%s: DBj.D: %4u 0x%016llx qid=0x%08x subId=%d J=%d Jf=%d Vmax=%d sqId=0x%016llx",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        n, D, qid, subId, J, Jf, Vmax, sqId );
        }
    }
#endif
    
    // move the Vmax list (just created by baseMaxV) to the low memory region
    m_pdbg->VmaxDx.ShuffleHiLo();

    if( m_isi < m_isiLimit )
    {
        // build a consolidated Du list that merges previously-processed D values with the D values that were just processed in the current iteration
        mergeDu();
    }



#if TODO_CHOP_WHEN_DEBUGGED
    thrust::device_ptr<UINT64> tpDx1( m_pdbg->Dx.p );
    isSorted = thrust::is_sorted( epCGA, tpDx1, tpDx1+m_pdbg->Dx.n, TSX::isLessD() );
    if( !isSorted ) DebugBreak();
#endif


    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs of unmapped mates
                Ru      subId bits for QIDs of unmapped mates (null)
                Dc      D values of mapped mates
                Dx      D list (prioritized by coverage)
                VmaxDx  Vmax values for Dx list

        high:   Di      isolated D values
                Du      previously-processed D values
    */

    // remove unmapped D values from the Dx list
    thrust::device_ptr<UINT64> tpDx( m_pdbg->Dx.p );
    thrust::device_ptr<UINT64> tpEolDx = thrust::remove_if( epCGA, tpDx, tpDx+m_pdbg->Dx.n, TSX::isUnmappedDvalue() );
    m_pdbg->Dx.n = static_cast<UINT32>(tpEolDx.get() - tpDx.get());

    // do the same for the corresponding Vmax list
    thrust::device_ptr<INT16> tpVmaxDx( m_pdbg->VmaxDx.p );
    thrust::device_ptr<INT16> tpEolVmaxDx = thrust::remove_if( epCGA, tpVmaxDx, tpVmaxDx+m_pdbg->VmaxDx.n, TSX::isZero<INT16>() );
    m_pdbg->VmaxDx.n = static_cast<UINT32>(tpEolVmaxDx.get() - tpVmaxDx.get());
    if( m_pdbg->Dx.n != m_pdbg->VmaxDx.n )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: Dx.n=%u VmaxDx=%u", m_pdbg->Dx.n, m_pdbg->VmaxDx.n );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: mapped Dx.n=%u", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_pdbg->Dx.n );
#endif

    // discard interleaved R sequence data for the Dx list
    m_pqb->DB.Ri.Free();

    if( m_pdbg->Dx.n )
    {
        // load interleaved R sequence data for the successfully-mapped reads
        RaiiPtr<tuBaseS> kLoadRi2 = baseLoadRix::GetInstance( m_ptum->Key, m_pqb, m_pdbg, riDx );
        kLoadRi2->Start();
        kLoadRi2->Wait();

#if TODO_CHOP_WHEN_DEBUGGED
    thrust::device_ptr<UINT64> tpDx4( epCGA, m_pdbg->Dx.p );
    isSorted = thrust::is_sorted( epCGA, tpDx4, tpDx4+m_pdbg->Dx.n, TSX::isLessD() );
    if( !isSorted ) DebugBreak();
#endif

        // do gapped alignment and traceback
        baseAlignG alignG( m_ptum->Key, m_pqb, m_pdbg, m_phb, riDx );
        alignG.Start();
        alignG.Wait();


#if TODO_CHOP_WHEN_DEBUGGED
    thrust::device_ptr<UINT64> tpDx5( m_pdbg->Dx.p );
    isSorted = thrust::is_sorted( epCGA, tpDx5, tpDx5+m_pdbg->Dx.n, TSX::isLessD() );
    if( !isSorted ) DebugBreak();
#endif



        // count per-Q mappings (update the Qwarp buffer in GPU global memory)
        baseCountA bCA( m_ptum->Key, m_pqb, m_pdbg, riDx );
        bCA.Start();
        bCA.Wait();
    }

#if TODO_CHOP_WHEN_DEBUGGED
    else
        CDPrint( cdpCD0, "%s: Dx.n == 0", __FUNCTION__ );
#endif


    // discard the Vmax buffer

#if TODO_CHOP_WHEN_DEBUGGED
    if( m_pdbg->VmaxDx.p == NULL ) DebugBreak();        // TODO: CHOP WHEN DEBUGGED
#endif


    m_pdbg->VmaxDx.Free();

    /* Count concordant mappings.
    
       We need a consolidated Dm list that contains
        - mapped D values from the Dx list
        - mapped D values from the Dc list (if the list is non-empty)
       We do this simply by appending the newly-mapped D values to the Dc list.
    */

    // allocate space for the Dm buffer
    m_pdbg->Dm.Alloc( cgaLow, m_pdbg->Dx.n+m_pdbg->Dc.n, false );

    // copy the mapped D values...
    m_pdbg->Dx.CopyInDevice( m_pdbg->Dm.p, m_pdbg->Dx.n );              // ...from the Dm buffer
    m_pdbg->Dc.CopyInDevice( m_pdbg->Dm.p+m_pdbg->Dx.n, m_pdbg->Dc.n ); // ...from the Dc buffer
    m_pdbg->Dm.n = m_pdbg->Dx.n + m_pdbg->Dc.n;

    

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: Dx.n=%u Dc.n=%u Dm.n=%u", __FUNCTION__, m_pdbg->Dx.n, m_pdbg->Dc.n, m_pdbg->Dm.n );
    
    // copy the D values to a host buffer
    WinGlobalPtr<UINT64> Dxxxx( m_pdbg->Dx.n, false );
    cudaMemcpy( Dxxxx.p, m_pdbg->Dx.p, m_pdbg->Dx.n*sizeof(UINT64), cudaMemcpyDeviceToHost );
    bool isBadDx = false;
    for( UINT32 n=0; n<m_pdbg->Dx.n; ++n )
    {
        UINT64 Dx = Dxxxx.p[n];
        INT16 subId = static_cast<INT16>((Dx >> 32) & 0x7F);
        if( subId >= 0x20 )
        {
            CDPrint( cdpCD0, "%s: n=%u Dx=0x%016llx subId=%d", __FUNCTION__, n, Dx, subId );
            isBadDx = true;
        }

        if( (n == 54630) || (n == 60529) )
            CDPrint( cdpCD0, "%s: n=%u Dx=0x%016llx", __FUNCTION__, n, Dx );

        UINT32 pos = static_cast<UINT32>(Dx & 0x7FFFFFFF);
        if( m_pab->M.p[subId] < pos )
        {
            CDPrint( cdpCD0, "%s: n=%u Dx=0x%016llx subId=%d pos=%u M=%u", __FUNCTION__, n, Dx, subId, pos, m_pab->M.p[subId] );
            isBadDx = true;
        }
    }

    if( isBadDx )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif


#if TODO_CHOP_WHEN_DEBUGGED
    // dump the D values for a specified QID
    UINT32 qidxxx = 0x000001;
    WinGlobalPtr<UINT64> Dmxxx( m_pdbg->Dm.n, false );
    m_pdbg->Dm.CopyToHost( Dmxxx.p, Dmxxx.Count );
    for( UINT32 n=0; n<m_pdbg->Dm.n; ++n )
    {
        UINT64 D = Dmxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid(D);
        if( (qid ^ qidxxx) > 1 )
            continue;
        
        INT16 subId = static_cast<INT16>((D & AriocDS::D::maskSubId) >> 32);
        INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
        INT32 Jf = (D & AriocDS::D::maskStrand) ? (m_pab->M.p[subId] - 1) - J : J;

        CDPrint( cdpCD0, "[%d] %s::%s: Dm 1: %4u 0x%016llx qid=0x%08x subId=%d J=%d Jf=%d",
                            m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                            n, D, qid, subId, J, Jf );
    }
#endif


    /* baseCountC wants a list of D values (not Df values), but the list must be sorted so that adjacent D values represent paired mates,
        i.e., the list needs to be sorted in Df order! */

    // translate to Df format; reset the flag bits
    tuXlatToDf xDf( m_pqb, m_pdbg->Dm.p, m_pdbg->Dm.n, m_pdbg->Dm.p, AriocDS::Df::maskFlags, 0 );
    xDf.Start();
    xDf.Wait();


#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dm2( m_pdbg->Dm.n, false );

    cudaMemcpy( Dm2.p, m_pdbg->Dm.p, m_pdbg->Dm.n*sizeof(UINT64), cudaMemcpyDeviceToHost );
    bool isBadDm2 = false;
    for( UINT32 n=0; n<m_pdbg->Dm.n; ++n )
    {
        UINT64 Dm = Dm2.p[n];
        if( ((Dm >> 33) & 0x7F) >= 0x20 )
        {
            CDPrint( cdpCD0, "%s: n=%u Dm=0x%016llx", __FUNCTION__, n, Dm );
            isBadDm2 = true;
        }

        if( (n == 54630) || (n == 60529) )
            CDPrint( cdpCD0, "%s: n=%u Dm=0x%016llx", __FUNCTION__, n, Dm );
    }

    if( isBadDm2 )
        CDPrint( cdpCD0, __FUNCTION__ );

#endif



    // sort in Df order for the benefit of baseCountC
    thrust::device_ptr<UINT64> tpDm( m_pdbg->Dm.p );
    thrust::stable_sort( epCGA, tpDm, tpDm+m_pdbg->Dm.n, TSX::isLessDf() );

    // unduplicate
    thrust::device_ptr<UINT64> tpEolDm = thrust::unique( epCGA, tpDm, tpDm+m_pdbg->Dm.n );
    m_pdbg->Dm.n = static_cast<UINT32>(tpEolDm.get() - tpDm.get());

    // translate back to D format
    tuXlatToD xD( m_pqb, m_pdbg->Dm.p, m_pdbg->Dm.n, m_pdbg->Dm.p, 0, 0 );
    xD.Start();
    xD.Wait();



#if TODO_CHOP_WHEN_DEBUGGED
    // copy the D values to a host buffer
    WinGlobalPtr<UINT64> Dmxxx( m_pdbg->Dm.n, false );
    cudaMemcpy( Dmxxx.p, m_pdbg->Dm.p, m_pdbg->Dm.n*sizeof(UINT64), cudaMemcpyDeviceToHost );
    bool isBadDm = false;
    for( UINT32 n=0; n<m_pdbg->Dm.n; ++n )
    {
        UINT64 Dm = Dmxxx.p[n];
        if( ((Dm >> 32) & 0x7F) >= 0x20 )
        {
            CDPrint( cdpCD0, "%s: n=%u Dm=0x%016llx", __FUNCTION__, n, Dm );
            isBadDm = true;
        }
    }

    if( isBadDm )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif
    
    
    // count concordant mappings in the Dm buffer
    baseCountC bCC( m_ptum->Key, m_pqb, m_pdbg, riDm, false );
    bCC.Start();
    bCC.Wait();

    // sort the Dm buffer
    thrust::stable_sort( epCGA, tpDm, tpDm+m_pdbg->Dm.n, TSX::isLessD() );

    // make a temporary copy of the consolidated Dm buffer
    CudaGlobalPtr<UINT64> tempDm( m_pqb->pgi->pCGA );
    tempDm.Alloc( cgaHigh, m_pdbg->Dm.n, false );
    m_pdbg->Dm.CopyInDevice( tempDm.p, tempDm.Count );
    tempDm.n = m_pdbg->Dm.n;

    // discard buffers
    m_pdbg->Dm.Free();
    m_pqb->DB.Ri.Free();
    m_pdbg->Dx.Free();

    // the consolidated Dm buffer (which includes any newly-mapped D values) becomes the Dc buffer for the subsequent iteration
    m_pdbg->Dc.Resize( tempDm.n );
    tempDm.CopyInDevice( m_pdbg->Dc.p, tempDm.n );
    m_pdbg->Dc.n = tempDm.n;
    tempDm.Free();


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s (at exit): m_pdbg->Dc.n=%u", __FUNCTION__, m_pdbg->Dc.n );

        // copy the D values to a host buffer
    WinGlobalPtr<UINT64> Dxxx( m_pdbg->Dc.n, false );
    cudaMemcpy( Dxxx.p, m_pdbg->Dc.p, m_pdbg->Dc.n*sizeof(UINT64), cudaMemcpyDeviceToHost );
    bool isBadDc = false;
    for( UINT32 n=0; n<m_pdbg->Dc.n; ++n )
    {
        UINT64 Dc = Dxxx.p[n];
        if( ((Dc >> 32) & 0x7F) >= 0x20 )
        {
            CDPrint( cdpCD0, "%s: n=%u Dc=0x%016llx", __FUNCTION__, n, Dc );
            isBadDc = true;
        }
    }

    if( isBadDc )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif
}

/// [private] method pruneQu
void baseMapGc::pruneQu()
{
    /* CUDA global memory layout at this point:

        low:    Qw      Qwarps
                Qi      interleaved Q sequence data
                Qu      QIDs of unmapped mates
                Ru      subId bits for QIDs of unmapped mates (null)
                Dc      D values of mapped mates

        high:   (unallocated)
    */


#if TODO_CHOP_WHEN_DEBUGGED
    UINT32 nD0 = m_pdbg->Dc.n;
    UINT32 nQ0 = m_pdbg->Qu.n;
    UINT32 nR0 = m_pdbg->Ru.n;
#endif

    if( m_pdbg->Dc.n )
    {
        // set the "candidate" flag for the Dc values
        thrust::device_ptr<UINT64> tpDc( m_pdbg->Dc.p );
        thrust::for_each( epCGA, tpDc, tpDc+m_pdbg->Dc.n, TSX::initializeDflags(AriocDS::D::flagCandidate) );

        // reset the "candidate" flag for D values in concordantly-mapped pairs
        baseFilterD filterD( m_ptum->Key, m_pqb, m_pdbg, riDc, m_pab->aas.ACP.AtN );
        filterD.Start();
        filterD.Wait();

        // remove concordantly-mapped D values from the Dc list
        thrust::device_ptr<UINT64> tptpEolDc = thrust::remove_if( epCGA, tpDc, tpDc+m_pdbg->Dc.n, TSX::isNotCandidateDvalue() );
        m_pdbg->Dc.n = static_cast<UINT32>(tptpEolDc.get() - tpDc.get());


#if TRACE_SQID
        // is the specified SqId in the Dc list
        WinGlobalPtr<UINT64> Dcxxx( m_pdbg->Dc.n, false );
        m_pdbg->Dc.CopyToHost( Dcxxx.p, Dcxxx.Count );
        bool inList = false;
        for( UINT32 n=0; n<m_pdbg->Dc.n; ++n )
        {
            UINT64 D = Dcxxx.p[n];
            UINT32 qid = AriocDS::D::GetQid(D);
            UINT32 iw = QID_IW(qid);
            UINT32 iq = QID_IQ(qid);
            Qwarp* pQw = m_pqb->QwBuffer.p + iw;
            UINT64 sqId = pQw->sqId[iq];

            if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
            {
                CDPrint( cdpCD0, "%s: D=0x%016llx  qid=0x%08x  sqId=0x%016llx in Dc list", __FUNCTION__, D, qid, sqId );
                inList = true;
            }
        }

        CDPrint( cdpCD0, "%s: sqId 0x%016llx is %s in the Dc list", __FUNCTION__, TRACE_SQID, (inList ? "" : "not ") );
#endif
    }
    

#if TODO_CHOP_WHEN_DEBUGGED
    thrust::device_ptr<UINT64> tpDc1( m_pdbg->Dc.p  );
    bool isSorted = thrust::is_sorted( epCGA, tpDc1, tpDc1+m_pdbg->Dc.n, TSX::isLessD() );
    if( !isSorted )
    {
        WinGlobalPtr<UINT64> Dcxx( m_pdbg->Dc.n, false );
        m_pdbg->Dc.CopyToHost( Dcxx.p, Dcxx.Count );
        for( UINT32 n=1; n<m_pdbg->Dc.n; ++n )
        {
            if( Dcxx.p[n] < Dcxx.p[n-1] )
                CDPrint( cdpCD0, "%s: D[%u]=0x%016llx < D[%u]=0x%016llx", n, Dcxx.p[n], n-1, Dcxx.p[n-1] ); 
        }
        DebugBreak();
    }
#endif


    // null the mapped QIDs in the list of unmapped QIDs
    baseFilterQu filterQu( m_ptum->Key, m_pqb, m_pdbg, m_pab->aas.ACP.AtN );
    filterQu.Start();
    filterQu.Wait();

    // remove mapped QIDs and the corresponding subId bits
    thrust::device_ptr<UINT32> tpQu( m_pdbg->Qu.p );
    thrust::device_ptr<UINT32> tpRu( m_pdbg->Ru.p );

    // remove Ru where Qu is null (all bits set)
    thrust::device_ptr<UINT32> tpEol = thrust::remove_if( epCGA, tpRu, tpRu+m_pdbg->Ru.n, tpQu, TSX::isEqualTo<UINT32>(_UI32_MAX) );
    m_pdbg->Ru.n = static_cast<UINT32>(tpEol.get() - tpRu.get());

    // remove Qu where null (all bits set)
    tpEol = thrust::remove( epCGA, tpQu, tpQu+m_pdbg->Qu.n, _UI32_MAX );
    m_pdbg->Qu.n = static_cast<UINT32>(tpEol.get() - tpQu.get());
    if( m_pdbg->Qu.n != m_pdbg->Ru.n )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent list count: Qu.n=%u Ru.n=%u", m_pdbg->Qu.n, m_pdbg->Ru.n );


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: Dc %u/%u, Qu %u/%u, Ru %u/%u",
                        m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__,
                        m_pdbg->Dc.n, nD0, m_pdbg->Qu.n, nQ0, m_pdbg->Ru.n, nR0 );
#endif



}
#pragma endregion
