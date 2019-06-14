/*
  baseLoadJs.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region CUDA device code and data
/* CUDA constant memory
*/
static __device__ __constant__ UINT16               ccSPSI[A21HashedSeed::celSPSI/2];   // see comments in baseCountJs.cu
static __device__ __constant__ __align__(4) UINT32  ccM[AriocDS::SqId::MaxSubId+1];

/// [device] function loadJ
static inline __device__ INT32 loadJ( UINT64&                          s,           // out: mask for strand bit
                                      UINT32&                          subId,       // out: R-sequence subunit ID
                                      INT32&                           spos,        // out: 0-based position of the seed within the Q sequence
                                      UINT32&                          qid,         // out: QID
                                      const UINT32* const __restrict__ pQuBuffer,   // in: QIDs
                                      const UINT32* const __restrict__ pRuBuffer,   // in: subId bits (one per Q sequence)
                                      const UINT32* const __restrict__ pJ,          // in: J lookup table
                                      const UINT64* const __restrict__ poJBuffer,   // in: J-list offsets (one per seed)
                                      const UINT32                     sps,         // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                      const UINT32                     nSeedPos,    // in: number of seed positions per Q sequence
                                      const UINT64                     Dq           // in: Dq value for the current thread
                                    )
{
    // get the QID for the current CUDA thread
    UINT32 iqid = static_cast<UINT32>(Dq >> 42) & AriocDS::QID::maskQID;    // index of QID
    qid = pQuBuffer[iqid];                                                  // QID

    // get the corresponding subId bits; if no subId was supplied by the caller, set all subId bits
    const UINT32 rbits = (pRuBuffer ? pRuBuffer[iqid] : _UI32_MAX);

    // position of the seed relative to the start of the Q sequence
    UINT32 ispos = static_cast<UINT32>(Dq >> 27) & 0x7FFF;  // index of the seed position in the list for the current seed iteration
    spos = ccSPSI[ispos];                                   // 0-based position of the seed relative to the start of the Q sequence

    // find the J list for the current thread
    UINT32 iJlist = ((iqid * nSeedPos) + ispos) * sps;
    if( Dq & AriocDS::Dq::maskRC )  // if the J list is associated with the reverse complement...
        iJlist++;                   // ... the list is in the adjacent odd-numbered element

    // get the offset of the start of the J list
    UINT64 oj = poJBuffer[iJlist];

    // add the offset of this thread's J value within its J list
    oj += (Dq & 0x07FFFFFF);

    UINT64 ofs32 = (oj * 5) >> 2;           // compute the offset of the UINT32 in which the specified Jvalue5 begins
    UINT64 u64 = static_cast<UINT64>(pJ[ofs32]) | (static_cast<UINT64>(pJ[ofs32+1]) << 32);
    INT32 shr = (oj & 3) << 3;              // number of bits to shift
    UINT64 j5 = u64 >> shr;                 // shift the Jvalue5 into the low-order bytes

    // extract the subId (bits 32-38) from the Jvalue
    subId = (j5 >> (Jvalue5::bfSize_J+Jvalue5::bfSize_s)) & Jvalue5::bfMaxVal_subId;

    /* Use the subId bits to filter the result:
        - bits in the rbits bitmap correspond to desired subId values between 0 and 31
        - we don't filter the J list if the bitmap is null (all bits set)
    */
    if( rbits != _UI32_MAX )
    {
        // if this thread's subId's bit is not set, return a null (all bits set) J value
        if( 0 == (rbits & (1 << subId)) )
            return -1;
    }
        
    // extract and return the strand bit
    s = j5 & (static_cast<UINT64>(1) << Jvalue5::bfSize_J);

    // return the 0-based position of the seed relative to the start of the reference strand (R sequence)
    return (j5 & Jvalue5::bfMaxVal_J);





#if TODO_CHOP_ASAP
    // get the offset of the start of the J list
    INT64 oj = poJBuffer[iJlist];

    // get the offset of this thread's J value within its J list
    UINT64 ij = (Dq & 0x07FFFFFF);

    // load the high-order bytes of the J value
    oj += (ij >> 2) * 5;                    // offset of the packed high-order bytes
    UINT32 hi4 = pJ[oj];

    // load the low-order 32 bits of the J value
    UINT32 mod4 = (ij & 3);
    oj += mod4 + 1;                         // offset of the low-order 32-bit word
    UINT32 lo = pJ[oj];

    // extract the subId (bits 32-38) from the Jvalue
    subId = __bfe32( hi4, (mod4 << 3), 7 );  // shift the correct high-order byte into bits 0-6

    /* Use the subId bits to filter the result:
        - bits in the rbits bitmap correspond to desired subId values between 0 and 31
        - we don't filter the J list if the bitmap is null (all bits set)
    */
    if( rbits != _UI32_MAX )
    {
        // if this thread's subId's bit is not set, return a null (all bits set) J value
        if( 0 == (rbits & (1 << subId)) )
            return -1;
    }

    // extract and return the strand bit
    s = JVALUE_MASK_S & lo;

    // return the 0-based position of the seed relative to the start of the reference strand (R sequence)
    return JVALUE_POS(lo);
#endif
}

/// [kernel] baseLoadJs_KernelD
static __global__ void baseLoadJs_KernelD( const UINT32* const __restrict__ pJ,         // in: J lookup table
                                           const UINT64* const __restrict__ poJBuffer,  // in: J-list offsets (one per seed)
                                           const UINT32                     nJ,         // in: total number of J values
                                           const UINT32* const __restrict__ pQuBuffer,  // in: QIDs
                                           const UINT32* const __restrict__ pRuBuffer,  // in: subId bits (one per Q sequence)
                                           const UINT32                     sps,        // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                           const UINT32                     nSeedPos,   // in: number of seed positions for the current seed iteration
                                                 UINT64* const              pDBuffer    // in,out: pointer to D-list buffer
                                         )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // each CUDA thread loads one J value and computes one corresponding D value
    if( tid >= nJ )
        return;

    // get the Dq info
    UINT64 D = pDBuffer[tid];

    // load the components of one J-list value (strand, subunit ID, reference position)
    UINT64 s;
    UINT32 subId;
    INT32 spos;
    UINT32 qid;
    INT32 pos = loadJ( s, subId, spos, qid, pQuBuffer, pRuBuffer, pJ, poJBuffer, sps, nSeedPos, D );

    if( pos > 0 )
    {
        /* Clamp the reference-sequence position so that it remains non-negative.  In the rare cases where this happens
            (i.e., at the start of the reference sequence), the seed will probably be covered anyway by the width of
            the diagonal band in the dynamic programming scoring matrix. */
        pos = (spos < pos) ? (pos - spos) : 0;

        /* Pack the position information into a 64-bit value:

                bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
                bits 31..31: strand (0: forward; 1: reverse complement)
                bits 32..38: subId
                bits 39..59: QID
                bits 60..60: seed from reverse complement
                bits 61..63: flags (bit 61: 0; bit 62:1; bit 63: 0)
                */
        D = pos |                                    // 64-bit value with 0-based position in bits 0-30 (see loadJ())
            s |                                      // strand
            (static_cast<UINT64>(subId) << 32) |     // subId
            (static_cast<UINT64>(qid) << 39) |       // QID
            ((D & AriocDS::Dq::maskRC) ? AriocDS::D::flagRCq : 0) |     // bit 60 (flagRCq)
            AriocDS::D::flagCandidate;                                  // bit 61: 0; bit 62: 1; bit 63: 0
    }
    else                    // (empty J list, or J value not on an interesting R sequence subunit)
        D = _UI64_MAX;      // return null (all bits set)

    // save the result
    pDBuffer[tid] = D;


#if TODO_CHOP_WHEN_THE_ABOVE_WORKS
    
    // TODO: WHAT HAPPENS IF A SEED HASHES TO AN EMPTY BUCKET (NO J VALUES ASSOCIATED WITH THE SEED)?

    UINT64 s;
    UINT32 subId;
    UINT32 lid;
    UINT32 qid;
    UINT64 pos = loadJ( s, subId, lid, qid, jid, pJ, poJBuffer, pcnJBuffer, pQuBuffer, pRuBuffer, nQtotal, seedsPerQ );

    if( pos != _UI64_MAX )
    {
        // compute the index of the seed position within the Q sequence
        INT32 iSeed = lid % seedsPerQ;

        // transform J to D (i.e., offset J by the seed position relative to the start of the Q sequence)
        UINT32 i = ccSPSI[iSeed];

        /* Clamp the D value so that it remains non-negative.  In the rare cases where this happens (i.e., at the start
            of the reference sequence), the seed will probably be covered anyway by the width of the diagonal band in the
            dynamic programming scoring matrix. */
        pos = (i < pos) ? (pos - i) : 0;

        /* Pack the position information into a 64-bit value:

                bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
                bits 31..31: strand (0: forward; 1: reverse complement)
                bits 32..38: subId
                bits 39..59: QID
                bits 60..60: seed from reverse complement
                bits 61..63: flags (bit 61: 0; bit 62:1; bit 63: 0)
        */
        UINT64 D = pos |                                    // 64-bit value with 0-based position in bits 0-30 (see loadJ())
                   s |                                      // strand
                   (static_cast<UINT64>(subId) << 32) |     // subId
                   (static_cast<UINT64>(qid) << 39) |       // QID
                   AriocDS::D::flagCandidate;               // bit 60: 0; bit 61: 0; bit 62: 1; bit 63: 0

        // save the result
        pDbuffer[tid] = D;
    }
#endif
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseLoadJs::initConstantMemory()
{
    CRVALIDATOR;

    // copy parameters into CUDA constant memory
    CRVALIDATE = cudaMemcpyToSymbol( ccM, m_pab->M.p, m_pab->M.cb );

    // copy the seed positions for the current seed iteration
    CRVALIDATE = cudaMemcpyToSymbol( ccSPSI,
                                     m_pab->a21hs.SPSI.p+m_iSeedPos,
                                     m_nSeedPos*sizeof(UINT16),
                                     0,
                                     cudaMemcpyHostToDevice );
}

/// [private] method initSharedMemory
UINT32 baseLoadJs::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
#if TODO_CHOP_IF_UNUSED
    CRVALIDATE = cudaFuncSetCacheConfig( baseLoadJs_KernelDj, cudaFuncCachePreferL1 );
#endif
    CRVALIDATE = cudaFuncSetCacheConfig( baseLoadJs_KernelD, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernelD
void baseLoadJs::launchKernelD( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    CRVALIDATOR;

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );


#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dxxx(m_pdbb->Diter.Count, false );
    m_pdbb->Diter.CopyToHost( Dxxx.p, Dxxx.Count );

    CDPrint( cdpCD0, "%s: m_isi=%u m_nJ=%d Diter.p=0x%016llx Diter.Count=%lld",
                        __FUNCTION__, m_isi, m_nJ, m_pdbb->Diter.p, m_pdbb->Diter.Count );

    for( UINT32 n=0; n<min2(100,m_nJ); n++ )
        CDPrint( cdpCD0, "%s: Diter.p[%4u]=%016llx", __FUNCTION__, n, Dxxx.p[n] );
    CDPrint( cdpCD0, __FUNCTION__ );
#endif




    // execute the kernel
    baseLoadJs_KernelD<<< d3g, d3b, cbSharedPerBlock >>>( m_pab->Jg.p,              // in: J lookup table
                                                          m_pqb->DBj.oJ.p,          // in: per-seed J-list offsets
                                                          m_nJ,                     // in: number of J values for the current iteration
                                                          m_pdbb->Qu.p,             // in: QIDs
                                                          m_pRuBuffer,              // in: subId bits
                                                          m_pab->StrandsPerSeed,    // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                                          m_nSeedPos,               // in: number of seed positions per Q sequence for the current seed iteration
                                                          m_pdbb->Diter.p           // in,out: D list
                                                        );
}
#pragma endregion
