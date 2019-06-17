/*
  baseLoadJn.cu

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
static __device__ __constant__ __align__(4) UINT32  ccM[AriocDS::SqId::MaxSubId+1];

/// [device] function loadJ
static inline __device__ INT32 loadJ( UINT64&                          s,           // out: mask for strand bit
                                      UINT32&                          subId,       // out: R-sequence subunit ID
                                      INT32&                           spos,        // out: 0-based position of the seed within the Q sequence
                                      UINT32&                          qid,         // out: QID
                                      const UINT32* const __restrict__ pJ,          // in: J lookup table
                                      const UINT64* const __restrict__ poJBuffer,   // in: J-list offsets (one per seed)
                                      const UINT32                     npos,        // in: number of seed positions per Q sequence
                                      const UINT32                     sps,         // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                      const UINT64                     Dq           // in: Dq value for the current thread
                                    )
{
    // find the J list for the current thread
    qid = static_cast<UINT32>(Dq >> 42) & AriocDS::QID::maskQID;    // QID
    spos = static_cast<UINT32>(Dq >> 27) & 0x7FFF;                  // position of the seed relative to the start of the Q sequence
    UINT32 iList = ((qid * npos) + spos) * sps;

    // for bsDNA alignments where the seeds come from Qrc, the J list resides in an odd-numbered element in the J-list buffer
    if( Dq & AriocDS::Dq::maskRC )
        iList++;

    // get the offset of the start of the J list
    UINT64 oj = poJBuffer[iList];

    // add the offset of this thread's J value within its J list
    oj += (Dq & 0x07FFFFFF);

    UINT64 ofs32 = (oj * 5) >> 2;           // compute the offset of the UINT32 in which the specified Jvalue5 begins
    UINT64 u64 = static_cast<UINT64>(pJ[ofs32]) | (static_cast<UINT64>(pJ[ofs32+1]) << 32);
    INT32 shr = (oj & 3) << 3;              // number of bits to shift
    UINT64 j5 = u64 >> shr;                 // shift the Jvalue5 into the low-order bytes

    // extract the subId (bits 32-38) from the Jvalue
    subId = (j5 >> (Jvalue5::bfSize_J+Jvalue5::bfSize_s)) & Jvalue5::bfMaxVal_subId;
        
    // extract and return the strand bit
    s = j5 & (static_cast<UINT64>(1) << Jvalue5::bfSize_J);

    return (j5 & Jvalue5::bfMaxVal_J);










#if TODO_CHOP_ASAP
    // get the offset of the start of the J list
    INT64 oj = poJBuffer[iList];

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
    subId = __bfe32( hi4, (mod4 << 3), 7 ); // shift the correct high-order byte into bits 0-6

    // extract and return the strand bit
    s = JVALUE_MASK_S & lo;

    // return the 0-based position of the seed relative to the start of the reference strand (R sequence)
    return JVALUE_POS(lo);
#endif
}

/// [kernel] baseLoadJn_KernelDf
static __global__ void baseLoadJn_KernelDf( const UINT32* const __restrict__ pJ,        // in: J lookup table
                                            const UINT64* const __restrict__ poJBuffer, // in: J-list offsets (one per seed)
                                            const UINT32                     nJtotal,   // in: total number of J values
                                            const UINT32                     npos,      // in: number of seed positions per Q sequence
                                            const UINT32                     sps,       // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                                  UINT64* const              pDfBuffer  // in,out: pointer to D-list buffer
                                          )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // each CUDA thread loads one J value
    if( tid >= nJtotal )
        return;

    /* Each CUDA thread computes one D value as follows:
        - identify the J list for the current thread
        - identify the J value in the J-list
        - convert to D value (pack QID and adjust J)
        - write to the output buffer
    */

    // get the Dq info
    const UINT64 Dq = pDfBuffer[tid];

    // load the components of one J-list value (strand, subunit ID, reference position)
    UINT64 s;
    UINT32 subId;
    INT32 iSeed;
    UINT32 qid;
    INT32 pos = loadJ( s, subId, iSeed, qid, pJ, poJBuffer, npos, sps, Dq );

    // transform J to D (i.e., offset J by the seed position relative to the start of the Q sequence)
    pos -= iSeed;

    // if the D value references the reverse-complement strand...
    if( s )
    {
        // get the total number of symbols in the reference sequence
        const INT32 M = static_cast<INT32>(ccM[subId]);

        /* Map the position to the forward strand.

            A Df value maps a 0-based position on the reverse-complement strand to the opposite position on the
            forward strand:

                    D
            --------+------->
            <-------+--------
                    Df
    
            For what it's worth, the Df position is not the same as the POS value reported in a SAM record for a
            read mapped to the reverse-complement strand, which would be the position of the last mapped base on the
            reverse-complement strand, i.e. (Df-N)+1.  (And the SAM POS value is also 1-based.)
        */
        pos = (M-1) - pos;
    }

    /* Pack the position information into a 64-bit value; the bitfields are ordered so as to permit sorting as
        an unsigned 64-bit integer:

            bits 00..00: QID bit 0 (0: mate 1; 1: mate 2)
            bits 01..01: strand (0: forward; 1: reverse complement)
            bits 02..32: position mapped relative to the start of the forward strand
            bits 33..39: subId
            bits 40..59: QID bits 1-20
            bits 60..60: seed from reverse complement
            bits 61..63: flags (bits 61-63: 0)
    */
    UINT64 Df = (static_cast<UINT64>(qid) & 1) |                        // QID low-order bit
                (static_cast<UINT64>(s >> 30)) |                        // strand
                (static_cast<UINT64>(AriocDS::D::maskPos & pos) << 2) | // position on forward strand
                (static_cast<UINT64>(subId) << 33) |                    // subId
                (static_cast<UINT64>(qid >> 1) << 40);                  // QID high-order bits

    // set a flag if the seed originated in Qrc (the reverse complement of the Q sequence)
    if( Dq & AriocDS::Dq::maskRC )
    {
        /* Bit 60 (flagRCq) is zero because all of the above values are zero in those bits.  The following
            XOR thus accomplishes two things:
            - set bit 60 to 1
            - toggle bit 1 (the strand bit)
        */
        Df ^= (AriocDS::Df::flagRCq | AriocDS::Df::maskStrand);
    }

    // save the result
    pDfBuffer[tid] = Df;
}

/// [kernel] baseLoadJn_KernelD
static __global__ void baseLoadJn_KernelD( const UINT32* const __restrict__ pJ,         // in: J lookup table
                                           const UINT64* const __restrict__ poJBuffer,  // in: J-list offsets (one per seed)
                                           const UINT32                     nJtotal,    // in: total number of J values
                                           const UINT32                     npos,       // in: number of seed positions per Q sequence
                                           const UINT32                     sps,        // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                                 UINT64* const              pDBuffer    // in,out: pointer to D-list buffer
                                          )
{
    // compute the 0-based index of the CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // each CUDA thread loads one J value
    if( tid >= nJtotal )
        return;

    /* Each CUDA thread computes one D value as follows:
        - identify the J list for the current thread
        - identify the J value in the J-list
        - convert to D value (pack QID and adjust J)
        - write to the output buffer
    */

    // get the Dq info
    const UINT64 Dq = pDBuffer[tid];

    // load the components of one J-list value (strand, subunit ID, reference position)
    UINT64 s;
    UINT32 subId;
    INT32 iSeed;
    UINT32 qid;
    INT32 pos = loadJ( s, subId, iSeed, qid, pJ, poJBuffer, npos, sps, Dq );

    // transform J to D (i.e., offset J by the seed position relative to the start of the Q sequence)
    pos -= iSeed;

    /* Pack the position information into a 64-bit value; the bitfields are ordered so as to permit sorting as
        an unsigned 64-bit integer:

            bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            bits 31..31: strand (0: forward; 1: reverse complement)
            bits 32..38: subId
            bits 39..59: QID
            bits 60..60: seed from reverse complement
            bits 61..63: flags (bit 61: 1; bit 62: 0; bit 63: 0) initialized for counting D-value occurrences
    */
    UINT64 D = (static_cast<UINT64>(pos) & AriocDS::D::maskPos) |   // 64-bit value with 0-based position in bits 0-30
               s |                                                  // strand
               (static_cast<UINT64>(subId) << 32) |                 // subId
               (static_cast<UINT64>(qid) << 39) |                   // QID
               AriocDS::D::flagX;                                   // bit 61: 1

    // set a flag if the seed originated in Qrc (the reverse complement of the Q sequence)
    if( Dq & AriocDS::Dq::maskRC )
        D |= AriocDS::D::flagRCq;

    // save the result
    pDBuffer[tid] = D;
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseLoadJn::initConstantMemory()
{
    CRVALIDATOR;

    // copy parameters into CUDA constant memory
    CRVALIDATE = cudaMemcpyToSymbol( ccM, m_pab->M.p, m_pab->M.cb );
}

/// [private] method initSharedMemory
UINT32 baseLoadJn::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseLoadJn_KernelDf, cudaFuncCachePreferL1 );
    CRVALIDATE = cudaFuncSetCacheConfig( baseLoadJn_KernelD, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernelDf
void baseLoadJn::launchKernelDf( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    CRVALIDATOR;

#if TODO_USE_CRITICAL_SECTION
    RaiiCriticalSection<baseLoadJn> rcs;
#endif


    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );

#if TODO_CHOP_WHEN_DEBUGGED
    // debug the Qu list for the SX aligner
    CDPrint( cdpCD0, "baseLoadJn::launchKernelDf: DBj.Qu.n=%u", m_pqb->DBj.Qu.n );

    if( m_ptum == AriocBase::GetTaskUnitMetrics( "tuSetupGs20" ) )
    {
        // dump a few of the QIDs of the pairs whose J lists are to be probed
        WinGlobalPtr<UINT32> Qujxxx( m_pqb->DBj.Qu.n, false );
        m_pqb->DBj.Qu.CopyToHost( Qujxxx.p, Qujxxx.Count );

        for( UINT32 n=0; n<256; ++n )
        {
            UINT32 qid = Qujxxx.p[n];
            Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
            CDPrint( cdpCD0, "baseLoadJn::launchKernelDf: qid=0x%08X sqId=0x%016llx", qid, pQw->sqId[QID_IQ(qid)] );
        }

        // get the minimum and maximum QIDs
        CDPrint( cdpCD0, "baseLoadJn::launchKernelDf: min QID = 0x%08x  max QID = 0x%08X", Qujxxx.p[0], Qujxxx.p[Qujxxx.Count-1] );
    }
#endif

    // execute the kernel
    baseLoadJn_KernelDf<<< d3g, d3b, cbSharedPerBlock >>>( m_pab->Jn.p,             // in: J lookup table
                                                           m_pqb->DBj.oJ.p,         // in: per-seed J-list offsets
                                                           m_pqb->DBj.D.n,          // in: number of Df values
                                                           m_npos,                  // in: number of seed positions per Q sequence
                                                           m_pab->StrandsPerSeed,   // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                                           m_pqb->DBj.D.p           // in,out: Df list
                                                         );
    

#if TODO_USE_CRITICAL_SECTION
    CREXEC( waitForKernel() );   // (do this within the scope of the critical section)
#endif
}

/// [private] method launchKernelD
void baseLoadJn::launchKernelD( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
    CRVALIDATOR;

#if TODO_USE_CRITICAL_SECTION
    RaiiCriticalSection<baseLoadJn> rcs;
#endif



#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT32> cnJxxx( m_pqb->DBj.cnJ.Count, false );
    m_pqb->DBj.cnJ.CopyToHost( cnJxxx.p, cnJxxx.Count );
#endif


    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );

    // execute the kernel
    baseLoadJn_KernelD<<< d3g, d3b, cbSharedPerBlock >>>( m_pab->Jn.p,              // in: J lookup table
                                                          m_pqb->DBj.oJ.p,          // in: per-seed J-list offsets
                                                          m_pqb->DBj.D.n,           // in: number of D values
                                                          m_npos,                   // in: number of seed positions per Q sequence
                                                          m_pab->StrandsPerSeed,    // in: 2: seeding from Q+ and Q-; 1: seeding from Q+ only
                                                          m_pqb->DBj.D.p            // in,out: D list
                                                        );

#if TODO_USE_CRITICAL_SECTION
    CREXEC( waitForKernel() );   // (do this within the scope of the critical section)
#endif
}
#pragma endregion
