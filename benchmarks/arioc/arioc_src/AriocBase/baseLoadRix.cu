/*
  baseLoadRix.cu

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
static __device__ __constant__ __align__(8) INT64   ccOfsRplus[AriocDS::SqId::MaxSubId+1];
static __device__ __constant__ __align__(8) INT64   ccOfsRminus[AriocDS::SqId::MaxSubId+1];

/* CUDA shared memory
*/
extern __device__ __shared__ UINT64 csRi[];

/// [device] method initializeSharedMemory
static __device__ void initializeSharedMemory( const INT32 _sharedMemStride, const INT32 _celPerRi )
{
    if( threadIdx.x < static_cast<UINT32>(_sharedMemStride) )
    {
        // each thread zeroes one column of shared memory
        UINT64* p = csRi + threadIdx.x;
        for( INT32 i=0; i<_celPerRi; ++i )
        {
#if TODO_CHOP_WHEN_DEBUGGED
            //*p = 0xBEEF000000000000 + (static_cast<UINT64>(blockIdx.x) << 32) + (i << 16) + threadIdx.x;
            *p = 0xAAAAAAAABBBBBBBB;
#endif
            *p = 0;

            p += _sharedMemStride;
        }
    }

    // all threads in the CUDA block must wait at this point for shared memory initialization to complete
    __syncthreads();
}

/// [device] method xformR
static __device__ UINT64 xformR( const UINT64*  const __restrict__ pR,      // in: pointer to the R sequence data
                                 const UINT32                      iR,      // in: 0-based index of the R value for the current thread
                                 const UINT64                      Dc,      // in: candidate D value
                                 const INT32                       Mr,      // in: maximum span of mapped R sequence symbols
                                 const INT32                       wcgsc,   // in: worst-case gap space count
                                 const INT32                       celPerR, // in: number of A21 values per R sequence
                                 const INT32                       celPerRi // in: number of A21 values per Ri sequence
                               )
{
    /* The candidate D value is bitmapped like this:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
            UINT64  flags :  3;     // bits 61..63: flags
    */

    // extract the the 0-based position on the reference strand
    INT32 d = static_cast<INT32>(Dc & AriocDS::D::maskPos); // isolate bits 0-30
    d = (d << 1) >> 1;                                      // sign-extend

    /* Adjust for the worst-case gap space count:
        - this value ensures that the scoring-matrix diagonal crosses the d'th diagonal at the end of the mapping; this
           also assumes that Mr has already been computed so as to take this into account
        - for gapped alignment, the Mr value should account for the worst-case gap space count, e.g.
                - N = 100
                - wcgsc = 20
                - Mr = N + 2*wcgsc + 1 = 141
        - for nongapped alignment, wcgsc should be specified as 0, e.g.:
                - N = 100
                - wcgsc = 0
                - Mr = 100
    */
    d -= wcgsc;

    // extract the subunit ID
    INT32 subId = static_cast<INT32>(Dc >> 32) & 0x7F;

    // point to the R sequence to be copied
    const UINT64* const __restrict__ pR0 = pR + ((Dc & AriocDS::D::maskStrand) ? ccOfsRminus[subId] : ccOfsRplus[subId]);

    /* Compute the offset of the first 64-bit value in R; the computation differs depending on whether d (the 0-based
        offset into the R sequence) is negative:
        - if d >= 0, the computation is straightforward
        - if d < 0, then we need to adjust the value to ensure that the offset is computed correctly
    */
    INT32 ofsFrom = ((d >= 0) ? d : (d-20)) / 21;

    // add the offset within the R sequence for the current thread
    ofsFrom += iR;

    /* Read this thread's R value.
    
       This is a coalesced read, although it's suboptimal because it probably isn't aligned to the hardware's memory-
        transaction granularity (e.g. 128 bytes), and all 128 bytes aren't used unless the read is long enough.
        But it's better than reading each 8-byte value separately in a loop.
    */
    UINT64 v = pR0[ofsFrom];

    /* Get the subsequent R value from the adjacent thread.

       In the thread that handles the "rightmost" R value in each sequence (i.e., where iR == (celPerRi-1)),
        the "next" value is the first value in the next sequence.  The symbols from the "next" value get
        appended but then ANDed out of the final R value.        
        
       This could perhaps be dealt with like this:
       
            const UINT32 iLane = threadIdx.x & (CUDATHREADSPERWARP-1);                  // zero for rightmost R values
            const UINT32 laneMask = __ballot_sync( __activemask(), (iLane+1)%celPerR ); // zero bits for rightmost R values
            vn = __shfl_down_sync( laneMask, v, 1 );
       
       But there's no good reason to do this fine-tuning since the result is the same without doing so.

       Also: because this subroutine executes conditionally, some threads in a warp may be inactive so we need
        to use __activemask() to ensure that the warp shuffle function doesn't hang.  (It actually doesn't
        matter in earlier microarchitectures, but Volta hangs if the mask specified to the warp shuffle function
        includes inactive threads.)
    */
    UINT64 vn = 0;
    vn = __shfl_down_sync( __activemask(), v, 1 );

    /* Compute the number of bits to shift each 64-bit value so that the first R symbol is in the low-order position:
        - if d >= 0, the computation is straightforward
        - if d < 0, the number of symbols to shift right is ((d+1) % 21) + 20
    */
    INT16 posToShift = (d >= 0) ? (d%21) : ((d+1)%21)+20;
    const INT16 shr = 3 * posToShift;
    const INT16 shl = 63 - shr;

    // shift the bits into position
    if( shr )
        v = ((v >> shr) | (vn << shl)) & MASKA21;

    // if this thread is computing the final Ri value...
    if( iR == (celPerRi-1) )
    {
        // mask the tail with zero
        const INT32 jRemaining = Mr % 21;
        if( jRemaining )
            v &= (MASKA21 >> (63 - (jRemaining*3)));
    }

    // return the bit-shifted A21-formatted value for the current thread
    return v;
}

/// baseLoadRix_Kernel
static __global__ void baseLoadRix_Kernel( const UINT64*  const __restrict__ pR,            // in: pointer to the R sequence data
                                           const UINT64*  const __restrict__ pDcBuffer,     // in: list of candidate D values
                                           const UINT32                      nDc,           // in: number of candidates
                                           const INT32                       Mr,            // in: maximum number of R symbols spanned by a successful mapping
                                           const INT32                       wcgsc,         // in: worst-case gap space count
                                           const INT32                       nRperBlock,    // in: R values per block
                                                 UINT64*  const              pRiBuffer      // out: pointer to interleaved R sequence data
                                         )
{ 
    // initialize shared memory
    const INT32 sharedMemStride = (nRperBlock & 0x1F) ? nRperBlock : (nRperBlock+1);
    const INT32 celPerRi = blockdiv(Mr,21);
    initializeSharedMemory( sharedMemStride, celPerRi );

    // compute the 0-based index of the current CUDA thread
    const UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // determine which warp and lane correspond to the current CUDA thread
    const UINT32 iWarpG = tid / CUDATHREADSPERWARP;             // 0-based index (in CUDA grid) of current warp
    const UINT32 iWarpB = threadIdx.x / CUDATHREADSPERWARP;     // 0-based index (in current block) of current warp
    const UINT32 iLane = threadIdx.x % CUDATHREADSPERWARP;      // index (in current warp) of current thread

    // compute the number of R values and threads used in each warp
    const INT32 celPerR = (Mr + 40) / 21;                       // number of R values (threads) per R sequence
    const INT32 nRperWarp = CUDATHREADSPERWARP / celPerR;       // number of R sequences per warp
    const INT32 iRinWarp = iLane / celPerR;                     // index (in current warp) of R sequence

    // transform R to Ri
    if( iLane < (nRperWarp * celPerR) )
    {
        /* Compute the index (in the Dc buffer) of the Dc value for the current CUDA thread:

                [index of first R in the current warp] + [index (within the current warp) of R sequence]
        */
        const INT32 iDc = (iWarpG * nRperWarp) + iRinWarp;

        if( iDc < nDc )
        {
            // load the candidate D value for this thread
            UINT64 Dc = pDcBuffer[iDc];

            // compute the index (in R sequence) of the A21-formatted 8-byte R value for the current CUDA thread
            const UINT32 iR = iLane % celPerR;

            // transform R to Ri
            const UINT64 a21Ri = xformR( pR, iR, Dc, Mr, wcgsc, celPerR, celPerRi );

            /* Save Ri for the current thread.

               The R symbols are left-shifted into the first celPerRi values, so only the corresponding lanes save
                an Ri value here. */
            const UINT32 iLane0forR = iRinWarp * celPerR;   // index of the first lane for the current R sequence
            if( (iLane-iLane0forR) < celPerRi )
            {
                /* Compute the index (within the shared-memory buffer) of the Ri value for the current CUDA thread:

                        [index (in current block) of current warp] * nRperWarp +
                        index (in current warp) of current R value +
                        [index (in current R sequence) of current R value] * sharedMemStride
                */
                const UINT32 iRi = (iWarpB * nRperWarp) + iRinWarp + (iR * sharedMemStride);

                // write the current thread's Ri value to shared memory
#if TODO_CHOP_WHEN_DEBUGGED
                //csRi[iRi] = (static_cast<UINT64>(iDc) << 48) + (static_cast<UINT64>(blockIdx.x) << 32) + (iR << 16) + threadIdx.x;
                //csRi[iRi] = Dc;
#endif
                csRi[iRi] = a21Ri;
            }
        }
    }

    /* Copy Ri values from shared memory to global memory.

       The buffer is tiled, with each tile having celPerRi rows and CUDATHREADSPERWARP columns, i.e.,
        one column per candidate D value and one row per A21-formatted R value for each candidate.
        
       See baseLoadRi::ComputeRiBufsize.
    */

    // flow of control in all warps must join at this point
    __syncthreads();

    // we emit one R sequence per thread
    if( threadIdx.x < nRperBlock )
    {
        /* The shared-memory buffer is laid out as a 2-dimensional array of Ri values, with
            width (stride) = nRperBlock
            height         = celPerRi
            
           The output buffer is laid out as a list of tiles, each of which has
            width (stride) = CUDATHREADSPERWARP
            height         = celPerRi

           Each warp in this kernel writes one tile.
        */

        // point to the current warp's R values in shared memory to the output buffer
        UINT64* pFrom = csRi + threadIdx.x;

        // point to the output buffer for the first Dc in the current block
        UINT64* pTo = pRiBuffer +
                        blockIdx.x * nRperBlock * celPerRi +    // index (in grid) of first Ri in current block
                        iWarpB * CUDATHREADSPERWARP * celPerRi +    // index (in block) of first Ri for the current warp
                        iLane;                                      // index (in warp) of Ri for the current thread
        
        for( INT32 i=0; i<celPerRi; ++i )
        {
#if TODO_CHOP_WHEN_DEBUGGED
//            *pTo = (static_cast<UINT64>(tid + (i<<16)) << 32) | (blockIdx.x << 16) | threadIdx.x;
#endif
            *pTo = *pFrom;              // coalesced write and no shared-memory bank collisions

            pFrom += sharedMemStride;
            pTo += CUDATHREADSPERWARP;
        }
    }
}
#pragma endregion

#pragma region protected methods
/// [protected] method initConstantMemory
void baseLoadRix::initConstantMemory()
{
    CRVALIDATOR;

    // load the offset of each R sequence into constant memory on the current GPU
    CRVALIDATE = cudaMemcpyToSymbol( ccOfsRplus, m_pab->ofsRplus.p, m_pab->ofsRplus.cb );
    CRVALIDATE = cudaMemcpyToSymbol( ccOfsRminus, m_pab->ofsRminus.p, m_pab->ofsRminus.cb );
}

/// [protected] method initSharedMemory
UINT32 baseLoadRix::initSharedMemory( UINT32 _cbSharedMem )
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory (although it probably makes no difference)
    if( _cbSharedMem <= 16*1024 )
        CRVALIDATE = cudaFuncSetCacheConfig( baseLoadRix_Kernel, cudaFuncCachePreferL1 );
    else
        CRVALIDATE = cudaFuncSetCacheConfig( baseLoadRix_Kernel, cudaFuncCachePreferShared );

    // we want 64-bit shared-memory banks (although it's not clear if this makes any difference in performance)
    CRVALIDATE = cudaDeviceSetSharedMemConfig( cudaSharedMemBankSizeEightByte );

    return _cbSharedMem;
}

/// [protected] method launchKernel
void baseLoadRix::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: m_pD=0x%016llx m_nD=%u", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_pD, m_nD );


    WinGlobalPtr<UINT64> Dxxx( m_nD, false );
    cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
    for( UINT32 n=0; n<100; ++n )
        CDPrint( cdpCD0, "%s: %3u 0x%016llx", __FUNCTION__, n, Dxxx.p[n] );


    CDPrint( cdpCD0, "%s: calling baseLoadRix_Kernel with DB.Ri.p=0x%016llx", __FUNCTION__, m_pqb->DB.Ri.p );
#endif


    // performance metrics
    m_usXferR = m_hrt.GetElapsed( true );       // (pre-launch time elapsed)


#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: m_pab->R.p=0x%016llx", __FUNCTION__, m_pab->R.p );
    for( UINT32 n=0; n<100; n+=8 )
    {
        CDPrint( cdpCD0, "%s: 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx", __FUNCTION__,
            m_pab->R.p[n+0], m_pab->R.p[n+1], m_pab->R.p[n+2], m_pab->R.p[n+3], m_pab->R.p[n+4], m_pab->R.p[n+5], m_pab->R.p[n+6], m_pab->R.p[n+7] );
    }
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    WinGlobalPtr<UINT64> Dcxxx( m_nD, false );
    cudaMemcpy( Dcxxx.p, m_pD, Dcxxx.cb, cudaMemcpyDeviceToHost );
    if( (Dcxxx.p[0] & 0x7FFFFFFF) == 0 )
    {
        CDPrint( cdpCD0, "%s: m_pD=0x%016llx m_nD=%u", __FUNCTION__, m_pD, m_nD );
    
    }
#endif


#if TODO_CHOP_WHEN_DEBUGGED
    cudaMemset( m_pqb->DB.Ri.p, 0xFD, m_pqb->DB.Ri.cb );

    CDPrint( cdpCD0, "%s: m_nRperBlock=%d cbSharedPerBlock=%d", __FUNCTION__, m_nRperBlock, cbSharedPerBlock );
#endif

    // launch the CUDA kernel
    baseLoadRix_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->pgi->pR,       // in: pointer to the R sequence data
                                                          m_pD,                 // in: list of candidate D values
                                                          m_nD,                 // in: number of candidates
                                                          m_pdbb->AKP.Mr,       // in: maximum number of R symbols spanned by a successful mapping
                                                          m_pdbb->AKP.wcgsc,    // in: worst-case gap space count
                                                          m_nRperBlock,         // in: shared-memory stride
                                                          m_pqb->DB.Ri.p );     // out: pointer to interleaved R sequence data
}
#pragma endregion
