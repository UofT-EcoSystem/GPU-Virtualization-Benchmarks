/*
  baseLoadRw.cu

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
static __device__ __constant__ __align__(8) INT64                       ccOfsRplus[AriocDS::SqId::MaxSubId+1];
static __device__ __constant__ __align__(8) INT64                       ccOfsRminus[AriocDS::SqId::MaxSubId+1];
static __device__ __constant__ __align__(4) UINT32                      ccM[AriocDS::SqId::MaxSubId+1];
static __device__ __constant__ __align__(8) AlignmentControlParameters  ccACP;
static __device__ __constant__ __align__(8) AlignmentScoreParameters    ccASP;

/// [device] function computeWindowRange
// TODO: CHOP: static __device__ bool computeWindowRange( INT32& minJw, INT32& maxJw, const INT32 Mrw, const INT32 M, const INT32 J, const UINT32 qid, const INT16 N, bool strand )
static __device__ bool computeWindowRange( INT32& dw, const INT32 Mrw, const INT32 M, const INT32 J, const UINT32 qid, const INT16 N, bool strand )
{
    /* Compute the "window" on the reference sequence.
    
       We already know the size of the window, which is determined by the range of acceptable fragment lengths (TLENs) plus
        sufficient additional bases to allow for a maximally-gapped mapping on the "inside" of the window.  (See the assignment
        to QBatch::Mrw in QBatch::Initialize().)

       The location of the window depends on the expected orientation of the paired ends and on which end is already mapped.

       The fragment length is computed as the "outer distance":

                ····------>········<------····          convergent
                    ^                    ^
                JminT                JmaxT

                ····<------········------>····          divergent
                    ^                    ^
                JminT                JmaxT

                ····------>········------>····          same
                    ^                    ^
                JminT                JmaxT

       We do the computation in three steps:
        - determine the maximum and minimum positions of the range on the same strand as the mapped mate
        - for convergent and divergent orientations, map the range to the opposite strand

       The computed minJw and maxJw range may extend beyond one of the ends of the reference sequence, so the reference
        sequence must be padded with null symbols (see AriocBase::loadR()).

       The computed range is inclusive, that is, maxJw is the last position in the range
    */

    INT32 JminT, JmaxT;
    switch( ccACP.arf & arfMaskOrientation )
    {
        case arfOrientationConvergent:
            JmaxT = J + ccACP.maxFragLen;   // map the far end of the range to the same strand as the mapped mate
            dw = (M-1) - JmaxT;             // map to the opposite strand
            strand = !strand;
            break;

        case arfOrientationDivergent:
            JminT = (J + N) - ccACP.maxFragLen; // map the far end of the range to the same strand as the mapped mate
            dw = (M-1) - JminT;              // map to the opposite strand
            dw -= Mrw;
            strand = !strand;
            break;

        default:    // arfOrientationSame
            if( qid & 1 )
            {
                dw = J + N - ccACP.maxFragLen;
            }
            else
            {
                dw = J + ccACP.maxFragLen;
                dw -= Mrw;
            }
            break;
    }

    return strand;
}

/// baseLoadRw_Kernel
static __global__ void baseLoadRw_Kernel( const UINT64*  const __restrict__ pR,         // in: pointer to the R sequence data
                                          const Qwarp*   const __restrict__ pQwBuffer,  // in: pointer to Qwarps
                                                UINT64*  const              pDcBuffer,  // in, out: pointer to D values for already-aligned mates
                                          const UINT32                      nDc,        // in: number of D values
                                          const INT32                       Mrw,        // in: number of R symbols per scoring matrix
                                                UINT64*  const __restrict__ pRi         // out: pointer to interleaved R sequence data
                                        )
{
    // compute the 0-based index of the CUDA thread
    UINT32 tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // abort this thread if there is no work to do
    if( tid >= nDc )
        return;


    // TODO: CHOP WHEN DEBUGGED
    //if( tid < (nDc-1) )
    //    return;





    /* Get the D value for the mapped mate:
            UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
            UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
            UINT64  subId :  7;     // bits 32..38: subId
            UINT64  qid   : 21;     // bits 39..59: QID
            UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
            UINT64  flags :  3;     // bits 61..63: flags
                                    //  bit 61: 0
                                    //  bit 62: set if the D value is a candidate for alignment
                                    //  bit 63: set if the D value is successfully mapped
    */
    const UINT64 Dc = pDcBuffer[tid];

    // unpack the QID
    UINT32 qid = static_cast<UINT32>(Dc >> 39) & AriocDS::QID::maskQID;

    // get the length of the mapped read
    UINT32 iw = QID_IW(qid);
    const Qwarp* pQw = pQwBuffer + iw;
    INT16 N = pQw->Nmax;



#if TODO_CHOP_WHEN_DEBUGGED
    if( pQw->sqId[QID_IQ(qid)] == 0x00000204000C907E )
        asm( "brkpt;" );
#endif



    // unpack the D value (0-based position of the mapped read), strand, and subunit ID
    UINT32 d = static_cast<UINT32>(Dc) & 0x7FFFFFFF;
    INT32 subId = static_cast<INT32>((Dc >> 32) & 0x7F);
    bool strand = static_cast<bool>(Dc & AriocDS::D::maskStrand);

    // get the R sequence length (see also AriocBase::loadR())
    const INT32 M = static_cast<INT32>(ccM[subId]);

    // compute the window range
    INT32 dw;
    strand = computeWindowRange( dw, Mrw, M, d, qid, N, strand );

    // point to the R sequence to be copied
    const UINT64* const __restrict__ pR0 = pR + (strand ? ccOfsRminus[subId] : ccOfsRplus[subId]);

    /* Update the specified D value:
        - save the QID, start position, strand, and subId of the "window" for the unmapped mate
        - zero the flags in bits 61..63 (the high-order bits of the QID are always zero)
    */
    pDcBuffer[tid] = (static_cast<UINT64>(qid^1) << 39) |   // QID of the unmapped mate
                     (static_cast<UINT64>(subId) << 32) |   // subId
                     (static_cast<UINT64>(strand) << 31) |  // strand
                     static_cast<UINT64>(dw & 0x7FFFFFFF);  // 0-based position on the reference

    /* Compute the offset of the first 64-bit value to copy; since dw (the 0-based offset into the R sequence) may be
        negative, there are two cases:
        - if dw >= 0, the computation is straightforward
        - if dw < 0, then we need to adjust the value to ensure that the offset is computed correctly
    */
    INT32 ofsFrom = ((dw >= 0) ? dw : (dw-20)) / 21;

    // initialize the pointers
    const UINT64* pFrom = pR0 + ofsFrom;

    UINT32 iRw = tid >> 5;              // offset of the first Ri for the CUDA warp that corresponds to the current thread ID
    UINT32 celMr = blockdiv(Mrw,21);    // number of 64-bit elements needed to represent Mrw

    UINT32 ofsTo = (iRw * celMr * CUDATHREADSPERWARP) + (tid & 0x1F);
    UINT64* pTo = pRi + ofsTo;

    /* Compute the number of bits to shift each 64-bit value so that the first R symbol is in the low-order position:
        - if d >= 0, the computation is straightforward
        - if d < 0, the number of symbols to shift right is ((d+1) % 21) + 20
    */
    INT16 posToShift = (dw >= 0) ? (dw%21) : ((dw+1)%21)+20;
    const INT16 shr = 3 * posToShift;
    const INT16 shl = 63 - shr;

    // copy the 64-bit R values from their linear layout into an interleaved layout
    UINT64 v = *(pFrom++);
    UINT64 vn = 0;

    // iterate until at least Mr symbols are copied
    INT32 jRemaining = Mrw;
    while( jRemaining >= 21 )
    {
        vn = *(pFrom++);

        // shift the bits into position
        if( shr )
            v = ((v >> shr) | (vn << shl)) & MASKA21;

        // save one 64-bit value
        *pTo = v;
        pTo += CUDATHREADSPERWARP;

        // set up the next iteration
        v = vn;
        jRemaining -= 21;
    }

    if( jRemaining )
    {
        // shift the remaining bits into position
        v >>= shr;

        // if necessary, read one more 64-bit value to capture the remaining bits
        if( jRemaining > (21-posToShift) )
        {
            vn = *pFrom;
            v |= (vn << shl);
        }

        // mask the tail with zero
        v &= (MASKA21 >> (63 - (jRemaining*3)));

        // save the final 64-bit value
        *pTo = v;
    }
}
#pragma endregion

#pragma region private methods
/// [private] method initConstantMemory
void baseLoadRw::initConstantMemory()
{
    CRVALIDATOR;

    CRVALIDATE = cudaMemcpyToSymbol( ccACP, &m_pab->aas.ACP, sizeof(AlignmentControlParameters) );
    CRVALIDATE = cudaMemcpyToSymbol( ccASP, &m_pab->aas.ASP, sizeof(AlignmentScoreParameters) );

    // load the offset of each R sequence into constant memory on the current GPU
    CRVALIDATE = cudaMemcpyToSymbol( ccOfsRplus, m_pab->ofsRplus.p, m_pab->ofsRplus.cb );
    CRVALIDATE = cudaMemcpyToSymbol( ccOfsRminus, m_pab->ofsRminus.p, m_pab->ofsRminus.cb );
    CRVALIDATE = cudaMemcpyToSymbol( ccM, m_pab->M.p, m_pab->M.cb );
}

/// [private] method initSharedMemory
UINT32 baseLoadRw::initSharedMemory()
{
    CRVALIDATOR;

    // provide a hint to the CUDA driver as to how to apportion L1 and shared memory
    CRVALIDATE = cudaFuncSetCacheConfig( baseLoadRw_Kernel, cudaFuncCachePreferL1 );

    return 0;
}

/// [private] method launchKernel
void baseLoadRw::launchKernel( dim3& d3g, dim3& d3b, UINT32 cbSharedPerBlock )
{
#if TRACE_SQID
    CDPrint( cdpCD0, "%s::baseLoadRw::launchKernel: candidates for windowed gapped alignment:", m_ptum->Key );
    // copy the D values to a host buffer
    WinGlobalPtr<UINT64> Dxxx( m_nD, false );
    cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );

    bool sqIdFound = false;
    for( UINT32 n=0; n<m_nD; ++n )
    {
        UINT64 D = Dxxx.p[n];
        UINT32 qid = AriocDS::D::GetQid( D );
        UINT32 iw = QID_IW(qid);
        INT16 iq = QID_IQ(qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        UINT64 sqId = pQw->sqId[iq];
            
        if( (sqId | AriocDS::SqId::MaskMateId) == (TRACE_SQID | AriocDS::SqId::MaskMateId) )
        {
            INT16 subId = static_cast<INT16>(D >> 32) & 0x007F;
            INT32 J = static_cast<INT32>(D & 0x7FFFFFFF);
            INT32 Jf = (D & 0x80000000) ? (m_pab->M.p[subId]-1) - J : J;

#if TRACE_SUBID
            if( subId == TRACE_SUBID )
#endif
            CDPrint( cdpCD0, "%s::baseLoadRw::launchKernel: %3d: D=0x%016llx sqId=0x%016llx qid=0x%08x subId=%d J=%d Jf=%d", m_ptum->Key, n, D, sqId, qid, subId, J, Jf );

            sqIdFound = true;
        }
    }

    if( sqIdFound )
        CDPrint( cdpCD0, "%s::baseLoadRw::launchKernel", m_ptum->Key );
#endif


#if TODO_CHOP_WHEN_DEBUGGED
    // copy the D values to a host buffer
    WinGlobalPtr<UINT64> Dxxx( m_nD, false );
    cudaMemcpy( Dxxx.p, m_pD, m_nD*sizeof(UINT64), cudaMemcpyDeviceToHost );
    bool isBad = false;
    for( UINT32 n=0; n<m_nD; ++n )
    {
        UINT64 Dc = Dxxx.p[n];
        if( ((Dc >> 32) & 0x7F) >= 0x20 )
        {
            CDPrint( cdpCD0, "%s: n=%u Dc=0x%016llx", __FUNCTION__, n, Dc );
            isBad = true;
        }
    }

    if( isBad )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif


    // performance metrics
    InterlockedExchangeAdd( &m_ptum->us.PreLaunch, m_hrt.GetElapsed(true) );
    
    // launch the CUDA kernel
    baseLoadRw_Kernel<<< d3g, d3b, cbSharedPerBlock >>>( m_pqb->pgi->pR,    // in: R sequence (reference) data
                                                         m_pqb->DB.Qw.p,    // in: pointer to Qwarps
                                                         m_pD,              // in: pointer to D values for already-mapped mates
                                                         m_nD,              // in: size of the D list
                                                         m_pqb->Mrw,        // in: number of R symbols per read
                                                         m_pqb->DB.Ri.p     // out: pointer to interleaved R sequence data
                                                        );

}
#pragma endregion
