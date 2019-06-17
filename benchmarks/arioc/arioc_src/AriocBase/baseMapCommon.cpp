/*
  baseMapCommon.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
baseMapCommon::baseMapCommon()
{
}

/// [protected] constructor
baseMapCommon::baseMapCommon( QBatch* pqb, DeviceBuffersG* pdbg, HostBuffers* phb ) : m_pqb(pqb),
                                                                                      m_pab(pqb->pab),
                                                                                      m_pdbg(pdbg),
                                                                                      m_phb(phb),
                                                                                      m_isiLimit(5),
                                                                                      m_ptum(NULL)
{
}

/// destructor
baseMapCommon::~baseMapCommon()
{
}
#pragma endregion

#pragma region private methods
/// [private] method nJforQIDrange
UINT32 baseMapCommon::nJforQIDrange( UINT32 ofs0, UINT32 ofs1 )
{
    /* compute the number of J values associated with the specified range of offsets into the
        cumulative list of J-list sizes */
    return static_cast<UINT32>(m_cnJ.p[ofs1] - m_cnJ.p[ofs0]);
}
#pragma endregion

#pragma region protected methods
/// [protected] method prepareIteration
void baseMapCommon::prepareIteration( const UINT32 nQremaining, const INT64 nJremaining, UINT32& iQ, UINT32& nQ, UINT32& nJ )
{
    /* The maximum number of D values in an iteration is limited by the amount of CUDA global memory
        available to contain the buffers.

       We leave a bit of "slop" in order to accommodate incidental allocations inside the thrust APIs.
    */
    static const UINT32 cbPerJ = sizeof(UINT64) +  // D list
                                 sizeof(UINT64) +  // copy of D list (in Thrust sort APIs)
                                 sizeof(UINT64) +  // copy of reduced D list
                                 sizeof(UINT32);   // nDc values

    // compute the maximum number of J values that can be loaded in the current iteration
    INT64 cbAvailable = m_pqb->pgi->pCGA->GetAvailableByteCount() - (1024*1024);    // (1MB of "slop")
    UINT32 nJiter = static_cast<UINT32>(cbAvailable / cbPerJ);

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: nQremaining=%u nJremaining=%lld iQ=%u nQ=%u nJ=%u nJiter=%u", __FUNCTION__, nQremaining, nJremaining, iQ, nQ, nJ, nJiter );
#endif

    // estimate the number of iterations that remain
    UINT32 nRemainingIterations = static_cast<UINT32>(blockdiv(nJremaining,nJiter));

    /* estimate the number of QIDs that can be processed in the current iteration; this assumes that
        the number of J values associated with each Q sequence is uniformly distributed */
    nQ = nQremaining / nRemainingIterations;
    
#ifdef _DEBUG
    if( (iQ+nQ) > m_pdbg->Qu.n ) DebugBreak();
#endif
    
    // compute the number of J lists associated with each Q sequence
    UINT32 nJlistsPerQ = m_pdbg->AKP.seedsPerQ * m_pab->StrandsPerSeed;
    UINT32 iJ = iQ * nJlistsPerQ;   // index of the first J value for the first seed in the specified Q sequence

    // compute the actual number of J values that would be loaded for the set of nQ QIDs starting at iQ
    UINT32 ofs1 = (iQ+nQ) * nJlistsPerQ;
    nJ = nJforQIDrange( iJ, ofs1 );

    // if this estimate is too high, reduce the number of Q sequences for the current iteration
    while( nJ > nJiter )
    {
        ofs1 -= nJlistsPerQ;
        nJ = nJforQIDrange( iJ, ofs1 );
    }

    // if this estimate is too low, increase the number of Q sequences for the current iteration
    while( (nJ < nJiter) && (nQ < nQremaining) )
    {
        ofs1 += nJlistsPerQ;
        nJ = nJforQIDrange( iJ, ofs1 );
    }

    // if we have overestimated, correct the number of QIDs to process
    if( nJ > nJiter )
    {
        ofs1 -= nJlistsPerQ;
        nJ = nJforQIDrange( iJ, ofs1 );
    }

    // use the offsets to compute the number of QIDs for the current iteration
    nQ = (ofs1 - iJ) / nJlistsPerQ;

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s::%s: m_isi=%u nQremaining=%u iQ=%u nQ=%u nJ=%u", m_pqb->pgi->deviceId, m_ptum->Key, __FUNCTION__, m_isi, nQremaining, iQ, nQ, nJ );
#endif

    // sanity check
    if( nQ == 0 )
    {
        // the estimated amount of global memory needed is a WAG that assumes that about half is consumed on each iteration
        throw new ApplicationException( __FILE__, __LINE__, "insufficient GPU global memory for batch size %u: estimated cbNeeded=%lld cbAvailable=%lld",
                                                            m_pab->BatchSize, nJremaining*cbPerJ*2/nRemainingIterations, cbAvailable );
    }
}

#pragma endregion
