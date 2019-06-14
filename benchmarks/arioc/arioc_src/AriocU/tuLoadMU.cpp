/*
  tuLoadMU.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuLoadMU::tuLoadMU()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pQFI">a reference to a <c>QfileInfo</c> instance</param>
/// <param name="pFileM">a reference to an <c>RaiiFile</c> instance</param>
/// <param name="pMFI">a reference to an <c>MfileInfo</c> (metadata file info) instance</param>
/// <param name="pMFB">a reference to an <c>MfileBuf</c> (metadata file buffer) instance</param>
tuLoadMU::tuLoadMU( QBatch* pqb, QfileInfo* pQFI, RaiiFile* pFileM, MfileInfo* pMFI, MfileBuf* pMFB ) : tuLoadM(pqb)
{
    m_pQFI[0] = pQFI;
    m_pFileM[0] = pFileM;
    m_pMFI[0] = pMFI;
    m_pMFB[0] = pMFB;
}

/// destructor
tuLoadMU::~tuLoadMU()
{
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Loads Q-sequence metadata for the current batch
/// </summary>
void tuLoadMU::main()
{
    CDPrint( cdpCD3, "[%d] %s: %s...", m_pqb->pgi->deviceId, __FUNCTION__, m_pMFI[0]->ext );

    // point to the last Qwarp in the current batch
    Qwarp* pQw = m_pqb->QwBuffer.p + m_pqb->QwBuffer.n - 1;

    // get the first and last sqIds in the current batch
    UINT64 sqIdInitial = m_pqb->QwBuffer.p->sqId[0];
    UINT64 sqIdFinal = pQw->sqId[pQw->nQ-1];

    // read the metadata
    readMetadata( m_pFileM[0], m_pQFI[0], m_pMFI[0], m_pMFB[0], sqIdInitial, sqIdFinal );

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.Elapsed, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s: %s completed in %dms", m_pqb->pgi->deviceId, __FUNCTION__, m_pMFI[0]->ext, m_hrt.GetElapsed(false) );
}
#pragma endregion
