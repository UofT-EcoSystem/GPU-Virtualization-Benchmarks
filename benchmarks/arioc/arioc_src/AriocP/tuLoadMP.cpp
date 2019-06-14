/*
  tuLoadMP.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuLoadMP::tuLoadMP()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="QFI">a reference to a <c>QfileInfo</c> instances</param>
/// <param name="fileM">a reference to a pair of <c>RaiiFile</c> instances</param>
/// <param name="MFI">a reference to a pair of <c>MfileInfo</c> (metadata file info) instances</param>
/// <param name="MFB">a reference to a pair of <c>MfileBuf</c> (metadata file buffer) instances</param>
tuLoadMP::tuLoadMP( QBatch* pqb, QfileInfo QFI[2], RaiiFile fileM[2], MfileInfo MFI[2], MfileBuf MFB[2] ) : tuLoadM(pqb)
{
    for( INT32 mate=0; mate<2; ++mate )
    {
        m_pQFI[mate] = QFI+mate;
        m_pFileM[mate] = fileM+mate;
        m_pMFI[mate] = MFI+mate;
        m_pMFB[mate] = MFB+mate;
    }
}

/// destructor
tuLoadMP::~tuLoadMP()
{
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Loads Q-sequence metadata for the current batch
/// </summary>
void tuLoadMP::main()
{
    CDPrint( cdpCD3, "[%d] %s: %s...", m_pqb->pgi->deviceId, __FUNCTION__, m_pMFI[0]->ext );

    // point to the last Qwarp in the current batch
    Qwarp* pQw = m_pqb->QwBuffer.p + m_pqb->QwBuffer.n - 1;

    for( INT32 mate=0; mate<2; ++mate )
    {
        // get the first and last sqIds for the mate in the current batch
        UINT64 sqIdInitial = m_pqb->QwBuffer.p->sqId[mate];
        UINT64 sqIdFinal = pQw->sqId[pQw->nQ-(2-mate)];

        // read the metadata
        readMetadata( m_pFileM[mate], m_pQFI[mate], m_pMFI[mate], m_pMFB[mate], sqIdInitial, sqIdFinal );
    }

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.Elapsed, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s: %s completed in %dms", m_pqb->pgi->deviceId, __FUNCTION__, m_pMFI[0]->ext, m_hrt.GetElapsed(false) );
}
#pragma endregion
