/*
  tuTailP.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuTailP::tuTailP()
{

}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuTailP::tuTailP( QBatch* pqb ) : m_pqb(pqb), m_pab(pqb->pab)
{
}

/// destructor
tuTailP::~tuTailP()
{
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Writes alignment results to disk files
/// </summary>
void tuTailP::main()
{
#if TODO_CHOP_WHEN_DEBUGGED
    HiResTimer hrt;
#endif

    CDPrint( cdpCD3, "[%d] %s (batch 0x%016llx)...", m_pqb->pgi->deviceId, __FUNCTION__, m_pqb );

    // classify the alignment results
    tuClassifyP classifyP( m_pqb );
    classifyP.Start();
    classifyP.Wait();

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s (batch 0x%016llx): after classifyP: %dms", m_pqb->pgi->deviceId, __FUNCTION__, m_pqb, hrt.GetElapsed(false) );
#endif

    // choose one of the row-writer implementations to count read-mapping categories (concordant, discordant, etc.)
    OutputFormatType oftCount = m_pab->SAMwriter.n ? oftSAM :
                                m_pab->SBFwriter.n ? oftSBF :
                                m_pab->TSEwriter.n ? oftTSE :
                                oftUnknown;

    // write alignment data to disk in the formats specified in the config file
    tuWriteSAM writeSAM( m_pqb, (oftCount == oftSAM) );
    tuWriteSBF writeSBF( m_pqb, (oftCount == oftSBF) );
    tuWriteTSE writeTSE( m_pqb, (oftCount == oftTSE) );
    tuWriteKMH writeKMH( m_pqb, false );
    writeSAM.Start();
    writeSBF.Start();
    writeTSE.Start();
    writeKMH.Start();

    // wait for all disk writes to complete
    writeSAM.Wait();
    writeSBF.Wait();
    writeTSE.Wait();
    writeKMH.Wait();

    // release the current QBatch instance
    m_pqb->Release();

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s (batch 0x%016llx): after QBatch release: %dms", m_pqb->pgi->deviceId, __FUNCTION__, m_pqb, hrt.GetElapsed( false ) );
#endif

    CDPrint( cdpCD3, "[%d] %s (batch 0x%016llx) completed", m_pqb->pgi->deviceId, __FUNCTION__, m_pqb );
}
