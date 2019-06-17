/*
  tuValidateQ.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Encodes FASTA-formatted sequence data.
/// </summary>
/// <param name="psip">Reference to a common parameter structure</param>
/// <param name="iInputFile">index of input file</param>
/// <param name="psem">Reference to an <c>RaiiSemaphore</c> instance</param>
tuValidateQ::tuValidateQ( AriocEncoderParams* psip, INT16 iInputFile, RaiiSemaphore* psem ) : baseValidateA21( psip, sqCatQ, iInputFile, psem )
{
}

/// [public] destructor
tuValidateQ::~tuValidateQ()
{
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Validates encoded sequence data.
/// </summary>
void tuValidateQ::main()
{
    CDPrint( cdpCD3, "%s...", __FUNCTION__ );

    /* Each "row" in the A21.sbf file is formatted as follows:
        sqId         bigint        not null,    -- 8 bytes
        N            smallint      not null,    -- 2 bytes
        B2164        varbinary(n)  not null     -- 2-byte count (n) followed by n data bytes
    */

    A21FIXEDFIELDS* pdrh = NULL;
    INT64 cbTotal = 0;
    INT64 cbInFile = m_inFile.FileSize();
    INT64 nRows = 0;

    while( cbTotal < cbInFile )
    {
        /* At this point we expect the fixed-size SQL column data:
            - sqId      8 bytes
            - N         2 bytes
            - cb        2 bytes
        */
        pdrh = reinterpret_cast<A21FIXEDFIELDS*>(getNext( m_cbFixed ));

        if( m_srcId != static_cast<INT32>(AriocDS::SqId::GetSrcId( pdrh->sqId )) )
            throw new ApplicationException( __FILE__, __LINE__, "unexpected srcId %d (expected %d) in sqId 0x%08llx at offset %lld in %s", AriocDS::SqId::GetSrcId( pdrh->sqId ), m_srcId, pdrh->sqId, cbTotal, m_inFile.FileSpec.p );

        if( m_subId != static_cast<INT32>(AriocDS::SqId::GetSubId( pdrh->sqId )) )
            throw new ApplicationException( __FILE__, __LINE__, "unexpected subId %d (expected %d) in sqId 0x%08llx at offset %lld in %s", AriocDS::SqId::GetSubId( pdrh->sqId ), m_subId, pdrh->sqId, cbTotal, m_inFile.FileSpec.p );

        if( (pdrh->cb < 0) || (pdrh->cb > 8000) )               // 8000: maximum size of SQL varbinary datatype
            throw new ApplicationException( __FILE__, __LINE__, "invalid binary data length %d (0x%04X) in sqId 0x%08llx at offset %lld in %s", pdrh->cb, pdrh->cb, pdrh->sqId, cbTotal, m_inFile.FileSpec.p );

        // get the expected number of bytes of binary data
        INT16 cbBinary = pdrh->cb;
        getNext( cbBinary );

        // update the counts
        nRows++;                            // number of rows
        cbTotal += (m_cbFixed + cbBinary);  // number of bytes in the file
    }

    if( cbTotal != cbInFile )
        throw new ApplicationException( __FILE__, __LINE__, "unexpected file byte count: read %lld bytes from file size %lld: %s", cbTotal, cbInFile, m_inFile.FileSpec.p );

    /* At this point we have successfully reached the end of the file. */
    CDPrint( cdpCD3, "%s completed: %lld rows validated in %s", __FUNCTION__, nRows, m_inFile.FileSpec.p );

    // signal that this thread has completed its work
    m_psemComplete->Release( 1 );
}
#pragma endregion
