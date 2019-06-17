/*
  tuEncodeFASTA.cpp

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
/// <param name="sqCat">sequence category (+ or - strand)</param>
/// <param name="iInputFile">index of input file</param>
/// <param name="psem">Reference to an <c>RaiiSemaphore</c> instance</param>
/// <param name="pscw">Reference to a <c>SAMConfigWriter</c> instance (may be NULL)</param>
tuEncodeFASTA::tuEncodeFASTA( AriocEncoderParams* psip, SqCategory sqCat, INT16 iInputFile, RaiiSemaphore* psem, SAMConfigWriter* pscw ) : baseEncode(psip,sqCat,iInputFile,psem,pscw)
{
}

/// [public] destructor
tuEncodeFASTA::~tuEncodeFASTA()
{
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Encodes FASTA-formatted sequence data.
/// </summary>
void tuEncodeFASTA::main()
{
    CDPrint( cdpCD4, "%s...", __FUNCTION__ );

    /* Set up to write a .CFG file for the input sequence file.  If the caller does not provide a
        SAMConfigWriter instance, we create one here.
    */
    if( m_pscw == NULL )
        m_pscw = new SAMConfigWriter( m_psip, m_psip->OutFilespecStubSq, m_baseName );

    // assign a 0-based ordinal to each sequence
    INT64 readId = 0;
    UINT64 nSqIn = 0;
    UINT64 nSqEncoded = 0;

    // read the input file in chunks
    m_inFile.Seek( 0, SEEK_SET );
    INT64 cbRead = readInputFile( m_inBuf.p, m_inBuf.cb );

    char* pLimit = m_inBuf.p + cbRead;
    char* pEOL = m_inBuf.p;
    char* p = findSOL( pEOL, &pLimit );

    while( p && (p < pLimit) )
    {
        // find the end of the next line of data and append a null terminator
        char* pEOL = findEOL( &p, &pLimit );

        /* compute the hash threshold for encoding the current sequence; that is, we only encode the current sequence if
            the 32-bit hash of its readId is below a threshold determined by the user-specified sampling ratio */
        bool encodeSq = (AriocE::Hash6432(readId) < m_psip->EncodeThreshold);

        // copy the sequence metadata into its output buffer
        p++ ;                                           // point past the '>' symbol

        if( encodeSq )
            (this->*m_writeSqm)( readId, p, static_cast<INT32>(pEOL-p) );    // copy the data

        // copy and encode the sequence data
        p = findSOL( pEOL+1, &pLimit );
        while( p && (*p != '>') )
        {
            // find the end of the next line of data and append a null terminator
            pEOL = findEOL( &p, &pLimit );

            // copy the raw and encoded sequence data into their output buffers
            if( encodeSq )
                (this->*m_writeRaw)( readId, p, static_cast<INT32>(pEOL-p) );

            // advance to the next line in the input file
            p = findSOL( pEOL+1, &pLimit );
        }

        if( encodeSq )
            (this->*m_endRowRaw)();

        // increment the counts
        readId++;           // ordinal (0-based)
        nSqIn++;            // number of sequences read
        if( encodeSq )
            nSqEncoded++ ;  // number of sequences encoded
    }

    // flush the output buffers
    flushSqm();
    flushRaw( true );
    flushA21();
    flushKmers();

    // update the total number of sequences imported
    InterlockedExchangeAdd( &m_psip->nSqIn, nSqIn );
    InterlockedExchangeAdd( &m_psip->nSqEncoded, nSqEncoded );

    // write the .cfg file (query sequences only)
    if( m_psip->Nenc == NencodingQ )
        writeConfigFile();

    // signal that this thread has completed its work
    m_psemComplete->Release( 1 );

    CDPrint( cdpCD4, "%s completed", __FUNCTION__ );
}
#pragma endregion
