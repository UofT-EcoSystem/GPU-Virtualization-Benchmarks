/*
  ARowWriter.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.

  Notes:
   The idea here is that each calling thread (i.e., one per GPU) fills its own output buffer; the
    ARowWriter implementation flushes that buffer to disk in a thread-safe manner.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// constructor
ARowWriter::ARowWriter( AriocBase* pabase, OutputFileInfo* pofi ) : baseARowWriter(pabase)
{
    // copy the specified output file info
    m_ofi = *pofi;

    // allocate one output buffer per GPU
    for( INT16 iBuf=0; iBuf<m_nBufs; ++iBuf )
    {
        m_outBuf[iBuf].Realloc( OUTPUT_BUFFER_SIZE, false );
        m_outBuf[iBuf].n = 0;
    }

    /* Build the output filespec stub:
        - full subdirectory path from the configuration file
        - base filename (e.g. "AriocA")
        - alignment result flags
        - 3-digit file sequence number
        - filename extension

       Example:  e:/data/BulkData/SqA/yanhuang/110114/AriocA.cdru.001.sbf
    */
    strcpy_s( m_filespecStub, sizeof m_filespecStub, m_ofi.path );

    // append the base filename to the file specification stub
    RaiiDirectory::AppendTrailingPathSeparator( m_filespecStub );       // append a path separator
    strcat_s( m_filespecStub, sizeof m_filespecStub, m_ofi.baseName );  // append the base filename

    // append the alignment result flags to the file specification stub
    strcat_s( m_filespecStub, sizeof m_filespecStub, "." );

    // map bits to strings
    if( m_ofi.arf & arfReportMapped )                 // (arfReportMapped == arfReportConcordant)
    {
        if( pabase->pifgQ->HasPairs )
            strcat_s( m_filespecStub, sizeof m_filespecStub, "c" );
        else
            strcat_s( m_filespecStub, sizeof m_filespecStub, "m" );
    }
    if( m_ofi.arf & arfReportUnmapped )   strcat_s( m_filespecStub, sizeof m_filespecStub, "u" );
    if( m_ofi.arf & arfReportDiscordant ) strcat_s( m_filespecStub, sizeof m_filespecStub, "d" );
    if( m_ofi.arf & arfReportRejected )   strcat_s( m_filespecStub, sizeof m_filespecStub, "r" );

    // save the filename extension for the output file specification
    switch( m_ofi.oft )
    {
        case oftSAM:
            strcpy_s( m_filespecExt, sizeof m_filespecExt, "sam" );
            break;

        case oftSBF:
        case oftTSE:
            strcpy_s( m_filespecExt, sizeof m_filespecExt, "sbf" );
            break;

        case oftKMH:
            strcpy_s( m_filespecExt, sizeof m_filespecExt, "kmh" );
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected output format type %d", m_ofi.oft );
    }

    // set a flag to indicate that this ARowWriter instance is active
    this->IsActive = true;
}

/// destructor
ARowWriter::~ARowWriter()
{
    // write anything that remains in the output buffers
    for( INT16 iBuf=0; iBuf<m_nBufs; ++iBuf )
        internalFlush( iBuf );

    // close the file
    internalClose();
}
#pragma endregion

#pragma region private methods

/// [private] method internalClose
void ARowWriter::internalClose()
{
    m_outFile.Close();
    this->TotalA += m_nA;
    m_nA = 0;
}

/// [private] method internalFlush
void ARowWriter::internalFlush( INT16 iBuf )
{
    /* Each calling thread is associated with a QBatch instance and with a corresponding buffer instance in m_outBuf,
        so the only thread synchronization required is right here to serialize writes to the output file for this ARowWriter
        instance.
    */
    m_mtx.Wait( BUFFER_LOCK_TIMEOUT );

    // if there is data to write ...
    if( m_outBuf[iBuf].n )
    {
        // if there is no currently-open output file, open a new output file
        if( m_outFile.Handle < 0 )
        {
            // increment the file sequence number
            ++m_outFileSeqNo;

            // build a complete file specification
            char filespec[FILENAME_MAX];
            sprintf_s( filespec, sizeof filespec, "%s.%03d.%s", m_filespecStub, m_outFileSeqNo, m_filespecExt );

            // open and initialize the file
            m_outFile.Open( filespec );
            if( m_ofi.oft == oftSAM )
                m_pab->SAMhdb.WriteHeader( m_outFile );
        }

        /* Write the data.
        
           The C++ _write method (which is used by RaiiFile.Write) may not be thread-safe in either Windows or Linux,
            so we use a critical section here.
        */
        INT64 cbWritten;
        {
            RaiiCriticalSection<ARowWriter> rcs;
            cbWritten = m_outFile.Write( m_outBuf[iBuf].p, m_outBuf[iBuf].n );
        }

        if( cbWritten != m_outBuf[iBuf].n )
            throw new ApplicationException( __FILE__, __LINE__, "write failed for %s: %lld/%u bytes written", m_outFile.FileSpec.p, cbWritten, m_outBuf[iBuf].n );

        // reset the buffer byte counters
        m_outBuf[iBuf].n = 0;     // bytes used
    }

    // close the current output file if we have reached the user-specified per-file limit for the number of alignments to write
    if( m_nA >= m_ofi.maxA )
        internalClose();

    // relinquish this thread's ownership of the mutex
    m_mtx.Release();
}
#pragma endregion

#pragma region public methods
/// [public] method Close
void ARowWriter::Close()
{
    for( INT16 iBuf=0; iBuf<m_nBufs; ++iBuf )
        internalFlush( iBuf );

    internalClose();
}

/// [public] method Flush
void ARowWriter::Flush( INT16 iBuf )
{
    internalFlush( iBuf );
}

/// [public] method Lock
char* ARowWriter::Lock( INT16 iBuf, UINT32 cb )
{
    // if necessary, flush to liberate free space in the output buffer
    if( (m_outBuf[iBuf].n + cb) > m_outBuf[iBuf].cb )
        internalFlush( iBuf );


#if TODO_CHOP_WHEN_DEBUGGED
    if( cb > m_outBuf[iBuf].cb )
        DebugBreak();
#endif


    return m_outBuf[iBuf].p + m_outBuf[iBuf].n;
}

/// [public] method Release
void ARowWriter::Release( INT16 iBuf, UINT32 cb )
{
    // track the number of bytes actually used in the buffer
    m_outBuf[iBuf].n += cb;

    // count the number of times the buffer was released (i.e., the number of alignments written to the buffer)
    m_nA++ ;
}
#pragma endregion
