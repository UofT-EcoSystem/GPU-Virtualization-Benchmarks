/*
  AriocE.Q.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region private methods
/// [private] method encodeQ
void AriocE::encodeQ()
{
    // prepare to encode input files
    RaiiSemaphore semEncoderWorkers( this->Params.nLPs, this->Params.nLPs );

    /* Encode input files:
        - there are three output files (raw.sbf, sqm.sbf, A21.sbf) for each FASTA input file
        - there are four output files (raw.sbf, sqm.sbf, A21.sbf, sqq.sbf) for each FASTQ input file
        - we import each input file concurrently
    */
    HiResTimer hrt( ms );

    UINT32 nFiles = this->Params.ifgRaw.InputFile.n;
    WinGlobalPtr<baseEncode*> ptuBaseEncode( nFiles, true );

    CDPrint( cdpCD0, "%s: encoding %u file%s (%d CPU thread%s available)...",
                        __FUNCTION__,
                        nFiles, PluralS( nFiles ),
                        this->Params.nLPs, PluralS( this->Params.nLPs) );

    for( UINT32 i=0; i<nFiles; ++i )
    {
        // wait for a CPU worker thread to become available
        semEncoderWorkers.Wait( m_encoderWorkerThreadTimeout );

        // launch a worker thread for the ith input file
        switch( this->Params.InputFileFormat )
        {
            case SqFormatFASTA:
                ptuBaseEncode.p[i] = new tuEncodeFASTA( &this->Params, sqCatQ, i, &semEncoderWorkers, NULL );
                break;

            case SqFormatFASTQ:
                ptuBaseEncode.p[i] = new tuEncodeFASTQ( &this->Params, sqCatQ, i, &semEncoderWorkers, NULL );
                break;

            default:
                throw new ApplicationException( __FILE__, __LINE__, "unsupported sequence file format" );
        }

        ptuBaseEncode.p[i]->Start();
    }

    // wait for the worker threads to exit
    for( UINT32 n=0; n<nFiles; ++n )
        ptuBaseEncode.p[n]->Wait( INFINITE );

    // destruct the tuBaseEncode instances
    for( UINT32 n=0; n<nFiles; ++n )
        delete ptuBaseEncode.p[n];

    // performance metrics
    AriocE::PerfMetrics.msEncoding = hrt.GetElapsed( false );

    CDPrint( cdpCD0, "%s: encoded %u file%s", __FUNCTION__, nFiles, PluralS(nFiles) );
}

/// [private] method validateQ
void AriocE::validateQ()
{
    // prepare to validate encoded files
    RaiiSemaphore semEncoderWorkers( this->Params.nLPs, this->Params.nLPs );

    /* Validate encoded (A21.sbf) files */
    HiResTimer hrt( ms );

    UINT32 nFiles = this->Params.ifgRaw.InputFile.n;
    WinGlobalPtr<baseValidateA21*> ptubaseValidateA21( nFiles, true );

    CDPrint( cdpCD0, "%s: validating %u file%s (%d CPU thread%s available)...",
                        __FUNCTION__,
                        nFiles, PluralS(nFiles),
                        this->Params.nLPs, PluralS(this->Params.nLPs) );

    for( UINT32 i=0; i<nFiles; ++i )
    {
        // wait for a CPU worker thread to become available
        semEncoderWorkers.Wait( m_encoderWorkerThreadTimeout );

        // launch a worker thread for the ith input file
        ptubaseValidateA21.p[i] = new tuValidateQ( &this->Params, i, &semEncoderWorkers );
        ptubaseValidateA21.p[i]->Start();
    }

    // wait for the worker threads to exit
    for( UINT32 n=0; n<nFiles; ++n )
        ptubaseValidateA21.p[n]->Wait( INFINITE );

    // destruct the tubaseValidateA21 instances
    for( UINT32 n=0; n<nFiles; ++n )
        delete ptubaseValidateA21.p[n];

    // performance metrics
    AriocE::PerfMetrics.msValidateQ = hrt.GetElapsed( false );

    CDPrint( cdpCD0, "%s: validated %u file%s", __FUNCTION__, nFiles, PluralS(nFiles) );
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Imports and encodes Q sequence data (&quot;reads&quot;)
/// </summary>
void AriocE::ImportQ()
{
    HiResTimer hrtApp(ms);

    // ensure that the output directory exists
    RaiiDirectory::OpenOrCreateDirectory( this->Params.OutDir );

    // encode Q sequence data (reads)
    encodeQ();

    // validate Q sequence data (A21.sbf files)
    validateQ();

    // performance metrics
    switch( this->Params.InputFileFormat )
    {
        case SqFormatFASTA:
            strcpy_s( AriocE::PerfMetrics.InputFileFormat, sizeof AriocE::PerfMetrics.InputFileFormat, "FASTA" );
            break;
                
        case SqFormatFASTQ:
            strcpy_s( AriocE::PerfMetrics.InputFileFormat, sizeof AriocE::PerfMetrics.InputFileFormat, "FASTQ" );
            break;
                
        default:
            strcpy_s( AriocE::PerfMetrics.InputFileFormat, sizeof AriocE::PerfMetrics.InputFileFormat, "(unrecognized input file format)" );
            break;
    }

    AriocE::PerfMetrics.nKmersIgnored = this->Params.nKmersWithN;
    AriocE::PerfMetrics.nSqEncoded = this->Params.nSqEncoded;
    AriocE::PerfMetrics.nSqIn = this->Params.nSqIn;
    AriocE::PerfMetrics.nSymbolsEncoded = this->Params.nSymbolsEncoded;
    AriocE::PerfMetrics.nConcurrentThreads = this->Params.nLPs;
    AriocE::PerfMetrics.nInputFiles = static_cast<INT32>(this->Params.ifgRaw.InputFile.n);
    AriocE::PerfMetrics.msApp = hrtApp.GetElapsed(false);
}
#pragma endregion
