/*
  AriocU.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// [public] constructor
AriocU::AriocU( const char* _pathR, A21SpacedSeed _a21ss, A21HashedSeed _a21hs, AlignmentControlParameters _acp, AlignmentScoreParameters _asp,
                InputFileGroup* _pifgQ,
                WinGlobalPtr<OutputFileInfo>& _ofi,
                UINT32 _gpuMask, INT32 _maxDOP, UINT32 _batchSize, INT32 _kmerSize, CIGARfmtType _cft, MDfmtType _mft,
                AppMain* _pam ) : AriocBase( _pathR, _a21ss, _a21hs, _acp, _asp, _pifgQ, _ofi, _gpuMask, _maxDOP, _batchSize, _kmerSize, _cft, _mft, _pam )
{
}

/// [public] destructor
AriocU::~AriocU()
{
}
#pragma endregion

#pragma region public methods
/// [public] method Main
void AriocU::Main()
{
    CDPrint( cdpCD4, "%s...", __FUNCTION__ );

    // sanity check
    if( this->a21hs.maxMismatches > 0 )
        throw new ApplicationException( __FILE__, __LINE__, "mismatches in seed-and-extend seeds are not supported (%d mismatch%s specified)", this->a21hs.maxMismatches, (this->a21hs.maxMismatches != 1) ? "es" : "" );

    // sniff the Q-sequence input files
    m_hrt.Restart();
    this->nReadsInFiles = this->pifgQ->Sniff( this->nGPUs, this->BatchSize );
    AriocBase::aam.ms.SniffQ = m_hrt.GetElapsed(false);           // performance metrics

    // load the R (reference sequence) data into a buffer
    loadR();

    // load the H and J lookup tables into pinned (page-locked) memory that can be accessed by all GPUs
    loadHJ();

    if( this->doMainLoopJoin )
    {
        // initialize an inverse semaphore that will be used to join the tuGpu threads in the inner loop
        this->semMainLoop.Reset( this->nGPUs );
    }

    // update progress
    updateProgress( 0 );

    /* Start a TaskUnit instance for each GPU:
        - Each tuGpu instance is associated with a 0-based ordinal that identifies the subset of the input data that is
            processed on an individual GPU.  This is not necessarily the same as the device ID associated with the GPU.
        - There is no RAII here so these tuGpu instances must be explicitly destructed later on.
    */
    WinGlobalPtr<tuGpuU*> ptuGpuU( this->nGPUs, true );
    for( INT16 n=0; n<this->nGPUs; ++n )
    {
        ptuGpuU.p[n] = new tuGpuU( n, this );
        ptuGpuU.p[n]->Start();
    }

    // wait for all work to complete
    for( INT16 n=0; n<this->nGPUs; ++n )
        ptuGpuU.p[n]->Wait( INFINITE );

    // finalize alignment output
    flushARowWriters( this->SAMwriter );
    flushARowWriters( this->SBFwriter );
    flushARowWriters( this->TSEwriter );

    // release shared GPU resources (specifically, pinned (page-locked) host memory)
    releaseGpuResources();

    // destruct the tuGpu instances
    for( INT16 n=0; n<this->nGPUs; ++n )
        delete ptuGpuU.p[n];

    // performance metrics
    AriocBase::aam.ms.App = m_hrt.GetElapsed( false );

    CDPrint( cdpCD4, "%s completed", __FUNCTION__ );
}
#pragma endregion
