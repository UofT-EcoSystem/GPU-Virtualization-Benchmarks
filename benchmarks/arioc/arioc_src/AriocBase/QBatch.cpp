/*
  QBatch.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// constructor
QBatch::QBatch( QBatchPool* _pqbp, INT16 _ibp, AriocBase* _pab, GpuInfo* _pgi ) : m_pqbp(_pqbp),
                                                                                  m_ibp(_ibp),
                                                                                  pab(_pab),
                                                                                  pgi(_pgi),
                                                                                  celBRLEAperQ(0),
                                                                                  Mrw(0),
                                                                                  DB(_pgi->pCGA),
                                                                                  DBj(_pgi->pCGA),
                                                                                  DBn(_pgi->pCGA),
                                                                                  DBgw(_pgi->pCGA),
                                                                                  DBgs(_pgi->pCGA),
                                                                                  DBkmh(_pgi->pCGA),
                                                                                  QwarpLimit(0),
                                                                                  Nmax(0),
                                                                                  pfiQ(NULL)
{
    CDPrint( cdpCDb, "QBatch[%d %d]::ctor", this->pgi->deviceId, this->m_ibp );

    // performance metrics
    InterlockedIncrement( &AriocBase::aam.n.BatchInstances );
}

/// destructor
QBatch::~QBatch()
{
    // release the metafile data buffers
    for( INT32 mate=0; mate<2; ++mate )
    {
        MFBm[mate].buf.Free();
        MFBm[mate].ofs.Free();
        MFBq[mate].buf.Free();
        MFBq[mate].ofs.Free();
    }
}
#pragma endregion


#pragma region public methods
/// <summary>
/// Initializes the state of a QBatch instance
/// </summary>
void QBatch::Initialize( INT16 estimatedNmax, InputFileGroup::FileInfo* pfi )
{
    // compute the number of Qwarps for this Qbatch instance
    this->QwarpLimit = this->pab->BatchSize / CUDATHREADSPERWARP;

    // if necessary, adjust the Qwarp buffer size
    if( static_cast<UINT32>(this->QwBuffer.Count) < this->QwarpLimit )
        this->QwBuffer.Realloc( this->QwarpLimit, false );

    // conditionally adjust the size of the Qi (interleaved Q sequence data) buffer
    UINT32 cel = this->QwarpLimit * CUDATHREADSPERWARP * blockdiv(estimatedNmax,21);       // number of 64-bit elements in the Qi buffer
    if( this->QiBuffer.Count < cel )
        this->QiBuffer.Realloc( cel, false );

    // reset the batch state
    this->Nmax = estimatedNmax;
    this->QwBuffer.n = 0;
    this->HBn.Reset();
    this->HBgwn.Reset();
    this->HBgs.Reset();
    this->HBgc.Reset();
    this->HBgwc.Reset();
    this->DBn.Reset();
    this->DBgw.Reset();
    this->DBgs.Reset();
    this->ComputeMrw();
    this->pfiQ = pfi;

    CDPrint( cdpCD3, "[%d] %s: m_ibp=%d QwarpLimit=%u", this->pgi->deviceId, __FUNCTION__, this->m_ibp, this->QwarpLimit );
}

/// <summary>
/// Computes the "window" size for windowed gapped alignment.
/// </summary>
void QBatch::ComputeMrw()
{
    /* For paired-end alignments, compute the number of R symbols spanned by the window determined by an opposite-mate mapping.
        The window accommodates:
        - the range of user-specified fragment lengths (TLEN)
        - a maximally-gapped mapping on the "inside" of the fragment (i.e. a mapping that contains the worst-case number of
            gap spaces)
       
        - maximum TLEN:  Jr = (J + maxFragLen)

             J
             -------->                         <---------
                                                        Jr
        - minimum TLEN:  Jr = (J + minFragLen)
             J
             -------->   <---------
                                  Jr
    
       The minimum fragment length is clamped so as to exclude dovetailed mappings.
    */
    INT32 clampedMinFragLen = max2(this->pab->aas.ACP.minFragLen, this->Nmax);
    this->Mrw = (this->pab->aas.ACP.maxFragLen - clampedMinFragLen) +
                    this->Nmax +
                    this->pab->aas.ComputeWorstCaseGapSpaceCount( this->Nmax ) +
                    1;
}

/// <summary>
/// Returns a QBatch instance to its containing pool
/// </summary>
void QBatch::Release()
{
    // performance metrics
    if( this->QwBuffer.n )
        InterlockedIncrement( &AriocBase::aam.n.BatchesCompleted );

    m_pqbp->Enpool( this );

    CDPrint( cdpCD3, "QBatch[%d %d]::Release", this->pgi->deviceId, this->m_ibp );
}

/// <summary>
/// Obtains GPU configuration parameters from the Arioc configuration
/// </summary>
/// <remarks>The parameter should be initialized with default values prior to calling this function.</remarks>
void QBatch::GetGpuConfig( const char* paramName, UINT32& cudaThreadsPerBlock )
{
    // look for the specified Xparam
    INT32 i = this->pab->paamb->Xparam.IndexOf( paramName );

    // if the Xparam exists, update the specified parameter
    if( i >= 0 )
        cudaThreadsPerBlock = static_cast<UINT32>(this->pab->paamb->Xparam.Value(i));
}
#pragma endregion
