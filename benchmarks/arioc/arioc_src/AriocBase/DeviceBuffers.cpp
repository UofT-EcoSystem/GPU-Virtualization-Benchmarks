/*
  DeviceBuffers.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region DeviceBuffersBase
/// constructor
DeviceBuffersBase::DeviceBuffersBase( CudaGlobalAllocator* pCGA ) : Dc(pCGA), Dx(pCGA), Dm(pCGA), Du(pCGA), Qu(pCGA), Ru(pCGA), Rx(pCGA),
                                                                    Diter(pCGA),
                                                                    nDm1(0), nDm2(0), flagRi(riNone)
{
    this->Reset();
}

/// destructor
DeviceBuffersBase::~DeviceBuffersBase()
{
}

/// [public] method Reset
void DeviceBuffersBase::Reset()
{
    // initialize the AlignmentKernelParameters struct
    memset( &this->AKP, 0, sizeof(AlignmentKernelParameters) );
}
#pragma endregion

#pragma region DeviceBuffersN
/// constructor
DeviceBuffersN::DeviceBuffersN( CudaGlobalAllocator* pCGA ) : DeviceBuffersBase(pCGA), Dl(pCGA)
{
}

/// destructor
DeviceBuffersN::~DeviceBuffersN()
{
}

/// virtual method implementation
void DeviceBuffersN::InitKernelParameters( QBatch* _pqb )
{
    /* Initialize CUDA kernel parameters for the nongapped aligner:
        - the number of R sequence symbols required to map one Q sequence is the maximum number of symbols in a Q sequence

       We assume here that all Q sequences have the same (maximum) length; this assumption is wasteful if the Q-sequence lengths
        vary, but it simplifies the bookkeeping.
    */
    this->AKP.Mr = _pqb->Nmax;              // maximum number of R symbols spanned by a nongapped alignment
}
#pragma endregion

#pragma region DeviceBuffersG
/// constructor
DeviceBuffersG::DeviceBuffersG( CudaGlobalAllocator* pCGA ) : DeviceBuffersBase(pCGA), VmaxDc(pCGA), VmaxDx(pCGA), VmaxDm(pCGA), BRLEA(pCGA), Di(pCGA), Cu(pCGA), nDc1(0)
{
}

/// destructor
DeviceBuffersG::~DeviceBuffersG()
{
}

/// virtual method implementation
void DeviceBuffersG::InitKernelParameters( QBatch* _pqb )
{
    if( this->AKP.Mr )
        return;

    /* Initialize CUDA kernel parameters for the windowed gapped aligner and the seed-and-extend gapped aligner:
        - the number of scoring matrix cells per row (i.e., per Q symbol) assumes a potential worst-case gap on either side of the
            target diagonal, i.e. the width of the computed band is
                bw = 2*wcgsc + 1
        - similarly, the number of R sequence symbols spanned by a scoring matrix is
                Mr = 2*wcgsc + N + 1
            where N is the number of symbols in the Q sequence
       We assume here that all Q sequences have the same (maximum) length; this assumption is wasteful if the Q-sequence lengths
        vary, but it simplifies the bookkeeping.
    */
    INT32 wcgsc = _pqb->pab->aas.ComputeWorstCaseGapSpaceCount( _pqb->Nmax );
    this->AKP.wcgsc = wcgsc;                            // worst-case gap space count
    this->AKP.bw = (2 * wcgsc) + 1;                     // width of the computed scoring-matrix band
    this->AKP.celBRLEAperQ = _pqb->celBRLEAperQ;        // number of 32-bit elements per Q sequence in BRLEA buffers
    this->AKP.Mr = _pqb->Nmax + (2*wcgsc) + 1;          // maximum number of R symbols spanned by a scoring matrix
    this->AKP.celSMperQ = this->AKP.bw * _pqb->Nmax;    // scoring-matrix cells per Q sequence (assumes that all Q sequences have the maximum length)
}
#pragma endregion
