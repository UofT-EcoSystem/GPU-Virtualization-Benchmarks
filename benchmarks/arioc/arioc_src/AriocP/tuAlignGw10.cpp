/*
  tuAlignGw10.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuAlignGw10::tuAlignGw10()
{
}

/// <param name="ptumKey">a string that identifies the <c>AriocTaskUnitMetrics</c> instance associated with the specialization of this base class</param>
/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="pdbb">a reference to a <c>DeviceBuffersBase</c> instance</param>
/// <param name="flagRi">D-value buffer identifier</param>
tuAlignGw10::tuAlignGw10( const char* ptumKey, QBatch* pqb, DeviceBuffersBase* pdbb, RiFlags flagRi ) : m_pqb(pqb),
                                                                                                        m_pab(pqb->pab),
                                                                                                        m_pdbb(pdbb),
                                                                                                        m_pD(NULL), m_nD(0),
                                                                                                        m_ptum(AriocBase::GetTaskUnitMetrics(ptumKey,"tuAlignGw10"))
{
    pqb->GetGpuConfig( "ThreadsPerBlock_tuAlignGw10", m_cudaThreadsPerBlock );

    // use a flag specified by the caller to determine which buffer to use
    pdbb->flagRi = flagRi;
    switch( pdbb->flagRi )
    {
        case riDc:
            m_pD = pdbb->Dc.p;
            m_nD = pdbb->Dc.n;
            break;

        case riDm:
            m_pD = pdbb->Dm.p;
            m_nD = pdbb->Dm.n;
            break;

        case riDx:
            m_pD = pdbb->Dx.p;
            m_nD = pdbb->Dx.n;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for flagRi: %d", flagRi );
            break;
    }
}

/// destructor
tuAlignGw10::~tuAlignGw10()
{
}
#pragma endregion

#pragma region private methods
/// [private] method initGlobalMemory
void tuAlignGw10::initGlobalMemory()
{
    CRVALIDATOR;

    try
    {
        /* CUDA global memory layout after initialization:

             low:   Qw      Qwarps
                    Qi      interleaved Q sequence data
                    DBgw.Dc mapped candidates with unmapped opposite mates
                    DBgw.Qu QIDs of candidates for seed-and-extend alignment (unmapped opposite mates)
                    DBgw.Ru subId bits for candidates for SX alignment

            high:   (unallocated)
        */

        /* Allocate two parallel lists:
            - QIDs of mates
            - corresponding subId bits (from D values for mapped mates with unmapped opposite mates)
        */
        CREXEC( m_pdbb->Qu.Alloc( cgaLow, m_nD, false ) );
        SET_CUDAGLOBALPTR_TAG(m_pdbb->Qu,"Qu");
        CREXEC( m_pdbb->Ru.Alloc( cgaLow, m_nD, false ) );
        SET_CUDAGLOBALPTR_TAG(m_pdbb->Ru,"Ru");
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method computeGridDimensions
void tuAlignGw10::computeGridDimensions( dim3& d3g, dim3& d3b )
{
    computeKernelGridDimensions( d3g, d3b, m_nD );
}

/// [private] method copyKernelResults
void tuAlignGw10::copyKernelResults()
{
    CRVALIDATOR;

    try
    {
        // save the number of elements in the Qu and Ru lists
        m_pdbb->Qu.n = m_nD;
        m_pdbb->Ru.n = m_nD;

        // block until the kernel completes
        CREXEC( waitForKernel() );

        // performance metrics
        InterlockedExchangeAdd( &m_ptum->ms.Launch, m_hrt.GetElapsed(true) );
    }
    catch( ApplicationException* pex )
    {
        CRTHROW;
    }
}

/// [private] method resetGlobalMemory
void tuAlignGw10::resetGlobalMemory()
{
    /* CUDA global memory layout after reset:

         low:   Qw      Qwarps
                Qi      interleaved Q sequence data
                Dc      mapped D values with unmapped opposite mates
                Qu      QIDs of D values
                Ru      subId bits for QIDs

        high:   (unallocated)
    */
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Extracts QIDs and subId bits for unmapped opposite mates of mapped candidates.
/// </summary>
void tuAlignGw10::main()
{
    CDPrint( cdpCD3, "[%d] %s...", m_pqb->pgi->deviceId, __FUNCTION__ );

    initConstantMemory();
    initGlobalMemory();

    dim3 d3b;
    dim3 d3g;
    computeGridDimensions( d3g, d3b );

    launchKernel( d3g, d3b, initSharedMemory() );

    copyKernelResults();
    resetGlobalMemory();
    
    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.PostLaunch, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "[%d] %s: completed", m_pqb->pgi->deviceId, __FUNCTION__ );
}
#pragma endregion
