/*
  tuSortJgpu.cu

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>


#pragma region alignN21
/// [private] method initGlobalMemory10
void tuSortJgpu::initGlobalMemory10( JsortChunk* _pjsc )
{
    CRVALIDATOR;

    CDPrint( cdpCD4, "%s: 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx",
                     __FUNCTION__, m_pJ[_pjsc->ofsJfirst+0], m_pJ[_pjsc->ofsJfirst+1], m_pJ[_pjsc->ofsJfirst+2], m_pJ[_pjsc->ofsJfirst+3], m_pJ[_pjsc->ofsJfirst+4], m_pJ[_pjsc->ofsJfirst+5], m_pJ[_pjsc->ofsJfirst+6], m_pJ[_pjsc->ofsJfirst+7] );

    // allocate a GPU buffer to contain J lists
    INT64 cel = (_pjsc->ofsJlast - _pjsc->ofsJfirst) + 1;
    CREXEC( m_pJbuf->Alloc( cgaLow, cel, true ) );

    // copy from host to device memory
    m_pJbuf->CopyToDevice( m_pJ+_pjsc->ofsJfirst, m_pJbuf->Count );
}

/// [private] method launchKernel10
void tuSortJgpu::launchKernel10()
{
    /* Sort the current J-list buffer chunk.  Since each 64-bit value contains a "tag" that associates the value
        with the J list that corresponds to an H (hash key) value, this is in effect a segmented operation. */
    thrust::device_ptr<UINT64> ttpJbuf( m_pJbuf->p );
    thrust::sort( epCGA, ttpJbuf, ttpJbuf+m_pJbuf->Count );

    // zero the tag and x fields
    thrust::transform( epCGA, ttpJbuf, ttpJbuf+m_pJbuf->Count, ttpJbuf, tuSortJgpu::zeroJvalue8Tag() );
}

/// [private] method copyKernelResults10
void tuSortJgpu::copyKernelResults10( JsortChunk* _pjsc )
{
    // copy from device to host memory
    m_pJbuf->CopyToHost( m_pJ+_pjsc->ofsJfirst, m_pJbuf->Count );

    CDPrint( cdpCD4, "%s: 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx 0x%016llx",
                     __FUNCTION__, m_pJ[_pjsc->ofsJfirst+0], m_pJ[_pjsc->ofsJfirst+1], m_pJ[_pjsc->ofsJfirst+2], m_pJ[_pjsc->ofsJfirst+3], m_pJ[_pjsc->ofsJfirst+4], m_pJ[_pjsc->ofsJfirst+5], m_pJ[_pjsc->ofsJfirst+6], m_pJ[_pjsc->ofsJfirst+7] );
}

/// [private] method resetGlobalMemory10
void tuSortJgpu::resetGlobalMemory10()
{
    // free the GPU buffer
    m_pJbuf->Free();
}
#pragma endregion

