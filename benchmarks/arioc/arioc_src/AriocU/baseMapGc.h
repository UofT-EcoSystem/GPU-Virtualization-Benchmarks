/*
  baseMapGc.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseMapGc__


/// <summary>
/// Class <c>baseMapGc</c> does seed-and-extend gapped alignment at reference-sequence locations prioritized by seed coverage.
/// </summary>
class baseMapGc : public tuBaseS, public baseMapCommon
{
    private:
        baseMapGc( void );
        virtual void main( void );
        void computeGridDimensions( dim3& d3g, dim3& d3b, UINT32 nThreads );
        UINT32 initSharedMemory( void );
        void initGlobalMemory( void );
        void resetGlobalMemory( void );
        void initGlobalMemory_LJSI( void );
        void initGlobalMemory_LJSIIteration( UINT32 nJ );
        void initConstantMemory5( void );
        void resetGlobalMemory_LJSIIteration( void );
        void resetGlobalMemory_LJSI( void );
        void resetGlobalMemoryForIteration( void );
        void resetGlobalMemoryForKernels( void );
        void accumulateJcounts( void );
        void loadJforQ( UINT32 iQ, UINT32 nQ, UINT32 nJ );
        void loadJforSeedInterval( void );
        void mergeDl( void );
        void filterJforSeedInterval( UINT32 cbSharedPerBlock );
        void mergeDu( void );
        void mapJforSeedInterval( void );
        void pruneQu( void );

    public:
        baseMapGc( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, HostBuffers* phb );
        virtual ~baseMapGc( void );
};
