/*
  baseMapGw.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseMapGw__

/// <summary>
/// Class <c>baseMapGw</c> performs seed-and-extend alignment on a list of Q sequences.
/// </summary>
class baseMapGw : public tuBaseS, public baseMapCommon
{
    protected:
        baseMapGw( void );
        virtual void main( void );

    private:
        void computeGridDimensions( dim3& d3g, dim3& d3b, UINT32 nThreads );
        UINT32 initSharedMemory( void );
        void initGlobalMemory( void );
        void initGlobalMemory_LJSI( void );
        void initGlobalMemory_LJSIIteration( UINT32 nJ );
        void resetGlobalMemory_LJSIIteration( void );
        void resetGlobalMemory_LJSI( void );
        void accumulateJcounts( void );
        void loadJforQ( UINT32 iQ, UINT32 nQ, UINT32 nJ );
        void resetGlobalMemory( void );
        void loadJforSeedInterval( void );
        void mergeDu( void );
        void filterJforSeedInterval( UINT32 cbSharedPerBlock );
        void mapJforSeedInterval( void );
        void pruneQu( void );

    public:
        baseMapGw( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, HostBuffers* phb );
        virtual ~baseMapGw( void );
};
