/*
  baseMapGs.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseMapGs__

/// <summary>
/// Class <c>baseMapGs</c> performs seed-and-extend alignment on a list of Q sequences.
/// </summary>
class baseMapGs : public tuBaseS, public baseMapCommon
{
    protected:
        baseMapGs( void );
        virtual void main( void );

    private:
        UINT32 initSharedMemory( void );
        void initGlobalMemory( void );
        void initGlobalMemory_LJSI( void );
        void initGlobalMemory_LJSIIteration( UINT32 nJ );
        void resetGlobalMemory_LJSIIteration( void );
        void resetGlobalMemory_LJSI( void );
        void resetGlobalMemory( void );
        void accumulateJcounts( void );
        void loadJforQ( UINT32 iQ, UINT32 nQ, UINT32 nJ );
        void combineRuForSeedInterval( void );
        void loadJforSeedInterval( void );
        void filterJforSeedInterval( UINT32 cbSharedPerBlock );
        void mergeDu( void );
        void mapJforSeedInterval( void );
        void pruneQu( void );

    public:
        baseMapGs( const char* ptumKey, QBatch* pqb, DeviceBuffersG* pdbg, HostBuffers* phb );
        virtual ~baseMapGs( void );
};
