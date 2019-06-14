/*
  tuClassifyP.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuClassifyP__

#ifndef __tuClassifyP1__
#include "tuClassifyP1.h"
#endif

#ifndef __tuClassifyP2__
#include "tuClassifyP2.h"
#endif


/// <summary>
/// Class <c>tuClassifyP</c> classifies paired-end mappings
/// </summary>
class tuClassifyP : public tuClassify
{
    protected:
        void main( void );

    private:
        tuClassifyP( void );


#if TODO_CHOP_WHEN_DEBUGGED
        INT16 computeMa( BRLEAheader* pBH );
#endif
        void computeMeanTLEN( void );
        void countMappings( UINT32& totalAn, UINT32& totalAg );
        void consolidateAlignmentInfoFromBRLEAs( HostBuffers* pHB, const QAIflags mappedBy );
        void consolidateAlignmentInfo( UINT32 totalAn, UINT32 totalAg );
        // TODO: CHOP   void initializePartitions( void );
        void sortAlignmentInfo( void );
        void findPairs( void );
        // TODO: CHOP   void consolidatePartitions( void );

    public:
        tuClassifyP( QBatch* pqb );
        virtual ~tuClassifyP( void );
};

