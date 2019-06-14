/*
  tuComputeKMH.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuComputeKMH__

/// <summary>
/// Class <c>tuComputeKMH</c> computes kmer hashes and "sketch bits" for unmapped reads.
/// </summary>
class tuComputeKMH : public tuBaseS
{
    private:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        AriocTaskUnitMetrics*   m_ptum;
        HiResTimer              m_hrt;

    protected:
        void main( void );

    private:
        tuComputeKMH( void );

        void computeKMH10( void );

        void computeKMH20( void );
        void launchKernel20( void );

        void computeKMH30( void );

    public:
        tuComputeKMH( QBatch* pqb );
        virtual ~tuComputeKMH( void );
};
