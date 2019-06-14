/*
  AriocTaskUnitMetrics.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocTaskUnitMetrics__

/// <summary>
/// Flags used with <c>AriocTaskUnitMetrics::Print</c> that cause zero-valued task unit performance metrics to be emitted.
/// </summary>
enum TumFlags
{
    tumDefault =                0x00000000,
    tumCandidateQ =             0x00000001,
    tumCandidateD =             0x00000002,
    tumCandidateP =             0x00000004,
    tumMappedQ =                0x00000008,
    tumMappedD =                0x00000010,
    tumMappedP =                0x00000020,
    tumMappedPairs =            0x00000040,
    tumInstances =              0x00000080,
    tumIterations =             0x00000100,
    tumMaxAvailable =           0x00000200,
    tumMaxAvailablePct =        0x00000400,
    tumMaxUsed =                0x00000800,
    tumMaxUsedPct =             0x00001000,
    tumTotalBytes =             0x00002000,
    tumElapsed =                0x00004000,
    tumPreLaunch =              0x00008000,
    tumLaunch =                 0x00010000,
    tumPostLaunch =             0x00020000,

    tumCandidatesPerSecond =    0x80000000
};

/// <summary>
/// Class <c>AriocTaskUnitMetrics</c> consolidates performance metrics for task units in the Arioc applications
/// </summary>
class AriocTaskUnitMetrics
{
    private:
        static const double m_usPerSec;
        static const double m_msPerSec;
        static const double m_bytesPerMB;
        static const double m_bytesPerGB;
        static const char*  m_fmt_n1;
        static const char*  m_fmt_n2;
        static const char*  m_fmt_n3;
        static const char*  m_fmt_n4;
        static const char*  m_fmt_cb1;
        static const char*  m_fmt_cb2;
        static const char*  m_fmt_ms1;

    public:
        struct
        {
            UINT64 CandidateQ;      // number of candidate Q sequences
            UINT64 CandidateD;      // number of candidate D values 
            UINT64 CandidateP;      // number of candidate pairs
            UINT64 MappedQ;         // number of mapped Q sequences
            UINT64 MappedD;         // number of mapped D values
            UINT64 MappedP;         // number of mapped pairs
            UINT64 MappedPairs[9];  // number of mapped mates, broken down as pairs

            UINT64 MaxQperBatch;    // maximum number of Q sequences in a batch
            UINT32 Batches;         // number of batches completed
            UINT32 BatchInstances;  // number of batch instances
            UINT32 GPUs;            // number of GPUs
            UINT32 CPUthreads;      // number of CPU threads per GPU

            UINT32 Instances;       // number of instances (e.g. of a CUDA kernel)
            UINT32 Iterations;      // number of iterations
        }
                    n;

        struct
        {
            UINT64 MaxUsed;         // maximum number of bytes of memory used
            double MaxUsedPct;      // maximum bytes used as a percentage of available capacity
            UINT64 MaxAvailable;    // maximum number of bytes of memory available
            double MaxAvailablePct; // maximum number of bytes of memory available
            UINT64 TotalBytes;      // total number of bytes

            UINT64 TotalRAM;        // total number of bytes of physical memory
        }
                    cb;

        struct
        {
            UINT64 Elapsed;
            UINT64 PreLaunch;
            UINT64 Launch;
            UINT64 PostLaunch;
        }
                    us;

        struct
        {
            UINT64 Elapsed;
            UINT64 PreLaunch;
            UINT64 Launch;
            UINT64 PostLaunch;
        }
                    ms;

        char Key[64];               // the "key" in the associative array in which this AriocTaskUnitMetrics instance is contained
        bool emitted;               // flag set when the contents of this AriocTaskUnitMetrics instance have been emitted

    private:
        bool isNull( void );
        UINT64 usToMs( const UINT64 us );

    public:
        AriocTaskUnitMetrics( void );
        ~AriocTaskUnitMetrics( void );
        void Print( CDPrintFlags cdp, const char* desc, TumFlags flags, INT32 cchIndent1, INT32 cchIndent2, INT32 cchSpan1, INT32 cchSpan2 = -1 );
        void AccumulateMax( TumFlags flag, UINT64 value );
        void AccumulateMax( TumFlags flag, double value );
        void AccumulateMaxPct( TumFlags flag, UINT64 value, UINT64 capacity );
};
