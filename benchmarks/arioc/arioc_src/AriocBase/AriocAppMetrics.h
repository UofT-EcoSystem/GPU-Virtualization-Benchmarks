/*
  AriocAppMetricsP.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocAppMetricsP__

/// <summary>
/// Class <c>AriocAppMetrics</c> records performance metrics.
/// </summary>
class AriocAppMetrics
{
    typedef void (AriocAppMetrics::*mfnEmitSamInfo)( CDPrintFlags cdp, const char* indent2, INT32 cchSpan1, INT32 cchSpan2 );

    private:
        static const double m_msPerSec;
        static const double m_perM;
        static const double m_perG;

        static const char*  m_fmt_n1;
        static const char*  m_fmt_n2;
        static const char*  m_fmt_n3;
        static const char*  m_fmt_cb1;
        static const char*  m_fmt_cb2;
        static const char*  m_fmt_ms1;

    public:
        UINT32  GPUmask;                    // GPU mask

        struct
        {
            UINT64  Available;              // number of bytes of physical host memory available to this process
            UINT64  R;                      // number of bytes in host memory buffers for R data
            UINT64  Hn;                     // number of bytes in host memory buffers for H table for nongapped aligner
            UINT64  Hg;                     // number of bytes in host memory buffers for H table for gapped aligners
            UINT64  Jn;                     // number of bytes in host memory buffers for J table for nongapped aligner
            UINT64  Jg;                     // number of bytes in host memory buffers for J table for gapped aligner
			UINT64  Hng;					// number of bytes in CUDA global memory buffers for nongapped H table
			UINT64  Hgg;					// number of bytes in CUDA global memory buffers for gapped H table
			UINT64  Pinned;                 // number of bytes in pinned host memory buffers
			UINT64	HJgmem;					// number of bytes in CUDA global memory for H and J tables
            UINT64  Q;                      // total bytes read from all Q-sequence files
        }
            cb;

        struct
        {
            UINT64  App;                    // total elapsed time for this application
            UINT64  Aligners;               // elapsed time for aligners only
            UINT64  InitializeGPUs;         // initialize GPUs
            UINT64  SniffQ;                 // estimate read lengths and find partition cutpoints in Q-sequence input files
            UINT64  LoadR;                  // read R sequence data from disk into CPU memory
            UINT64  LoadQ;                  // load Q from disk and transfer to GPU memory
            UINT64  BuildQwarps;            // build Qwarps
            UINT64  LoadHJ;                 // read H and J lookup tables into CPU memory
            UINT64  XferMappings;           // transfer mappings (D values or BRLEAs) from GPU to CPU memory

            UINT64  ReleasePinned;          // free page-locked (pinned) CPU memory
        }
            ms;

        struct
        {
            UINT64  XferR;                  // transfer R sequence data to GPU memory
            UINT64  XferQ;                  // transfer interleaved Q sequence data to GPU memory
            UINT64  XferJ;                  // transfer J lookup table info from GPU memory
        }
            us;

        struct
        {
            INT16   CPUThreads;             // number of CPU threads (hyperthreads) available to this process
            INT16   GPUs;                   // number of available GPUs

            UINT32  maxnJn;                 // maximum J-list size (spaced seeds)
            UINT32  maxnJg;                 // maximum J-list size (seed-and-extend seeds)
            INT8    maxSubId;               // maximum subId in J tables

            UINT32  BatchInstances;         // number of QBatch instances (e.g. 2 * nGPUs)
            UINT32  BatchesCompleted;       // number of non-empty batches processed

            UINT64  Q;                      // total Q sequences (mates)
            double  pctReadsMapped;         // overall percentage of Q sequences for which at least one mapping was found
            UINT64  DuplicateMappings;      // total number of duplicate mappings
            UINT32  Nmax;                   // maximum-length N
            UINT32  bwmax;                  // maximum band width for gapped alignment

            UINT32  SketchBits;             // expected number of bits to set in kmer-hash "sketch bitmap"

            struct                          // AriocU only
            {
                UINT64  nReads_Mapped1;         // total reads with 1 mapped alignment
                UINT64  nReads_Mapped2;         // total reads with 2 or more mapped alignments
            }
                u;

            struct                          // AriocP only
            {
                UINT64  Pairs_Concordant1;      // total concordant pairs (1 mapped alignment)
                UINT64  Pairs_Concordant2;      // total concordant pairs (2 or more mapped alignments)
                UINT64  Pairs_Discordant;       // total discordant pairs
                UINT64  Pairs_Rejected;         // total unmapped pairs (i.e., no concordant or discordant alignments) having both ends mapped
                UINT64  Pairs_Unmapped;         // total unmapped pairs (i.e., no concordant or discordant alignments) having one or both ends unmapped
                UINT64  Mates_Unmapped;         // total unmapped mates in unmapped pairs
                UINT64  Mates_Mapped1;          // total mates with 1 mapped alignment
                UINT64  Mates_Mapped2;          // total mates with 2 or more mapped alignments
                UINT64  Mates_MappedU1;         // total mates (in unmapped pairs) with 1 mapped alignment
                UINT64  Mates_MappedU2;         // total mates (in unmapped pairs) with 2 or more mapped alignments

                double  TLEN_mean;              // TLEN (paired-end fragment length) mean
                double  TLEN_sd;                // TLEN (paired-end fragment length) standard deviation
            }
                p;
        }
            n;

        struct
        {
            UINT64  nCandidateQ;            // total number of candidate Q sequences

            struct                          // AriocU only
            {
                UINT64  nReads_Mapped1;         // SAM rows for reads with 1 mapped alignment
                UINT64  nReads_Mapped2;         // SAM rows for reads with 2 or more mapped alignments
                UINT64  nReads_Unmapped;        // SAM rows for reads without mappings
            }
                u;

            struct                          // AriocP only
            {
                UINT64  nPairs_Concordant1;     // total concordant pairs (1 mapped alignment)
                UINT64  nPairs_Concordant2;     // total concordant pairs (2 or more mapped alignments)
                UINT64  nPairs_Discordant;      // total discordant pairs
                UINT64  nPairs_Rejected;        // total unmapped pairs (i.e., no concordant or discordant alignments) having both ends mapped
                UINT64  nPairs_Unmapped;        // total unmapped pairs (i.e., no concordant or discordant alignments) having one or both ends unmapped
                UINT64  nMates_Unmapped;        // total unmapped mates in unmapped pairs
                UINT64  nMates_Mapped1;         // total mates with 1 mapped alignment
                UINT64  nMates_Mapped2;         // total mates with 2 or more mapped alignments
                UINT64  nMates_MappedU1;        // total mates (in unmapped pairs) with 1 mapped alignment
                UINT64  nMates_MappedU2;        // total mates (in unmapped pairs) with 2 or more mapped alignments

                UINT64  nRowsConcordant;        // SAM rows written
                UINT64  nRowsDiscordant;
                UINT64  nRowsRejected;
                UINT64  nRowsUnmapped;
            }
                p;

        }
            sam;

    private:
        void emitGpuInfo( CDPrintFlags cdp, const char* indent2, INT32 cchSpan1 );
        void emitCpuInfo( CDPrintFlags cdp, const char* indent2, INT32 cchSpan1, INT32 cchSpan2 );
        void emitSummaryInfo( CDPrintFlags cdp, const char* indent2, INT32 cchSpan1, INT32 cchSpan2 );
        void emitSamInfoU( CDPrintFlags cdp, const char* indent2, INT32 cchSpan1, INT32 cchSpan2 );
        void emitSamInfoP( CDPrintFlags cdp, const char* indent2, INT32 cchSpan1, INT32 cchSpan2 );
        void emitSamInfo( CDPrintFlags cdp, const char* indent2, INT32 cchSpan1, INT32 cchSpan2 );

    public:
        AriocAppMetrics( void );
        ~AriocAppMetrics( void );
        void Print( CDPrintFlags cdp, const char* desc, INT32 cchIndent1, INT32 cchIndent2, INT32 cchSpan1, INT32 cchSpan2 = -1 );
        void PrintPipelineInfo( CDPrintFlags cdp, INT32 cchIndent1, INT32 cchIndent2, INT32 cchSpan1, INT32 cchSpan2 = -1);
};
