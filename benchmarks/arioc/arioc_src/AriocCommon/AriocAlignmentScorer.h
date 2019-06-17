/*
  AriocAlignmentScorer.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocAlignmentScorer__

#pragma region enums
enum OutputFormatType
{
    oftUnknown =    0,
    oftSAM =        1,  // SAM
    oftSBF =        2,  // SAM fields in Microsoft SQL Server "native" bulk format
    oftTSE =        3,  // Terabase Search Engine fields in Microsoft SQL Server "native" bulk format
    oftKMH =        4   // Terabase Search Engine kmer-hashed sequences in Microsoft SQL Server "native" bulk format
};

enum AlignmentResultFlags : UINT16 // modelled after Bowtie2 functionality
{
    arfNone =                   0x0000,

    arfReportUnmapped =         0x0001,             // (paired or unpaired reads)
    arfReportMapped =           0x0002,             // (unpaired reads only)
    arfReportConcordant =       arfReportMapped,    // (paired reads only)
    arfReportDiscordant =       0x0004,             // (paired reads only)
    arfReportRejected =         0x0008,             // (paired reads only)
    arfMaskReport =             (arfReportUnmapped | arfReportMapped | arfReportConcordant | arfReportDiscordant | arfReportRejected),

    arfMaxReportPaired =        arfReportRejected,  // maximum values for arfReport* flags
    arfMaxReportUnpaired =      arfReportMapped,

    arfUnmapped =               arfReportUnmapped,  // (synonyms for arfReport* flags)
    arfMapped =                 arfReportMapped,
    arfConcordant =             arfReportConcordant,
    arfDiscordant =             arfReportDiscordant,
    arfRejected =               arfReportRejected,

    arfOrientationSame =        0x0010,
    arfOrientationConvergent =  0x0020,
    arfOrientationDivergent =   0x0040,
    arfMaskOrientation =        (arfOrientationSame | arfOrientationConvergent | arfOrientationDivergent),

    arfCollisionOverlap =       0x0080,
    arfCollisionCover =         0x0100,
    arfCollisionDovetail =      0x0200,
    arfMaskCollision =          (arfCollisionOverlap | arfCollisionCover | arfCollisionDovetail),

    arfMate1Unmapped =          0x0400,
    arfMate2Unmapped =          0x0800,
    arfBothMatesUnmapped =      (arfMate1Unmapped | arfMate2Unmapped),
    arfUnique =                 0x1000,
    arfPrimary =                0x2000,             // SAM "primary alignment"

    arfWriteMate1 =             0x4000,
    arfWriteMate2 =             0x8000
};
#pragma endregion

#pragma region structs
#pragma pack(push, 2)
struct AlignmentScoreParameters
{
    /* nongapped alignments */
    INT16               maxMismatches;  // maximum number of mismatches

    /* gapped alignments */
    ScoreFunctionType   sft;            // score function: type
    double              sfA;            // score function: coefficient
    double              sfB;            // score function: constant term

    INT16               Wm;             // per-symbol score for matching symbols
    INT16               Wx;             // per-symbol score for mismatched symbols
    INT16               Wg;             // per-gap penalty (specified as a positive number)
    INT16               Ws;             // per-space penalty (specified as a positive number)

    INT16               W[8][8];        // W-score lookup table

    AlignmentScoreParameters()
    {
    }

    AlignmentScoreParameters( INT16 _maxMismatches,
                              Wmxgs _Wmxgs, ScoreFunctionType _sft, double _sfCoeff, double _sfConst,
                              INT16 _baseConvert ) : maxMismatches(_maxMismatches),
                                                     sft(_sft),
                                                     sfA(_sfCoeff),
                                                     sfB(_sfConst),
                                                     Wm( _Wmxgs >> 24),
                                                     Wx((_Wmxgs >> 16) & 0xFF),
                                                     Wg((_Wmxgs >>  8) & 0xFF),
                                                     Ws( _Wmxgs        & 0xFF)
    {
        /* Initialize the W lookup table:
            - the "mismatch penalty" score is stored as a negative number
            - A,C,G,T, and N encodings use the configured Wm and Wx scores
            - all other potential encodings (including null) are assigned a negative (Wx) score regardless
               of how they pair; this simplifies scoring where a mapping spans the end of the R (reference)
               sequence because R can be padded at both ends with null symbols

                000     :   Wx
                001     :   Wx
                010 (Nq):   Wm/Wx
                011 (Nr):   Wm/Wx
                100 (A) :   Wm/Wx
                101 (C) :   Wm/Wx
                110 (G) :   Wm/Wx
                111 (T) :   Wm/Wx

            Ns in the same sequence type match each other, but Ns in R sequences do not match Ns in Q sequences.
        */
        for( INT32 i=0; i<8; ++i )
        {
            for( INT32 j=0; j<8; ++j )
            {
                bool isMatch = (i == j) && (i >= 2) && (j >= 2);
                this->W[i][j] = isMatch ? this->Wm : -this->Wx;
            }
        }

        /* For bsDNA alignments, C in R matches T in Q. */
        if( _baseConvert == 1 )             // 1: CT conversion
            this->W[7][5] = this->Wm;
    }
};

struct AlignmentControlParameters
{
    /* nongapped alignment */
    INT32   maxJn;              // maximum size of a hash-table J list ("bucket") for any Q sequence
    INT32   maxAn;              // maximum number of successful nongapped alignments to report for each Q sequence
    INT16   celDhitList;        // number of 32-bit elements in the Dhit list (see below)
    INT32   seedCoverageN;      // minimum number of spaced seeds required for a seed location to be a candidate for nongapped alignment

    /* gapped alignment */
    INT16   AtN;                    // threshold number of nongapped mappings to suppress subsequent alignments for a Q sequence
    INT16   AtG;                    // threshold number of gapped mappings to suppress subsequent alignments for a Q sequence
    INT32   maxJg;                  // maximum size of a hash-table J list ("bucket") for any Q sequence
    INT32   maxAg;                  // maximum number of successful gapped alignments to report for each Q sequence
    INT16   seedDepth;              // number of "seed iterations" for gapped alignment
    UINT32  minPosSep;              // minimum amount of separation to unduplicate two potentially duplicate mapping positions

    /* paired-end reads */
    INT32                   minFragLen;     // minimum paired-end fragment length
    INT32                   maxFragLen;     // maximum paired-end fragment length
    AlignmentResultFlags    arfSAM;         // alignment results to be reported in SAM format
    AlignmentResultFlags    arfSBF;         // alignment results to be reported in SAM-like SBF format
    AlignmentResultFlags    arfTSE;         // alignment results to be reported in Terabase Search Engine SBF format
    AlignmentResultFlags    arfKMH;         // alignment results to be reported in Terabase Search Engine kmer-hash format
    AlignmentResultFlags    arf;            // alignment results to be reported
    UINT8                   mapqDefault;    // default SAM MAPQ score
    UINT8                   mapqVersion;    // MAPQ computation version (see AriocBase::baseMAPQ)

    AlignmentControlParameters()
    {
    }

    AlignmentControlParameters( INT32 _maxJn, INT16 _maxAn, INT32 _seedCoverageN,
                                INT16 _AtN, INT16 _AtG,
                                INT32 _maxJg, INT16 _maxAg, INT16 _seedDepth,
                                UINT32 _minPosSep,
                                INT32 _minFragLen, INT32 _maxFragLen,
                                AlignmentResultFlags _arfSAM, AlignmentResultFlags _arfSBF, AlignmentResultFlags _arfTSE, AlignmentResultFlags _arfKMH,
                                UINT8 _mapqDefault, UINT8 _mapqVersion ) :
                                    maxJn(_maxJn), maxAn(_maxAn), celDhitList(0), seedCoverageN(_seedCoverageN),
                                    AtN(_AtN), AtG(_AtG),
                                    maxJg(_maxJg), maxAg(_maxAg), seedDepth(_seedDepth),
                                    minPosSep(_minPosSep),
                                    minFragLen(_minFragLen), maxFragLen(_maxFragLen),
                                    arfSAM(_arfSAM), arfSBF(_arfSBF), arfTSE(_arfTSE), arfKMH(_arfKMH),
                                    arf(static_cast<AlignmentResultFlags>(_arfSAM|_arfSBF|_arfTSE|_arfKMH)),
                                    mapqDefault(_mapqDefault),
                                    mapqVersion(_mapqVersion)
    {
        // sanity check
        if( _maxAg > 32 )
            throw new ApplicationException( __FILE__, __LINE__, "The maximum number of gapped alignments that can be reported per query sequence is 32" );

        /* Compute the number of 32-bit values required to represent the maximum number of successful nongapped alignments for each Q sequence; each successful
            alignment ("Dhit") is recorded as a 5-byte value:
                - bytes 0-3 in a single 32-bit value
                - byte 4 packed into an additional 32-bit value that contains the fifth byte of four successive Dhit values
        */
        this->celDhitList = _maxAn + blockdiv(_maxAn,sizeof(UINT32));
    }
};
#pragma pack(pop)
#pragma endregion


/// <summary>
/// Class <c>AriocAlignmentScorer</c> wraps scoring and control functionality for nongapped and gapped alignment implementations.
/// </summary>
class AriocAlignmentScorer : public AlignmentScorerCommon
{
    public:
        AlignmentScoreParameters    ASP;
        AlignmentControlParameters  ACP;

    private:
        AriocAlignmentScorer( void );

    public:
        AriocAlignmentScorer( const AlignmentControlParameters& _acp, const AlignmentScoreParameters& _asp );
        ~AriocAlignmentScorer( void );
        static char* AlignmentResultFlagsToString( const AlignmentResultFlags arfValue, const AlignmentResultFlags arfMask );
        static AlignmentResultFlags ParsePairOrientation( const char* pAlignmentResultFlags );
        INT16 ComputeThresholdScore( INT32 N );
        INT16 ValidateScoreFunction( INT32 N );
        INT32 ComputeWorstCaseMismatchCount( INT16 Nmax );
        INT32 ComputeWorstCaseGapSpaceCount( INT16 Nmax );
        INT32 ComputeWorstCaseDPSize( INT16 Nmax, INT16 Dspread );
        INT32 ComputeWorstCaseBRLEAsize( INT16 Nmax );
        INT32 ComputeMinimumMappingSize( INT16 Nmax );
};
