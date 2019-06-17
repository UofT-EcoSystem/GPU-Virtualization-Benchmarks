/*
  AriocEncoderParams.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocEncoderParams__

#pragma region enums
enum SqFileFormat
{
    SqFormatUnknown = 0x00,
    SqFormatFASTA   = 0x01,
    SqFormatFASTQ   = 0x02
};

enum SqCategory
{
    sqCatQ =            0x01,
    sqCatR =            0x02,
    sqCatRstrandMask =  0x04,
    sqCatRplus =        sqCatR,                     // 0x02
    sqCatRminus =       (sqCatR|sqCatRstrandMask)   // 0x06
};
#pragma endregion

#pragma region structs
#pragma pack(push, 1)
struct Cvalue
{
    UINT32            hashKey;  // hash key
    volatile UINT32   nJ;       // number of associated J values

    Cvalue( UINT32 _hashKey, INT32 _nJ )
    {
        this->hashKey = _hashKey;
        this->nJ = _nJ;
    }
};
#pragma pack(pop)

struct AriocEncoderParams
{
    AppMain*                pam;                                // reference to this app's AppMain instance
    Nencoding               Nenc;                               // N (non-ACGT) symbol encoding
    INT32                   nLPs;                               // number of CPU "logical processors" (cores or hyperthreads)
    INT32                   DataSourceId;                       // user-specified data source ID
    volatile UINT64         nSqIn;                              // total number of sequences read from the input file(s)
    volatile UINT64         nSqEncoded;                         // total number of sequences encoded
    volatile UINT64         nSymbolsEncoded;                    // total number of symbols encoded
    volatile UINT64         nKmersWithN;                        // total number of kmers containing too many Ns to align
    A21SeedBase*            pa21sb;                             // seed parameters
    bool                    emitKmers;                          // flag set if kmers should be written to disk
    UINT32                  gpuMask;                            // GPU available-device bitmask
    UINT32                  EncodeThreshold;                    // hash-value threshold for random sampling of sequences to be encoded
    INT8                    QualityScoreBias;                   // adjustment for rationalizing sanger or illumina format quality scores
    char*                   qmdQNAME;                           // Q-sequence metadata pattern: QNAME
    INT64                   QSB8;                               // quality score bias replicated in all 8 bytes of a 64-bit value
    char*                   ConfigFilespec;                     // AriocE configuration file specification
    SqFileFormat            InputFileFormat;                    // input sequence file format
    InputFileGroup          ifgRaw;                             // input file info
    char                    OutFilespecStubSq[FILENAME_MAX];    // file specification "stub" for sequence files
    char                    OutFilespecStubLUT[FILENAME_MAX];   // file specification "stub" for lookup table files
    char                    LUTtypeStub[4];                     // lookup-table file type
    char                    OutDir[_MAX_PATH];                  // directory path for output files
    char                    OutDirLUT[_MAX_PATH];               // subdirectory path for lookup table files
    UINT8                   SymbolEncode[128];                  // encodings for ASCII symbols
    char                    DnaComplement[128];                 // DNA complements for ASCII symbols
    Cvalue*                 C;                                  // J list sizes
    UINT32*                 CH;                                 // index into C list
    Hvalue8*                H;                                  // H table
    Jvalue8*                J;                                  // J table

    AriocEncoderParams( AppMain* _pam, Nencoding _enc, INT32 _nLPs, INT32 _DataSourceId,
                        char* _cfgFile, InputFileGroup& _ifgRaw, char* _OutDir, double _samplingRatio,
                        char _qualityScoreBias, bool _emitKmers, UINT32 _gpuMask, char* _qmdQNAME, A21SeedBase* _pa21sb ) :
                            pam(_pam),
                            Nenc(_enc),
                            nLPs(_nLPs),
                            DataSourceId(_DataSourceId),
                            nSqIn(0),
                            nSqEncoded(0),
                            nSymbolsEncoded(0),
                            nKmersWithN(0),
                            pa21sb(_pa21sb),
                            emitKmers(_emitKmers),
                            gpuMask(_gpuMask),
                            EncodeThreshold(static_cast<UINT32>(_samplingRatio*UINT_MAX)),
                            QualityScoreBias(_qualityScoreBias),
                            qmdQNAME(_qmdQNAME),
                            QSB8(_qualityScoreBias),
                            ConfigFilespec(_cfgFile),
                            InputFileFormat(SqFormatUnknown),
                            ifgRaw(_ifgRaw),
                            C(NULL), CH(NULL), H(NULL), J(NULL)
    {
        // sanity check
        if( (_samplingRatio < 0) || (_samplingRatio > 1) )
            throw new ApplicationException( __FILE__, __LINE__, "invalid value %4.2f for samplingRatio; valid values range from 0.0 to 1.0", _samplingRatio );
        if( (_qualityScoreBias != static_cast<char>(-1)) && (_qualityScoreBias != 33) && (_qualityScoreBias != 64) )
            throw new ApplicationException( __FILE__, __LINE__, "invalid value %d for qualityScoreBias; valid values are 33 (Sanger) or 64 (Solexa, Illumina)", _qualityScoreBias );

        memset( LUTtypeStub, 0, sizeof LUTtypeStub );
        memset( OutFilespecStubSq, 0, sizeof OutFilespecStubSq );
        memset( OutFilespecStubLUT, 0, sizeof OutFilespecStubLUT);
        strcpy_s( OutDir, _MAX_PATH, _OutDir );
        memset( OutDirLUT, 0, sizeof OutDirLUT );
        memset( SymbolEncode, 0, sizeof SymbolEncode );
        memset( DnaComplement, 0, sizeof DnaComplement );

        QSB8 = (QSB8 << 8) | QSB8;
        QSB8 = (QSB8 << 16) | QSB8;
        QSB8 = (QSB8 << 32) | QSB8;
    }
};
#pragma endregion
