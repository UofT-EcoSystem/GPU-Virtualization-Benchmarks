/*
  QBatch.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __QBatch__

// forward declaration
class QBatchPool;

#pragma region enums
enum QAIflags : UINT16
{
    qaiNone =       0x0000,
    qaiRCr =        0x0001,     // bit  0: the mapping is on the reverse-complement strand of the reference sequence
    qaiBest =       0x0002,     // bit  1: 1 for the "best" mapping (see tuClassifyP2 and tuClassifyU2)
    qaiInPair =     0x0004,     // bit  2: 1 if read is aligned in a pair (paired-end mappings)
    qaiParity =     0x0008,     // bit  3: 0 if read is first mate in pair; 1 if read is second mate in pair
    qaiRCq =        0x0010,     // bit  4: 1 if the mapping uses the reverse complement of the read (query) sequence
    qaiRCdup =      0x0020,     // bit  5: 1 if the mapping is duplicated using either strand of the read (query) sequence
    qaiMapperN =    0x0100,     // bit  8: the mapping was found by the nongapped aligner
    qaiMapperGwn =  0x0200,     // bit  9: the mapping was found by the windowed gapped aligner for unpaired nongapped mappings
    qaiMapperGs =   0x0400,     // bit 10: the mapping was found by the seed-and-extend gapped aligner using paired-end criteria
    qaiMapperGc  =  0x0800,     // bit 11: the mapping was found by the seed-and-extend gapped aligner using seed coverage prioritization
    qaiMapperGwc =  0x1000,     // bit 12: the mapping was found by the windowed gapped aligner for unpaired seed-and-extend gapped mappings
    qaiIgnoreDup =  0x8000,     // bit 15: this is a duplicate mapping and can be ignored
    qaiMapped =     (qaiMapperN|qaiMapperGwn|qaiMapperGs|qaiMapperGc|qaiMapperGwc)
};

#define qaiShrRC    4           // number of bits to shift between qaiRCr and qaiRCq
#pragma endregion

#pragma region structs
#pragma pack(push, 1)
struct QAI      // Q sequence alignment info
{
    BRLEAheader*    pBH;            // pointer to BRLEA info
    QAIflags        flags;          // QAI flags
    UINT32          nAu;            // number of unduplicated mappings
    UINT32          qid;            // Qwarp identifier
    UINT32          Jf;             // reportable J value (0-based position in the forward R sequence of the first aligned symbol in the local alignment)
    INT16           N;              // Q sequence length
    INT16           editDistance;   // edit distance (Levenshtein distance)
    INT16           Vsec;           // second-best V score (for a Q sequence with multiple mappings)
    UINT16          mcm;            // maximum number of contiguous matches in the mapping
    UINT8           mapQ;           // SAM mapping quality
    UINT8           mapQp2;         // SAM mapping quality (for secondary paired-end mapping)
    UINT8           subId;          // subunit Id (e.g. chromosome number)
    UINT8           nTBO;           // number of traceback origins
};

struct PAI      // pair alignment info
{
    UINT32                  iw;                 // Qwarp index for mate 1 (the first read of the pair)
    INT16                   iq;                 // index within Qwarp for mate 1 (the first read of the pair)
    INT16                   Vpair;              // combined V scores for the mates
    AlignmentResultFlags    arf;                // flags
    QAI*                    pQAI1;              // Q alignment info for mate 1
    QAI*                    pQAI2;              // Q alignment info for mate 2
    INT32                   FragmentLength;     // SAM-compatible "outer distance" between the mates (TLEN)
    INT32                   diffTLEN;           // difference between actual TLEN and estimated mean TLEN for the batch

    PAI() : iw(0), iq(0), Vpair(0), arf(arfNone), pQAI1(NULL), pQAI2(NULL), FragmentLength(0), diffTLEN(0)
    {
    }

    PAI( UINT32 _qid ) : iw(QID_IW(_qid)), iq(QID_IQ(_qid)), Vpair(0), arf(arfNone), pQAI1(NULL), pQAI2(NULL), FragmentLength(0), diffTLEN(0)
    {
    }
};

struct QAIReference
{
    UINT32  ofsQAI;     // offset into QAI buffer
    UINT32  nAq;        // total number of alignments for the corresponding Q sequence

    QAIReference( UINT32 _ofsQAI, INT16 _nAq )
    {
        this->ofsQAI = _ofsQAI;
        this->nAq = _nAq;
    }
};
#pragma pack(pop)

struct GpuInfo
{
    CudaDeviceBinding*      pCDB;
    CudaGlobalAllocator*    pCGA;
    INT16                   deviceId;
    INT16                   deviceOrdinal;
    CudaGlobalPtr<UINT64>   bufR;
    CudaGlobalPtr<UINT32>   bufHn;
    CudaGlobalPtr<UINT32>   bufHg;
    UINT64*                 pR;
    UINT32*                 pHn;
    UINT32*                 pHg;

    GpuInfo( CudaDeviceBinding* _pCDB, INT16 _gpuOrdinal) : pCDB(_pCDB), pCGA(NULL),
                                                            deviceId(_pCDB->GetDeviceProperties()->cudaDeviceId),
                                                            deviceOrdinal(_gpuOrdinal),
                                                            pR(NULL), pHn(NULL), pHg(NULL)
    {
    }
};

/// <summary>
/// Metadata buffers
/// </summary>
struct MfileBuf
{
    WinGlobalPtr<char>      buf;        // metadata loaded from .sqm file
    WinGlobalPtr<UINT32>    ofs;        // per-Q offsets to buffered metadata
};
#pragma endregion

/// <summary>
/// Class <c>QBatch</c> maintains data and state for a group of Q sequences
/// </summary>
class QBatch
{
    private:
        QBatchPool*                 m_pqbp;         // references the QBatchPool that contains the QBatch instance
        INT16                       m_ibp;          // batch pool index

    public:
        AriocBase*                  pab;            // references the AriocBase instance for this application
        GpuInfo*                    pgi;            // per-GPU info

        WinGlobalPtr<Qwarp>         QwBuffer;       // Qwarps
        WinGlobalPtr<UINT64>        QiBuffer;       // interleaved encoded Q sequences
        UINT32                      celBRLEAperQ;   // number of 32-bit values required to contain a worst-case BRLEA for one mapping
        UINT32                      Mrw;            // number of R symbols spanned by the "window" in windowed gapped alignment

        DeviceBuffers               DB;             // GPU buffers
        DeviceBuffersJ              DBj;            // GPU buffers (setup for paired-end alignment)
        DeviceBuffersN              DBn;            // GPU buffers (nongapped aligner)
        DeviceBuffersG              DBgw;           // GPU buffers (windowed gapped aligner)
        DeviceBuffersG              DBgs;           // GPU buffers (seed-and-extend gapped aligner)
        DeviceBuffersKMH            DBkmh;          // GPU buffers (kmer hash)
        
        HostBuffers                 HBn;            // buffers (nongapped aligner)
        HostBuffers                 HBgs;           // buffers (seed-and-extend gapped aligner, seeds filtered by position)
        HostBuffers                 HBgc;           // buffers (seed-and-extend gapped aligner, seeds prioritized by coverage)
        HostBuffers                 HBgwn;          // buffers (windowed gapped aligner for mates unpaired after nongapped alignment)
        HostBuffers                 HBgwc;          // buffers (windowed gapped aligner for mates unpaired after prioritized seed-and-extend gapped alignment)
        
        WinGlobalPtr<QAI>           AaBuffer;       // consolidated alignment info
        WinGlobalPtr<QAIReference>  ArBuffer;       // per-Q offsets into the consolidated alignment info buffer
        WinGlobalPtr<PAI>           ApBuffer;       // info for each paired alignment

        WinGlobalPtr<UINT64>        S64;            // S64 ("sketch bits") for unmapped Q sequences
        
        UINT32                      QwarpLimit;     // maximum number of Qwarps that can be allocated for the current QBatch instance
        INT16                       Nmax;           // maximum length of a Q sequence in the QBatch
        
        MfileBuf                    MFBm[2];        // metadata file buffers
        MfileBuf                    MFBq[2];

        InputFileGroup::FileInfo*   pfiQ;           // Q-sequence (read) file info

    public:
        QBatch( QBatchPool* _pqbp, INT16 _ibp, AriocBase* _pab, GpuInfo* _pgi );
        ~QBatch( void );
        void Initialize( INT16 estimatedNmax, InputFileGroup::FileInfo* pfi );
        void ComputeMrw( void );
        void Release( void );
        void GetGpuConfig( const char* paramName, UINT32& cudaThreadsPerBlock );
};
