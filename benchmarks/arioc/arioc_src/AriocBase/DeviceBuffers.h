/*
  DeviceBuffers.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __DeviceBuffers__

#pragma region enums
/// <summary>
/// Enum <c>RiFlags</c> identifies device buffers that contain D values.
/// </summary>
enum RiFlags
{
    riNone =    0,
    riDc =      1,
    riDx =      2,
    riDm =      3,
    riDi =      4,
    riDu =      5
};
#pragma endregion

/// <summary>
/// Struct <c>DeviceBuffersKMH</c> defines buffers common to the CUDA kernels that manipulate kmer hash values for unmapped reads
/// </summary>
struct DeviceBuffersKMH
{
    CudaGlobalPtr<UINT64>   KMH;        // kmer hash values
    CudaGlobalPtr<UINT64>   S64;        // 64-bit "sketch bits" values
    UINT32                  stride;     // number of kmer hash values per Q sequence

    DeviceBuffersKMH()
    {
    }

    DeviceBuffersKMH( CudaGlobalAllocator* pCGA ) : KMH(pCGA), S64(pCGA), stride(0)
    {
    }
};

/// <summary>
/// Struct <c>DeviceBuffersJ</c> defines buffers common to the CUDA kernels that join J lists (Df lists) for paired-end alignment
/// </summary>
struct DeviceBuffersJ
{
    CudaGlobalPtr<UINT64>   oJ;         // per-seed J-list offsets
    CudaGlobalPtr<UINT32>   nJ;         // per-seed J-list sizes
    CudaGlobalPtr<UINT32>   cnJ;        // cumulative per-seed J-list sizes
    CudaGlobalPtr<UINT64>   D;          // D list
    UINT32                  celJ;       // number of elements in the oJ, nJ, and cnJ lists (not counting padding)
    UINT32                  totalD;     // total number of D values

    DeviceBuffersJ()
    {
    }

    DeviceBuffersJ( CudaGlobalAllocator* pCGA ) : oJ(pCGA), nJ(pCGA), cnJ(pCGA), D(pCGA), celJ(0), totalD(0)
    {
    }
};

/// <summary>
/// Struct <c>AlignmentKernelParameters</c> defines constant parameters common to the CUDA kernels that do nongapped and gapped alignment.
/// </summary>
struct AlignmentKernelParameters
{
    INT32   Mr;             // maximum number of R symbols for a scoring matrix
    INT32   wcgsc;          // worst-case gap space count
    UINT32  celBRLEAperQ;   // worst-case BRLEA size for one Q sequence
    UINT32  celSMperQ;      // number of scoring-matrix cells for one Q sequence
    UINT32  bw;             // width of the computed scoring-matrix band
    INT32   seedsPerQ;      // number of seed-and-extend seeds per maximum-length Q sequence
};

/// <summary>
/// Class <c>DeviceBuffersBase</c> defines buffers and parameters common to the CUDA kernels that do nongapped and gapped alignment.
/// </summary>
class DeviceBuffersBase
{
    public:
        CudaGlobalPtr<UINT64>       Dc;     // candidates for alignment
        CudaGlobalPtr<UINT64>       Dx;     // non-candidates for alignment
        CudaGlobalPtr<UINT64>       Dm;     // successfully-mapped candidates
        CudaGlobalPtr<UINT64>       Du;     // unmapped or previously aligned candidates
        CudaGlobalPtr<UINT32>       Qu;     // QIDs for J lists
        CudaGlobalPtr<UINT32>       Ru;     // subId bits for J lists
        CudaGlobalPtr<UINT32>       Rx;     // combined subId bits for pairs
        CudaGlobalPtr<UINT64>       Diter;  // J-list D values for one iteration
        AlignmentKernelParameters   AKP;    // parameters for CUDA kernels
        UINT32                      nDm1;   // number of mappings for paired D values (Dc list)
        UINT32                      nDm2;   // number of mappings for unpaired D values (Dx list)
        RiFlags                     flagRi; // indicates which buffer to use for loading interleaved R sequence data (see baseLoadRi implementation)

    public:
        DeviceBuffersBase( CudaGlobalAllocator* pCGA );
        virtual ~DeviceBuffersBase();
        virtual void InitKernelParameters( QBatch* _pqb ) = 0;
        void Reset( void );
};

/// <summary>
/// Class <c>DeviceBuffersN</c> defines buffers and parameters constant parameters common to the CUDA kernels that do nongapped alignment.
/// </summary>
class DeviceBuffersN : public DeviceBuffersBase
{
    public:
        CudaGlobalPtr<UINT64>   Dl;             // "leftover" D values (failed nongapped alignments)

    public:
        DeviceBuffersN( CudaGlobalAllocator* pCGA );
        virtual ~DeviceBuffersN( void );
        void InitKernelParameters( QBatch* _pqb );
};

/// <summary>
/// Class <c>DeviceBuffersG</c> defines buffers and parameters constant parameters common to the CUDA kernels that do gapped alignment.
/// </summary>
class DeviceBuffersG : public DeviceBuffersBase
{
    public:
        CudaGlobalPtr<INT16>    VmaxDc;         // Vmax values for Dc buffer
        CudaGlobalPtr<INT16>    VmaxDx;         // Vmax values for Dx buffer
        CudaGlobalPtr<INT16>    VmaxDm;         // Vmax values for Dx buffer
        CudaGlobalPtr<UINT32>   BRLEA;          // BRLEAs for mapped Q sequences
        CudaGlobalPtr<UINT64>   Di;             // isolated candidates
        CudaGlobalPtr<UINT32>   Cu;             // seed coverage for unmapped candidates
        UINT32                  nDc1;           // (used for candidates for windowed gapped alignment)

    public:
        DeviceBuffersG( CudaGlobalAllocator* pCGA );
        virtual ~DeviceBuffersG( void );
        void InitKernelParameters( QBatch* _pqb );
};

/// <summary>
/// Struct <c>DeviceBuffers</c> defines buffers common to multiple CUDA kernels in a QBatch instance.
/// </summary>
struct DeviceBuffers
{
    CudaGlobalPtr<Qwarp>    Qw;     // Qwarp buffer
    CudaGlobalPtr<UINT64>   Qi;     // interleaved Q sequence buffer
    CudaGlobalPtr<UINT64>   Ri;     // interleaved R sequence data
    UINT32                  nQ;     // number of Q sequences in the Qwarp buffer

    DeviceBuffers( CudaGlobalAllocator* pCGA ) : Qw(pCGA), Qi(pCGA), Ri(pCGA), nQ(0)
    {
    }
};
