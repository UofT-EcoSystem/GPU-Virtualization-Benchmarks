/*
  InputFileGroup.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __InputFileGroup__

#pragma pack(push, 1)
struct MetadataRowHeader    // Q-sequence metadata row header (used in AriocE, AriocU, and AriocP)
{
    UINT64  sqId;
    UINT8   byteVal;
    INT16   cb;
};
#pragma pack(pop)

/// <summary>
/// Class <c>InputFileGroup</c> wraps a group of input files.
/// </summary>
class InputFileGroup
{
    public:
        static const INT64          DefaultReadIdFrom = 0;
        static const INT64          DefaultReadIdTo = _I64_MAX;
        static const INT32          MaxPartsPerGPU = 100;

        struct FileInfo
        {
            INT16                   Index;              // 0-based index into a list of FileInfo instances
            char*                   Filespec;           // pointer to a file specification string
            INT32                   SrcId;              // data source ID value
            UINT8                   SubId;              // subunit ID value
            UINT32                  MateId;             // mate ID value (0: missing; >0: valid value)
            INT64                   ReadIdFrom;         // user-specified readId range
            INT64                   ReadIdTo;
            INT16                   EstimatedNmax;      // estimated maximum read length
            INT16                   EstimatedNmin;      // estimated minimum read length
            INT16                   EstimatedNavg;      // estimated average read length
            UINT64                  EstimatedSqIdFrom;  // estimated range of SqId values
            UINT64                  EstimatedSqIdTo;
            bool                    IsPaired;           // flag set if the file contains paired-end reads
            char*                   URI;                // pointer to a URI for the file (for the corresponding @SQ row in a SAM file)
            WinGlobalPtr<char>      QmdRG;              // read-group ID pattern in FASTQ deflines
            INT32                   RGOrdinal;          // per-file read-group ordinal
            WinGlobalPtr<INT64>     Cutpoints;          // starting file positions for partitioned file access
            WinGlobalPtr<UINT64>    SqIds;              // sqId at each cutpoint

            FileInfo() : Index(-1),
                         Filespec(NULL),
                         SrcId(0), SubId(0), MateId(0), ReadIdFrom(InputFileGroup::DefaultReadIdFrom), ReadIdTo(InputFileGroup::DefaultReadIdTo),
                         EstimatedNmax(0), EstimatedNmin(0), EstimatedNavg(0), EstimatedSqIdFrom(0), EstimatedSqIdTo(0),
                         IsPaired(false), URI(NULL), RGOrdinal(-1)
            {
            }
        };

#pragma pack(push, 2)
        struct DataRowHeader
        {
            INT64 sqId;         // read sequence ID
            INT16 N;            // number of symbols in read
            INT16 cb;           // number of bytes in encoded symbol data
        };

        struct InputFilePartitionInfo
        {
            INT16 iFile;        // index of input file
            INT16 iPart;        // index of partition in input file

            static void Swap( InputFilePartitionInfo& ifpi1, InputFilePartitionInfo& ifpi2 )
            {
                InputFilePartitionInfo ifpix = ifpi1;
                ifpi1 = ifpi2;
                ifpi2 = ifpix;
            }
        };
#pragma pack(pop)

    private:
// TODO: CHOP WHEN DEBUGGED        INT32                                   m_partsPerFile; // number of partitions per file
// TODO: CHOP WHEN DEBUGGED        UINT32                                  m_partCount;    // number of file partitions to be processed
        volatile INT32                          m_partSeqNo;    // 0-based partition sequence number
        WinGlobalPtr<InputFilePartitionInfo>    m_ifpi;         // list of all input files and partitions

    public:
        WinGlobalPtr<char>      FilePath;       // base file path
        WinGlobalPtr<char>      UriPath;        // base URI path (for @SQ rows in a SAM file)
        WinGlobalPtr<char>      FilespecBuf;    // buffer that contains the file-specification strings
        WinGlobalPtr<char>      UriBuf;         // buffer that contains URI strings
        WinGlobalPtr<FileInfo>  InputFile;      // a list of InputFileGroup::FileInfo structs
        bool                    HasPairs;       // flag set if input files contain paired-end reads

    private:
        static int fileInfoComparer( const void* a, const void* b );
        void findPartitionCutpoints( FileInfo* pfi, INT16 nPartitions );
        void estimateReadLengths( INT16 iFile );
        void estimateSqIdRange( INT16 iFile );

    public:
        InputFileGroup( const char* filePath, const char* uriPath );
        virtual ~InputFileGroup( void );
        InputFileGroup( const InputFileGroup& other );
        void Init( INT32 nFiles );
        void Append( const char* fileSpec, INT32 srcId, UINT8 subId, INT64 readIdFrom, INT64 readIdTo, INT32 mateId, const char* uriPathForFile );
        void CleanMateIds( void );
        void ValidateSubunitIds( void );
        UINT64 Sniff( INT16 nGPUs, UINT32 readsPerBatch );
        FileInfo* GetNextPartition( INT32& iPart );
        INT16 GetEstimatedNmax( void );
};
