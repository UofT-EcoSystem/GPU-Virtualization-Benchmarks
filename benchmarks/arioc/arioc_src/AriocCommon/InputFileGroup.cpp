/*
  InputFileGroup.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// [public] constructor (const char*, const char*)
InputFileGroup::InputFileGroup( const char* filePath, const char* uriPath ) : m_partSeqNo(-1), HasPairs(false)
{
    // ensure that the base file path ends with a trailing separator
    if( (filePath != NULL) && *filePath )
    {
        this->FilePath.Realloc( strlen(filePath)+2, true );
        strcpy_s( this->FilePath.p, this->FilePath.cb, filePath );
        RaiiDirectory::AppendTrailingPathSeparator( this->FilePath.p );
    }

    // ensure that the base URI path ends with a trailing separator
    if( (uriPath != NULL) && *uriPath )
    {
        this->UriPath.Realloc( strlen(uriPath)+2, true );
        strcpy_s( this->UriPath.p, this->UriPath.cb, uriPath );
        RaiiDirectory::AppendTrailingPathSeparator( this->UriPath.p );
    }
}

/// [public] copy constructor
InputFileGroup::InputFileGroup( const InputFileGroup& other ) : FilePath(other.FilePath),
                                                                UriPath(other.UriPath),
                                                                FilespecBuf(other.FilespecBuf.Count,false),
                                                                InputFile(other.InputFile.Count,false)
{
    // copy the file specification strings
    memcpy_s( this->FilespecBuf.p, this->FilespecBuf.cb, other.FilespecBuf.p, other.FilespecBuf.cb );
    this->FilespecBuf.n = other.FilespecBuf.n;

    // copy the input file info
    memcpy_s( this->InputFile.p, this->InputFile.cb, other.InputFile.p, other.InputFile.cb );
    this->InputFile.n = other.InputFile.n;
    this->HasPairs = other.HasPairs;
    m_partSeqNo = other.m_partSeqNo;

    this->m_ifpi.Realloc( other.m_ifpi.Count, false );
    memcpy_s( this->m_ifpi.p, this->m_ifpi.cb, other.m_ifpi.p, other.m_ifpi.cb );
}

/// [public] destructor
InputFileGroup::~InputFileGroup()
{
    InputFile.Free();
}
#pragma endregion

#pragma region private methods
/// [private static] method fileInfoComparer
int InputFileGroup::fileInfoComparer( const void* a, const void* b )
{
    const FileInfo* pa = reinterpret_cast<const FileInfo*>(a);
    const FileInfo* pb = reinterpret_cast<const FileInfo*>(b);

    // order by subId, mateId, Filespec
    int rval = pa->SubId - pb->SubId;
    if( rval == 0 )
    {
        rval = pa->MateId - pb->MateId;
        if( rval == 0 )
            rval = _strcmpi( pa->Filespec, pb->Filespec );
    }

    return rval;
}

/// [private] method findPartitionCutpoints
void InputFileGroup::findPartitionCutpoints( FileInfo* pfi, INT16 nPartitions )
{
    /* Each "cutpoint" is the file offset of the first data row in a partition in the specified file. */

    // open the input file for read-only access
    RaiiFile f( pfi->Filespec, true );

    // compute the expected partition size
    const INT64 cbFile = f.FileSize();
    const INT64 cbPartition = cbFile / nPartitions;

    // initialize the cutpoint list; we save the file size in the last element so that each partition range is defined by a pair of adjacent cutpoints
    pfi->Cutpoints.Realloc( nPartitions+1, false );
    memset( pfi->Cutpoints.p, 0xFF, pfi->Cutpoints.cb );    // initialize all cutpoints to -1 (0xFFFFFFFFFFFFFFFF)
    pfi->Cutpoints.n = 0;
    pfi->Cutpoints.p[nPartitions] = cbFile;

    // we sample the input file 16KB at a time; this is more than sufficient to contain a row with a maximum read size of 16K symbols
    WinGlobalPtr<char> sampleBuf( 16*1024, false );

    // compute the number of sqIds per partition
    const UINT64 nSqIds = ((pfi->EstimatedSqIdTo - pfi->EstimatedSqIdFrom) / AriocDS::SqId::AdjacentSqIdDiff) + 1;
    const UINT64 nSqIdsPerPartition = blockdiv(nSqIds,nPartitions);

    CDPrint( cdpCDf, "%s: %s: 0x%016llx-0x%016llx (%llu reads)", __FUNCTION__, pfi->Filespec, pfi->EstimatedSqIdFrom, pfi->EstimatedSqIdTo, nSqIds );

    // initialize the list of sqIds for each partition
    pfi->SqIds.Realloc( nPartitions+1, true );
    pfi->SqIds.n = nPartitions;
    for( INT16 n=0; n<nPartitions; ++n )
        pfi->SqIds.p[n] = pfi->EstimatedSqIdFrom + (n * (AriocDS::SqId::AdjacentSqIdDiff * nSqIdsPerPartition));
    pfi->SqIds.p[nPartitions] = pfi->EstimatedSqIdTo;

    INT64 fpSample = 0;
    INT16 t = 0;
    while( t < nPartitions )
    {
        /* Look for the sqId in the file:
            - Sample the file using the estimated partition size (i.e., file size / number of partitions).
            - If the sqId is in the sample, we're done; otherwise, resample until we do find the sqId.
        */

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s: t=%d reads %d bytes at fpSample=%lld", __FUNCTION__, t, sampleBuf.cb, fpSample );
#endif

        // fill the sample buffer
        f.Seek( fpSample, SEEK_SET );
        INT64 ofsLimit = f.Read( sampleBuf.p, sampleBuf.cb ) - sizeof(DataRowHeader);
        DataRowHeader* pdrhLimit = reinterpret_cast<DataRowHeader*>(sampleBuf.p+ofsLimit);

        // look for the DataRowHeader that contains the first sqId in the nth partition
        INT64 ofs = 0;
        INT64 ofsIncr = 1;
        while( ofs < ofsLimit )
        {
            DataRowHeader* pdrh = reinterpret_cast<DataRowHeader*>(sampleBuf.p + ofs);

            /* Find the first DataRowHeader; we assume we have one if
                - the subId field of the sqId is correct
                - the N and cb fields are greater than 0
                - the address of the next expected DataRowHeader lies within the sample buffer
                - the cb field is the relative offset of another sqId
                - the difference between two consecutive sqIds is 1
            */
            if( (AriocDS::SqId::GetSubId( pdrh->sqId ) == pfi->SubId) && (pdrh->N > 0) && (pdrh->cb > 0) )
            {
                // find the subsequent DataRowHeader
                DataRowHeader* pdrh2 = reinterpret_cast<DataRowHeader*>(reinterpret_cast<char*>(pdrh+1) + pdrh->cb);

                // if it's a valid data row header...
                bool isValidDRH = (pdrh2 < pdrhLimit) && AriocDS::SqId::IsAdjacent( pdrh2->sqId, pdrh->sqId ) && (pdrh2->N > 0) && (pdrh2->cb > 0);
                if( !isValidDRH )
                    isValidDRH = (pdrh2 >= pdrhLimit) && f.EndOfFile();

                if( isValidDRH )
                {
                    if( pdrh->sqId == static_cast<INT64>(pfi->SqIds.p[t]) )
                    {
                        // save the cutpoint
                        pfi->Cutpoints.p[t] = fpSample + ofs;
                        break;
                    }

                    if( pdrh->sqId > static_cast<INT64>(pfi->SqIds.p[t]) )
                    {
                        // back up and resample
                        fpSample -= (sampleBuf.cb - sizeof(DataRowHeader));
                        break;
                    }

                    // look at the next sqId in the buffer
                    ofsIncr = sizeof(DataRowHeader) + pdrh->cb;
                }
            }

            // advance to the next offset at which a valid DataRowHeader might be found
            ofs += ofsIncr;
        }

        if( pfi->Cutpoints.p[t] >= 0 )
        {
            // we found the sqId for the t'th partition
            fpSample = pfi->Cutpoints.p[t] + cbPartition;
            ++t;
        }
        else
        {
            // if the sqId lies after the current sample, prepare to resample
            if( ofs >= ofsLimit )
                fpSample += (ofs-ofsIncr);
        }    
    }

    pfi->Cutpoints.n = t;

#ifdef _DEBUG
    for( INT32 i=0; i<(nPartitions+1); ++i )
        CDPrint( cdpCD0, "%s: %s Cutpoints.p[%d] = %lld", __FUNCTION__, pfi->Filespec, i, pfi->Cutpoints.p[i] );
    CDPrint( cdpCD0, __FUNCTION__ );
#endif
}

/// [private] method estimateReadLengths
void InputFileGroup::estimateReadLengths( INT16 iFile )
{
    INT16 nReads = 0;
    INT32 Ntotal = 0;
    FileInfo* pfi = this->InputFile.p + iFile;
    pfi->EstimatedNmax = SHRT_MIN;
    pfi->EstimatedNmin = SHRT_MAX;

    // open the input file for read-only access
    RaiiFile f( pfi->Filespec, true );

    // sample the first 10 partitions in the file
    UINT32 nSampledPartitions = min2( pfi->Cutpoints.n, 10 );
    for( UINT32 p=0; p<nSampledPartitions; ++p )
    {
        // set the file pointer to the first read in the pth partition
        INT64 fp = pfi->Cutpoints.p[p];
        f.Seek( fp, SEEK_SET );

        // get the first 256 read lengths in the pth partition of the file
        INT64 fpMax = pfi->Cutpoints.p[p+1];
        for( INT16 r=0; (r<256)&&(fp<fpMax); ++r )
        {
            // get the "row header" for the rth read
            DataRowHeader drh;
            f.Read( &drh, sizeof drh );

            // track the maximum, minimum, and total read lengths and count the number of reads
            pfi->EstimatedNmax = max2(pfi->EstimatedNmax, drh.N);
            pfi->EstimatedNmin = min2(pfi->EstimatedNmin, drh.N);
            Ntotal += drh.N;
            ++nReads;

            // point to the next "row header"
            f.Seek( drh.cb, SEEK_CUR );
        }
    }

    // compute the average for the sampled reads
    pfi->EstimatedNavg = Ntotal / nReads;
}

/// [private] method estimateSqIdRange
void InputFileGroup::estimateSqIdRange( INT16 iFile )
{
    FileInfo* pfi = this->InputFile.p + iFile;

    // open the input file for read-only access
    RaiiFile f( pfi->Filespec, true );

    // set the file pointer to the first read in the file
    f.Seek( 0, SEEK_SET );

    // get the "row header" for the first read
    DataRowHeader drh;
    f.Read( &drh, sizeof drh );
    pfi->EstimatedSqIdFrom = drh.sqId;

    // create a mask for the invariant portion of the sqId (srcId and subId only)
    UINT64 sqIdKey = drh.sqId & AriocDS::SqId::MaskDataSource;

    // set the file pointer near the end of the file
    const INT64 cbFile = f.FileSize();
    INT64 ofs = cbFile - static_cast<INT64>((sizeof drh)+sizeof(UINT64));
    f.Seek( ofs, SEEK_SET );

    // scan backward from the end of the file
    pfi->EstimatedSqIdTo = 0;
    while( ofs >= 0 )
    {
        // look for a DataRowHeader at the current file position
        f.Read( &drh, sizeof drh );

        // if the DataRowHeader fields do not look like they represent a valid DataRowHeader...
        if( ((drh.sqId & AriocDS::SqId::MaskDataSource) != sqIdKey) ||
            (static_cast<INT16>(blockdiv(drh.N,21)*sizeof(UINT64)) != drh.cb) ||
            ((ofs+static_cast<INT64>(sizeof drh) + drh.cb) != cbFile) )
        {
            // back up one byte in the file and try again
            f.Seek( --ofs, SEEK_SET );
            continue;
        }

        /* At this point the DataRowHeader should contain the last sqId in the file. */
        pfi->EstimatedSqIdTo = drh.sqId;
        break;
    }

    if( pfi->EstimatedSqIdTo < sqIdKey )
        throw new ApplicationException( __FILE__, __LINE__, "unable to find SqId range (sqIdKey=0x%016llx)", sqIdKey );
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Initializes the current <c>InputFileGroup</c> instance
/// </summary>
void InputFileGroup::Init( INT32 nFiles )
{
    // allocate and zero the file specification buffer
    FilespecBuf.Realloc( nFiles*FILENAME_MAX, true );

    // allocate and zero the file info buffer
    InputFile.New( nFiles, true );

    // initialize each FileInfo struct
    for( INT16 iInputFile=0; iInputFile<static_cast<INT16>(nFiles); ++iInputFile )
    {
        InputFile.p[iInputFile].Index = iInputFile;     // associate an integer index value with the InputFile instance
        InputFile.p[iInputFile].RGOrdinal = -1;         // initialize the read group ordinal associated with the InputFile instance
    }

    // allocate a buffer to contain a URI string for each input file
    if( this->UriPath.cb )
        UriBuf.Realloc( nFiles*(this->UriPath.cb+FILENAME_MAX), true );
}

/// <summary>
/// Adds input file information to the list maintained in this <c>InputFileGroup</c> object
/// </summary>
void InputFileGroup::Append( const char* fileSpec, INT32 srcId, UINT8 subId, INT64 readIdFrom, INT64 readIdTo, INT32 mateId, const char* uriPathForFile )
{
    // build the complete input file specification
    char fullFileSpec[FILENAME_MAX] = { 0 };
    if( this->FilePath.cb )
        strcpy_s( fullFileSpec, sizeof fullFileSpec, this->FilePath.p );
    strcat_s( fullFileSpec, sizeof fullFileSpec, fileSpec );

    // append to the list of pointers to the file specification strings
    InputFile.p[InputFile.n].Filespec = FilespecBuf.p + FilespecBuf.n;                          // InputFile.n counts the number of file specifications; FilespecBuf.n counts the bytes used in the string buffer
    strcpy_s( InputFile.p[InputFile.n].Filespec, FilespecBuf.cb-FilespecBuf.n, fullFileSpec );  // copy the file specification
    FilespecBuf.n += static_cast<UINT32>(strlen(InputFile.p[InputFile.n].Filespec) + 1);        // update the offset of the next file-specification string in the buffer

    // look for a URI path
    char* pUriPathForFile = uriPathForFile ? const_cast<char*>(uriPathForFile) :    // look for the specified URI path for the file
                            this->UriPath.cb ? this->UriPath.p :                    // look for a default URI path for all files
                            NULL;

    if( pUriPathForFile )
    {
        // build the complete URI path
        char* puri = UriBuf.p + UriBuf.n;
        UINT32 cbBuf = static_cast<UINT32>(UriBuf.cb - UriBuf.n);
        strcpy_s( puri, cbBuf, pUriPathForFile );
        RaiiDirectory::ChopTrailingPathSeparator( puri );
        strcat_s( puri, cbBuf, "/" );                       // assume that a URI uses forward slashes
        strcat_s( puri, cbBuf, fileSpec );

        // append to the list of pointers to the URI strings
        InputFile.p[InputFile.n].URI = puri;                // UriBuf.n counts the bytes used in the string buffer
        UriBuf.n += static_cast<UINT32>(strlen(puri) + 1);  // update the offset of the next URI string in the buffer
    }

    // append the srcId, subId, and readId range values
    InputFile.p[InputFile.n].SrcId = srcId;
    InputFile.p[InputFile.n].SubId = subId;
    InputFile.p[InputFile.n].ReadIdFrom = readIdFrom;
    InputFile.p[InputFile.n].ReadIdTo = readIdTo;

    // append the mateId value
    InputFile.p[InputFile.n].MateId = (mateId > 0) ? 1 : 0;

    // count the number of file specifications in the list
    InputFile.n++ ;
}

/// <summary>
/// Ensures that the pair IDs specified in the application configuration file are consistent
/// </summary>
void InputFileGroup::CleanMateIds()
{
    // reorder the list by subId and mateId
    qsort( InputFile.p, InputFile.n, sizeof(FileInfo), fileInfoComparer );

    // set flags to indicate which files are paired
    UINT32 n = 0;
    while( n < (InputFile.n-1) )
    {
        /* a pair of files contains paired reads if ...
            - adjacent files have the same subId, AND
            - the second adjacent file's mateId flag is set
        */
        if( (InputFile.p[n].SubId == InputFile.p[n+1].SubId) && InputFile.p[n+1].MateId )
        {
            // set the IsPaired flags on the files
            InputFile.p[n].IsPaired = InputFile.p[n+1].IsPaired = true;

            // normalize the mate IDs; the list is sorted by subunit ID and mate ID, so we rely on this order to set the mate IDs to 0 and 1 respectively
            InputFile.p[n].MateId = 0;
            InputFile.p[n+1].MateId = 1;

            // set a flag to indicate that paired reads exist in at least one pair of input files
            this->HasPairs = true;

            n += 2;
        }

        else
            n++ ;
    }

    // sanity check: if at least one pair exists, all of the files must be paired
    if( HasPairs )
    {
        for( UINT32 n=0; n<InputFile.n; ++n )
        {
            if( !InputFile.p[n].IsPaired )
                throw new ApplicationException( __FILE__, __LINE__, "non-paired Q sequence file %s cannot be processed along with paired sequence files", InputFile.p[n].Filespec );
        }
    }
}

/// <summary>
/// For each input file:
/// <list type="bullet">
///    <item><description>Verifies that the subunit ID specified in the application configuration file matches the subunit ID embedded in the file data</description></item>
///    <item><description>For paired-end data, verifies that the pair ID specified in the configuration file matches the pair ID flag in the file data</description></item>
/// </summary>
void InputFileGroup::ValidateSubunitIds()
{
    // for each file, sample the first sqId to verify that the encoded subunit ID and pair ID are valid
    for( UINT32 n=0; n<InputFile.n; ++n )
    {
        RaiiFile fileX( InputFile.p[n].Filespec, true );        // true: read only

        // ensure that there is at least one sqId in the file
        if( fileX.FileSize() < static_cast<INT64>(sizeof(INT64)) )
            throw new ApplicationException( __FILE__, __LINE__, "unable to validate subunit IDs for file %s: file contains no data", fileX.FileSpec.p );

        // read the first sqId
        INT64 sqId;
        fileX.Read( &sqId, sizeof sqId );

        // validate the srcId value
        if( AriocDS::SqId::GetSrcId( sqId ) != static_cast<UINT32>(InputFile.p[n].SrcId) )
            throw new ApplicationException( __FILE__, __LINE__, "inconsistent value for attribute srcId= for %s: expected %u specified %d", fileX.FileSpec.p, AriocDS::SqId::GetSrcId( sqId ), InputFile.p[n].SrcId );

        // validate the subId value
        if( AriocDS::SqId::GetSubId(sqId) != InputFile.p[n].SubId )
            throw new ApplicationException( __FILE__, __LINE__, "inconsistent value for attribute subId= for %s: expected %u specified %d", fileX.FileSpec.p, AriocDS::SqId::GetSubId(sqId), InputFile.p[n].SubId ); 

        // validate the mateId value
        if( this->HasPairs && (AriocDS::SqId::GetMateId(sqId) != InputFile.p[n].MateId) )

            throw new ApplicationException( __FILE__, __LINE__, "inconsistent paired-end data flag (verify file order in <paired> element): %s", fileX.FileSpec.p );
    }
}

/// <summary>
/// For each input file:
/// <list type="bullet">
///   <item><description>Determines cutpoints at which the file can be split into a specified number of equal-size partitions</description></item>
///   <item><description>Samples the first few reads in each partition to estimate the average read length</description></item>
/// </list>
/// </summary>
UINT64 InputFileGroup::Sniff( INT16 nGPUs, UINT32 readsPerBatch )
{
    // reset the input-file partition-info list
    m_ifpi.Free();

    UINT64 nTotalReads = 0;
    for( UINT32 iFile=0; iFile<InputFile.n; ++iFile )
    {
        estimateSqIdRange( iFile );

        /* The specified number of partitions is a bare minimum (e.g., the number of GPUs).  But we want each
            input file to be split into more than one partition per GPU if possible so as to more evenly
            distribute the work among the GPUs.
        */

        // AriocE assigns sqIds in ascending order, so we can compute the number of reads in the file
        UINT64 sqIdFrom = InputFile.p[iFile].EstimatedSqIdFrom;
        UINT64 sqIdTo = InputFile.p[iFile].EstimatedSqIdTo;
        UINT64 nReadsInFile = (AriocDS::SqId::GetReadId(sqIdTo) - AriocDS::SqId::GetReadId(sqIdFrom)) + 1;
        nTotalReads += nReadsInFile;

        // if the file contains a sufficient number of reads, split it into a reasonable number of partitions per GPU
        UINT32 nPartitions = 0;
        if( nReadsInFile > static_cast<UINT64>(readsPerBatch) )
        {
            UINT64 nReadsPerGPU = nReadsInFile / nGPUs;
            UINT32 nPartsPerGPU = static_cast<UINT32>(min2( blockdiv( nReadsPerGPU, readsPerBatch ), InputFileGroup::MaxPartsPerGPU ));
            nPartitions = nPartsPerGPU * nGPUs;
        }
        else
            nPartitions = 1;

        // find the partition cutpoints for the file
        findPartitionCutpoints( InputFile.p+iFile, nPartitions );

        /* Update the input-file partition-info list.
            - For unpaired reads, the list contains one entry for each partition in each input file.
            - For paired reads, the list contains one entry for each partition in the even-numbered
               input files only.
        */
        if( !this->HasPairs || ((iFile & 1) == 0) )
        {
            size_t cel = m_ifpi.Count + nPartitions;
            m_ifpi.Realloc( cel, false );
            UINT32 seqNo = m_ifpi.n;
            for( UINT32 iPart=0; iPart<nPartitions; ++iPart )
            {
                m_ifpi.p[seqNo].iFile = iFile;
                m_ifpi.p[seqNo].iPart = iPart;
                seqNo++;
            }
            m_ifpi.n = seqNo;
        }

        estimateReadLengths( iFile );

#if TODO_THIS_DOES_NOT_WORK
        // shuffle all but the final partition to avoid clusters of hard-to-map reads in presorted input
        Hash hasher;
        UINT32 nPartsToShuffle = m_ifpi.n - 1;
        for( UINT32 s=0; s<nPartsToShuffle; ++s )
        {
            // swap
            UINT32 h = hasher.ComputeH32( reinterpret_cast<UINT64>(m_ifpi.p+s) );
            UINT32 sx = h % nPartsToShuffle;
            InputFilePartitionInfo::Swap( m_ifpi.p[s], m_ifpi.p[sx] );
        }
#endif
    }

#if TODO_CHOP_WHEN_DEBUGGED
    // dump InputFileGroup info
    for( UINT32 iFile=0; iFile<InputFile.n; ++iFile )
    {
        UINT64 sqIdFrom = InputFile.p[iFile].EstimatedSqIdFrom;
        UINT64 sqIdTo = InputFile.p[iFile].EstimatedSqIdTo;
        CDPrint( cdpCD0, "%s: iFile=%d: %s, estimated sqId range 0x%016llx-0x%016llx", __FUNCTION__, iFile, InputFile.p[iFile].Filespec, sqIdFrom, sqIdTo );
        
        for( UINT32 n=0; n<=InputFile.p[iFile].Cutpoints.n; ++n )
            CDPrint( cdpCD0, "%s:  cutpoint %u at %lld", __FUNCTION__, n, InputFile.p[iFile].Cutpoints.p[n] );
    }
    CDPrint( cdpCD0, __FUNCTION__ );
#endif

    return nTotalReads;
}

/// <summary>
/// Returns the next file partition to be processed
/// </summary>
InputFileGroup::FileInfo* InputFileGroup::GetNextPartition( INT32& iPart )
{
    // advance the partition sequence number in a thread-safe manner
    UINT32 seqNo = InterlockedIncrement(reinterpret_cast<volatile UINT32*>(&m_partSeqNo));

    // return NULL if no partitions remain to be processed
    if( seqNo >= m_ifpi.n )
        return NULL;

    // return the index of the partition and a pointer to the FileInfo for the input file
    iPart = m_ifpi.p[seqNo].iPart;
    return InputFile.p + m_ifpi.p[seqNo].iFile;

#if TODO_CHOP_WHEN_THE_ABOVE_WORKS
    // return NULL if no partitions remain to be processed
    if( seqNo >= m_partCount )
        return NULL;

    // return the indexes of the input file and the partition within that file
    div_t qr = div( seqNo, m_partsPerFile );

    CDPrint( cdpCD4, "%s: returning InputFile %d iPart %d", __FUNCTION__, qr.quot, qr.rem );

    iPart = qr.rem;
    return InputFile.p + (this->HasPairs ? 2*qr.quot : qr.quot);
#endif
}

/// <summary>
/// Returns the estimated maximum N (Q sequence length) across all InputFileGroups.
/// </summary>
INT16 InputFileGroup::GetEstimatedNmax()
{
    INT16 rval = _I16_MIN;
    for( UINT32 n=0; n<InputFile.n; ++n )
    {
        if( InputFile.p[n].EstimatedNmax > rval )
            rval = InputFile.p[n].EstimatedNmax;
    }

    return rval;
}
#pragma endregion
