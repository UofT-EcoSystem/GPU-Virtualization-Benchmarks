/*
  baseQReader.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// constructor
baseQReader::baseQReader( INT16 partitionId ) : m_cbRead(0), m_nQ(0), m_iPart(partitionId)
{
    // initialize file info for Q-sequence metadata
    MFIm[0].ext = MFIm[1].ext = "sqm";    // row metadata
    MFIq[0].ext = MFIq[1].ext = "sqq";    // quality scores
}

/// [public] destructor
baseQReader::~baseQReader()
{
    // performance metrics
    AriocTaskUnitMetrics* ptum = AriocBase::GetTaskUnitMetrics( "baseQReader" );
    InterlockedExchangeAdd( &ptum->n.CandidateQ, m_nQ );
    InterlockedExchangeAdd( &ptum->cb.TotalBytes, m_cbRead );
}
#pragma endregion

#pragma region protected methods
/// <summary>
/// Opens a Q-sequence input file.
/// </summary>
INT64 baseQReader::openQfile( RaiiFile* pFileQ, QfileInfo* pqfi )
{
    // open the Q sequence input file
    pFileQ->OpenReadOnly( pqfi->pfi->Filespec );

    // sanity check
    if( pFileQ->FileSize() == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "file %s is empty", pFileQ->FileSpec.p );

    // initialize the file position
    pqfi->fp0 = pqfi->pfi->Cutpoints.p[m_iPart];
    pqfi->fpLimit = pqfi->pfi->Cutpoints.p[m_iPart+1];

    // seek to (and return) the initial file position
    pqfi->fp = pqfi->fp0;
    return pFileQ->Seek( pqfi->fp, SEEK_SET );
}

/// <summary>
/// Opens a metadata file.
/// </summary>
void baseQReader::openMfile( RaiiFile* pFileM, MfileInfo* pmfi, QfileInfo* pqfi )
{
    // build the metadata file specification
    char filespec[FILENAME_MAX];
    strcpy_s( filespec, FILENAME_MAX, pqfi->pfi->Filespec );    // $a21.sdf
    INT32 cb = static_cast<INT32>(strlen( filespec ));
    memcpy_s( filespec+(cb-7), 3, pmfi->ext, 3 );               // replace "a21" with "sqm" or "sqq"

    // open the file
    pFileM->OpenReadOnly( filespec );

    // estimate the initial file position for the specified partition in the metadata file
    INT64 cbM = pFileM->FileSize();
    INT64 initialFp = static_cast<INT64>( static_cast<double>(m_iPart) * cbM / pqfi->pfi->SqIds.Count );

    // find the initial file position in the metadata file
    pmfi->fp = baseQReader::sqIdHunt( pFileM, initialFp, pqfi->pfi->SqIds.p[m_iPart] );

    // find the limiting file position in the metadata file
    INT16 iPartNext = m_iPart + 1;
    if( iPartNext == static_cast<INT16>(pqfi->pfi->SqIds.n) )
        pmfi->fpLimit = cbM;
    else
    {
        INT64 finalFp = static_cast<INT64>( static_cast<double>(iPartNext) * cbM / pqfi->pfi->SqIds.Count );
        pmfi->fpLimit = baseQReader::sqIdHunt( pFileM, finalFp, pqfi->pfi->SqIds.p[iPartNext] );
    }

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: iPartNext=%d pmfi->fpLimit=%lld (%s)", __FUNCTION__, iPartNext, pmfi->fpLimit, pqfi->pfi->Filespec );
#endif

}

/// [private] method initQFB
void baseQReader::initQFB( WinGlobalPtr<UINT8>* pQFB, QfileInfo* pqfi, UINT32 nQwarpsPerBatch )
{
    // estimate the number of bytes in one encoded read; this estimate is exact if all reads have the same length
    pqfi->cbPerRead = sizeof( InputFileGroup::DataRowHeader ) + blockdiv( pqfi->pfi->EstimatedNmax, 21 )*sizeof( UINT64 );

    // compute the amount of data to load from each of the files, the idea being to load all of the reads in the batch
    INT64 cb = (nQwarpsPerBatch * CUDATHREADSPERWARP) * pqfi->cbPerRead;

    // allocate the input buffer and prepare for the initial call to loadFromQfile()
    pQFB->Realloc( cb, false );
    pQFB->n = static_cast<UINT32>(cb);
}

/// [protected] method loadFromQfile
UINT8* baseQReader::loadFromQfile( RaiiFile* pFileQ, QfileInfo* pqfi, WinGlobalPtr<UINT8>* pQFB, INT32 cb )
{
    /* The first call to this method (to initialize the buffer) should have pQFB->n = pQFB->cb. */

    // if all of the requested bytes are not in the buffer...
    if( (pQFB->n+cb) > pQFB->cb )
    {
        m_hrt.Restart();

        // move all remaining bytes to the start of the buffer
        size_t cbx = pQFB->cb - pQFB->n;
        memmove( pQFB->p, pQFB->p+pQFB->n, cbx );
        pQFB->n = 0;

        // fill the remaining bytes in the buffer
        INT64 cbReadExpected = min2( static_cast<INT64>(pQFB->cb-cbx), pqfi->fpLimit-pqfi->fp );
        INT64 cbReadActual = pFileQ->Read( pQFB->p+cbx, cbReadExpected );
        if( cbReadActual < cbReadExpected )
            throw new ApplicationException( __FILE__, __LINE__, "unexpected end of file near %lld in %s", pqfi->fp, pFileQ->FileSpec.p );

        // update the local copy of the file pointer
        pqfi->fp += cbReadActual;

        // track the number of usable bytes in the buffer
        pqfi->cbInBuf = static_cast<UINT32>(cbx + cbReadActual);

        // performance metrics
        InterlockedExchangeAdd(&AriocBase::aam.ms.LoadQ, m_hrt.GetElapsed(false));
    }

    // return a pointer to the requested bytes
    UINT8* rval = pQFB->p + pQFB->n;

    // update the current offset within the buffer
    pQFB->n += cb;

    // set a flag if the end of the input data has been reached (i.e. a subsequent call to this method will fail)
    if( (pqfi->fp >= pqfi->fpLimit) && (pQFB->n >= pqfi->cbInBuf) )
        pqfi->eod = true;

    // return a pointer to the data bytes
    return rval;
}

/// [protected] method copyToQwarp
void baseQReader::copyToQwarp( Qwarp* pQw, INT16 iq, UINT64* const pQiBuffer, InputFileGroup::DataRowHeader* pdrh, UINT64* pA21 )
{
    // save the sqId and sequence length for the iq'th Q sequence
    pQw->sqId[iq] = pdrh->sqId;
    pQw->N[iq] = pdrh->N;
    pQw->Nmax = max2( pQw->Nmax, pdrh->N );

    /* Interleave the Q sequence data:
        - Each "column" of data contains the 64-bit encoded values for one Q sequence.
        - Since there are 32 columns of data, each of which is 8 bytes wide, each "row" of interleaved data is aligned on a 32*8=256 byte
            boundary, so the data can be accessed using coalesced reads in a CUDA kernel.
    */
    UINT64* pFrom = pA21;
    UINT64* pTo = pQiBuffer + pQw->ofsQi + iq;
    for( size_t n=0; n<(pdrh->cb/sizeof(UINT64)); ++n )
    {
        *pTo = *(pFrom++);          // copy one 64-bit value
        pTo += CUDATHREADSPERWARP;  // interleave
    }
}

/// [protected] static method scanBufferForSqId
INT32 baseQReader::scanBufferForSqId( UINT64 _sqId, WinGlobalPtr<UINT8>* _pbuf, INT64* _pofsFrom, INT64* _pofsTo )
{
    // initialize the output parameters
    *_pofsFrom = *_pofsTo = -1;
    INT32 nSqIdsInBuffer = 0;

    // isolate the invariant portion of sqId (srcId and subId only)
    INT64 sqIdKey = _sqId & AriocDS::SqId::MaskDataSource;

    // we can only scan up to the point where a complete MetadataRowHeader and sqId can fit in the buffer
    UINT32 ofsLimit = _pbuf->n - (sizeof(MetadataRowHeader) + sizeof(_sqId));

    // find the first sqId in the buffer
    UINT32 ofs = 0;
    MetadataRowHeader* pmrhTest;
    while( ofs < ofsLimit )
    {
        bool bFound = false;
        pmrhTest = reinterpret_cast<MetadataRowHeader*>(_pbuf->p + ofs);

        // we have tentatively found a MetadataRowHeader if the invariant portion of the sqId is valid
        if( sqIdKey == static_cast<INT64>(pmrhTest->sqId & AriocDS::SqId::MaskDataSource) )
        {
            // if we have a valid sqId, then it should be followed either by another sqId or by the end of the buffer
            UINT8* pNext = reinterpret_cast<UINT8*>(pmrhTest+1) + pmrhTest->cb;
            if( pNext == (_pbuf->p + _pbuf->n) )
                bFound = true;                          // end of the buffer
            else
            {
                if( pNext <= (_pbuf->p + ofsLimit) )
                {
                    MetadataRowHeader* pmrhNext = reinterpret_cast<MetadataRowHeader*>(pNext);
                    if( sqIdKey == static_cast<INT64>(pmrhNext->sqId & AriocDS::SqId::MaskDataSource) )
                        bFound = true;
                }
            }

            if( bFound )
            {
                ++nSqIdsInBuffer;

                if( *_pofsFrom < 0 )
                {
                    // save the offset of the first MetadataRowHeader discovered in the buffer
                    *_pofsFrom = ofs;
                }

                // save the offset of the last MetadataRowHeader discovered in the buffer
                *_pofsTo = ofs;

                // we're finished if we have found the specified sqId
                if( pmrhTest->sqId == _sqId )
                    break;
            }

            /* At this point we assume that we have happened to hit a metadata byte pattern that mimics a sqId,
                so we need to keep iterating. */
        }

        // move to the next byte in the buffer
        if( bFound )
            ofs += sizeof(MetadataRowHeader) + pmrhTest->cb;
        else
            ++ofs;
    }

    return nSqIdsInBuffer;
}
#pragma endregion

/// <summary>
/// Finds the specified sqId in the specified metadata file.
/// </summary>
INT64 baseQReader::sqIdHunt( RaiiFile* pFileM, INT64 initialFp, UINT64 sqId )
{
    // allocate a temporary buffer
    WinGlobalPtr<UINT8> buf( SQIDHUNTBUFSIZE, true );

    // seek to the initial position in the metadata file
    INT64 cbFile = pFileM->FileSize();
    initialFp = min2( initialFp, cbFile-static_cast<INT64>(buf.cb) );
    INT64 fp = max2( initialFp, 0 );

    // track the range of file positions that remain to be examined
    INT64 minFp = 0;
    INT64 maxFp = cbFile;

    do
    {
        // fill a buffer with metadata rows
        if( fp != pFileM->Seek( fp, SEEK_SET ) )
            throw new ApplicationException( __FILE__, __LINE__, "%s: error seeking position %lld in %s", __FUNCTION__, fp, pFileM->FileSpec.p );

        buf.n = static_cast<UINT32>(pFileM->Read( buf.p, buf.cb ));

        INT64 ofsFrom;
        INT64 ofsTo;
        INT32 nSqIdsInBuffer = scanBufferForSqId( sqId, &buf, &ofsFrom, &ofsTo );

#if TODO_CHOP
        CDPrint( cdpCD0, "%s: scanBufferForSqId( 0x%016llx... ), fp=%lld returns ofsFrom=%lld ofsTo=%lld",
                            __FUNCTION__, sqId, fp, ofsFrom, ofsTo );
#endif

        if( ofsTo < 0 )
            throw new ApplicationException( __FILE__, __LINE__, "metadata sqId hunt failed for 0x%016llx in %s", sqId, pFileM->FileSpec.p );

        /* At this point we have found at least one sqId in the buffer. */

        // if the specified SqId is in the buffer, it is referenced by the "to" offset returned by scanBufferForSqId()
        MetadataRowHeader* pmrhTo = reinterpret_cast<MetadataRowHeader*>(buf.p + ofsTo);
        if( pmrhTo->sqId == sqId )
        {
            // return the file pointer for the metadata 
            fp += ofsTo;
            break;
        }

        // estimate the number of bytes per metadata row
        MetadataRowHeader* pmrhFrom = reinterpret_cast<MetadataRowHeader*>(buf.p + ofsFrom);
        INT64 cbPerRow = (ofsTo - ofsFrom) / nSqIdsInBuffer;

        // if the last identified sqId is smaller than the specified sqId...
        if( pmrhTo->sqId < sqId )
        {
            // set a minimum value for the file position
            minFp = fp + ofsTo + sizeof(MetadataRowHeader) + pmrhTo->cb;
            

            INT64 nRowsToSkip = (AriocDS::SqId::GetReadId(sqId) - AriocDS::SqId::GetReadId(pmrhTo->sqId)) + 1;
            fp += nRowsToSkip * cbPerRow;

            // clamp the file position
            fp = min2( fp, maxFp );
        }
        else
        {
            /* At this point the first identified sqId must be larger than the specified sqId. */

            // set a maximum value for the file position
            maxFp = fp + ofsFrom - (sizeof(MetadataRowHeader) + sizeof(sqId));

            INT64 nRowsToSkip = (AriocDS::SqId::GetReadId(pmrhFrom->sqId) - AriocDS::SqId::GetReadId(sqId)) + 1;
            fp -= nRowsToSkip * cbPerRow;

            // clamp the file position
            fp = max2( fp, minFp );
        }
    }
    while( minFp <= maxFp );

    // return the file pointer to the metadata row header
    return fp;
}
#pragma endregion
