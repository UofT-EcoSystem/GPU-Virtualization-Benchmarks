/*
  tuLoadM.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuLoadM::tuLoadM()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuLoadM::tuLoadM( QBatch* pqb ) : m_pqb(pqb)
{
    // performance metrics
    m_ptum = AriocBase::GetTaskUnitMetrics( "tuLoadM" );
}

/// destructor
tuLoadM::~tuLoadM()
{
}
#pragma endregion

#pragma region protected methods
/// [protected] method readMetadata
void tuLoadM::readMetadata( RaiiFile* pFileM, QfileInfo* pqfi, MfileInfo* pmfi, MfileBuf* pmfb, UINT64 sqIdInitial, UINT64 sqIdFinal )
{
    // do nothing if there is no metadata
    if( pFileM->Handle < 0 )
        return;

    /* Prepare to handle either unpaired or paired-end metadata:
        For the inner read loop:
        - Unpaired data: use qid as an index into the list of offsets; increment the loop count by 1.
        - Paired-end data: use qid/2 as an index into the list of offsets; increment the loop count by 2.
        For determining the number of rows of metadata to read:
        - Unpaired data: same as the number of Q sequences in the current batch.
        - Paired-end data: 1/2 the number of Q sequences.
    */
    const UINT32 px = pqfi->pfi->IsPaired ? 2 : 1;

    // compute the number of rows of metadata to read
    UINT32 nRows = ((m_pqb->QwBuffer.n-1) * CUDATHREADSPERWARP) +   // all but the final Qwarp
                    m_pqb->QwBuffer.p[m_pqb->QwBuffer.n-1].nQ;      // final Qwarp

#if TODO_CHOP_WHEN_DEBUGGED
    if( nRows != m_pqb->DB.nQ )
        DebugBreak();
#endif

    nRows /= px;                                                    // for paired reads, there is a pair of metadata files

    /* Estimate the file position of the first metadata row.
        - If no rows have yet been read from the metadata file, estimate the start position in the metadata file
           as the same relative position as the start position in the Q-sequence file.
        - Otherwise, the start position is where it was after the previous batch iteration.
    */
    INT64 fpInitial;
    if( pmfi->nRows == 0 )
    {
        double frac;
        if( (sqIdInitial >= pqfi->pfi->EstimatedSqIdFrom) && (sqIdInitial <= pqfi->pfi->EstimatedSqIdTo) )
        {
            // we have a range of sqId values in the Q-sequence file, so we can estimate the relative file position
            frac = static_cast<double>(sqIdInitial - pqfi->pfi->EstimatedSqIdFrom) / (pqfi->pfi->EstimatedSqIdTo - pqfi->pfi->EstimatedSqIdFrom);
        }
        else
        {
            /* We use the relative position for the current Q file to estimate the position in the metadata file.
                This has a potentially time-consuming drawback in that the Q file position does not account for
                reads that are skipped because of a user-specified range of readIds. */
            frac = static_cast<double>(pqfi->fp0) / pqfi->fpLimit;
        }

        fpInitial = static_cast<INT64>(frac * pFileM->FileSize());

        // scrape the metadata until we find the actual file position of the initial sqId
        fpInitial = baseQReader::sqIdHunt( pFileM, fpInitial, sqIdInitial );
    }
    else
        fpInitial = pmfi->fp;

    /* Estimate the file position of the (last+1)th metadata row:
        - If this is the final file partition, it's the end of the metadata file.
        - Otherwise:
            - We assume the metadata is fairly uniform in size for each Q sequence.
            - The end position is assumed to be in the same relative position in the metadata file as the end position in
                the Q-sequence file.
            - At the time this method executes, the Q-sequence file position has already advanced past the last Q sequence
                in the current batch.  (See QReaderP::LoadQ).
    */
    INT64 fpFinal;
    if( pmfi->nRows == 0 )
    {
        if( (sqIdFinal >= pqfi->pfi->EstimatedSqIdFrom) && (sqIdFinal <= pqfi->pfi->EstimatedSqIdTo) )
        {
            // we have a range of sqId values in the Q-sequence file, so we can estimate the relative file position
            double frac = static_cast<double>(sqIdFinal - pqfi->pfi->EstimatedSqIdFrom) / (pqfi->pfi->EstimatedSqIdTo - pqfi->pfi->EstimatedSqIdFrom);
            fpFinal = static_cast<INT64>(frac * pFileM->FileSize());
        }
        else
        {
            // we cannot estimate where the metadata corresponding to sqIdFinal might be
            fpFinal = fpInitial;
        }
    }
    else
    {
        // use the average metadata row size to estimate the file position of the final sqId in the specified range
        fpFinal = fpInitial + static_cast<INT64>((static_cast<double>(nRows)) * pmfi->cb / pmfi->nRows);
    }

    // scrape the metadata until we find the actual file position of the final sqId
    fpFinal = min2( fpFinal, pmfi->fpLimit );
    fpFinal = baseQReader::sqIdHunt( pFileM, fpFinal, sqIdFinal );

    if( fpFinal < pmfi->fpLimit )
    {
        // point past the last sqId and its associated metadata
        MetadataRowHeader mrh;
        pFileM->Seek( fpFinal, SEEK_SET );
        pFileM->Read( &mrh, sizeof mrh );
        fpFinal += (sizeof mrh) + mrh.cb;
    }

    // ensure that the metadata buffer is large enough to accommodate the metadata
    size_t cbBuf = fpFinal - fpInitial;
    if( cbBuf > pmfb->buf.Count )
    {
        // the buffer allocation is padded to accommodate 8-byte chunked reads at the end of the buffer
        pmfb->buf.Realloc( cbBuf+sizeof(UINT64), false );
    }

    UINT32 nQ = m_pqb->QwBuffer.n * CUDATHREADSPERWARP;
    if( nQ >= static_cast<UINT32>(pmfb->ofs.Count) )
        pmfb->ofs.Realloc( nQ, false );

    // seek to the starting position in the metadata file for the first sqId in the batch
    pFileM->Seek( fpInitial, SEEK_SET );



#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: sqIdInitial=0x%016llx sqIdFinal=0x%016llx nSqIds=%llu", __FUNCTION__, sqIdInitial, sqIdFinal, 1+(sqIdFinal-sqIdInitial)/4 );
    memset( pmfb->buf.p, 0xFE, pmfb->buf.cb );
   
    if( sqIdFinal == 0x00000408002d5cd5 )
        CDPrint( cdpCD0, __FUNCTION__ );
#endif

    // load the metadata into the buffer
    pFileM->Read( pmfb->buf.p, cbBuf );

#ifdef _DEBUG
    if( fpFinal != pFileM->Seek( 0, SEEK_CUR ) )
        throw new ApplicationException( __FILE__, __LINE__, "file %s: fpFinal=%lld actual=%lld", pFileM->FileSpec.p, fpFinal, pFileM->Seek( 0, SEEK_CUR ) );
#endif

    // update the metadata file info
    pmfi->fp = fpFinal;
    pmfi->cb += cbBuf;
    pmfi->nRows += nRows;
    
    // build per-Q references to the metadata
    Qwarp* pQw = m_pqb->QwBuffer.p;
    MetadataRowHeader* pmrh = reinterpret_cast<MetadataRowHeader*>(pmfb->buf.p);
    UINT32 ofsMetadataRow = 0;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        UINT32 qid = AriocDS::QID::Pack(iw, 0);
        UINT32 iofs = qid / px;
        for( INT16 iq=0; iq<pQw->nQ; iq+=px )
        {
            // save an offset into the metadata buffer for the mate
            pmfb->ofs.p[iofs] = ofsMetadataRow;
            ++iofs;

            // update the buffer offset
            ofsMetadataRow += (sizeof(MetadataRowHeader) + pmrh->cb);

            // point to the next row of metadata
            pmrh = reinterpret_cast<MetadataRowHeader*>(pmfb->buf.p+ofsMetadataRow);
        }

        // point to the next Qwarp
        pQw++ ;
    }
}
#pragma endregion
