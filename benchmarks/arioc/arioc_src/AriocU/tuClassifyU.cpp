/*
  tuClassifyU.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor/destructor
/// [private] constructor
tuClassifyU::tuClassifyU() : tuClassify()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
tuClassifyU::tuClassifyU( QBatch* pqb ) : tuClassify(pqb)
{
    m_ptum = AriocBase::GetTaskUnitMetrics( "tuClassifyU" );
}

/// destructor
tuClassifyU::~tuClassifyU()
{
}
#pragma endregion


#pragma region private methods
#if TODO_CHOP_WHEN_DEBUGGED
/// [private] method computeMa
INT16 tuClassifyU::computeMa( BRLEAheader* pBH )
{
    // accumulate the total number of symbols in the R sequence that are spanned by the specified BRLEA
    INT16 Ma = 0;

    // point to the first BRLEA byte
    UINT8* p = reinterpret_cast<UINT8*>(pBH + 1);

    // compute the loop limit
    UINT8* pLimit = p + pBH->cb;

    if( (*p >> 6) != bbMatch )      // always start with a match
        DebugBreak();

    // read the first BRLEA byte (which, by definition, always has bbMatch type)
    BRLEAbyteType bbTypePrev = bbMatch;
    INT16 cbRun = static_cast<INT16>(*p++);       // bbMatch is defined as zero, so we can get the first run length without using a bit mask
    while( p < pLimit )
    {
        // get the current BRLEA byte type
        BRLEAbyteType bbType = static_cast<BRLEAbyteType>(*p >> 6);

        // if we have a new BRLEA byte type, 
        if( bbType != bbTypePrev )
        {
            // conditionally add the accumulated run length to the total number of R symbols
            if( bbTypePrev != bbGapR )
                Ma += cbRun;

            // start again to accumulate the run length
            cbRun = *p & 0x3F;
            bbTypePrev = bbType;
        }

        else
            cbRun = (cbRun << 6) + (*p & 0x3F);     // accumulate the current run length

        // point to the next BRLEA byte
        p++ ;
    }


    if( bbTypePrev != bbMatch )     // always end with a match
        DebugBreak();


    // an alignnment cannot end with a gap, so add the accumulated run length to the total number of R symbols
    Ma += cbRun;

    // return the R symbol count
    return Ma;
}
#endif

/// [private] method countMappings
void tuClassifyU::countMappings( UINT32& totalAn, UINT32& totalAg )
{
    // traverse the list of Qwarps to count mappings for each Q sequence
    totalAn = 0;
    totalAg = 0;
    UINT32 nMappedQn = 0;
    UINT32 nMappedQ = 0;

    // performance metrics
    AriocTaskUnitMetrics* ptumN = AriocBase::GetTaskUnitMetrics( "tuFinalizeN" );

    Qwarp* pQw = m_pqb->QwBuffer.p;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {       
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
#if TODO_CHOP
            if( (pQw->nAn[iq] + pQw->nAg[iq]) == 0 )
            {
                if( ++nUnmapped < 256 )
                {
                    UINT32 qid = PACK_QID(iw,iq);
                    CDPrint( cdpCD0, "tuClassifyU::countMappings: %3u: 0x%08X 0x%016llX", nUnmapped, qid, pQw->sqId[iq] );
                }
            }
#endif


            totalAn += pQw->nAn[iq];            // nongapped
            totalAg += pQw->nAg[iq];            // gapped

            if( pQw->nAn[iq] ) ++nMappedQn;                 // number of Q sequences with at least one nongapped mapping
            if( pQw->nAn[iq] + pQw->nAg[iq] ) ++nMappedQ;   // number of Q sequences with at least one mapping (nongapped or gapped)
        }

        ++pQw;
    }

    // sanity check
    UINT32 nMappedActual = totalAn + totalAg;
    UINT32 nMappedExpected = m_pqb->HBn.nMapped +                           // totalAn
                             m_pqb->HBgwn.nMapped +                         // totalAg
                             m_pqb->HBgs.nMapped +
                             m_pqb->HBgc.nMapped +
                             m_pqb->HBgwc.nMapped;
    if( nMappedExpected != nMappedActual )
        throw new ApplicationException( __FILE__, __LINE__, "inconsistent mapping counts: expected/actual = %u/%u", nMappedExpected, nMappedActual );

    // performance metrics
    InterlockedExchangeAdd( &ptumN->n.CandidateQ, m_pqb->DB.nQ );
    InterlockedExchangeAdd( &ptumN->n.MappedQ, nMappedQn );
    InterlockedExchangeAdd( &m_ptum->n.CandidateQ, m_pqb->DB.nQ );
    InterlockedExchangeAdd( &m_ptum->n.MappedQ, nMappedQ );

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCDb, "tuClassifyU::countMappings: totalAn=%u totalAg=%u nMappedQ=%u", totalAn, totalAg, nMappedQ );
#endif
}

/// [private] method consolidateAlignmentInfoFromBRLEAs
void tuClassifyU::consolidateAlignmentInfoFromBRLEAs( HostBuffers* pHB, const QAIflags mappedBy )
{
    // point to the start of the BRLEA buffer
    BRLEAheader* pBH = reinterpret_cast<BRLEAheader*>(pHB->BRLEA.p);
    BRLEAheader* pBHlimit = reinterpret_cast<BRLEAheader*>(pHB->BRLEA.p + pHB->BRLEA.n);

    
#if TODO_CHOP_WHEN_DEBUGGED
    INT16 Vperfect = m_pab->aas.ASP.Wm * m_pqb->Nmax;
#endif
    
    // traverse the list of BRLEAs
    while( pBH < pBHlimit )
    {
        // point to the Qwarp for the current BRLEA
        UINT32 iw = QID_IW(pBH->qid);
        UINT32 iq = QID_IQ(pBH->qid);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;


        
#if TODO_CHOP_WHEN_DEBUGGED
        if( (pBH->V == 0) || (pBH->V > Vperfect) )
        {
            CDPrint( cdpCD0, "tuClassifyU::consolidateAlignmentInfoFromBRLEAs: invalid V = %d qid=0x%08x sqId=0x%016llx", pBH->V, pBH->qid, pQw->sqId[iq] );
            //DebugBreak();
            goto LX4361;
        }
#endif


#if TODO_CHOP_WHEN_DEBUGGED
        if( (pQw->sqId[iq] & ~DVALUE_MASK_S) == 0x0000020400000C0F )
        {
            UINT32 n = static_cast<UINT32>(pBH - reinterpret_cast<BRLEAheader*>(pHB->BRLEA.p));
            CDPrint( cdpCD0, "tuClassifyU::consolidateAlignmentInfoFromBRLEAs: %u: sqId=0x%016llx", n, pQw->sqId[iq] );
        }

#endif

        // add a QAI to the buffer
        appendQAI( pBH, pQw->N[iq], mappedBy );



#if TODO_CHOP_WHEN_DEBUGGED
LX4361:
#endif


        // point to the next BRLEA
        pBH = reinterpret_cast<BRLEAheader*>(reinterpret_cast<UINT32*>(pBH) + m_pqb->celBRLEAperQ);
    }
}

/// [private] method consolidateAlignmentInfo
void tuClassifyU::consolidateAlignmentInfo( UINT32 totalAn, UINT32 totalAg )
{
#if TODO_CHOP_WHEN_DEBUGGED
    HiResTimer hrt(ms);
#endif


    /* (Re)allocate and zero a buffer to contain consolidated alignment info:
        - the buffer contains one QAI struct for each mapping
        - the 0th element in the buffer is null
    */
    size_t celAq = totalAn + totalAg + 1;                 // (including space for a null QAI struct)
    m_pqb->AaBuffer.Reuse( celAq, false );
    memset( m_pqb->AaBuffer.p, 0, m_pqb->AaBuffer.cb );
    m_pqb->AaBuffer.n = 1;

    /* Initialize a list of offsets into the consolidated alignment info buffer:
        - there is one element in this list for each Q sequence
        - if a Q sequence has no associated mappings, the corresponding offset defaults to zero (a reference to
            the null QAI in the 0th element of the QAI buffer)
    */
    size_t nQAI = m_pqb->QwBuffer.Count * CUDATHREADSPERWARP;       // (one QAI reference per Q sequence)
    m_pqb->ArBuffer.Reuse( nQAI, false );
    memset( m_pqb->ArBuffer.p, 0, m_pqb->ArBuffer.cb );

    Qwarp* pQw = m_pqb->QwBuffer.p;
    UINT32 ofsQAI = 0;
    for( UINT32 iw=0; iw<m_pqb->QwBuffer.n; ++iw )
    {
        for( INT16 iq=0; iq<pQw->nQ; ++iq )
        {
            // initialize the QAIReference instance for the current Q sequence
            UINT32 qid = AriocDS::QID::Pack(iw,iq);
            m_pqb->ArBuffer.p[qid] = QAIReference( ofsQAI, 0 );

            // track the total number of items in the QAI list
            ofsQAI += (pQw->nAn[iq] + pQw->nAg[iq]);
        }

        ++pQw;
    }

    // build a list of consolidated alignment info ...    
    consolidateAlignmentInfoFromBRLEAs( &m_pqb->HBn, qaiMapperN );      // ... for mappings found by the nongapped aligner
    consolidateAlignmentInfoFromBRLEAs( &m_pqb->HBgc, qaiMapperGc );    // ... for mappings found by the seed-and-extend gapped aligner (seed coverage filter)

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "tuClassifyU::consolidateAlignmentInfo: %dms", hrt.GetElapsed(false) );
#endif
}

/// [private] method sortAlignmentInfo
void tuClassifyU::sortAlignmentInfo()
{
    // prepare for multithreaded access to Qwarps
    m_qid = 0;

    // sort the mappings concurrently
    WinGlobalPtr<tuClassifyU1*> classifyU1( m_pab->nLPs, true );
    for( INT32 n=0; n<m_pab->nLPs; ++n )
    {
        classifyU1.p[n] = new tuClassifyU1( m_pqb, &m_qid );
        classifyU1.p[n]->Start();
    }

    // wait for completion
    for( INT32 n=0; n<m_pab->nLPs; ++n )
    {
        classifyU1.p[n]->Wait();
        delete classifyU1.p[n];
    }
}

/// [private] method computeMapq
void tuClassifyU::computeMapq()
{
    // prepare for multithreaded access to Qwarps
    m_qid = 0;

    // use multiple CPU threads
    WinGlobalPtr<tuClassifyU2*> classifyU2( m_pab->nLPs, true );
    for( INT32 n=0; n<m_pab->nLPs; ++n )
    {
        classifyU2.p[n] = new tuClassifyU2( m_pqb, &m_qid );
        classifyU2.p[n]->Start();
    }

    // wait for completion
    for( INT32 n=0; n<m_pab->nLPs; ++n )
    {
        classifyU2.p[n]->Wait();
        delete classifyU2.p[n];
    }
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Classifies unpaired mappings
/// </summary>
void tuClassifyU::main()
{
    CRVALIDATOR;

    CDPrint( cdpCD3, "tuClassifyU[%d]::main ...", m_pqb->pgi->deviceId );

    // count the total number of nongapped and gapped mappings for the current batch
    UINT32 totalAn;
    UINT32 totalAg;
    countMappings( totalAn, totalAg );

    // build a consolidated list of references to all mappings (nongapped and gapped) for each Q sequence
    consolidateAlignmentInfo( totalAn, totalAg );

#if TODO_CHOP_WHEN_DEBUGGED
    m_pab->nLPs = 1;
#endif

    // split the work into concurrent partitions
    sortAlignmentInfo();
    computeMapq();

    // performance metrics
    InterlockedExchangeAdd( &m_ptum->ms.Elapsed, m_hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "tuClassifyU[%d]::main ends (%dms)", m_pqb->pgi->deviceId, m_hrt.GetElapsed(false) );
}
