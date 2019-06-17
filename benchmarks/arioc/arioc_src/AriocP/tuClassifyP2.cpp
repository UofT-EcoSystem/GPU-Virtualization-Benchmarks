/*
  tuClassifyP2.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"


#pragma region constructor/destructor
/// [private] constructor
tuClassifyP2::tuClassifyP2()
{
}

/// <param name="pqb">a reference to a <c>QBatch</c> instance</param>
/// <param name="ppi">partition info</param>
tuClassifyP2::tuClassifyP2( QBatch* pqb, volatile UINT32* pqid ) :
                                baseMAPQ( pqb->pab ),
                                m_pqb(pqb),
                                m_pab(pqb->pab),
                                m_arfMaskCollision(static_cast<AlignmentResultFlags>(pqb->pab->aas.ACP.arf & arfMaskCollision)),
                                m_pqid(pqid),
                                m_paiq(2048,true)
{
    // set up the correct member function for evaluating DP candidate pairs
    switch( m_pab->aas.ACP.arf & arfMaskOrientation )
    {
        case arfOrientationSame:
            m_evaluateCandidatePair = &tuClassifyP2::evaluateCandidatePair_same;
            break;

        case arfOrientationConvergent:
            m_evaluateCandidatePair = &tuClassifyP2::evaluateCandidatePair_convergent;
            break;

        case arfOrientationDivergent:
            m_evaluateCandidatePair = &tuClassifyP2::evaluateCandidatePair_divergent;
            break;

        default:
            throw new ApplicationException( __FILE__, __LINE__, "unexpected value for AlignmentResultFlags orientation: 0x%08x", m_pab->aas.ACP.arf & arfMaskOrientation ); 
    }
}

/// destructor
tuClassifyP2::~tuClassifyP2()
{
}
#pragma endregion

#pragma region private methods
/// [private static] method PAIComparer
int tuClassifyP2::PAIComparer( const void* a, const void* b )
{
    const PAI* pa = reinterpret_cast<const PAI*>(a);
    const PAI* pb = reinterpret_cast<const PAI*>(b);

    // order by whether the pair is mapped concordantly
    int rval = static_cast<int>(pb->arf & arfConcordant) - static_cast<int>(pa->arf & arfConcordant);
    if( rval == 0 )
    {
        // order by total Vmax (descending)
        rval = static_cast<int>(pb->Vpair) - static_cast<int>(pa->Vpair);

        if( (rval == 0) && (pa->arf & arfConcordant) )
        {
        
#if TODO_CHOP_WHEN_DEBUGGED
            if( (pa->pQAI1 == NULL) || (pa->pQAI2 == NULL) || (pb->pQAI1 == NULL) || (pb->pQAI2 == NULL) )
                DebugBreak();
#endif

            /* At this point we are comparing two concordant pairs with the same total V scores.
            
               The following sequence of comparisons is arbitrary; we have no empirical evidence that one
                is better than any other.
            */

#if TODO_CHOP_WHEN_DEBUGGED
            if( pa->pQAI1->mcm == 0 ) DebugBreak();
            if( pa->pQAI2->mcm == 0 ) DebugBreak();
            if( pb->pQAI2->mcm == 0 ) DebugBreak();
            if( pb->pQAI2->mcm == 0 ) DebugBreak();
#endif
            
            // order by total edit distance
            rval = (pa->pQAI1->editDistance+pa->pQAI2->editDistance) - (pb->pQAI1->editDistance+pb->pQAI2->editDistance);
            if( rval == 0 )
            {            
                // order by the number of contiguous mapped positions
                rval = static_cast<int>(pb->pQAI1->mcm+pb->pQAI2->mcm) - static_cast<int>(pa->pQAI1->mcm+pa->pQAI2->mcm);
                if( rval == 0 )
                {
                    // order by proximity to the estimated mean fragment length (TLEN) for the batch
                    rval = pa->diffTLEN - pb->diffTLEN;
                    if( rval == 0 )
                    {
                        // order by maximum V score for the individual mates (i.e. 200+180 vs 195+185)
                        rval = max2(pb->pQAI1->pBH->V,pb->pQAI2->pBH->V) - max2(pa->pQAI1->pBH->V,pa->pQAI2->pBH->V);
                        if( rval == 0 )
                        {
                            // order by Q-sequence strand (prefer Qf to Qrc)
                            rval = ((pa->pQAI1->flags & qaiRCq) + (pa->pQAI2->flags & qaiRCq)) - ((pb->pQAI1->flags & qaiRCq) + (pb->pQAI2->flags & qaiRCq));
                        }
                    }
                }
            }
        }
    }
    return rval;
}

/// [private] method verifyCandidatePairFragmentLength
inline bool tuClassifyP2::verifyCandidatePairFragmentLength( PAI& pai )
{
    // set a flag if the magnitude of the fragment length lies within the configured range
    INT32 absFragLen = (pai.FragmentLength <= 0) ? -pai.FragmentLength : pai.FragmentLength;
    bool rval = (absFragLen >= m_pab->aas.ACP.minFragLen) && (absFragLen <= m_pab->aas.ACP.maxFragLen);

    // if the fragment length lies within the configured range, record its distance from the estimated mean fragment length
    if( rval )
        pai.diffTLEN = abs(absFragLen-m_pab->iMeanTLEN);

    return rval;
}

/// [private] method evaluateCandidatePairCollision
bool tuClassifyP2::evaluateCandidatePairCollision( PAI& pai, QAI* pQAIu, QAI* pQAId )
{
    /* Compute the "footprint" of each of the two ends relative to the start of the forward R sequence:
        - For determining fragment length, we compute what the SAM spec requires, i.e., "the number of bases from the
            leftmost mapped base to the rightmost mapped base", even though it might make more sense to account for
            symbols that are soft-clipped:

                Jul -= pQAIu->pBH->Il;
                Jdr += pQAId->pBH->Il;
                Jdl -= ((pQAId->pBH->N - pQAId->pBH->Ir) - 1);
                Jur += ((pQAIu->pBH->N - pQAIu->pBH->Ir) - 1);
    */
    UINT32 Jul = pQAIu->Jf;                 // upstream read, leftmost (upstream) end
    UINT32 Jur = Jul + pQAIu->pBH->Ma - 1;  // upstream read, rightmost (downstream) end
    UINT32 Jdl = pQAId->Jf;                 // downstream read, leftmost end
    UINT32 Jdr = Jdl + pQAId->pBH->Ma - 1;  // downstream read, rightmost end

    /* Look for a simple noncolliding orientation:

            Jul           Jur
            ---------------·························
            ······················--------------····
                                  Jdl          Jdr
    */
    if( Jur < Jdl )
    {
        pai.FragmentLength = (Jdr - Jul) + 1;
        if( !verifyCandidatePairFragmentLength( pai ) )
            return false;

        pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfConcordant);
        return true;
    }

    /* Look for a non-covering dovetail
    
                                  Jul          Jur
            ······················--------------····
            ---------------·························
            Jdl           Jdr
    */
    if( Jul > Jdr )
    {
        pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfCollisionDovetail);
        pai.FragmentLength = (Jur - Jdl) + 1;
    }

    else
    {
        /* Look for the following cases:

                                 Jul           Jur
                overlap:         ---------------·············
                                    ··········--------------····
                                              Jdl          Jdr

                                     Jul             Jur
                cover:           ····-----------------·······
                                    ·······-----------··········
                                           Jdl       Jdr
        */
        if( Jul < Jdl )
        {
            if( Jur < Jdr )
            {
                /* the reads overlap */
                pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfCollisionOverlap);
                pai.FragmentLength = (Jdr - Jul) + 1;
            }
            else
            {
                /* the "upstream" read covers the "downstream" read */
                pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfCollisionCover);
                pai.FragmentLength = (Jur - Jul) + 1;             // same as pai.pQAIu->pBH->Ma
            }
        }

        /* Look for the following cases:

                                      Jul          Jur
                dovetail:   ··········--------------····
                            ···---------------·············
                               Jdl           Jdr

                                        Jul      Jur
                cover:      ············----------·······
                            ·······---------------··········
                                   Jdl           Jdr
        */
        else
        {
            if( Jur > Jdr )
            {
                /* the reads dovetail */
                pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfCollisionDovetail);
                pai.FragmentLength = (Jur - Jdl) + 1;
            }
            else
            {
                /* the "downstream" read covers the "upstream" read */
                pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfCollisionCover);
                pai.FragmentLength = (Jdr - Jdl) + 1;             // same as pai.pQAId->pBH->Ma
            }
        }
    }

    // the pair is not concordant if its collision type is not one of the user-specified collision types
    if( (pai.arf & m_arfMaskCollision) != (pai.arf & arfMaskCollision) )
        return false;

    // verify whether the fragment length is within the user-configured range
    if( !verifyCandidatePairFragmentLength( pai ) )
        return false;

    // at this point we have a concordant mapping
    pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfConcordant);
    return true;
}

/// [private] method computeVpair
inline
INT16 tuClassifyP2::computeVpair( PAI& pai )
{
    /* Possible ways to combine the V scores of the mates:
        - sum
        - average
        - scaled product (or some other probabilistic computation)
       We use the composite V score to prioritize the list of mappings when two or more valid paired mappings
        exist, so a simple sum ought to do it.
    */
    return pai.pQAI1->pBH->V + pai.pQAI2->pBH->V;
}

/// [private] method appendPAI
void tuClassifyP2::appendPAI( PAI& pai )
{
    UINT32 ofs = m_paiq.n++;
    if( ofs == static_cast<UINT32>(m_paiq.Count) )
    {
        INT64 cel = (static_cast<INT64>(m_paiq.n) * 3) / 2;    // grow by about 50%
        m_paiq.Realloc( cel, false );

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s: m_paiq.Count=%lld", __FUNCTION__, m_paiq.Count );
#endif
    }

#if TODO_CHOP_WHEN_DEBUGGED
    if( (pai.arf & arfMate1Unmapped) && pai.pQAI1 )
        CDPrint( cdpCD0, "tuClassifyP2::appendPAI" );
    if( (pai.arf & arfMate2Unmapped) && pai.pQAI2 )
        CDPrint( cdpCD0, "tuClassifyP2::appendPAI" );
#endif

    // copy the specified PAI data into the buffer
    m_paiq.p[ofs] = pai;
}

/// [private] method appendPAIforQ
void tuClassifyP2::appendPAIforQ()
{
    RaiiCriticalSection<tuClassifyP2> rcs;

    // get the offset of the next available element in the consolidated PAI buffer
    UINT32 ofs = m_pqb->ApBuffer.n;
    
    // compute the new total number of elements in the consolidated PAI buffer
    m_pqb->ApBuffer.n += m_paiq.n;

    // conditionally grow the consolidated buffer by about 10%
    if( m_pqb->ApBuffer.n >= static_cast<UINT32>(m_pqb->ApBuffer.Count) )
    {
        const INT64 cel = (m_pqb->ApBuffer.n < 100) ? 2*m_pqb->ApBuffer.n : (static_cast<INT64>(m_pqb->ApBuffer.n) * 110) / 100;
        m_pqb->ApBuffer.Reuse( cel, false );

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s: m_pqb->ApBuffer.Count increased to %lld", __FUNCTION__, m_pqb->ApBuffer.Count );
#endif
    }

    // copy the specified PAI data into the consolidated buffer
    memcpy( m_pqb->ApBuffer.p+ofs, m_paiq.p, m_paiq.n*sizeof(PAI) );
}

/// [private] method evaluateCandidatePair_same
bool tuClassifyP2::evaluateCandidatePair_same( PAI& pai )
{
    // verify that the mates map to the same reference strand
    bool mappedToRC1 = (((pai.pQAI1->flags >> qaiShrRC) ^ pai.pQAI1->flags) & qaiRCr) != 0;
    bool mappedToRC2 = (((pai.pQAI2->flags >> qaiShrRC) ^ pai.pQAI2->flags) & qaiRCr) != 0;
    if( mappedToRC1 != mappedToRC2 )
        return false;

    // determine which mate is expected to be upstream
    QAI* pQAIu;
    QAI* pQAId;
    if( mappedToRC1 & qaiRCr )
    {
        pQAIu = pai.pQAI2;
        pQAId = pai.pQAI1;
    }
    else
    {
        pQAIu = pai.pQAI1;
        pQAId = pai.pQAI2;
    }

    // look for collisions between the mapped mates
    bool isConcordant = evaluateCandidatePairCollision( pai, pQAIu, pQAId );
    if( isConcordant )
        pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfOrientationSame);

    return isConcordant;
}

/// [private] method evaluateCandidatePair_divergent
bool tuClassifyP2::evaluateCandidatePair_divergent( PAI& pai )      /*** TODO: VERIFY THAT THIS WORKS ***/
{
    // verify that the mates map to different reference strands
    bool mappedToRC1 = (((pai.pQAI1->flags >> qaiShrRC) ^ pai.pQAI1->flags) & qaiRCr) != 0;
    bool mappedToRC2 = (((pai.pQAI2->flags >> qaiShrRC) ^ pai.pQAI2->flags) & qaiRCr) != 0;
    if( mappedToRC1 == mappedToRC2 )
        return false;

    // determine which mate is mapped to which strand
    QAI* pQAIf;
    QAI* pQAIr;
    if( mappedToRC1 )
    {
        pQAIf = pai.pQAI1;
        pQAIr = pai.pQAI2;
    }
    else
    {
        pQAIf = pai.pQAI2;
        pQAIr = pai.pQAI1;
    }

    // look for collisions between the mapped mates
    bool isConcordant = evaluateCandidatePairCollision( pai, pQAIr, pQAIf );   // the "reverse complement" mate is expected to map upstream
    if( isConcordant )
        pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfOrientationDivergent);

    return isConcordant;
}


#if TODO_CHOP_WHEN_DEBUGGED
static UINT32 nConcordant = 0;
#endif


/// [private] method evaluateCandidatePair_convergent
bool tuClassifyP2::evaluateCandidatePair_convergent( PAI& pai )
{
#if TODO_CHOP_WHEN_DEBUGGED
    //sqId=0x000004080001d388 pQAI1=0x000000001c219db8 1/242589685  pQAI2=0x000000001c21af82 1/242589568
    //pQAI1->subId, pQAI1->Jf,  pQAI2, pQAI2->subId, pQAI2->Jf
    if( m_pqb->QwBuffer.p[pai.iw].sqId[pai.iq] == 0x0000040800000030 )
    {
        if( pai.pQAI1 )
            CDPrint( cdpCD0, "%s: arf=0x%08x mate1 flags=0x%04x subId=%d Jf=%d", __FUNCTION__, pai.arf,
                                pai.pQAI1->flags, pai.pQAI1->subId, pai.pQAI1->Jf );
        if( pai.pQAI2 )
            CDPrint( cdpCD0, "%s: arf=0x%08x mate2 flags=0x%04x subId=%d Jf=%d", __FUNCTION__, pai.arf,
                                pai.pQAI2->flags, pai.pQAI2->subId, pai.pQAI2->Jf );

        CDPrint( cdpCD0, "%s: sqId=0x%016llx", __FUNCTION__, m_pqb->QwBuffer.p[pai.iw].sqId[pai.iq] );
    }
#endif

    // verify that the mates map to different reference strands
    bool mappedToRC1 = (((pai.pQAI1->flags >> qaiShrRC) ^ pai.pQAI1->flags) & qaiRCr) != 0;
    bool mappedToRC2 = (((pai.pQAI2->flags >> qaiShrRC) ^ pai.pQAI2->flags) & qaiRCr) != 0;
    if( mappedToRC1 == mappedToRC2 )
        return false;

    // determine which mate is mapped to which strand
    QAI* pQAIf;
    QAI* pQAIr;
    if( mappedToRC2 )
    {
        pQAIf = pai.pQAI1;
        pQAIr = pai.pQAI2;
    }
    else
    {
        pQAIf = pai.pQAI2;
        pQAIr = pai.pQAI1;
    }

    // look for collisions between the mapped mates
    bool isConcordant = evaluateCandidatePairCollision( pai, pQAIf, pQAIr );   // the "forward" mate is expected to map upstream
    if( isConcordant )
        pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfOrientationConvergent);

#if TODO_CHOP_WHEN_DEBUGGED
    if( isConcordant && (++nConcordant < 256) )
    {
        Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(pai.pQAI1->qid);
        UINT32 iq = QID_IQ(pai.pQAI1->qid);
        UINT8 subId = pai.pQAI1->subId;
        UINT64 sqId1 = pQw->sqId[iq];
        UINT64 sqId2 = pQw->sqId[iq^1];
        UINT32 Jf1 = pai.pQAI1->Jf;
        UINT32 Jf2 = pai.pQAI2->Jf;

        CDPrint( cdpCD0, "tuClassifyP2::evaluateCandidatePair_convergent: concordant: 0x%08X: 0x%016llx %u / 0x%016llx %u",
                                                                                     pai.pQAI1->qid, sqId1, Jf1, sqId2, Jf2 );

    }
#endif

#if TODO_CHOP_WHEN_DEBUGGED
    if( m_pqb->QwBuffer.p[pai.iw].sqId[pai.iq] == 0x000004080001D388 )
        CDPrint( cdpCD0, "%s: sqId=0x%016llx pos1=%d/%d pos2=%d/%d V1=%d V2=%d isConcordant=%d", __FUNCTION__, m_pqb->QwBuffer.p[pai.iw].sqId[pai.iq],
                            pai.pQAI1->subId, pai.pQAI1->Jf,
                            pai.pQAI2->subId, pai.pQAI2->Jf,
                            pai.pQAI1->pBH->V, pai.pQAI2->pBH->V,
                            isConcordant );
#endif

    return isConcordant;
}

/// [private] method findPairUnique
INT32 tuClassifyP2::findPairUnique( UINT32 qid, QAI* pQAI1, QAI* pQAI2, pairTypeCounts* pptc )
{
    // initialize a PAI struct
    PAI pai( qid );
    pai.arf = arfUnique;

    // add the alignment info for each mate
    pai.pQAI1 = pQAI1;
    pai.pQAI2 = pQAI2;
    pai.Vpair = computeVpair( pai );

    if( pQAI1->subId == pQAI2->subId )
    {
        // both ends map to the same reference
        if( (this->*m_evaluateCandidatePair)( pai ) )
        {
            // performance metrics
            pptc->nConcordant1++ ;
        }
        else
        {
            // the mates map to the same R sequence but do not meet the collision or fragment-length criteria for a concordant alignment
            pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfDiscordant);

            // performance metrics
            pptc->nDiscordant++ ;
        }
    }
    else
    {
        // the mates map to different R sequences
        pai.arf = static_cast<AlignmentResultFlags>(pai.arf | arfDiscordant);

        // performance metrics
        pptc->nDiscordant++ ;
    }

    // set a flag to indicate that each of the mates is in a mapped pair
    pQAI1->flags = static_cast<QAIflags>(pQAI1->flags | qaiInPair);
    pQAI2->flags = static_cast<QAIflags>(pQAI2->flags | qaiInPair);

    // append the PAI struct
    appendPAI( pai );

    // return 1 to indicate that the pair has one valid paired-end mapping (either concordant or discordant)
    return 1;
}

// [private] method trackMappedV
void tuClassifyP2::trackMappedV( INT16* pV, INT16 nV, INT16 v )
{
    for( INT16 n=0; n<nV; ++n )
    {
        // if V is at least as great as the nth value in the list...
        if( v >= pV[n] )
        {
            // expand the list
#pragma warning( push )
#pragma warning( disable:4996 )                 // (don't nag us about memmove being "unsafe")
            memmove( pV+n+1, pV+n, ((nV-n)-1)*sizeof(INT16) );
#pragma warning( pop )

            pV[n] = v;
            break;
        }
    }
}

/// [private] method findPairMultiple
INT32 tuClassifyP2::findPairMultiple( UINT32 qid, QAI* pQAI1, QAI* pQAI2, QAIReference* pqair0, QAIReference* pqair1, pairTypeCounts* pptc )
{
#if TODO_CHOP_WHEN_DEBUGGED
    HiResTimer hrt;
#endif


#if defined(TRACE_SQID)
    Qwarp* pQw = m_pqb->QwBuffer.p + QID_IW(qid);
    UINT64 sqId = pQw->sqId[QID_IQ(qid)];
    bool tracing = ((sqId | 1) == (TRACE_SQID | 1) );
#endif
       

    /* We are interested here only in concordant mappings, so we use the following heuristics to try to decrease the number of
        potential pairs we examine when there are a great many pairs to check out.
        
        1.  We track the top alignment scores for each pair and use them to exclude mappings whose scores are too low.

        2.  When we have two or more maximum-score mappings, we stop looking for more.  The idea is that pairs with thousands
             or more mappings are likely to have two or more maximum-score mappings, in which case MAPQ <= 3 anyway.
    */

    // allocate a list of the top high-scoring total V (sum of V for both mates)
    INT16 nV = max2(2, m_pab->aas.ACP.maxAg);
    INT16* pVtotal = reinterpret_cast<INT16*>(alloca( nV*sizeof(INT16) ));
    memset( pVtotal, 0, nV*sizeof(INT16) );

    // find the maximum V score for each mate
    INT16 Vp1 = 0;
    INT16 Vp2 = 0;
    for( UINT32 n=0; n<pqair0->nAq; ++n )
        Vp1 = max2( Vp1, pQAI1[n].pBH->V );
    for( UINT32 n=0; n<pqair1->nAq; ++n )
        Vp2 = max2( Vp2, pQAI2[n].pBH->V );

    // compute the maximum possible V for the pair
    INT16 Vp = Vp1 + Vp2;

    // evaluate all combinations of the two ends
    INT32 nConcordantMappings = 0;
    QAI* peol0 = pQAI1 + pqair0->nAq;
    QAI* peol1 = pQAI2 + pqair1->nAq;

    bool eol = false;
    do
    {
        if( pQAI1->subId < pQAI2->subId )
            eol = ((++pQAI1) == peol0);         // bump the first QAI pointer and set the end-of-list flag

        else
        if( pQAI1->subId > pQAI2->subId )
            eol = ((++pQAI2) == peol1);         // bump the second QAI pointer and set the end-of-list flag

        else
        {
            // save the second QAI pointer
            QAI* pQAIreset1 = pQAI2;
                        
            if( (pQAI1->pBH->V + Vp2) >= pVtotal[nV-1] )    // (heuristic 1)
            {
                // loop through the second list of QAIs until the mates no longer have the same subId
                while( !eol && (pQAI1->subId == pQAI2->subId) )
                {
                    if( (pQAI1->pBH->V + pQAI2->pBH->V) >= pVtotal[nV-1] )  // (heuristic 1)
                    {
                        PAI pai( qid );
                        pai.pQAI1 = pQAI1;
                        pai.pQAI2 = pQAI2;

#if TRACE_SQID
                        if( tracing )
                            CDPrint( cdpCD0, "%s: sqId=0x%016llx pQAI1=0x%016llx %d/%u  pQAI2=0x%016llx %d/%u",
                            __FUNCTION__, sqId,
                            pQAI1, pQAI1->subId, pQAI1->Jf,  pQAI2, pQAI2->subId, pQAI2->Jf );
#endif

                        if( (this->*m_evaluateCandidatePair)(pai) )
                        {
                            // set a flag to indicate that each of the mates is in a mapped pair
                            pQAI1->flags = static_cast<QAIflags>(pQAI1->flags | qaiInPair);
                            pQAI2->flags = static_cast<QAIflags>(pQAI2->flags | qaiInPair);

                            pai.Vpair = computeVpair( pai );
                            appendPAI( pai );
                            trackMappedV( pVtotal, nV, (pQAI1->pBH->V + pQAI2->pBH->V) );
                            nConcordantMappings++;

#if TRACE_SQID
                            if( tracing )
                                CDPrint( cdpCD0, "%s after evaluateCandidatePair: sqId=0x%016llx subId=%d Jf1=%u Jf2=%u",
                                __FUNCTION__,
                                sqId, pQAI1->subId, pQAI1->Jf, pQAI2->Jf );
#endif

                            if( Vp == pVtotal[nV-1] )    // (heuristic 2)
                            {
                                pQAI1 = peol0 - 1;  // set up the end-of-list condition for the outer loop
                                break;              // fall out of the inner loop
                            }
                        }
                    }

                    // bump the second QAI pointer
                    eol = ((++pQAI2) == peol1);     // bump the second QAI pointer and set the end-of-list flag
                }
            }

            // reset the second QAI pointer
            pQAI2 = pQAIreset1;

            // bump the first QAI pointer and set the end-of-list flag
            eol = ((++pQAI1) == peol0);
        }
    }
    while( !eol );

    switch( nConcordantMappings )
    {
        case 0:                 // the pair has no paired-end mapping (either concordant or discordant)
            break;

        case 1:                 // the pair has one concordant mapping
            pptc->nConcordant1++ ;
            break;

        default:                // the pair has two or more concordant mappings
            pptc->nConcordant2++ ;
            break;
    }

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "%s: %dms", __FUNCTION__, hrt.GetElapsed(false) );
#endif

    return nConcordantMappings;
}

/// [private] method findUnpairedMappings
void tuClassifyP2::findUnpairedMappings( UINT32 qid, QAIReference* pqair, AlignmentResultFlags arf0, INT32 nPairedMappings, pairTypeCounts* pptc )
{
#if TODO_CHOP_WHEN_DEBUGGED
    UINT64 sqId = m_pqb->QwBuffer.p[QID_IW(qid)].sqId[QID_IQ(qid)];
    if( sqId == 0x00000418000155c0 )
        CDPrint( cdpCD0, "%s: sqId=0x%016llx", __FUNCTION__, sqId );
#endif



    INT32 nUnpairedMappings = 0;

    // point to the first QAI struct for the read
    QAI* pQAI = m_pqb->AaBuffer.p + pqair->ofsQAI;

    // traverse the list of mappings for the read
    for( UINT32 n=0; n<pqair->nAq; ++n )
    {
        // if the nth mapping for the read is not in a paired-end mapping ...
        if( !(pQAI->flags & qaiInPair) )
        {
            // report the nth mapping as a "pair" with a null mate
            PAI pai( qid );
            if( pQAI->flags & qaiParity )
            {
                pai.arf = static_cast<AlignmentResultFlags>(arf0 | arfMate1Unmapped);
                pai.pQAI2 = pQAI;
            }
            else
            {
                pai.pQAI1 = pQAI;
                pai.arf = static_cast<AlignmentResultFlags>(arf0 | arfMate2Unmapped);
            }

            // the "combined" V score for the pair is just the V score for the mapped mate
            pai.Vpair = pQAI->pBH->V;

            appendPAI( pai );

            nUnpairedMappings++ ;
        }

        // point to the next QAI struct
        pQAI++ ;
    }

    // performance metrics
    switch( nUnpairedMappings )
    {
        case 0:
            break;

        case 1:
            pptc->nMatesMapped1++ ;
            if( nPairedMappings == 0 )
                pptc->nMatesMappedU1++ ;
            break;

        default:
            pptc->nMatesMapped2++ ;
            if( nPairedMappings == 0 )
                pptc->nMatesMappedU2++ ;
            break;
    }
}

/// [private] method findUnpairedMates
void tuClassifyP2::findUnpairedMates( UINT32 qid, QAIReference* pqair0, QAIReference* pqair1, INT32 nPairedMappings, pairTypeCounts* pptc )
{
#if TODO_FIGURE_THIS_OUT
    pptc->nPairsUnmapped++ ;        // count pairs without paired-end mappings
#endif

    // if mate 1 is unmapped ...
    if( pqair0->nAq == 0 )
    {
        pptc->nMatesUnmapped++ ;    // count unmapped mates

        // if mate 2 is unmapped ...
        if( pqair1->nAq == 0 )
        {
            PAI pai( qid );
            pai.arf = static_cast<AlignmentResultFlags>(arfUnmapped|arfMate1Unmapped|arfMate2Unmapped);
            appendPAI( pai );
            pptc->nMatesUnmapped++ ;
        }
        else    // mate 2 has at least one mapping
            findUnpairedMappings( qid, pqair1, static_cast<AlignmentResultFlags>(arfUnmapped|arfMate1Unmapped), nPairedMappings, pptc );
        
        return;
    }

    // if mate 2 is unmapped ...
    if( pqair1->nAq == 0 )
    {
        pptc->nMatesUnmapped++ ;    // count unmapped mates

        // we know mate 1 has at least one mapping ...
        findUnpairedMappings( qid, pqair0, static_cast<AlignmentResultFlags>(arfUnmapped|arfMate2Unmapped), nPairedMappings, pptc );

        return;
    }

    /* At this point both mates have at least one mapping; since the mappings don't meet criteria for a concordant
        or discordant paired-end mapping, they are "rejected" mappings. */
    findUnpairedMappings( qid, pqair0, arfRejected, nPairedMappings, pptc );
    findUnpairedMappings( qid, pqair1, arfRejected, nPairedMappings, pptc );
}

/// [private] method computeMappingQualities
void tuClassifyP2::computeMappingQualities()
{
    // accumulate mapping counts for the mates, both separately (unpaired) and as pairs
    baseMAPQ::Vinfo mvi1(m_pab);      // mate 1
    baseMAPQ::Vinfo mvi2(m_pab);      // mate 2
    baseMAPQ::Vinfo mviPair(m_pab);   // mates 1 and 2 (where both mates have mappings)
    
    PAI* pPAI = m_paiq.p;
    for( UINT32 n=0; n<m_paiq.n; ++n )
    {
#if TODO_CHOP_WHEN_DEBUGGED
        Qwarp* pQw = m_pqb->QwBuffer.p + pPAI->iw;
        if( pQw->sqId[pPAI->iq] == 0x0000040800745590 )
            CDPrint( cdpCD0, __FUNCTION__ );
#endif

        /* We use the qaiRCdup flag to exclude duplicate mappings where both the forward and
            reverse-complement Q sequence have identical mappings.
            
           The MAPQ computation must therefore allow for missing V-score info.  See the computeMAPQ_* methods
            in baseMAPQ.cpp.
        */
        if( (pPAI->arf & arfConcordant) && !((pPAI->pQAI1->flags | pPAI->pQAI2->flags) & qaiRCdup) )
            mviPair.TrackV( pPAI->pQAI1, pPAI->pQAI2 );

        if( pPAI->pQAI1 && !(pPAI->pQAI1->flags & qaiRCdup) )
            mvi1.TrackV( pPAI->pQAI1 );

        if( pPAI->pQAI2 && !(pPAI->pQAI2->flags & qaiRCdup) )
            mvi2.TrackV( pPAI->pQAI2 );

        // point to the next PAI
        pPAI++ ;
    }

    // finalize the mapqVinfo structs
    mvi1.FinalizeV();
    mvi2.FinalizeV();
    mviPair.FinalizeV();

    // compute MAPQ for mates that will eventually be written
    pPAI = m_paiq.p;
    for( UINT32 n=0; n<m_paiq.n; ++n )
    {

#if TODO_CHOP_WHEN_DEBUGGED
        Qwarp* pQw = m_pqb->QwBuffer.p + pPAI->iw;
        if( pQw->sqId[pPAI->iq] == 0x0000040800000010 )
            CDPrint( cdpCD0, __FUNCTION__ );
#endif

        if( (pPAI->arf & arfConcordant) &&
            (pPAI->arf & (arfWriteMate1|arfWriteMate2)) )
        {
            (this->*m_computeMAPQp)( pPAI, mviPair, mvi1, mvi2 );   // compute MAPQ for both mates

            if( pPAI->arf & arfPrimary )
            {
                // record the second-best V scores (if any)
                pPAI->pQAI1->Vsec = (mvi1.nVmax1 >= 2) ? mvi1.Vmax1 : mvi1.Vmax2;
                pPAI->pQAI2->Vsec = (mvi2.nVmax1 >= 2) ? mvi2.Vmax1 : mvi2.Vmax2;
            }            
        }
        else
        {
            if( pPAI->pQAI1 && (pPAI->arf & arfWriteMate1) )
            {
                (this->*m_computeMAPQ)( pPAI->pQAI1, mvi1 );    // compute MAPQ for mate 1 only
                if( pPAI->arf & arfPrimary )                    // record the second-best V score (if any) for mate 1
                    pPAI->pQAI1->Vsec = (mvi1.nVmax1 >= 2) ? mvi1.Vmax1 : mvi1.Vmax2;
            }

            if( pPAI->pQAI2 && (pPAI->arf & arfWriteMate2) )
            {
                (this->*m_computeMAPQ)( pPAI->pQAI2, mvi2 );    // compute MAPQ for mate 2 only
                if( pPAI->arf & arfPrimary )                    // record the second-best V score (if any) for mate 2
                    pPAI->pQAI2->Vsec = (mvi2.nVmax1 >= 2) ? mvi2.Vmax1 : mvi2.Vmax2;
            }
        }

        // point to the next PAI
        pPAI++ ;
    }
}

/// [private] method filterMappings
void tuClassifyP2::filterMappings( QAIReference* pqair0, QAIReference* pqair1 )
{
    /* We report mappings as follows:
        - If at least one paired (concordant or discordant) mapping exists, only the paired mapping(s) are reported.  The maximum number
            of paired mappings reported is limited by the configured value of maxAg.
        - Otherwise, each mate's unpaired mappings are reported.  The maximum number of unpaired mappings per mate is limited by maxAg.
        - The first reported mapping is the SAM "primary alignment".
    */

    PAI* pPAI = m_paiq.p;
    UINT32 cel = m_paiq.n;


#if TODO_CHOP_WHEN_DEBUGGED
    Qwarp* pQwx = m_pqb->QwBuffer.p + pPAI->iw;
    if( (pQwx->sqId[pPAI->iq] & 0xFFFFFFFFFFFFFFFC) == 0x0000040800006170 )
    {
        for( UINT32 n=0; n<cel; ++n )
        {
            if( pPAI[n].arf & arfConcordant )
                CDPrint( cdpCD0, "%s: n=%u diffTLEN=%d concordant mate1: flags=0x%04x subId=%d Jf=%d mcm=%d, mate2: flags=0x%04x subId=%d Jf=%d mcm=%d", __FUNCTION__, n, pPAI[n].diffTLEN,
                                pPAI[n].pQAI1->flags, pPAI[n].pQAI1->subId, pPAI[n].pQAI1->Jf, pPAI[n].pQAI1->mcm,
                                pPAI[n].pQAI2->flags, pPAI[n].pQAI2->subId, pPAI[n].pQAI2->Jf, pPAI[n].pQAI2->mcm
                       );
            else
            {
                if( pPAI[n].pQAI1 )
                    CDPrint( cdpCD0, "%s: n=%u arf=0x%08x mate1 flags=0x%04x subId=%d Jf=%d", __FUNCTION__, n, pPAI[n].arf,
                                        pPAI[n].pQAI1->flags, pPAI[n].pQAI1->subId, pPAI[n].pQAI1->Jf );
                if( pPAI[n].pQAI2 )
                    CDPrint( cdpCD0, "%s: n=%u arf=0x%08x mate2 flags=0x%04x subId=%d Jf=%d", __FUNCTION__, n, pPAI[n].arf,
                                        pPAI[n].pQAI2->flags, pPAI[n].pQAI2->subId, pPAI[n].pQAI2->Jf );
            }
        }
        CDPrint( cdpCD0, __FUNCTION__ );
    }
#endif



    // set "write" flags on the mates in the list
    if( cel == 1 )
    {
        /* This can happen if...
            - this is a unique mapping, OR
            - one mate has one mapping and the other is unmapped, OR
            - both mates are unmapped
        */
        pPAI->arf = static_cast<AlignmentResultFlags>(pPAI->arf | arfWriteMate1 | arfWriteMate2 | arfPrimary);
        if( pPAI->pQAI1 )
            pPAI->pQAI1->flags = static_cast<QAIflags>(pPAI->pQAI1->flags | qaiBest);
        if( pPAI->pQAI2 )
            pPAI->pQAI2->flags = static_cast<QAIflags>(pPAI->pQAI2->flags | qaiBest);
    }

    else
    {

#if TODO_CHOP_WHEN_DEBUGGED
        Qwarp* pQwx = m_pqb->QwBuffer.p + pPAI->iw;
//        if( (pQwx->sqId[pPAI->iq] & 0xFFFFFFFFFFFFFFFC) == 0x0000040800000038 )
        if( (cel >= 2) && (pPAI[0].arf & arfConcordant) && (pPAI[1].arf & arfConcordant) )
        {
            // dump the list
            for( UINT32 n=0; n<cel; ++n )
            {
                Qwarp* pQw1;
                Qwarp* pQw2;
                UINT32 iq1;
                UINT32 iq2;
                if( pPAI[n].pQAI1 )
                {
                    UINT32 iw = QID_IW(pPAI[n].pQAI1->qid);
                    iq1 = QID_IQ(pPAI[n].pQAI1->qid);
                    pQw1 = m_pqb->QwBuffer.p + iw;
                }
                if( pPAI[n].pQAI2 )
                {
                    UINT32 iw = QID_IW(pPAI[n].pQAI2->qid);
                    iq2 = QID_IQ(pPAI[n].pQAI2->qid);
                    pQw2 = m_pqb->QwBuffer.p + iw;
                }

                if( pPAI[n].pQAI1 && pPAI[n].pQAI2 )
                    CDPrint( cdpCD0, "%s: %d: 0x%016llx 0x%016llx", __FUNCTION__, n, pQw1->sqId[iq1], pQw2->sqId[iq2] );
                else
                {
                    if( pPAI[n].pQAI1 )
                        CDPrint( cdpCD0, "%s: %d: 0x%016llx (null)", __FUNCTION__, n, pQw1->sqId[iq1] );
                    else
                    if( pPAI[n].pQAI2 )
                        CDPrint( cdpCD0, "%s: %d: (null) 0x%016llx ", __FUNCTION__, n, pQw2->sqId[iq2] );
                    else
                        CDPrint( cdpCD0, "%s: %d: (null) (null)", __FUNCTION__, n );
                }
            }

            CDPrint( cdpCD0, "%s: sqId=0x%016llx cel=%d", __FUNCTION__, pQwx->sqId[pPAI->iq], cel );
        }
#endif

        // sort the list by mapping type (paired or unpaired) and Vmax
        qsort( pPAI, cel, sizeof(PAI), PAIComparer );

#if TODO_CHOP_WHEN_DEBUGGED
    for( UINT32 n=0; n<cel; ++n )
    {
        if( ((pPAI[n].arf & arfMate1Unmapped) == 0) && (pPAI[n].pQAI1 == NULL) )
            CDPrint( cdpCD0, __FUNCTION__ );
        if( ((pPAI[n].arf & arfMate2Unmapped) == 0) && (pPAI[n].pQAI2 == NULL) )
            CDPrint( cdpCD0, __FUNCTION__ );
    }
#endif

        // if the list contains at least one concordant pair, flag only the concordant mappings
        if( pPAI->arf & arfConcordant )
        {
            // the first concordant mapping is the "primary" mapping
            pPAI[0].arf = static_cast<AlignmentResultFlags>(pPAI[0].arf | arfWriteMate1 | arfWriteMate2 | arfPrimary);
            INT16 nPairsFlagged = 1;

            for( INT16 n=1; (nPairsFlagged<m_pab->aas.ACP.maxAg) && (static_cast<UINT32>(n)<cel) && (pPAI[n].arf & arfConcordant); ++n )
            {
                /* With bsDNA mappings, we align both strands of the Q sequence, so we may end up with adjacent pairs
                    that have identical mappings.  We check for that occurrence here. */
                if( (pPAI[n].pQAI1->Jf == pPAI[n-1].pQAI1->Jf) && (pPAI[n].pQAI2->Jf == pPAI[n-1].pQAI2->Jf) &&
                    (pPAI[n].pQAI1->subId == pPAI[n-1].pQAI1->subId) && (pPAI[n].pQAI2->subId == pPAI[n-1].pQAI2->subId) )
                {
                    continue;
                }

                // set write flags on the nth pair
                pPAI[n].arf = static_cast<AlignmentResultFlags>(pPAI[n].arf | arfWriteMate1 | arfWriteMate2);
                ++nPairsFlagged;
            }
        }
        else    // the list contains only mapped, unpaired mates
        {
            // flag each mapping until the configured limit is reached
            UINT32 nLimit1 = min2( pqair0->nAq, static_cast<UINT32>(m_pab->aas.ACP.maxAg) );
            UINT32 nLimit2 = min2( pqair1->nAq, static_cast<UINT32>(m_pab->aas.ACP.maxAg) );
            UINT32 nFlagged1 = 0;
            UINT32 nFlagged2 = 0;
            INT32 ipQAI1 = -1;
            INT32 ipQAI2 = -1;
            
            /* Flag mapped, unpaired mates for writing until the maximum per-mate count is reached.

               With bsDNA reads, we must ignore reads that have the "reverse-complement duplicate" (qaiRCdup)
                flag set.  Such reads are in the list only to make it possible for either Qf or Qrc to pair
                with the opposite mate and should not be flagged for writing.
            */
                
            for( UINT32 n=0; n<cel; ++n )
            {
                if( (nFlagged1 < nLimit1) && pPAI[n].pQAI1 && !(pPAI[n].pQAI1->flags & qaiRCdup) )
                {
                    // compact the pair list by moving the nth pQAI1 toward the start of the list
                    QAI* pqai1 = pPAI[n].pQAI1;
                    pPAI[n].pQAI1 = NULL;
                    while( pPAI[++ipQAI1].pQAI1 );
                    pPAI[ipQAI1].pQAI1 = pqai1;

                    nFlagged1++;
                }

                if( (nFlagged2 < nLimit2) && pPAI[n].pQAI2 && !(pPAI[n].pQAI2->flags & qaiRCdup) )
                {
                    // compact the pair list by moving the nth pQAI2 toward the start of the list
                    QAI* pqai2 = pPAI[n].pQAI2;
                    pPAI[n].pQAI2 = NULL;
                    while( pPAI[++ipQAI2].pQAI2 );
                    pPAI[ipQAI2].pQAI2 = pqai2;

                    nFlagged2++;
                }

                if( (nFlagged1 == nLimit1) && (nFlagged2 == nLimit2) )
                {
                    // ignore any remaining mappings in the PAI list
                    m_paiq.n = max2(ipQAI1,ipQAI2) + 1;
                    break;
                }
            }

            // finalize the flags
            if( pPAI[0].pQAI1 )
                pPAI[0].pQAI1->flags = static_cast<QAIflags>(pPAI[0].pQAI1->flags | qaiBest);
            if( pPAI[0].pQAI2 )
                pPAI[0].pQAI2->flags = static_cast<QAIflags>(pPAI[0].pQAI2->flags | qaiBest);
            if( pPAI[0].pQAI1 || pPAI[0].pQAI2 )
                pPAI[0].arf = static_cast<AlignmentResultFlags>(pPAI[0].arf | arfPrimary);

            for( UINT32 n=0; n<m_paiq.n; ++n )
            {
                /* SAM wants records for both mates, even if they are unmapped.  In picard's ValidateSamFile
                    (at least through v2.18.1), this conflicts with the secondary-mapping bit in the FLAG bitmap,
                    so IGNORE=INVALID_FLAG_NOT_PRIM_ALIGNMENT may be needed. */
                pPAI[n].arf = static_cast<AlignmentResultFlags>(pPAI[n].arf | (arfWriteMate1|arfWriteMate2));

                // ensure that the flags correspond to the QAI pointers
                pPAI[n].arf = pPAI[n].pQAI1 ? static_cast<AlignmentResultFlags>(pPAI[n].arf & ~arfMate1Unmapped) : static_cast<AlignmentResultFlags>(pPAI[n].arf | arfMate1Unmapped);
                pPAI[n].arf = pPAI[n].pQAI2 ? static_cast<AlignmentResultFlags>(pPAI[n].arf & ~arfMate2Unmapped) : static_cast<AlignmentResultFlags>(pPAI[n].arf | arfMate2Unmapped);
            }

#if TODO_CHOP_IF_UNUSED
            // if there are no mate 1 mappings...
            if( nFlagged1 == 0 )
            {
                // flag the first "pair" in the list so that the unmapped mate will be written
                pPAI->arf = static_cast<AlignmentResultFlags>(pPAI->arf | arfWriteMate1);
            }

            // if there are no mate 2 mappings...
            if( nFlagged2 == 0 )
            {
                // flag the first "pair" in the list so that the unmapped mate will be written
                pPAI->arf = static_cast<AlignmentResultFlags>(pPAI->arf | arfWriteMate2);
            }
#endif
        }
    }

#if TODO_CHOP_WHEN_DEBUGGED
    if( (pQwx->sqId[pPAI->iq] & 0xFFFFFFFFFFFFFFFC) == 0x0000040800006170 )
    {
        CDPrint( cdpCD0, "%s: filtered list:", __FUNCTION__ );
        for( UINT32 n=0; n<cel; ++n )
        {
            if( pPAI[n].arf & (arfWriteMate1|arfWriteMate2) )
            {
                if( pPAI[n].arf & arfConcordant )
                    CDPrint( cdpCD0, "%s: n=%u diffTLEN=%d concordant mate1: flags=0x%04x subId=%d Jf=%d mcm=%d, mate2: flags=0x%04x subId=%d Jf=%d mcm=%d", __FUNCTION__, n, pPAI[n].diffTLEN,
                    pPAI[n].pQAI1->flags, pPAI[n].pQAI1->subId, pPAI[n].pQAI1->Jf, pPAI[n].pQAI1->mcm,
                    pPAI[n].pQAI2->flags, pPAI[n].pQAI2->subId, pPAI[n].pQAI2->Jf, pPAI[n].pQAI2->mcm
                    );
                else
                {
                    if( pPAI[n].pQAI1 )
                        CDPrint( cdpCD0, "%s: n=%u arf=0x%08x mate1 flags=0x%04x subId=%d Jf=%d", __FUNCTION__, n, pPAI[n].arf,
                                            pPAI[n].pQAI1->flags, pPAI[n].pQAI1->subId, pPAI[n].pQAI1->Jf );
                    else
                        CDPrint( cdpCD0, "%s: n=%u arf=0x%08x mate1 (pQAI1 == null)", __FUNCTION__, n, pPAI[n].arf );
                    if( pPAI[n].pQAI2 )
                        CDPrint( cdpCD0, "%s: n=%u arf=0x%08x mate2 flags=0x%04x subId=%d Jf=%d", __FUNCTION__, n, pPAI[n].arf,
                                            pPAI[n].pQAI2->flags, pPAI[n].pQAI2->subId, pPAI[n].pQAI2->Jf );
                    else
                        CDPrint( cdpCD0, "%s: n=%u arf=0x%08x mate2 (pQAI2 == null)", __FUNCTION__, n, pPAI[n].arf );
                }
            }

            //if( ((pPAI[n].arf & arfMate1Unmapped) == 0) && (pPAI[n].pQAI1 == NULL) )
            //    CDPrint( cdpCD0, __FUNCTION__ );
            //if( ((pPAI[n].arf & arfMate2Unmapped) == 0) && (pPAI[n].pQAI2 == NULL) )
            //    CDPrint( cdpCD0, __FUNCTION__ );
        }
        CDPrint( cdpCD0, __FUNCTION__ );
    }
#endif

    // now that we know which mappings to report, we can compute mapping quality (MAPQ) probabilities for them
    computeMappingQualities();
}

/// [private] method findPairs
void tuClassifyP2::findPairs()
{
    // performance metrics
    pairTypeCounts ptc = { 0 };

    // get a reference to the next pair
    UINT32 qid0 = InterlockedExchangeAdd( m_pqid, 2 );
    while( qid0 < m_pqb->DB.nQ )
    {
        // reuse the PAI buffer
        m_paiq.n = 0;


#if TODO_CHOP_WHEN_DEBUGGED
        // sanity check
        if( pQw->sqId[iq] != (pQw->sqId[iq+1] ^ AriocDS::SqId::MaskMateId) )
            throw new ApplicationException( __FILE__, __LINE__, "unpaired sqIds 0x%016llx and 0x%016llx", pQw->sqId[iq], pQw->sqId[iq+1] );
#endif

#if TODO_CHOP_WHEN_DEBUGGED
        UINT32 iw = QID_IW(qid0);
        INT16 iq = QID_IQ(qid0);
        Qwarp* pQw = m_pqb->QwBuffer.p + iw;
        if( (pQw->sqId[iq] | 1) == (TRACE_SQID | 1) )
            CDPrint( cdpCD0, "%s: sqId=0x%016llx", __FUNCTION__, pQw->sqId[iq] );
#endif

        // point to the QAIReference struct for the first Q sequence
        QAIReference* pqair0 = m_pqb->ArBuffer.p + qid0;

        // point to the QAIReference struct for the second Q sequence
        QAIReference* pqair1 = pqair0 + 1;  // (the pairs are ordered in even-odd positions in the Qwarp)

        // count the number of paired (concordant or discordant) mappings
        INT32 nPairedMappings;

        // if both ends have at least one mapping ...
        if( pqair0->nAq && pqair1->nAq )
        {
            // point to the first QAI struct in the set of alignments for each end
            QAI* pQAI1 = m_pqb->AaBuffer.p + pqair0->ofsQAI;
            QAI* pQAI2 = m_pqb->AaBuffer.p + pqair1->ofsQAI;

            if( (pqair0->nAq == 1) && (pqair1->nAq == 1) )
                nPairedMappings = findPairUnique( qid0, pQAI1, pQAI2, &ptc );
            else
                nPairedMappings = findPairMultiple( qid0, pQAI1, pQAI2, pqair0, pqair1, &ptc );
        }
        else
            nPairedMappings = 0;

        // if we have not already found the configured maximum number of paired mappings ...
        if( nPairedMappings < m_pab->aas.ACP.maxAg )
        {
            // count unmapped pairs (one or both ends unmapped)
            if( nPairedMappings == 0 )
            {
                if( (pqair0->nAq == 0) || (pqair1->nAq == 0) )
                    ptc.nPairsUnmapped++ ;
                else
                if( pqair0->nAq || pqair1->nAq )
                    ptc.nPairsRejected++ ;
            }

            /* record results for unmapped pairs (mates that are either unmapped or individually mapped
                but not in a paired mapping) */
            findUnpairedMates( qid0, pqair0, pqair1, nPairedMappings, &ptc );
        }

        // filter the mappings for reporting
        filterMappings( pqair0, pqair1 );

        // copy the list of paired mappings to a consolidated buffer
        appendPAIforQ();

        // get another pair of mappings
        qid0 = InterlockedExchangeAdd( m_pqid, 2 );
    }

    // performance metrics
    InterlockedExchangeAdd( &AriocBase::aam.sam.p.nPairs_Concordant1, ptc.nConcordant1 );
    InterlockedExchangeAdd( &AriocBase::aam.sam.p.nPairs_Concordant2, ptc.nConcordant2 );
    InterlockedExchangeAdd( &AriocBase::aam.sam.p.nPairs_Discordant, ptc.nDiscordant );
    InterlockedExchangeAdd( &AriocBase::aam.sam.p.nPairs_Rejected, ptc.nPairsRejected );
    InterlockedExchangeAdd( &AriocBase::aam.sam.p.nPairs_Unmapped, ptc.nPairsUnmapped );
    InterlockedExchangeAdd( &AriocBase::aam.sam.p.nMates_Unmapped, ptc.nMatesUnmapped );
    InterlockedExchangeAdd( &AriocBase::aam.sam.p.nMates_Mapped1, ptc.nMatesMapped1 );
    InterlockedExchangeAdd( &AriocBase::aam.sam.p.nMates_Mapped2, ptc.nMatesMapped2 );
    InterlockedExchangeAdd( &AriocBase::aam.sam.p.nMates_MappedU1, ptc.nMatesMappedU1 );
    InterlockedExchangeAdd( &AriocBase::aam.sam.p.nMates_MappedU2, ptc.nMatesMappedU2 );
}

#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Find pairs
/// </summary>
void tuClassifyP2::main()
{

#if TODO_CHOP_WHEN_DEBUGGED
    HiResTimer hrt;
#endif

    findPairs();



#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD0, "[%d] %s: after findPairs: %dms", m_pqb->pgi->deviceId, __FUNCTION__, hrt.GetElapsed(false) );
#endif
}
#pragma endregion
