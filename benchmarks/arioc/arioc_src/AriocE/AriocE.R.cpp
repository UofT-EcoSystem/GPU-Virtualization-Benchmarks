/*
  AriocE.R.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region static methods
/// [private static] method CvalueComparer_nJ
int AriocE::CvalueComparer_nJ( const void* a, const void* b )
{
    const Cvalue* pa = reinterpret_cast<const Cvalue*>(a);
    const Cvalue* pb = reinterpret_cast<const Cvalue*>(b);

    // order by number of J values descending
    return (pb->nJ > pa->nJ) ? 1 : ((pb->nJ < pa->nJ) ? -1 : 0);
}
#pragma endregion

#pragma region private methods
/// [private] method preallocateCHJ
void AriocE::preallocateCHJ( RaiiFile& rfFile, RaiiSyncEventObject& rseoComplete, INT64 cb, const char cLUTtype[2] ) 
{
    // open/create the output file
    char outFileSpec[FILENAME_MAX];
    strcpy_s( outFileSpec, FILENAME_MAX, this->Params.OutFilespecStubLUT );
    strcat_s( outFileSpec, FILENAME_MAX, cLUTtype );            // "C", "H", or "J"
    strcat_s( outFileSpec, FILENAME_MAX, this->Params.LUTtypeStub );
    strcat_s( outFileSpec, FILENAME_MAX, ".sbf" );
    rfFile.OpenNoTruncate( outFileSpec );

    // preallocate space in the C output file
    rfFile.Preallocate( cb, &rseoComplete );
}

/// [private] method writeCHJ
INT64 AriocE::writeCHJ( RaiiFile& rfFile, RaiiSyncEventObject* prseoComplete, const void* buf, INT64 cb, INT32* pmsElapsed )
{
    HiResTimer hrt(ms);

    CDPrint( cdpCD3, "%s for %s...", __FUNCTION__, rfFile.FileSpec.p );

    // wait for the file preallocation to complete
    prseoComplete->Wait( FILE_PREALLOCATE_TIMEOUT );

    // write the file
    INT64 cbWritten = rfFile.ConcurrentFill( 0, buf, cb, 4 );       // (use 4 concurrent threads)

    // performance metrics
    InterlockedExchangeAdd( reinterpret_cast<volatile UINT32*>(pmsElapsed), hrt.GetElapsed(false) );

    CDPrint( cdpCD3, "%s completed for %s", __FUNCTION__, rfFile.FileSpec.p );

    return cbWritten;
}

/// [private] method sortJcpu
void AriocE::sortJcpu()
{
    WinGlobalPtr<tuSortJcpu*> ptuSortJ( this->Params.nLPs, true );
    volatile INT64 nSortedH = 0;

    CDPrint( cdpCD0, "%s: sorting J lists on %d worker threads...", __FUNCTION__, this->Params.nLPs );

    for( INT32 n=0; n<this->Params.nLPs; ++n )
    {
        // launch a worker thread
        ptuSortJ.p[n] = new tuSortJcpu( &this->Params, m_H.Count, &nSortedH );
        ptuSortJ.p[n]->Start();
    }

    // wait for the worker threads to exit
    for( INT32 n=0; n<this->Params.nLPs; ++n )
        ptuSortJ.p[n]->Wait( INFINITE );

    // destruct the tuSortJ instances
    for( INT32 n=0; n<this->Params.nLPs; ++n )
        delete ptuSortJ.p[n];

    CDPrint( cdpCD0, "%s: J list sort complete", __FUNCTION__ );
}

/// [private] method sortJgpu
void AriocE::sortJgpu()
{
    CDPrint( cdpCD0, "%s: sorting J lists on GPU...", __FUNCTION__ );

    // look for the optional "cgaReserved" Xparam
    INT64 cbCgaReserved = CUDAMINRESERVEDGLOBALMEMORY;
    INT32 i = m_pam->Xparam.IndexOf( "cgaReserved" );
    if( i >= 0 )
        cbCgaReserved = max2( CUDAMINRESERVEDGLOBALMEMORY, m_pam->Xparam.Value(i) );

    tuSortJgpu sjg( m_J.p, m_celJ, cbCgaReserved );
    sjg.Start();
    sjg.Wait();

    CDPrint( cdpCD0, "%s: J list sort complete", __FUNCTION__ );
}

/// [private] method accumulateJlistSize
void AriocE::accumulateJlistSize( size_t* _piC, UINT32* _pnCard )
{
    Cvalue* pC = m_C.p + *_piC;

    // count J values
    const UINT32 nJ = pC->nJ;
    m_nJ += nJ;

    // accumulate the J-list cardinality
    DWORD iCard = 0;
    _BitScanReverse( &iCard, nJ );     // (same as log2(nJ))
    _pnCard[iCard]++ ;

    // save a reference to the specified Cvalue
    m_CH.p[pC->hashKey] = static_cast<UINT32>(*_piC);

    // increment the C-list index
    (*_piC)++;
}

/// [private] method computeJlistSizes
void AriocE::computeJlistSizes()
{
    CDPrint( cdpCD3, "%s...", __FUNCTION__ );

    HiResTimer hrt(ms);

    // add the hash key to each "row" in the C table
    for( size_t iC=0; iC<m_C.Count; ++iC )
        m_C.p[iC].hashKey = static_cast<UINT32>(iC);

CDPrint( cdpCD0, "%s: before C list sort", __FUNCTION__ );

    // sort the C list in descending order by the number of J values associated with each hash key
    qsort( m_C.p, m_C.Count, sizeof(Cvalue), CvalueComparer_nJ );
CDPrint( cdpCD0, "%s: after C list sort", __FUNCTION__ );

    /* Find the total number of hash keys that have more associated J values than can be represented
        in the nJ bitfield of the associated H value (hash key). */

    // create a table of J-list cardinality counts
    WinGlobalPtr<UINT32> nCard( 32, true );

    // count the J values
    m_nJ = 0;
    INT64 nJvaluesLong = 0;     // number of J values in "long" lists where list size is stored in the list (not in the Hvalue)

    // accumulate the number of J values in the C list (which is sorted in descending order by nJ)

    // count J values in the "long" lists
    size_t iC = 0;
    while( iC < m_C.Count )
    {
        if( m_C.p[iC].nJ <= HVALUE_MAX_NJ )
            break;

        // accumulate J-list size and increment the list pointer
        accumulateJlistSize( &iC, nCard.p );
    }

    /* At this point the ih'th Cvalue is the first of the "short" J lists. */

    nJvaluesLong = m_nJ;    // number of J values in "long" lists
    m_iCshort = iC;         // index of first "short" list where list size is stored in the Hvalue (not in the list)

    // count J values in the "short" lists
    while( iC < m_C.Count )
    {
        if( m_C.p[iC].nJ == 0 )
            break;

        accumulateJlistSize( &iC, nCard.p );
    }

    // save the index of the first element in the C list where nJ == 0
    m_iCzero = iC;
    
    CDPrint( cdpCD0, "%s: m_nJ=%lld (%lld+%lld) m_iCshort=%lld m_iCzero=%lld", __FUNCTION__, m_nJ, nJvaluesLong, m_nJ-nJvaluesLong, m_iCshort, m_iCzero );

    // summarize J-list cardinalities
    CDPrint( cdpCD1, "%s: log2(nJ)  # J-lists", __FUNCTION__ );
    for( size_t n=0; n<nCard.Count; ++n )
    {
        if( nCard.p[n] )
            CDPrint( cdpCD1, "%s:%9llu%11u", __FUNCTION__, n, nCard.p[n] );
    }

    /* Allocate the J table.
    
       The total number of elements in the J table is the sum of...
        - 1             unused element 0 (so that no J list has a start offset of zero)
        - m_nJ          J values
        - m_iCshort     index of the first "short" J list (i.e., count of "long" J lists)
    */

    INT64 celJbuffer = 1 + m_nJ + m_iCshort;
    if( celJbuffer > HVALUE_MAX_OFSJ )
        throw new ApplicationException( __FILE__, __LINE__, "%s: too many J values (%lld) for J table (maximum = %lld)", __FUNCTION__, celJbuffer, HVALUE_MAX_OFSJ );

    m_J.Realloc( celJbuffer, true );

    // initialize J-list sizes for "long" lists
    m_celJ = 1;       // start at offset 1
    Cvalue* pC = m_C.p;
    Cvalue* pClimit = pC + m_iCshort;
    while( pC < pClimit )
    {
        // save the offset in the J table where the J list will start
        m_H.p[pC->hashKey].ofsJ = m_celJ;

        // save the J-list size in the element immediately preceding the values in the J list
        m_J.p[m_celJ].J = pC->nJ;

        // accumulate the total number of elements in the J list
        m_celJ += (pC->nJ + 1);

        // sanity check
        if( m_celJ > static_cast<INT64>(m_J.Count) )
            throw new ApplicationException( __FILE__, __LINE__, "%s: J table size mismatch", __FUNCTION__ );

        pC++;
    }

    // initialize J-list sizes for "short" lists
    pClimit = m_C.p + m_iCzero;
    while( pC < pClimit )
    {
        // save a reference to the offset in the J table where the J list will start
        m_H.p[pC->hashKey].ofsJ = m_celJ;

        // save the J-list size in the H table
        m_H.p[pC->hashKey].nJ = pC->nJ;

        // accumulate the total number of elements in the J list
        m_celJ += pC->nJ;

        // sanity check
        if( m_celJ > static_cast<INT64>(m_J.Count) )
            throw new ApplicationException( __FILE__, __LINE__, "%s: J table size mismatch", __FUNCTION__ );

        pC++;
    }

    // performance metrics
    AriocE::PerfMetrics.msJlistSizes = hrt.GetElapsed(false);
    INT64 celJused = m_celJ - 1;        // number of elements used by J lists, excluding the 0th element in the buffer
    CDPrint( cdpCD0, "%s exits: J table uses %lld elements for %lld J values (%5.3f%% for list counts in J lists)", __FUNCTION__, celJused, m_nJ, 100.0f*(celJused-m_nJ)/celJused );
}

/// [private] setEOLflags
void AriocE::setEOLflags()
{
    for( size_t h=0; h<m_H.Count; ++h )
    {
        // point to the h'th element in the H table
        Hvalue8* pH = m_H.p + h;

        // do nothing if the h'th hash key is unused
        if( pH->ofsJ == 0 )
            continue;

        // point to the 0th element in the corresponding J list
        Jvalue8* pJ = m_J.p + pH->ofsJ;


        
#if TODO_CHOP_WHEN_DEBUGGED
        if( h == 0x22 )
            CDPrint( cdpCD0, "%s: pH->ofsJ=0x%016llx nJ=%u nJx=%u, pJ->J=0x%08x s=%u subId=%u tag=0x%08x",
                             __FUNCTION__, pH->ofsJ, (UINT32)pH->nJ, (UINT32)pH->nJx, (UINT32)pJ->J, (UINT32)pJ->s, (UINT32)pJ->subId, (UINT32)pJ->tag );
#endif



        // extract the list size from the H table
        UINT32 nJ = pH->nJ;

        // if we have a "long" list, extract the list size from the first element of the J list
        if( nJ == 0 )
        {
            nJ = pJ->J;

            // sanity check
            if( nJ > HVALUE_MAX_NJX )
                throw new ApplicationException( __FILE__, __LINE__, "%s: J list for hash key 0x%08X contains %u values (maximum = %u)", __FUNCTION__, h, nJ, HVALUE_MAX_NJX );

            ++pJ;
        }

        // set the end-of-list flag on the last element in the J list
        pJ[nJ-1].x = 1;
    }
}

/// [private] method countBBJ
INT64 AriocE::countBBJ( WinGlobalPtr<INT64>* pnBBJ )
{
    /* count big bucket lists on worker threads */
    {
        CDPrint( cdpCD0, "%s: counting big-bucket lists...", __FUNCTION__ );

        WinGlobalPtr<tuCountBB*> ptuCountBB( this->Params.nLPs, true );
        volatile INT64 h = 0;
        volatile INT64 nBB = 0;

        for( INT32 n=0; n<this->Params.nLPs; ++n )
        {
            // launch a worker thread
            ptuCountBB.p[n] = new tuCountBB( m_maxJ, &m_H, m_H.Count, &m_J, &h, &nBB, pnBBJ );
            ptuCountBB.p[n]->Start();
        }

        // wait for the worker threads to exit
        for( INT32 n=0; n<this->Params.nLPs; ++n )
            ptuCountBB.p[n]->Wait( INFINITE );

        // destruct the tuCountBB instances
        for( INT32 n=0; n<this->Params.nLPs; ++n )
            delete ptuCountBB.p[n];

        CDPrint( cdpCD0, "%s: counted %u big-bucket lists", __FUNCTION__, nBB );
    }

    // count the total number of big-bucket lists
    INT64 totalBBJ = 0;
    for( INT32 subId=0; subId<=static_cast<INT32>(Jvalue8::bfMaxVal_subId); ++subId )
    {
        if( pnBBJ->p[subId] )
        {
            CDPrint( cdpCD3, "%s: nBBJ.p[%d]=%u", __FUNCTION__, subId, pnBBJ->p[subId] );
            totalBBJ += pnBBJ->p[subId];
        }
    }

    CDPrint( cdpCD3, "%s: totalBBJ=%u", __FUNCTION__, totalBBJ );

    return totalBBJ;
}

/// [private] method buildBBJlists
void AriocE::buildBBJlists( WinGlobalPtr<INT64>* pnBBJ )
{
    CDPrint( cdpCD0, "%s: identifying J values in big-bucket lists...", __FUNCTION__ );

    /* This function is called by identifyBBJ() to create a list of J values for each "big bucket". */

    // build a table of offsets into the BBHJ buffer
    WinGlobalPtr<INT64> ofsBBJ( Jvalue8::bfMaxVal_subId+1, true );
    for( INT32 subId=1; subId<=static_cast<INT32>(Jvalue8::bfMaxVal_subId); ++subId )
    {
        // the offset for the current subId is the exclusive prefix sum of the previous BBJ list counts
        ofsBBJ.p[subId] = ofsBBJ.p[subId-1] + pnBBJ->p[subId-1];
    }
    
    // iterate through the list of H values
    m_nBBHJ = 0;
    for( size_t h=0; h<m_H.Count; ++h )
    {
        // get a pointer to the H value corresponding to the h'th hash key
        Hvalue8* pH = m_H.p + h;

        // do nothing if the h'th hash key is unused
        if( pH->ofsJ == 0 )
            continue;

        // get a pointer to the start of the J list for the current H value
        Jvalue8* pJ = m_J.p + pH->ofsJ;

        UINT32 nJ = pH->nJ;
        if( nJ == 0 )
        {
            nJ = pJ->J;     // copy the J list count from the first element in the list
            pJ++ ;          // point to the first J value in the list
        }

        // if the J-list size exceeds the configured threshold...
        if( nJ > static_cast<UINT32>(m_maxJ) )
        {
            // append to the list of J values in "big bucket" lists
            for( UINT32 ij=0; ij<nJ; ++ij )
            {
                HJpair* phj = m_BBHJ.p + ofsBBJ.p[pJ->subId];
                phj->h = static_cast<UINT32>(h);                // clamp to 32 bits
                phj->ij = ij;
                phj->j = *pJ;

                ofsBBJ.p[pJ->subId]++;
                ++pJ;
            }
            m_nBBHJ += nJ;
        }
    }

    CDPrint( cdpCD3, "%s: %lld J values in big-bucket J lists", __FUNCTION__, m_nBBHJ );
}

/// [private] sortBBHJ
void AriocE::sortBBHJ( WinGlobalPtr<INT64>* pnBBJ )
{
    HiResTimer hrt;

    // build a table of offsets into the BBHJ buffer
    WinGlobalPtr<INT64> ofsBBJ( Jvalue8::bfMaxVal_subId+1, true );
    for( INT32 subId=1; subId<=static_cast<INT32>(Jvalue8::bfMaxVal_subId); ++subId )
        ofsBBJ.p[subId] = ofsBBJ.p[subId-1] + pnBBJ->p[subId-1];

    CDPrint( cdpCD0, "%s: sorting big-bucket lists...", __FUNCTION__ );

    WinGlobalPtr<tuSortBB*> ptuSortBB( this->Params.nLPs, true );
    volatile UINT32 nSorted = 0;

    for( INT32 n=0; n<this->Params.nLPs; ++n )
    {
        // launch a worker thread
        ptuSortBB.p[n] = new tuSortBB( &m_BBHJ, pnBBJ, &ofsBBJ, &nSorted );
        ptuSortBB.p[n]->Start();
    }

    // wait for the worker threads to exit
    for( INT32 n=0; n<this->Params.nLPs; ++n )
        ptuSortBB.p[n]->Wait( INFINITE );

    // destruct the tuSortBB instances
    for( INT32 n=0; n<this->Params.nLPs; ++n )
        delete ptuSortBB.p[n];

    // performance metrics
    AriocE::PerfMetrics.msSortBB = hrt.GetElapsed(false);

    CDPrint( cdpCD0, "%s: sorted %lld big-bucket J values", __FUNCTION__, m_nBBHJ );
}

/// [private] identifyBBJ
void AriocE::identifyBBJ()
{
    // count big-bucket J values and initialize a table of per-subId counts 
    WinGlobalPtr<INT64> nBBJ( Jvalue8::bfMaxVal_subId+1, true );
    INT64 nnBBJ = countBBJ( &nBBJ );
    CDPrint( cdpCD3, "%s: total big-bucket J values = %lld/%lld (%5.3f%%)", __FUNCTION__, nnBBJ, m_nJ, (100.0f*nnBBJ)/m_nJ );

    // build a table of J values from the J lists (hashtable buckets) whose cardinality exceeds the configured maximum
    m_BBHJ.Realloc( nnBBJ, true );
    buildBBJlists( &nBBJ );

    // return if there are no J values in big-bucket lists
    if( m_nBBHJ == 0 )
        return;

    /* At this point m_BBHJ contains J values from one or more "big bucket" J lists. */

    // sort the BBHJ lists
    sortBBHJ( &nBBJ );

    /* Build a list of big-bucket regions.  Each item in the list represents one or more adjacent locations
        on the reference, each of which hashes to a "big" hashtable bucket. */

    // initialize the big-bucket region-info table
    m_BBRI.Realloc( m_nBBHJ, true );
    BBregionInfo* pbbri = m_BBRI.p;

    // reset the region info
    pbbri->n = 1;
    pbbri->ofs = 0;

    for( INT64 iBB=1; iBB<m_nBBHJ; ++iBB )
    {
        if( m_BBHJ.p[iBB].j.J != (m_BBHJ.p[iBB-1].j.J+1) )
        {
            /* at this point, the iBB'th bucket represents the start of a different region */
            pbbri++ ;           // point to a new region-info struct
            pbbri->n = 1;       // reset the region info
            pbbri->ofs = iBB;
        }
        else
        {
            /* at this point, the iBB'th bucket's location is adjacent to that of the previous bucket */
            pbbri->n++ ;        // track the number of adjacent buckets in the current region
        }
    }

    // resize the region-info buffer
    m_BBRI.Realloc( pbbri - m_BBRI.p, false );

    // do simple statistics
    double avg = static_cast<double>(m_nBBHJ) / m_BBRI.Count;
    UINT32 navg = static_cast<UINT32>(avg);
    UINT32 minRegionSize = _I16_MAX;
    UINT32 maxRegionSize = 0;
    UINT64 totalDiff2 = 0;
    pbbri = m_BBRI.p;
    for( size_t i=0; i<m_BBRI.Count; ++i )
    {
        // track maximum and minimum region size
        if( pbbri->n > maxRegionSize )
            maxRegionSize = pbbri->n;
        else
        if( pbbri->n < minRegionSize )
            minRegionSize = pbbri->n;

        // track total squared difference between mean and region size
        UINT32 diff = pbbri->n - navg;
        totalDiff2 += (diff*diff);

        // iterate
        pbbri++ ;
    }

    // compute standard deviation
    double sd = sqrt( static_cast<double>(totalDiff2) / m_BBRI.Count );

    CDPrint( cdpCD0, "%s: big bucket region info: %llu regions (size %u-%u, avg=%3.1f sd=%3.1f)", __FUNCTION__, m_BBRI.Count, minRegionSize, maxRegionSize, avg, sd );
}

/// [private] setBBJlistFlags
void AriocE::setBBJlistFlags()
{
    HiResTimer hrt;

    CDPrint( cdpCD3, "%s...", __FUNCTION__ );


#if TODO_CHOP_WHEN_DEBUGGED
INT64 nFlagged = 0;
#endif


    /* Flag all but every 10th J value for removal from the table.  The idea here is that the J values for a Q sequence
        that spans a big-bucket region need to cover 30 consecutive positions in order for a "hit" in the region to be
        guaranteed.
    */
    BBregionInfo* pbbri = m_BBRI.p;
    for( size_t iRegion=0; iRegion<m_BBRI.Count; ++iRegion )
    {
        HJpair* phj = m_BBHJ.p + pbbri->ofs;
        for( UINT32 ijx=0; ijx<pbbri->n; ++ijx )
        {
            if( (ijx % 10) != 9 )
            {
                // get a pointer to the H value corresponding to the h'th hash key
                Hvalue8* pH = m_H.p + phj->h;

                // get a pointer to the start of the J list for the current H value
                Jvalue8* pJ = m_J.p + pH->ofsJ;     // point to the number of J values in the list

                // get the number of J values in the list
                UINT32 nJ = pH->nJ;
                if( nJ == 0 )
                {
                    nJ = pJ->J;     // copy the J list count from the first element in the list
                    pJ++ ;          // point to the first J value in the list
                }

                // sanity check
                if( nJ < static_cast<UINT32>(m_maxJ) )
                    throw new ApplicationException( __FILE__, __LINE__, "unexpected nJ=%d for hashkey=0x%08x (should be at least %d)", pH->nJ, phj->h, m_maxJ );

                // track the number of excluded J values for the current H value (hash key)
                pH->nJx++ ;

                // indicate that the J value is to be excluded
                pJ[phj->ij].tag = 1;

                /* If the just-excluded J value is at the end of the J list and there is at least one
                    non-excluded J value remaining in the list... */
                if( pJ[phj->ij].x && (pH->nJx < nJ) )
                {
                    // scan toward the start of the J list to find the first non-excluded J value
                    Jvalue8* pjx = pJ + phj->ij - 1;
                    while( (pjx >= pJ) && pjx->tag )
                        --pjx;

                    if( pjx >= pJ )
                    {
                        // set the end-of-list flag
                        pjx->x = 1;
                    }
                    else
                        throw new ApplicationException( __FILE__, __LINE__, "cannot set end-of-list flag for hashkey 0x%08x", phj->h );
                }

#if TODO_CHOP_WHEN_DEBUGGED
                nFlagged++ ;
#endif

            }

            // point to the next big-bucket J value in the region
            phj++ ;
        }

        // point to the next big-bucket region
        pbbri++ ;
    }

#if TODO_CHOP_WHEN_DEBUGGED
CDPrint( cdpCD0, "%s: nFlagged=%lld", __FUNCTION__, nFlagged );
#endif


#if TODO_CHOP_IF_UNUSED
    /*** CAN WE JUST GET RID OF THIS CRAP???? ***/




    HiResTimer hrt(ms);

    CDPrint( cdpCD3, "%s...", __FUNCTION__ );

    /* Now we use the Jvalue.tag field to flag J-list elements to be excluded. */

    for( INT64 h=0; h<m_nH; ++h )
    {

#if TODO_CHOP_WHEN_DEBUGGED
        UINT32 nFlaggedForExcision = 0;
#endif

        // get a pointer to the H value corresponding to the h'th hash key
        Hvalue8* pH = m_H.p + h;

        // do nothing if the h'th hash key is unused
        if( pH->ofsJ == 0 )
            continue;

        // get a pointer to the start of the J list for the current H value
        Jvalue8* pJ = m_J.p + pH->ofsJ;

        // get the number of values in the h'th J list
        INT32 nJ = static_cast<INT32>(pH->nJ);
        if( nJ == 0 )
        {
            nJ = pJ->J;     // copy the J list count from the first element in the list
            pJ++ ;          // point to the first J value in the list
        }

        // if the J list contains more than the configured maximum number of values ...
        if( nJ > m_maxJ )
        {

#if TODO_CHOP_WHEN_DEBUGGED
            INT32 jLast = -1;
            for( INT32 j=0; j<nJ; j++ )
            {
                if( pJ[j].x == 0 )
                    jLast = j;              // offset of last non-excised J value
            }

            for( INT32 j=0; j<nJ; j++ )
            {
                if( pJ[j].x )
                    ++nFlaggedForExcision;
            }

            for( INT32 j=0; j<nJ; j++ )
            {
                if( j && (pJ[j].J <= pJ[j-1].J ) )
                {
                    CDPrint( cdpCD0, "%s: h=0x%08x list is not sorted at j=%d", __FUNCTION__, h, j );
                    break;
                }
            }
#endif

            INT32 iLo = 0;
            INT32 iHi = nJ - 1;

            do
            {
                // find the first J value marked for exclusion (starting at the start of the list)
                for( ; iLo<=iHi; iLo++ )
                {
                    if( pJ[iLo].tag )
                        break;
                }

                // find the last J value not marked for exclusion (starting at the end of the list)
                for( ; iHi>iLo; iHi-- )
                {
                    if( pJ[iHi].tag == 0 )
                        break;
                }

                if( iLo < iHi )
                {

                    
#if TODO_CHOP_WHEN_DEBUGGED
                    CDPrint( cdpCD0, "%s: h=0x%08x copying pJ[%d] to pJ[%d] (0x%016llx)", __FUNCTION__, h, iHi, iLo, reinterpret_cast<UINT64*>(pJ)[iHi] );
#endif


                    // copy the J value and bump the low and high list indices
                    pJ[iLo++] = pJ[iHi--];
                }
            }
            while( iLo < iHi );

            // include the last J value if it is not flagged to be excluded from the list
            if( (iLo == iHi) && !pJ[iLo].tag )
                ++iLo;

            /* At this point iLo references the J value after the last one in the list. */
            if( iLo < nJ )
            {
                // save the new J-list count
                pH->nJx = iLo;
                
                // set a flag to indicate that one or more J values should be excluded
                UINT8* px = m_xH.p + (h>>3);
                *px |= (1 << (h&7));
            }


#if TODO_CHOP_WHEN_DEBUGGED
            CDPrint( cdpCD0, "%s: bucket for hash key 0x%08llx now contains %d(%u)/%d J values (nJ=%d nJx=%d jLast=%d)", __FUNCTION__, h, iLo, nFlaggedForExcision, pJ[-1].J, nJ, pH->nJx, jLast );
            if( (iLo + nFlaggedForExcision) != nJ )
            {
                CDPrint( cdpCD0, "%s: *** bucket count error ***", __FUNCTION__ );
            }


#endif


#if TODO_RANDOM_SHUFFLE
            if( nJ > 16 )       // TODO: PARAMETERIZE THIS?
            {
                // use the current LUT key to "seed" a pseudorandom sequence of hash values
                UINT32 hash = h;

        // TODO: CHOP WHEN DEBUGGED:        CDPrint( cdpCD0, "AriocE::t: h=%lld nJ=%d", h, nJ );
                // shuffle the J list
                for( INT32 n=(nJ-1); n>0; --n )
                {
                    hash = AriocE::Hash6432( 0xE8D7C6B5A4938271 | static_cast<UINT64>(hash) );
                    INT32 r = hash % nJ;   // random offset between 0 and nJ-1
        // TODO: CHOP WHEN DEBUGGED:            CDPrint( cdpCD0, " AriocE::editJ: r=%d", r );

                    Jvalue8 Jtemp = pJ[r];    // swap the nth and rth Jvalues
                    pJ[r] = pJ[n];
                    pJ[n] = Jtemp;
                }
            }
#endif

            // performance metrics
            AriocE::PerfMetrics.nJclamped++ ;
        }
    }
#endif

#if TODO_CHOP_IF_UNUSED
    // flag in-list counts for exclusion
    for( INT64 h=0; h<m_nH; h++ )
    {
        // point to the h'th element in the H table
        Hvalue8* pH = m_H.p + h;

        // do nothing if the h'th hash key is unused
        if( pH->ofsJ == 0 )
            continue;

        // point to the 0th element in the corresponding J list
        Jvalue8* pJ = m_J.p + pH->ofsJ;

        // extract the list size from the H table
        UINT32 nJ = pH->nJ;

        // if we have a "long" list, extract the list size from the first element of the J list
        if( nJ == 0 )
        {
            nJ = pJ->J;

            // sanity check
            if( nJ > MAX_HVALUE_NJ )
                throw new ApplicationException( __FILE__, __LINE__, "%s: J list for hash key 0x%08X contains %u values (maximum = %u)", __FUNCTION__, h, nJ, MAX_HVALUE_NJ );

            ++pJ;
        }

        // set the end-of-list flag on the last element in the J list
        pJ[nJ-1].x = 1;
    }
#endif

    CDPrint( cdpCD3, "%s completed", __FUNCTION__ );
}

#if TODO_CHOP_WHEN_DEBUGGED
UINT8 cjOrdinal = 0;
#endif

/// [private] method encodeR
void AriocE::encodeR( SAMConfigWriter* pscw )
{
    // prepare to encode the input file(s)
    RaiiSemaphore semEncoderWorkers( this->Params.nLPs, this->Params.nLPs );

    /* Encode input files:
        - there are three output files (raw.sbf, sqm.sbf, a21.sbf) for each input file
        - multiple input files are processed concurrently
    */
    {
        HiResTimer hrt(ms);

        // there are two reference sequences (+ and -) for each input sequence
        UINT32 nSq = 2 * this->Params.ifgRaw.InputFile.n;
        WinGlobalPtr<baseEncode*> ptuBaseEncode( nSq, true );

        CDPrint( cdpCD0, "%s: encoding %u file%s (%u sequence%s) (%d CPU thread%s available)...",
                            __FUNCTION__,
                            this->Params.ifgRaw.InputFile.n, PluralS(this->Params.ifgRaw.InputFile.n),
                            nSq, PluralS(nSq),
                            this->Params.nLPs, PluralS(this->Params.nLPs) );

        UINT32 n = 0;
        while( n < nSq )
        {
            // use the same file for both strands
            UINT32 i = n / 2;

            // wait for a CPU worker thread to become available
            semEncoderWorkers.Wait( m_encoderWorkerThreadTimeout );

            // launch a worker thread for the ith input file
            switch( this->Params.InputFileFormat )
            {
                case SqFormatFASTA:
                    ptuBaseEncode.p[n] = new tuEncodeFASTA( &this->Params, sqCatRplus, i, &semEncoderWorkers, pscw );
                    break;

                case SqFormatFASTQ:
                    ptuBaseEncode.p[n] = new tuEncodeFASTQ( &this->Params, sqCatRplus, i, &semEncoderWorkers, pscw );
                    break;

                default:
                    throw new ApplicationException( __FILE__, __LINE__, "unsupported sequence file format" );
            }
            ptuBaseEncode.p[n++]->Start();

            // wait for another CPU worker thread to become available
            semEncoderWorkers.Wait( m_encoderWorkerThreadTimeout );

            // launch a worker thread for the reverse complement of ith input file
            switch( this->Params.InputFileFormat )
            {
                case SqFormatFASTA:
                    ptuBaseEncode.p[n] = new tuEncodeFASTA( &this->Params, sqCatRminus, i, &semEncoderWorkers, pscw );
                    break;

                case SqFormatFASTQ:
                    ptuBaseEncode.p[n] = new tuEncodeFASTQ( &this->Params, sqCatRminus, i, &semEncoderWorkers, pscw );
                    break;

                default:
                    break;
            }
            ptuBaseEncode.p[n++]->Start();

            // if the config file contains a reference sequence URI path, save a URI for the nth input filename
            pscw->AppendReferenceURI( this->Params.ifgRaw.InputFile.p[i].SubId, this->Params.ifgRaw.InputFile.p[i].URI );
        }

        // wait for the worker threads to exit
        for( UINT32 n=0; n<nSq; ++n )
            ptuBaseEncode.p[n]->Wait( INFINITE );

        // destruct the tuBaseEncode instances
        for( UINT32 n=0; n<nSq; ++n )
            delete ptuBaseEncode.p[n];

        // performance metrics
        AriocE::PerfMetrics.msEncoding = hrt.GetElapsed(false);

        // finalize and write the config file
        pscw->AppendExecutionTime( AriocE::PerfMetrics.msEncoding );
        pscw->Write();

        CDPrint( cdpCD0, "%s: encoded %u files (%u sequences)", __FUNCTION__, this->Params.ifgRaw.InputFile.n, nSq );
    }
}

/// [private] method buildJlists()
void AriocE::buildJlists()
{
    /* We fill the J table with hash values for each subsequence in the R sequences.*/

    // prepare to build hash tables from the encoded input file(s)
    RaiiSemaphore semEncoderWorkers( this->Params.nLPs, this->Params.nLPs );
    HiResTimer hrt(ms);

    // zero the J-list counts in the C buffer
    for( size_t iC=0; iC<m_C.Count; ++iC )
        m_C.p[iC].nJ = 0;

    // allocate and zero a buffer for use in thread synchronization
    WinGlobalPtr<UINT32> waitBits( blockdiv(m_H.Count,32), true );

    // there are two reference sequences (+ and -) for each input sequence
    UINT32 nSq = 2 * this->Params.ifgRaw.InputFile.n;
    WinGlobalPtr<tuLoadR*> ptuLoadR( nSq, true );

    CDPrint( cdpCD0, "%s: building J lists for %u file%s (%u sequence%s) (%d CPU thread%s available)...",
                        __FUNCTION__,
                        this->Params.ifgRaw.InputFile.n, PluralS( this->Params.ifgRaw.InputFile.n ),
                        nSq, PluralS(nSq),
                        this->Params.nLPs, PluralS(this->Params.nLPs) );
        
    UINT32 n = 0;
    while( n < nSq )
    {
        // use the same file for both strands
        UINT32 i = n / 2;

        // wait for a CPU worker thread to become available
        semEncoderWorkers.Wait( m_encoderWorkerThreadTimeout );

        // launch a worker thread for the ith input file
        ptuLoadR.p[n] = new tuLoadR( &this->Params, sqCatRplus, i, waitBits.p, &semEncoderWorkers );
        ptuLoadR.p[n++]->Start();

        // wait for another CPU worker thread to become available
        semEncoderWorkers.Wait( m_encoderWorkerThreadTimeout );

        // launch a worker thread for the reverse complement of the input file
        ptuLoadR.p[n] = new tuLoadR( &this->Params, sqCatRminus, i, waitBits.p, &semEncoderWorkers );
        ptuLoadR.p[n++]->Start();
    }

    // wait for the worker threads to exit
    for( UINT32 n=0; n<nSq; ++n )
        ptuLoadR.p[n]->Wait( INFINITE );

    // destruct the tuLoadR instances
    for( UINT32 n=0; n<nSq; ++n )
        delete ptuLoadR.p[n];

    CDPrint( cdpCD0, "%s: built J lists for %u files (%u sequences)", __FUNCTION__, this->Params.ifgRaw.InputFile.n, nSq );

    // performance metrics
    AriocE::PerfMetrics.msBuildJ = hrt.GetElapsed(false);
}

/// [private] method compactHtable
size_t AriocE::compactHtable()
{
    HiResTimer hrt;

    // point to the start of the H table
    Hvalue8* pH8 = m_H.p;
    Hvalue5* pH5 = reinterpret_cast<Hvalue5*>(m_H.p);

    // copy Hvalue8 values to Hvalue5 values
    for( size_t h=0; h<m_H.Count; ++h )
        *(pH5++) = *reinterpret_cast<Hvalue5*>(pH8++);

    // performance metrics
    AriocE::PerfMetrics.msCompactHtable = hrt.GetElapsed(false);

    // return the number of bytes used
    return m_H.Count * sizeof(Hvalue5);
}

/// [private] method compactJtable
size_t AriocE::compactJtable()
{
    CDPrint( cdpCD3, "%s...", __FUNCTION__ );

    HiResTimer hrt(ms);

    /* Point to the start of the J table.
    
        - The J table is laid out as a 5-byte JtableHeader followed by a list of 5-byte Jvalue5 values.
        - J values start immediately after the JtableHeader, so J-table offsets in the H list are 0-based
           relative to the start of the J table (and the first J value is at offset 1).
    */
    Jvalue5* const pJ5base = reinterpret_cast<Jvalue5*>(m_J.p);
    Jvalue5* pJ5 = pJ5base + 1;     // (element 0 of the J table is unused)

    // performance metrics
    AriocE::PerfMetrics.nJclamped = 0;
    AriocE::PerfMetrics.nJexcluded = 0;

    for( size_t iC=0; iC<m_C.Count; ++iC )
    {
        // get the H value for the iC'th hash key
        Hvalue8* pH = m_H.p + m_C.p[iC].hashKey;
        if( pH->ofsJ == 0 )
            continue;

        // get the total number of J values (including excluded J values) in the current J list
        Jvalue8* pJ8 = m_J.p + pH->ofsJ;


#if TODO_CHOP_WHEN_DEBUGGED
        if( m_C.p[iC].hashKey == 0x0b70d009 )
        {
            CDPrint( cdpCD0, "%s pJ8=0x%016llx pH->ofsJ=0x%016llx pH->nJ=%u pH->nJx=%u, pJ8->J=0x%08x s=%u subId=%u tag=0x%08x x=%u pJ5=0x%016llx",
                             __FUNCTION__, pJ8, pH->ofsJ, (UINT32)pH->nJ, (UINT32)pH->nJx, (UINT32)pJ8->J, (UINT32)pJ8->s, (UINT32)pJ8->subId, (UINT32)pJ8->tag, (UINT32)pJ8->x, pJ5 );
        }
#endif




        UINT32 nJ = pH->nJ;
        if( pH->nJ == 0 )
        {
            nJ = *reinterpret_cast<UINT32*>(pJ8);
            pJ8++;
        }

        /* At this point:
            - pJ5 --> next target Jvalue5
            - pJ8 --> next source Jvalue8 (i.e., the first J value in the list)
        */

        // update the H-value offset
        pH->ofsJ = pJ5 - pJ5base;

        // update the list count in the C table
        UINT32 nJnew = nJ - static_cast<UINT32>(pH->nJx);
        m_C.p[iC].nJ = nJnew;

        // update the list count in the H and J tables
        if( nJnew <= HVALUE_MAX_NJ )
        {
            pH->nJ = nJnew;
            if( nJnew == 0 )
                pH->ofsJ = 0;
        }
        else
        {
            // insert a Jvalue5 list count if needed
            *reinterpret_cast<INT64*>(pJ5) = nJnew;
            pJ5++;
        }

        /* Traverse the J list and copy non-excluded values. */

        // initialize pointer limiting values
        Jvalue8* pJ8limit = pJ8 + nJ;
        Jvalue5* pJ5limit = pJ5 + nJnew;

        // iterate through the J list
        UINT32 nJcopied = 0;
        while( (pJ8 < pJ8limit) && (pJ5 < pJ5limit) )
        {
            // if the ij'th J value is flagged, exclude it from the final J table
            if( !pJ8->tag )
            {
                /* Copy Jvalue8 to Jvalue5.  The low-order 5 bytes of Jvalue8 have the same bitfield mapping
                    as Jvalue5, so a simple integer copy will suffice. */
                *pJ5 = *reinterpret_cast<Jvalue5*>(pJ8);
                pJ5++;
                nJcopied++;
            }

            pJ8++;
        }

        // sanity check
        if( nJcopied != m_C.p[iC].nJ )
            throw new ApplicationException( __FILE__, __LINE__, "copied %u/%u J values for hash key 0x%08x", nJcopied, m_C.p[iC].nJ, m_C.p[iC].hashKey );
        
        // performance metrics
        if( pH->nJx )
            AriocE::PerfMetrics.nJclamped++;
        AriocE::PerfMetrics.nJexcluded += pH->nJx;
    }

    // performance metrics
    AriocE::PerfMetrics.msCompactJtable = hrt.GetElapsed(false);


#if TODO_CHOP_WHEN_DEBUGGED
    for( INT64 n=0; n<m_nH; ++n )
    {
        Hvalue8* pH = m_H.p + n;
        UINT32 nJ = pH->nJ;
        if( nJ == 0 )
            nJ = m_J.p[pH->ofsJ].J;

        if( m_C.p[n].nJ != nJ )
        {
            if( nJ == m_maxJ )
                CDPrint( cdpCD0, "clamped hashkey %u", n );
            else
                CDPrint( cdpCD0, "error at hashkey %u", n );
        }
    }
#endif

    CDPrint( cdpCD3, "%s completed", __FUNCTION__ );

    
    // return the number of bytes used
    return sizeof(JtableHeader) + ((pJ5 - pJ5base) * sizeof(Jvalue5));
}

/// [private] method validateHJ
void AriocE::validateHJ()
{
    CDPrint( cdpCD0, "%s...", __FUNCTION__ );

    HiResTimer hrt(ms);

    // point to the start of 5-byte data in the H and J buffers
    Hvalue5* const pH5base = reinterpret_cast<Hvalue5*>(m_H.p);
    Jvalue5* const pJ5base = reinterpret_cast<Jvalue5*>(m_J.p);

    // validate the C list against the H and J lists
    for( size_t iC=0; iC<m_C.Count; ++iC )
    {
        Hvalue5* pH = pH5base + m_C.p[iC].hashKey;
        INT64 ofsJ = HVALUE5_OFS(pH);

        // do nothing if the h'th hash key is unused
        if( ofsJ == 0 )
            continue;

        // get the J-list size from the H value or J value
        UINT32 nJ = pH->nJ;
        if( nJ == 0 )
            nJ = pJ5base[ofsJ].J;

        if( m_C.p[iC].nJ != nJ )
            throw new ApplicationException( __FILE__, __LINE__, "%s: inconsistent J-list count for hash key 0x%08X: C value = %d, H value = %d", __FUNCTION__, m_C.p[iC].hashKey, m_C.p[iC].nJ, nJ );
            //CDPrint( cdpCD0, "%s: inconsistent J-list count for hash key 0x%08X: C value = %d, H value = %d", __FUNCTION__, m_C.p[iC].hashKey, m_C.p[iC].nJ, nJ );

        /* The Arioc aligners' D-value format imposes a limit on the cardinality of a J list, but unless there's a bug
            somewhere, this limit should never be reached here because MAX_HVALUE_NJ is smaller than that limit. */
        if( nJ >= AriocDS::Dq::limitIj )
            throw new ApplicationException( __FILE__, __LINE__, "%s: J list for H=0x%08X contains %u elements (maximum supported cardinality = %u)", __FUNCTION__, m_C.p[iC].hashKey, nJ, AriocDS::Dq::limitIj-1 );
    }

    // validate the H list pointers
    INT64 nJtotal = 0;          // total number of J values in the J table
    INT64 nHcountInList = 0;    // total number of H values with the corresponding J count saved in the J list
    INT64 hJmax = 0;            // hash key with the largest J list
    UINT32 maxnJ = 0;           // size of the largest J list
    INT64 nHused = 0;           // number of hash keys used

    for( size_t h=0; h<m_C.Count; ++h )     // (we use the C table count because the H-table buffer has been reallocated)
    {
        // get a pointer to the H value corresponding to the h'th hash key
        Hvalue5* pH = pH5base + h;

        // do nothing if the h'th hash key is unused
        INT64 ofsJ = HVALUE5_OFS(pH);
        if( ofsJ == 0 )
        {
            // performance metrics
            AriocE::PerfMetrics.nNullH++ ;      // count the number of null (unused) hash keys
            continue;
        }

        // track the number of hash keys used
        ++nHused;

        // get a pointer to the start of the J list for the current H value
        Jvalue5* pJ = pJ5base + ofsJ;

        UINT32 nJexpected = pH->nJ;
        if( nJexpected == 0 )
        {
            nJexpected = pJ->J;     // copy the J list count from the first element in the list
            pJ++ ;                  // point to the first J value in the list
            nHcountInList++ ;       // track the number of J lists that contain their counts
        }

        // track the biggest collision bucket
        if( nJexpected > maxnJ )
        {
            hJmax = h;
            maxnJ = nJexpected;
        }

        // track the total number of J values
        nJtotal += nJexpected;

        /* Validate that the J-list is ordered by
            - subId
            - strand
            - J
           The Jvalue struct is laid out so that a comparison of the containing 39-bit value will suffice.
        */
        while( --nJexpected )
        {
            // verify that the current J value in the list has its end-of-list flag cleared
            if( pJ->x )
                throw new ApplicationException( __FILE__, __LINE__, "unexpected end-of-list flag for hash 0x%08llX: J=0x%08X s=%d subId=%d",
                                                                    h,
                                                                    pJ->J, pJ->s, pJ->subId );

            Jvalue5* pJprev = pJ;
            ++pJ;
            const UINT64 j5mask = (static_cast<UINT64>(Jvalue5::bfMaxVal_subId) << (Jvalue5::bfSize_J+Jvalue5::bfSize_s)) |
                                  (static_cast<UINT64>(Jvalue5::bfMaxVal_s)     << Jvalue5::bfSize_J) |
                                  static_cast<UINT64>(Jvalue5::bfMaxVal_J);
            if( ((*reinterpret_cast<UINT64*>(pJprev)) & j5mask) >= ((*reinterpret_cast<UINT64*>(pJ)) & j5mask) )
                throw new ApplicationException( __FILE__, __LINE__, "J list out of order for hash 0x%08llX: J=0x%08X s=%d subId=%d is followed by J=%08X s=%d subId=%d",
                                                                    h,
                                                                    pJprev->J, pJprev->s, pJprev->subId,
                                                                    pJ->J, pJ->s, pJ->subId );
        }

#if TODO_CHOP_IF_UNUSED
        // verify that the last J value in the list has its end-of-list flag cleared
        if( pJ->x )
            throw new ApplicationException( __FILE__, __LINE__, "unexpected end-of-list flag for hash 0x%08llX: J=0x%08X s=%d subId=%d",
                                                                h,
                                                                pJ->J, pJ->s, pJ->subId );
#endif

        // verify that the last J value in the list has its end-of-list flag set
        if( !pJ->x )
            throw new ApplicationException( __FILE__, __LINE__, "missing end-of-list flag for hash 0x%08llX: J=0x%08X s=%d subId=%d",
                                                                h,
                                                                pJ->J, pJ->s, pJ->subId );
    }

    // save the maximum J value in the unused 0th element of the J-list buffer
    reinterpret_cast<JtableHeader*>(m_J.p)->maxnJ = maxnJ;

    // performance metrics
    AriocE::PerfMetrics.msValidateHJ = hrt.GetElapsed(false);
    AriocE::PerfMetrics.nKmersUsed = nJtotal;

    CDPrint( cdpCD1, "%s: H table uses %lld/%llu (%6.3f%%) hash keys", __FUNCTION__, nHused, m_C.Count, (100.0*nHused)/m_C.Count );
    CDPrint( cdpCD1, "%s: J table contains %lld J values (list counts in lists: %lld (%5.3f%%))",
                        __FUNCTION__,
                        nJtotal+nHcountInList, nHcountInList, 100.0*nHcountInList/(nJtotal+nHcountInList) );
    CDPrint( cdpCD1, "%s: maximum nJ=%d for hash key 0x%08llX", __FUNCTION__, maxnJ, hJmax );
    CDPrint( cdpCD0, "%s completed", __FUNCTION__ );
}

/// [private] method validateSubIds
void AriocE::validateSubIds( size_t _cbJ )
{
    CDPrint( cdpCD0, "%s...", __FUNCTION__ );

    HiResTimer hrt(ms);

    INT8 maxSubId = -1;
    INT8 minSubId = 0x7F;
    WinGlobalPtr<UINT32> nSubIdPerH( Jvalue5::bfMaxVal_subId+1, true );
    WinGlobalPtr<UINT64> totalJ( nSubIdPerH.Count, true );

    /* validate subIds on worker threads */
    {
        // compute the number of H values to evaluate in each worker thread
        INT64 nHperThread = blockdiv(m_C.Count, this->Params.nLPs);

        WinGlobalPtr<tuValidateSubIds*> ptuValidateSubIds( this->Params.nLPs, true );
        for( INT32 n=0; n<this->Params.nLPs; ++n )
        {
            INT64 iC0 = n * nHperThread;
            INT64 iClimit = min2(iC0+nHperThread, static_cast<INT64>(m_C.Count));
        
            // launch a worker thread
            ptuValidateSubIds.p[n] = new tuValidateSubIds( iC0, iClimit, &m_C, &m_H, &m_J, _cbJ );
            ptuValidateSubIds.p[n]->Start();
        }

        // wait for the worker threads to exit
        for( INT32 n=0; n<this->Params.nLPs; ++n )
            ptuValidateSubIds.p[n]->Wait( INFINITE );

        // accumulate the counts
        for( INT32 n=0; n<this->Params.nLPs; ++n )
        {
            minSubId = min2(minSubId, ptuValidateSubIds.p[n]->MinSubId);
            maxSubId = max2(maxSubId, ptuValidateSubIds.p[n]->MaxSubId);

            for( size_t s=0; s<nSubIdPerH.Count; ++s )
            {
                nSubIdPerH.p[s] += ptuValidateSubIds.p[n]->SubIdsPerH.p[s];
                totalJ.p[s] += ptuValidateSubIds.p[n]->TotalJ.p[s];
            }
        }

        // destruct the tuValidateSubIds instances
        for( INT32 n=0; n<this->Params.nLPs; ++n )
            delete ptuValidateSubIds.p[n];

        CDPrint( cdpCD0, "%s: validated %llu hash values", __FUNCTION__, m_C.Count );
    }

    // save the maximum subId value in the unused 0th element of the J-list buffer
    reinterpret_cast<JtableHeader*>(m_J.p)->maxSubId = maxSubId;
    reinterpret_cast<JtableHeader*>(m_J.p)->minSubId = min2(minSubId, 1);           // save either 0 or 1

    // find the maximum number of different subIds associated with any hash key
    INT32 maxRforH = static_cast<INT32>(nSubIdPerH.Count-1);
    for( ; maxRforH>=0; --maxRforH )
    {
        if( nSubIdPerH.p[maxRforH] )
            break;
    }

    // list the number of different subIds for a hash key
    CDPrint( cdpCD1, "%s: # subIds  # hash keys  avg J-list size", __FUNCTION__ );
    for( INT32 nRforH=0; nRforH<=maxRforH; nRforH++ )
        CDPrint( cdpCD1, "%s:%9d%13u%17.1f", __FUNCTION__, nRforH, nSubIdPerH.p[nRforH], static_cast<double>(totalJ.p[nRforH])/max2(1,nSubIdPerH.p[nRforH]) );

    // performance metrics
    AriocE::PerfMetrics.msValidateSubIds = hrt.GetElapsed(false);
    CDPrint( cdpCD0, "%s completed", __FUNCTION__ );
}
#pragma endregion

#pragma region public methods
/// <summary>
/// Imports and encodes R sequence data and builds hashed lookup tables
/// </summary>
void AriocE::ImportR()
{
    // sanity check
    if( this->Params.InputFileFormat == SqFormatFASTQ )
        throw new ApplicationException( __FILE__, __LINE__, "FASTQ format not supported for reference sequences" );

    // sanity check: ensure that the reference sequence IDs are in ascending order
    for( UINT32 n=1; n<this->Params.ifgRaw.InputFile.n; ++n )
    {
        if( this->Params.ifgRaw.InputFile.p[n].SubId <= this->Params.ifgRaw.InputFile.p[n-1].SubId )
            throw new ApplicationException( __FILE__, __LINE__, "input file subunit IDs must be distinct and in ascending order (%d precedes %d)", this->Params.ifgRaw.InputFile.p[n-1].SubId, this->Params.ifgRaw.InputFile.p[n].SubId );
    }

    HiResTimer hrtApp(ms);

    // ensure that the output directory exists
    RaiiDirectory::OpenOrCreateDirectory( this->Params.OutDir );

    // allocate and zero the H buffer
    if( this->Params.pa21sb->hashKeyWidth > 32 )
        throw new ApplicationException( __FILE__, __LINE__, "invalid hash key width = %d (supported maximum = 32)", this->Params.pa21sb->hashKeyWidth );
    size_t celH = static_cast<size_t>(1) << this->Params.pa21sb->hashKeyWidth;
    m_H.Realloc( celH, true );
    this->Params.H = m_H.p;         // make a copy of the buffer pointer that worker threads can reference

    // allocate and zero the xH buffer
    m_xH.Realloc( blockdiv(celH,8), true );   // one bit flag for each possible H value

    // allocate and zero the C buffer
    m_C.Realloc( celH, true );
    this->Params.C = m_C.p;         // make a copy of the buffer pointer that worker threads can reference

    // allocate and zero a list of offsets into the C buffer (ordered by hash key)
    m_CH.Realloc( celH, true );
    this->Params.CH = m_CH.p;       // make a copy of the buffer pointer that worker threads can reference

    // preallocate space in the C and H files
    RaiiSyncEventObject rseoCompleteH( true, false );
    RaiiFile fileH;
    preallocateCHJ( fileH, rseoCompleteH, m_H.cb, "H" );

    RaiiSyncEventObject rseoCompleteC( true, false );
    RaiiFile fileC;
    preallocateCHJ( fileC, rseoCompleteC, m_C.cb, "C" );

    // prepare to write configuration info destined to be included in SAM files
#pragma warning ( push )
#pragma warning ( disable : 4996 )		// don't nag us about _splitpath being "unsafe"
    // use the input filename as the "base name" for the associated output file
    char baseName[FILENAME_MAX] = { 0 };
    _splitpath( this->Params.OutDirLUT, NULL, NULL, baseName, NULL );
#pragma warning ( pop )   
    
    SAMConfigWriter scw( &this->Params, this->Params.OutFilespecStubLUT, baseName );

    // encode the input files
    encodeR( &scw );

    // count the J values and initialize the buffer that contains the J table
    computeJlistSizes();

    // make a copy of the J-list buffer pointer that worker threads can reference
    this->Params.J = m_J.p;             

    // preallocate space in the J file
    RaiiSyncEventObject rseoCompleteJ( true, false );
    RaiiFile fileJ;
    preallocateCHJ( fileJ, rseoCompleteJ, m_J.cb, "J" );

    // hash the R sequence again and save J values in the J lists
    buildJlists();

    /* sort the J lists */
    {
        HiResTimer hrt;

        if( this->Params.gpuMask )
            sortJgpu();
        else
            sortJcpu();

        // performance metrics
        AriocE::PerfMetrics.msSortJ = hrt.GetElapsed(false);
    }

    // set a flag in the last element of each J list
    setEOLflags();

    if( m_maxJ != _I32_MAX )
    {
        // identify J values to exclude from long J lists ("big buckets")
        identifyBBJ();

        // flag J values to be excluded
        setBBJlistFlags();
    }

    /* Compact the J table by excluding J values and converting the remaining J-table values from 8 bytes each
        to 5 bytes each.

       This weakens the J-table sort-order "heuristic" (largest J lists first in the buffer) but avoiding this
        would involve reordering the C, H and J tables so that the largest lists were again first in the buffer.
        (Maybe someday...)
    */
    size_t cbJ = compactJtable();
    m_J.Realloc( blockdiv(cbJ,sizeof(Jvalue8)), false );    // release unused space in the J buffer

    // compact the H table by converting H values from 8 bytes each to 5 bytes each
    size_t cbH = compactHtable();
    m_H.Realloc( blockdiv(cbH,sizeof(Hvalue8)), false );    // release unused space in the H buffer

    // validate the H and J lookup tables
    validateHJ();

    // validate the reformatted J and H tables
    validateSubIds( cbJ );

    // write the C, J and H tables
    AriocE::PerfMetrics.cbC = writeCHJ( fileC, &rseoCompleteC, m_C.p, m_C.cb, &AriocE::PerfMetrics.msWriteC );
    AriocE::PerfMetrics.cbJ = writeCHJ( fileJ, &rseoCompleteJ, m_J.p, cbJ, &AriocE::PerfMetrics.msWriteJ );
    AriocE::PerfMetrics.cbH = writeCHJ( fileH, &rseoCompleteH, m_H.p, cbH, &AriocE::PerfMetrics.msWriteH );

    // performance metrics
    switch( this->Params.InputFileFormat )
    {
        case SqFormatFASTA:
            strcpy_s( AriocE::PerfMetrics.InputFileFormat, sizeof AriocE::PerfMetrics.InputFileFormat, "FASTA" );
            break;
                
        case SqFormatFASTQ:
            strcpy_s( AriocE::PerfMetrics.InputFileFormat, sizeof AriocE::PerfMetrics.InputFileFormat, "FASTQ" );
            break;
                
        default:
            strcpy_s( AriocE::PerfMetrics.InputFileFormat, sizeof AriocE::PerfMetrics.InputFileFormat, "(unrecognized input file format)" );
            break;
    }

    // performance metrics
    AriocE::PerfMetrics.nKmersIgnored = this->Params.nKmersWithN;
    AriocE::PerfMetrics.nSqEncoded = this->Params.nSqEncoded;
    AriocE::PerfMetrics.nSqIn = this->Params.nSqIn;
    AriocE::PerfMetrics.nSymbolsEncoded = this->Params.nSymbolsEncoded;
    AriocE::PerfMetrics.nConcurrentThreads = this->Params.nLPs;
    AriocE::PerfMetrics.nInputFiles = static_cast<INT32>(this->Params.ifgRaw.InputFile.n);
    AriocE::PerfMetrics.msApp = hrtApp.GetElapsed(false);

    // write the SAM.cfg file
    scw.AppendExecutionTime( AriocE::PerfMetrics.msApp );
    scw.AppendMaxJ( m_maxJ );
    scw.Write();
}
#pragma endregion
