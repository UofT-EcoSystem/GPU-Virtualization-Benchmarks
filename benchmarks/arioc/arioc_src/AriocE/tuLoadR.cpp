/*
  tuLoadR.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Load R sequence data.
/// </summary>
/// <param name="psip">Reference to a common parameter structure</param>
/// <param name="sqCat">sequence category (+ or - strand)</param>
/// <param name="iInputFile">index of input file</param>
/// <param name="waitBits">array of bits used for thread synchronization (initialized to zero by the caller)</param> 
/// <param name="psem">Reference to an <c>RaiiSemaphore</c> instance</param>
tuLoadR::tuLoadR( AriocEncoderParams* psip, SqCategory sqCat, INT16 iInputFile, volatile UINT32* waitBits, RaiiSemaphore* psem) :
                        m_psip(psip),
                        m_sqCategory(sqCat),
                        m_iInputFile(iInputFile),
                        m_WaitBits(waitBits),
                        m_cbUnread(0),
                        m_maxEncodeHash(0),
                        m_subId( psip->ifgRaw.InputFile.p[iInputFile].SubId ),
                        m_isRminus(sqCat==sqCatRminus),
                        m_computeKmerHash(NULL),
                        m_cbHash(0),
                        m_psemComplete(psem),
                        m_hrt(us)
{
    // extract the "base name" from the input file specification
    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    char ext[_MAX_EXT];
    char baseName[_MAX_FNAME];

#pragma warning ( push )
#pragma warning ( disable : 4996 )		// don't nag us about _splitpath being "unsafe"
    // use the input filename as the "base name" for the associated output file
    _splitpath( psip->ifgRaw.InputFile.p[iInputFile].Filespec, drive, dir, baseName, ext );
    if( *baseName == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: no base name in '%s'", __FUNCTION__, psip->ifgRaw.InputFile.p[iInputFile].Filespec );
#pragma warning ( pop )

    // build the file specification "stub" (full path plus base name)
    char stubFileSpec[FILENAME_MAX];
    strcpy_s( stubFileSpec, FILENAME_MAX, m_psip->OutDir );
    strcat_s( stubFileSpec, FILENAME_MAX, FILEPATHSEPARATOR );
    strcat_s( stubFileSpec, FILENAME_MAX, baseName );
    strcat_s( stubFileSpec, FILENAME_MAX, "$" );

    // create a filename extension for the current worker thread
    char inFileExt[8];

    // for a reverse-complement reference sequence, include "rc" in the file specification
    if( m_sqCategory == sqCatRminus )
        strcpy_s( inFileExt, sizeof inFileExt, ".rc.sbf" );
    else
        strcpy_s( inFileExt, sizeof inFileExt, ".sbf" );

    // open the encoded data file
    char AriocFileSpec[FILENAME_MAX];
    strcpy_s( AriocFileSpec, FILENAME_MAX, stubFileSpec );
    strcat_s( AriocFileSpec, FILENAME_MAX, "a21" );
    strcat_s( AriocFileSpec, FILENAME_MAX, inFileExt );
    m_inFileArioc.OpenReadOnly( AriocFileSpec );

    // sanity check
    INT64 cbIn = m_inFileArioc.FileSize();
    if( cbIn & 7 )
        throw new ApplicationException( __FILE__, __LINE__, "the size of file %s is not a multiple of 8 bytes", m_inFileArioc.FileSpec.p );

    /* Allocate the input file buffer:
        - for a reference sequence: the buffer is large enough to hold the entire sequence
        - for query sequences: the buffer is a predefined size
    */

    // compute the number of elements in the buffer
    m_inBufArioc.n = static_cast<UINT32>((m_sqCategory & sqCatR) ? cbIn/sizeof(INT64) : INPUTBUFFERSIZE);

    // allocate the buffer and pad right to ensure that the computeKmerHash* functions do not overrun the buffer
    m_inBufArioc.Realloc( m_inBufArioc.n+2, true );

    // save the number of bytes required to represent a hash value
    m_cbHash = (m_psip->pa21sb->hashKeyWidth > 32) ? sizeof(INT64) : sizeof(INT32);

    // point to the member function that corresponds to the size of the k-mer seed
    if( m_psip->pa21sb->seedWidth <= 21 )
        m_computeKmerHash = &tuLoadR::computeKmerHash21;
    else
    if( m_psip->pa21sb->seedWidth <= 42 )
        m_computeKmerHash = &tuLoadR::computeKmerHash42;
    else
        m_computeKmerHash = &tuLoadR::computeKmerHash84;
}

/// [public] destructor
tuLoadR::~tuLoadR()
{
}
#pragma endregion

#pragma region private methods
// [private] method computeKmerHash21
bool tuLoadR::computeKmerHash21( UINT64& hashKey, INT64*& p64, INT32& shr, INT32& shl )
{
    // get the encoded symbols for the kmer
    UINT64 i64 = *p64;

    if( shr )
    {
        UINT64 i64n = *(++p64);
        i64 = ((i64 >> shr) | (i64n << shl)) & MASKA21;
    }

    // do not hash the kmer if it contains too many Ns to align
    if( static_cast<INT16>(__popcnt64( i64 & m_psip->pa21sb->maskNonN )) < m_psip->pa21sb->minNonN )
        return false;

    // convert DNA bases
    i64 = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64 );

    // compute the hash key
    hashKey = m_psip->pa21sb->ComputeH32( i64 );    // (computes a 32-bit hash key; see baseEncode::computeKmerHash21R)
    return true;
}

// [private] method computeKmerHash42
bool tuLoadR::computeKmerHash42( UINT64& hashKey, INT64*& p64, INT32& shr, INT32& shl )
{
    // get the encoded symbols for the kmer
    UINT64 i64lo = *p64;            // low-order 64 bits (21 symbols)
    UINT64 i64hi = *(++p64);        // high-order 64 bits (21 symbols)

    if( shr )
    {
        i64lo = ((i64lo >> shr) | (i64hi << shl)) & MASKA21;
        UINT64 i64n = *(++p64);
        i64hi = ((i64hi >> shr) | (i64n << shl)) & MASKA21;
    }

    // do not hash the kmer if it contains too many Ns to align
    if( static_cast<INT16>(__popcnt64( i64lo & MASK100 ) + __popcnt64( i64hi & m_psip->pa21sb->maskNonN )) < m_psip->pa21sb->minNonN )
        return false;

    // convert DNA bases
    i64lo = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64lo );
    i64hi = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64hi );

    // compute the hash key
    hashKey = m_psip->pa21sb->ComputeH32( i64lo, i64hi );   // computes a 32-bit hash; see AriocWorkerE_Base::computeKmerHash42R)
    return true;
}

// [private] method computeKmerHash84
bool tuLoadR::computeKmerHash84( UINT64& hashKey, INT64*& p64, INT32& shr, INT32& shl )
{
    // get the encoded symbols for the kmer
    UINT64 i64lo = *p64;            // low-order 64 bits (21 symbols)
    UINT64 i64m1 = *(++p64);
    UINT64 i64m2 = *(++p64);
    UINT64 i64hi = *(++p64);        // high-order 64 bits (21 symbols)

    if( shr )
    {
        i64lo = ((i64lo >> shr) | (i64m1 << shl)) & MASKA21;
        i64m1 = ((i64m1 >> shr) | (i64m2 << shl)) & MASKA21;
        i64m2 = ((i64m2 >> shr) | (i64hi << shl)) & MASKA21;
        UINT64 i64n = *(++p64);
        i64hi = ((i64hi >> shr) | (i64n << shl)) & MASKA21;
    }

    // do not hash the kmer if it contains too many Ns to align
    INT16 nNs = static_cast<INT16>(__popcnt64( i64lo & MASK100 ) +
                                    __popcnt64( i64m1 & MASK100 ) +
                                    __popcnt64( i64m2 & MASK100 ) +
                                    __popcnt64( i64hi & m_psip->pa21sb->maskNonN ));
    if( nNs < m_psip->pa21sb->minNonN )
        return false;

    // convert DNA bases
    i64lo = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64lo );
    i64m1 = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64m1 );
    i64m2 = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64m2 );
    i64hi = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64hi );

    // compute the hash key
    hashKey = m_psip->pa21sb->ComputeH32( i64lo, i64m1, i64m2, i64hi );   // computes a 32-bit hash; see AriocWorkerE_Base::computeKmerHash84R)
    return true;
}

/// [private] method buildJR
void tuLoadR::buildJR()
{
    m_hrt.Restart();

    // load the entire reference sequence into a buffer
    m_inFileArioc.Read( m_inBufArioc.p, m_inBufArioc.n*sizeof(INT64) );

    // performance metrics
    InterlockedExchangeAdd64( &AriocE::PerfMetrics.usReadA21, m_hrt.GetElapsed(false) );

    /* The input buffer is formatted like this:
        byte 0x00-0x07: sqId
        byte 0x08-0x0F: M (number of symbols in the sequence)
        byte 0x10-0x17: len(sq)
        byte 0x18-    : sq
    */

    // get a copy of the number of symbols in the reference sequence
    INT64 M = m_inBufArioc.p[1];

    // point to the first 64-bit value in the encoded reference sequence
    INT64* pB2164 = m_inBufArioc.p + 3;

    // compute the 0-based rightmost possible position for a kmer in the specified sequence
    INT32 posMax = static_cast<INT32>(M - m_psip->pa21sb->seedWidth);

    // save a flag that indicates the strand (forward or reverse complement) of the sequence
    bool isReverseComplement = (m_sqCategory == sqCatRminus);

#if TODO_CHOP_IF_UNUSED

    INT64 maxJcomparisons = 0;
    INT64 totalJcomparisons = 0;
    INT64 totalJinsertions = 0;
    INT64 totalBytesMoved = 0;
    INT64 maxBytesMoved = 0;

#endif

#if TODO_CHOP_WHEN_DEBUGGED
    UINT32 nSpinWaits = 0;
    UINT32 maxSleeps = 0;
#endif

    for( INT32 j=0; j<=posMax; j+=m_psip->pa21sb->seedInterval )
    {
        // point to the 64-bit value that contains the first 3-bit encoded symbol in the j'th kmer to be processed
        INT64* p64 = pB2164 + (j/21);

        // compute the number of bits to shift
        INT32 shr = 3 * (j%21);     // shifts bits in i64lo and i64hi rightward
        INT32 shl = 63 - shr;       // shifts bits in i64hi and i64n leftward so that they can be OR'd into i64lo and i64hi respectively

        // compute the kmer hash
        UINT64 hashKey;
        if( (this->*m_computeKmerHash)( hashKey, p64, shr, shl ) )
        {
            // compute the byte offset of the bit in the waitbits buffer that corresponds to the hash key
            UINT64 byteOffset = hashKey / 32;
            INT32 bitOffset = hashKey % 32;

            /* Spin-wait for the J list that corresponds to the hash key
                - as long as some other thread is concurrently appending to the J list, the current thread remains in a WHILE loop that
                    calls Sleep() to give up the remainder of its time slice
                - on a hyperthreaded CPU this may not be the most efficient way to implement a low-level spin lock (i.e. the current thread
                    may waste CPU cycles by checking the synchronization bit too frequently), but the code is simple and contention for
                    the same J list is empirically very unusual
            */

#if TODO_CHOP_WHEN_DEBUGGED
            UINT32 nSleeps = 0;
            while( InterlockedBitTestAndSet( reinterpret_cast<volatile LONG*>(m_WaitBits+byteOffset), bitOffset ) )
            {
                nSleeps++ ;
                Sleep( 0 );
            }

            if( nSleeps )
            {
                nSpinWaits++ ;
                maxSleeps = max2(maxSleeps, nSleeps);
            }
#endif

            while( InterlockedBitTestAndSet( reinterpret_cast<volatile LONG*>(m_WaitBits+byteOffset), bitOffset ) )
#ifdef _WIN32
                Sleep( 0 );
#endif
#ifdef __GNUC__
                sched_yield();
#endif

            // point to the H value for the hash key
            Hvalue8* pH = reinterpret_cast<Hvalue8*>(m_psip->H+hashKey);

            // point to the start of the J list for the hash key
            Jvalue8* pJ = m_psip->J + pH->ofsJ;

            // get the total J-list count and ensure that the J-list pointer references the start of the J-list
            UINT32 nJ = pH->nJ;

            // if the J-list count is not in the H value ...
            if( nJ == 0 )
            {
                nJ = *reinterpret_cast<UINT32*>(pJ);    // ... get the count
                pJ++;                                   // ... point past the count to the first J value
            }

            /* Get the offset at which to store the next J value from the C list, which was initialized with the expected
                number of J values for each hash key (see for example baseEncode::computeKmerHash21R()). */
            UINT32 iC = m_psip->CH[hashKey];
            if( m_psip->C[iC].nJ >= nJ )
                throw new ApplicationException( __FILE__, __LINE__, "%s: unexpected J value for hash key 0x%08x", __FUNCTION__, hashKey );

            UINT32 ofsJ = m_psip->C[iC].nJ++;

            // append the new J value to the list
            pJ[ofsJ].J = j;
            pJ[ofsJ].s = isReverseComplement;
            pJ[ofsJ].subId = m_subId;

            if( m_psip->gpuMask )
            {
                /* Prepare for segmented J-list sort by tagging the J value with the offset of the start of the J list
                    in the J-list buffer.

                   For J lists with a count preceding the list in the buffer, we tag the count as well.  The Jvalue8.x
                    bitfield is positioned so that after the sort the count precedes the values in the rest of each J list.
                */
                if( (ofsJ == 0) && (pH->nJ == 0) )
                {
                    // the J-list count precedes the J values (i.e., it is in the element before the 0th J value)
                    pJ[-1].tag = pH->ofsJ & 0x00FFFFFF;
                    pJ[-1].x = 0;
                }

                pJ[ofsJ].tag = pH->ofsJ & 0x00FFFFFF;
                pJ[ofsJ].x = 1;
            }


#if TODO_CHOP_WHEN_DEBUGGED
            if( hashKey == 0x0b70d009 )
                CDPrint( cdpCD0, "%s: hashKey=0x%08X pJ+ofsJ=0x%016llx ofsJ=0x%08X, J=0x%08X s=%u subId=%u x=%d tag=0x%08x",
                                    __FUNCTION__, hashKey, pJ+ofsJ, ofsJ,
                                    pJ[ofsJ].J, (UINT32)pJ[ofsJ].s, (UINT32)pJ[ofsJ].subId, (UINT32)pJ[ofsJ].x, (UINT32)pJ[ofsJ].tag );
#endif



            // reset the spin-wait bit
            InterlockedBitTestAndReset( reinterpret_cast<volatile LONG*>(m_WaitBits+byteOffset), bitOffset );
        }
    }

#if TODO_CHOP_WHEN_DEBUGGED
    CDPrint( cdpCD4, "AriocWorkerL::buildJR: nSpinWaits=%u maxSleeps=%u", nSpinWaits, maxSleeps );
#endif

}

/// [private] method buildJQ
void tuLoadR::buildJQ()
{
    throw new ApplicationException( __FILE__, __LINE__, "not yet implemented" );
}
#pragma endregion

#pragma region virtual method implementations
/// <summary>
/// Loads R (reference) sequence data.
/// </summary>
void tuLoadR::main()
{
    CDPrint( cdpCD4, "%s (subId=%d)...", __FUNCTION__, m_subId );

    if( m_sqCategory & sqCatR )
        buildJR();
    else
        buildJQ();

    // signal that this thread has completed its work
    m_psemComplete->Release( 1 );

    CDPrint( cdpCD4, "%s (subId=%d) completed", __FUNCTION__, m_subId );
}
#pragma endregion
