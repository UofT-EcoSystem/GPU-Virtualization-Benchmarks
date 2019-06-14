/*
  baseEncode.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Base implementation for the sequence data encoders.
/// </summary>
/// <param name="psip">Reference to a common parameter structure</param>
/// <param name="sqCat">sequence category (+ or - strand)</param>
/// <param name="iInputFile">index of input file</param>
/// <param name="psem">Reference to an <c>RaiiSemaphore</c> instance</param>
baseEncode::baseEncode( AriocEncoderParams* psip, SqCategory sqCat, INT16 iInputFile, RaiiSemaphore* psem, SAMConfigWriter* pscw ) :
                                m_cbUnread(0),
                                m_cbRawCurrent(0),
                                m_cbSqqCurrent(0),
                                m_maxEncodeHash(0),
                                m_subId( psip->ifgRaw.InputFile.p[iInputFile].SubId ),
                                m_isRminus(sqCat==sqCatRminus),
                                m_pairFlag(psip->ifgRaw.InputFile.p[iInputFile].MateId != 0),
                                m_deleteSCW(pscw==NULL),
                                m_computeKmerHash(NULL),
                                m_psip(psip),
                                m_sqCategory(sqCat),
                                m_iInputFile(iInputFile),
                                m_pscw(pscw),
                                m_cbHash(0),
                                m_writeSqm(NULL),
                                m_writeRaw(NULL),
                                m_writeSqq(NULL),
                                m_endRowRaw(NULL),
                                m_psemComplete(psem)
{
    // open the input file
    m_inFile.OpenReadOnly( m_psip->ifgRaw.InputFile.p[iInputFile].Filespec );

    // get the file size
    INT64 cbIn = m_inFile.FileSize();
    m_cbUnread = cbIn;

    // append the "base name" to the output filespec stub
    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    char ext[_MAX_EXT];
    char stubFileSpec[FILENAME_MAX];

#pragma warning ( push )
#pragma warning ( disable : 4996 )		// don't nag us about _splitpath being "unsafe"
    // use the input filename as the "base name" for the associated output file
    _splitpath( psip->ifgRaw.InputFile.p[iInputFile].Filespec, drive, dir, m_baseName, ext );
    if( *m_baseName == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: no base name in '%s'", __FUNCTION__, psip->ifgRaw.InputFile.p[iInputFile].Filespec );
#pragma warning ( pop )

    strcpy_s( stubFileSpec, FILENAME_MAX, m_psip->OutFilespecStubSq );
    strcat_s( stubFileSpec, FILENAME_MAX, m_baseName );
    strcat_s( stubFileSpec, FILENAME_MAX, "$" );

    // create a filename extension for the current worker thread
    char outFileExt[32];

    // for a reverse-complement reference sequence, include "rc" in the file specification
    if( m_sqCategory == sqCatRminus )
        strcpy_s( outFileExt, sizeof outFileExt, ".rc.sbf" );
    else
        strcpy_s( outFileExt, sizeof outFileExt, ".sbf" );

    // open/create the sequence metadata output file
    char outFileSpec[FILENAME_MAX];
    strcpy_s( outFileSpec, FILENAME_MAX, stubFileSpec );
    strcat_s( outFileSpec, FILENAME_MAX, "sqm" );
    strcat_s( outFileSpec, FILENAME_MAX, outFileExt );
    m_outFileSqm.Open( outFileSpec );

    // open/create the raw data output file
    strcpy_s( outFileSpec, FILENAME_MAX, stubFileSpec );
    strcat_s( outFileSpec, FILENAME_MAX, "raw" );
    strcat_s( outFileSpec, FILENAME_MAX, outFileExt );
    m_outFileRaw.Open( outFileSpec );

    if( m_psip->InputFileFormat == SqFormatFASTQ)
    {
        // open/create the quality score output file
        strcpy_s( outFileSpec, FILENAME_MAX, stubFileSpec );
        strcat_s( outFileSpec, FILENAME_MAX, "sqq" );
        strcat_s( outFileSpec, FILENAME_MAX, outFileExt );
        m_outFileSqq.Open( outFileSpec );
    }

    // open/create the encoded data output file
    strcpy_s( outFileSpec, FILENAME_MAX, stubFileSpec );
    strcat_s( outFileSpec, FILENAME_MAX, "a21" );
    strcat_s( outFileSpec, FILENAME_MAX, outFileExt );
    m_outFileA21.Open( outFileSpec );

    /* Allocate the file buffers:
        - the input buffer is either large enough to hold all of the input data or else a predefined "reasonable" size
        - for a reference sequence: the output buffers are large enough to hold the entire sequence
        - for query sequences: the output buffers are a predefined size
    */
    m_inBuf.Realloc( min2(INPUTBUFFERSIZE,cbIn+1), false );
    if( m_sqCategory & sqCatR )
    {
        m_outBufSqm.Realloc( cbIn+1, false );
        m_outBufRaw.Realloc( cbIn+1, false );
        m_outBufA21.Realloc( sizeof(INT64)*blockdiv(cbIn,21), false );     // (there are 21 encoded symbols per 64-bit value, so this is a reasonable buffer-size estimate)
    }
    else
    {
        m_outBufSqm.Realloc( OUTPUTBUFFERSIZE, false );
        m_outBufRaw.Realloc( OUTPUTBUFFERSIZE, false );
        m_outBufA21.Realloc( OUTPUTBUFFERSIZE, false );
        if( m_psip->InputFileFormat == SqFormatFASTQ )
            m_outBufSqq.Realloc( OUTPUTBUFFERSIZE, false );
    }

    if( m_psip->pa21sb->seedInterval )
    {
        if( m_psip->emitKmers )
        {
            // open/create the kmers output file
            sprintf_s( outFileSpec, FILENAME_MAX, "%sK%d", stubFileSpec, m_psip->pa21sb->seedInterval );
            strcat_s( outFileSpec, FILENAME_MAX, outFileExt );
            m_outFileKmers.Open( outFileSpec );
        }

        // allocate the output buffer
        m_outBufKmers.Realloc( OUTPUTBUFFERSIZE, false );
    }

    // save the number of bytes required to represent a hash value
    m_cbHash = (m_psip->pa21sb->hashKeyWidth > 32) ? sizeof(INT64) : sizeof(INT32);

    // point to the member functions that correspond to the type of sequence (R or Q)
    if( m_sqCategory & sqCatR )
    {
        m_writeSqm = &baseEncode::writeSqmR;
        m_writeRaw = &baseEncode::writeRawR;
        m_writeSqq = &baseEncode::writeSqqR;
        m_endRowRaw = &baseEncode::endRowRawR;

        // point to the member function that corresponds to the size of the k-mer seed
        if( m_psip->pa21sb->seedWidth <= 21 )
            m_computeKmerHash = &baseEncode::computeKmerHash21R;
        else
        if( m_psip->pa21sb->seedWidth <= 42 )
            m_computeKmerHash = &baseEncode::computeKmerHash42R;
        else
            m_computeKmerHash = &baseEncode::computeKmerHash84R;
    }
    else
    {
        m_writeSqm = &baseEncode::writeSqmQ;
        m_writeRaw = &baseEncode::writeRawQ;
        m_writeSqq = &baseEncode::writeSqqQ;
        m_endRowRaw = &baseEncode::endRowRawQ;

        // point to the member function that corresponds to the size of the k-mer seed
        m_computeKmerHash = (m_psip->pa21sb->seedWidth <= 21) ? &baseEncode::computeKmerHash21Q : &baseEncode::computeKmerHash42Q;
    }
}

/// [public] destructor
baseEncode::~baseEncode()
{
    if( m_deleteSCW && m_pscw )
    {
        // delete the SAMConfigWriter instance (see main() in the derived class)
        delete m_pscw;
    }
}
#pragma endregion

#pragma region private methods
/// [private] method reverseComplement
void baseEncode::reverseComplement( char* pSqRaw, INT64 cchRaw )
{
    // point to the ends of the string
    char* l = pSqRaw;                   // leftmost symbol
    char* r = pSqRaw + cchRaw - 1;      // rightmost symbol

    while( l < r )
    {
        // swap the leftmost and rightmost symbols and replace them with their complements
        char c = *l;
        *(l++) = m_psip->DnaComplement[static_cast<UINT32>(*r)];
        *(r--) = m_psip->DnaComplement[static_cast<UINT32>(c)];
    }

    // if there is an odd number of symbols in the sequence, complement the "middle" symbol
    if( l == r )
        *l = m_psip->DnaComplement[static_cast<UINT32>(*l)];
}

// [private] method computeKmerHash21R
bool baseEncode::computeKmerHash21R( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl )
{
    /* (see also tuLoadR::computeKmerHash21) */

    // get the encoded symbols for the kmer
    UINT64 i64lo = *p64;            // low-order 64 bits (21 symbols)

    // since k <= 21, a hash seed (i.e., a kmer) covers a maximum of two adjacent 64-bit values
    if( shr )
    {
        UINT64 i64hi = *(++p64);
        i64lo = ((i64lo >> shr) | (i64hi << shl)) & MASKA21;
    }

    // do not hash the kmer if it contains too many Ns to align
    if( static_cast<INT16>(__popcnt64( i64lo & m_psip->pa21sb->maskNonN )) < m_psip->pa21sb->minNonN )
        return false;

    // copy the sequence ID and position into the output buffer
    *(p++) = sqId;                          // sqId; point past sqId
    *reinterpret_cast<INT32*>(p) = pos;     // pos
    p = PINT64INCR( p, sizeof(INT32) );     // point past pos

    // convert DNA bases
    i64lo = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64lo );

    // save the hash of the kmer at the current position in the output buffer
    UINT32 hashKey = m_psip->pa21sb->ComputeH32( i64lo );
    *reinterpret_cast<UINT32*>(p) = hashKey;

    // count the number of times the hashKey occurs in the reference sequence(s)
    InterlockedIncrement( &m_psip->C[hashKey].nJ );

    // point to the next available position in the output buffer
    p = PINT64INCR( p, sizeof(UINT32) );

    return true;
}

// [private] method computeKmerHash42R
bool baseEncode::computeKmerHash42R( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl )
{
    // get the encoded symbols for the kmer
    UINT64 i64lo = *p64;            // low-order 64 bits (21 symbols)

    // k > 21 and k <= 42, so a hash seed (i.e., a kmer) covers a maximum of three adjacent 64-bit values
    UINT64 i64hi = *(++p64);
    if( shr )
    {
        i64lo = ((i64lo >> shr) | (i64hi << shl)) & MASKA21;
        UINT64 i64n = *(++p64);
        i64hi = ((i64hi >> shr) | (i64n << shl)) & MASKA21;
    }

    // do not hash the kmer if it contains too many Ns to align
    if( static_cast<INT16>(__popcnt64( i64lo & MASK100 ) + __popcnt64( i64hi & m_psip->pa21sb->maskNonN )) < m_psip->pa21sb->minNonN )
        return false;

    // copy the sequence ID and position into the output buffer
    *(p++) = sqId;                          // sqId; point past sqId
    *reinterpret_cast<INT32*>(p) = pos;     // pos
    p = PINT64INCR( p, sizeof(INT32) );     // point past pos

    // convert DNA bases
    i64lo = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64lo );
    i64hi = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64hi );

    // save the hash of the kmer at the current position in the output buffer
    if( m_cbHash == sizeof(INT64) )
        *p = m_psip->pa21sb->ComputeH64( i64lo, i64hi );
    else
    {
        UINT32 hashKey = m_psip->pa21sb->ComputeH32( i64lo, i64hi );
        *reinterpret_cast<INT32*>(p) = hashKey;

        // count the number of times the hashKey occurs in the reference sequence(s)
        InterlockedIncrement( &m_psip->C[hashKey].nJ );

#if TODO_CHOP_WHEN_DEBUGGED
        if( hashKey == 0x0060c712 )
            CDPrint( cdpCD0, "%s: hashKey 0x%08x counted for pos=%d sqId=0x%016llx", __FUNCTION__, hashKey, pos, sqId );
#endif
    }

    // point to the next available position in the output buffer
    p = PINT64INCR( p, m_cbHash );

    return true;
}

// [private] method computeKmerHash84R
bool baseEncode::computeKmerHash84R( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl )
{
#if TODO_CHOP_WHEN_DEBUGGED
    if( sqId == 23 )
    {
        if( (pos >= 69726649) && (pos <= 69726655) )
        {
            CDPrint( cdpCD0, "pos = %d", pos );
            AriocCommon::DumpB2164( *p64, pos );
        }
    }
#endif

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

#if TODO_CHOP_WHEN_DEBUGGED
    if( (i64lo == 0x7df6fa5df4dbdda5) && (i64m1 == 0x7b74bbd97dd3cbe6) )
        CDPrint( cdpCD0, "here we are!" );
#endif

    // do not hash the kmer if it contains too many Ns to align
    INT16 nNonN = static_cast<INT16>(__popcnt64( i64lo & MASK100 ) +
                                     __popcnt64( i64m1 & MASK100 ) +
                                     __popcnt64( i64m2 & MASK100 ) +
                                     __popcnt64( i64hi & m_psip->pa21sb->maskNonN ));
    if( nNonN < m_psip->pa21sb->minNonN )
        return false;

    // copy the sequence ID and position into the output buffer
    *(p++) = sqId;                          // sqId; point past sqId
    *reinterpret_cast<INT32*>(p) = pos;     // pos
    p = PINT64INCR( p, sizeof(INT32) );     // point past pos

    i64lo = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64lo );
    i64m1 = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64m1 );
    i64m2 = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64m2 );
    i64hi = (m_psip->pa21sb->*(m_psip->pa21sb->fnBaseConvert))( i64hi );

    // save the hash of the kmer at the current position in the output buffer
    if( m_cbHash == sizeof(INT64) )
        *p = m_psip->pa21sb->ComputeH64( i64lo, i64m1, i64m2, i64hi );
    else
    {
        UINT32 hashKey = m_psip->pa21sb->ComputeH32( i64lo, i64m1, i64m2, i64hi );
        *reinterpret_cast<INT32*>(p) = hashKey;

        // count the number of times the hashKey occurs in the reference sequence(s)
        InterlockedIncrement( &m_psip->C[hashKey].nJ );


#if TODO_CHOP_WHEN_DEBUGGED
        // first Q sequence in Q20110915 (random 100-mers)
if( (i64lo == 0x7f27fff97fffff3f)  &&   // TTATTTTTTTCATTTTTAATT
    (i64m1 == 0x4f2cb2c9e49e7de6) &&    // GATGTATAAATAACACACATA
    (i64m2 == 0x7fe492ff3ef2ef64) &&    // AACTGCATGTATTCAAAATTT
    (i64hi == 0x4f3efffffc97f92c) )     // ACAATTCAATTTTTTTGTATA
{
    CDPrint( cdpCD0, "hashKey = 0x%08X at pos = %d", hashKey, pos );
}
#endif



    }

    // point to the next available position in the output buffer
    p = PINT64INCR( p, m_cbHash );

    return true;
}

// [private] method computeKmerHash21Q
bool baseEncode::computeKmerHash21Q( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl )
{
    // get the encoded symbols for the kmer
    UINT64 i64lo = *p64;            // low-order 64 bits (21 symbols)

    // since k <= 21, a hash seed (i.e., a kmer) covers a maximum of two adjacent 64-bit values
    if( shr )
    {
        UINT64 i64hi = *(++p64);
        i64lo = ((i64lo >> shr) | (i64hi << shl)) & MASKA21;
    }

    // do not hash the kmer if it contains too many Ns to align
    if( static_cast<INT16>(__popcnt64( i64lo & m_psip->pa21sb->maskNonN )) < m_psip->pa21sb->minNonN )
        return false;

    // copy the sequence ID and position into the output buffer
    *(p++) = sqId;                                          // sqId; point past sqId
    *reinterpret_cast<INT16*>(p) = static_cast<INT16>(pos); // pos
    p = PINT64INCR( p, sizeof(INT16) );                     // point past pos

    // save the hash of the kmer at the current position in the output buffer
    UINT32 hashKey = m_psip->pa21sb->ComputeH32( i64lo );
    *reinterpret_cast<UINT32*>(p) = hashKey;

    // point to the next available position in the output buffer
    p = PINT64INCR( p, sizeof(UINT32) );

    return true;
}

// [private] method computeKmerHash42Q
bool baseEncode::computeKmerHash42Q( INT64& sqId, INT32 pos, INT64*& p, INT64*& p64, INT32& shr, INT32& shl )
{
    // get the encoded symbols for the kmer
    UINT64 i64lo = *p64;            // low-order 64 bits (21 symbols)

    // k > 21 and k <= 42, so a hash seed (i.e., a kmer) covers a maximum of three adjacent 64-bit values
    UINT64 i64hi = *(++p64);
    if( shr )
    {
        i64lo = ((i64lo >> shr) | (i64hi << shl)) & MASKA21;
        UINT64 i64n = *(++p64);
        i64hi = ((i64hi >> shr) | (i64n << shl)) & MASKA21;
    }

    // do not hash the kmer if it contains too many Ns to align
    if( static_cast<INT16>(__popcnt64( i64lo & MASK100 ) + __popcnt64( i64hi & m_psip->pa21sb->maskNonN )) < m_psip->pa21sb->minNonN )
        return false;

    // copy the sequence ID and position into the output buffer
    *(p++) = sqId;                          // sqId; point past sqId
    *reinterpret_cast<INT16*>(p) = pos;     // pos
    p = PINT64INCR( p, sizeof(INT16) );     // point past pos

    // save the hash of the kmer at the current position in the output buffer
    if( m_cbHash == sizeof(INT64) )
        *p = m_psip->pa21sb->ComputeH64( i64lo, i64hi );
    else
        *reinterpret_cast<INT32*>(p) = m_psip->pa21sb->ComputeH32( i64lo, i64hi );

    // point to the next available position in the output buffer
    p = PINT64INCR( p, m_cbHash );

    return true;
}

// [private] method writeKmersR
void baseEncode::writeKmersR( INT64 sqId, INT64* pB2164, INT64 N )
{
    // compute the 0-based rightmost possible position for a kmer in the specified sequence
    INT32 posMax = static_cast<INT32>(N - m_psip->pa21sb->seedWidth);

#if TODO_CHOP_IF_UNUSED
    // compute the number of kmers to be generated for the row
    INT64 nKmers = blockdiv( posMax+1, m_psip->pa21sb->seedInterval );
#endif

    // compute the number of bytes required to write one kmer
    INT16 cbKmer = static_cast<INT16>(sizeof(INT64) +       // sqId
                                      sizeof(INT32) +       // pos
                                      m_cbHash);            // hashKey (hash of kmer)

    // if the buffer is full, flush it
    if( (m_outBufKmers.n+cbKmer) > m_outBufKmers.cb )
        flushKmers();

    /* Copy the specified encoded sequence data into the buffer in the following format:

                byte 0x00-0x07:  sequence ID
                byte 0x08-0x0B:  0-based position of the seed (first symbol of the kmer) in the sequence
                byte 0x0C-0x0F:  kmer hash (32-bit)
           <or> byte 0x0C-0x13:  kmer hash (64-bit)

       This happens to be the binary format used to import and export SQL Server bulk data to a table with the following schema:

                sqId     bigint    not null,
                pos      int       not null,
                hashKey  int       not null  -- for 32-bit hash keys
           <or> hashKey  bigint    not null  -- for 64-bit hash keys
    */
    INT64* p = reinterpret_cast<INT64*>(m_outBufKmers.p + m_outBufKmers.n);     // point to sqId for the first "row" to be written

    for( INT32 pos=0; pos<=posMax; pos+=m_psip->pa21sb->seedInterval )
    {
        // point to the 64-bit value that contains the first 3-bit encoded symbol in the pos'th kmer to be processed
        INT64* p64 = pB2164 + (pos/21);

        // compute the number of bits to shift
        INT32 shr = 3 * (pos%21);
        INT32 shl = 63 - shr;

        // compute the kmer hash
        if( (this->*m_computeKmerHash)( sqId, pos, p, p64, shr, shl ) )
        {
            // update the number of bytes written into the buffer
            m_outBufKmers.n += cbKmer;

            // if the buffer is full, flush it
            if( (m_outBufKmers.n+cbKmer) > m_outBufKmers.cb )
            {

#if TODO_CHOP_WHEN_DEBUGGED
                if( reinterpret_cast<char*>(p) != (m_outBufKmers.p + m_outBufKmers.n) )
                    DebugBreak();
#endif

                flushKmers();                                   // write the contents of the buffer
                p = reinterpret_cast<INT64*>(m_outBufKmers.p);  // reset the buffer pointer
            }
        }
        else
            InterlockedIncrement( &m_psip->nKmersWithN );
    }

#if TODO_CHOP_WHEN_DEBUGGED
    if( reinterpret_cast<char*>(p) != (m_outBufKmers.p + m_outBufKmers.n) )
        DebugBreak();
#endif
}

// [private] method writeKmersQ
void baseEncode::writeKmersQ( INT64 sqId, INT64* pB2164, INT64 N )
{
    // compute the 0-based rightmost possible position for a kmer in the specified sequence
    INT16 posMax = static_cast<INT16>(N - m_psip->pa21sb->seedWidth);

#if TODO_CHOP_IF_UNUSED
    // compute the number of kmers to be generated for the row
    INT64 nKmers = blockdiv( posMax+1, m_psip->pa21sb->seedInterval );
#endif

    // compute the number of bytes required to write one kmer
    INT16 cbKmer = static_cast<INT16>(sizeof(INT64) +       // sqId
                                      sizeof(INT16) +       // pos
                                      m_cbHash);            // hashKey (hash of kmer)

    // if the buffer is full, flush it
    if( (m_outBufKmers.n+cbKmer) > m_outBufKmers.cb )
        flushKmers();

    /* Copy the specified encoded sequence data into the buffer in the following format:

                byte 0x00-0x07:  sequence ID
                byte 0x08-0x09:  0-based position of the seed (first symbol of the kmer) in the sequence
                byte 0x0A-0x0D:  kmer hash (32-bit)
           <or> byte 0x0A-0x11:  kmer hash (64-bit)

       This happens to be the binary format used to import and export SQL Server bulk data to a table with the following schema:

                sqId     bigint    not null,
                pos      smallint  not null,
                hashKey  int       not null  -- for 32-bit hash keys
           <or> hashKey  bigint    not null  -- for 64-bit hash keys
    */
    INT64* p = reinterpret_cast<INT64*>(m_outBufKmers.p + m_outBufKmers.n);     // point to sqId for the first "row" to be written

    for( INT16 pos=0; pos<=posMax; pos+=m_psip->pa21sb->seedInterval )
    {
        // point to the 64-bit value that contains the first 3-bit encoded symbol in the ih'th kmer to be processed
        INT64* p64 = pB2164 + (pos/21);

        // compute the number of bits to shift
        INT32 shr = 3 * (pos%21);
        INT32 shl = 63 - shr;

        // compute the kmer hash
        if( (this->*m_computeKmerHash)( sqId, pos, p, p64, shr, shl ) )
        {
            // update the number of bytes written into the buffer
            m_outBufKmers.n += cbKmer;

            // performance metrics
            InterlockedIncrement( &AriocE::PerfMetrics.nKmersUsed );

            // if the buffer is full, flush it
            if( (m_outBufKmers.n+cbKmer) > m_outBufKmers.cb )
            {

#if TODO_CHOP_WHEN_DEBUGGED
                if( reinterpret_cast<char*>(p) != (m_outBufKmers.p + m_outBufKmers.n) )
                    DebugBreak();
#endif

                flushKmers();                                   // write the contents of the buffer
                p = reinterpret_cast<INT64*>(m_outBufKmers.p);  // reset the buffer pointer
            }
        }
        else
            InterlockedIncrement( &m_psip->nKmersWithN );
    }

#if TODO_CHOP_WHEN_DEBUGGED
    if( reinterpret_cast<char*>(p) != (m_outBufKmers.p + m_outBufKmers.n) )
        DebugBreak();
#endif
}

/// [private] method getNextCaptureInPattern
void baseEncode::getNextCaptureInPattern( char** ppcstart, char** ppcend, char* ppat )
{
    // point to the open and close parenthesis
    *ppcstart = strchr( ppat, '(' );
    *ppcend = *ppcstart ? strchr( (*ppcstart)+1, ')' ) : NULL;
}

/// [private] method getNextOperatorInPattern
bool baseEncode::getNextOperatorInPattern( char** pppat, char* pcstart, char* pcend )
{
    // skip over parentheses (which indicate a range of operators to capture)
    if( **pppat == '(' )
        ++(*pppat);
    if( **pppat == ')' )
        ++(*pppat);

    // return true if the next operator lies within the specified capture range in the pattern
    return ((pcstart < *pppat) && (*pppat < pcend));
}

/// [private] method parseQmetadata
INT32 baseEncode::parseQmetadata( char* capture, char* md, INT32 cb, char* pattern )
{
    // handle the easy cases first (see comments below)
    if( 0 == strcmp( pattern, "(*)" ) )     // capture everything
    {
        strncpy_s( capture, QMDBUFFERSIZE, md, cb );
        return cb;
    }

    if( 0 == strcmp( pattern, "*" ) )       // ignore everything
    {
        *capture = 0;
        return 0;
    }

    // At this point we assume that we have to use the specified pattern to capture a subset of the characters in the
    //  metadata buffer.  We use the specified pattern to do that.
    //
    // The pattern is specified using "operators" (symbols to match) and "captures" (parentheses that indicate which
    //  symbols to extract).  Symbols in the pattern are:
    //
    //      - * represents a wildcard, i.e., one or more non-separator characters
    //      - all other characters (except, of course, ()*) represent separators
    // 
    // Multiple captures within the same pattern are concatenated.
    //
    // Examples:
    //
    //  pattern                captures
    //   (*)                    everything
    //   *                      nothing
    //   (*) *(/*)              everything prior to the first space and everything after the first slash (including the slash)
    //   * (*:*):               the first 2 colon-separated fields after the first space (e.g. <machine_id>:<lane>)
    //   * (*:*:*:*):           the first 4 colon-separated fields after the first space (e.g. <machine_id>:<run_number>:<FCID>:<lane>)
    //
    // This approach is quite inflexible in comparison with regular expressions, but regex's are much harder to write
    //  and undoubtedly slower than the simple implementation here.  Since FASTQ deflines are rigidly formatted and
    //  punctutation is the same for every defline in a FASTQ file, we choose to keep things simple and fast.
    //

    // initialize buffer pointers
    char* ppat = pattern;       // pattern string
    char* pc = capture;         // output buffer

    char* p = md;               // Q-sequence metadata
    char* pLimit = md + cb;

    // initialize a flag that indicates whether we are capturing
    char* pcstart = NULL;
    char* pcend = NULL;
    getNextCaptureInPattern( &pcstart, &pcend, ppat );
    bool capturing = getNextOperatorInPattern( &ppat, pcstart, pcend );

    // find the first separator in the pattern
    char* psep = ppat + strspn( ppat, "()*" );

    // parse each character in the Q-sequence metadata string
    do
    {
        bool atSep = (*p == *psep);
        if( atSep )
        {
            // reset the capture flag if the separator does not lie within a capture range
            capturing = ((pcstart < psep) && (psep < pcend));
        }

        // optionally extract the current character from the Q-sequence metadata
        if( capturing )
            *(pc++) = *p;

        if( atSep )
        {
            // point past the separator in the pattern
            ppat = psep + 1;

            // conditionally update the capture range
            if( pcend && (ppat >= pcend) )
                getNextCaptureInPattern( &pcstart, &pcend, pcend+1 );

            // get the next operator from the pattern
            capturing = getNextOperatorInPattern( &ppat, pcstart, pcend );
            if( *ppat == 0 )
                break;

            // find the next separator in the pattern
            psep = ppat + strspn( ppat, "()*" );
        }
    }
    while( (++p) < pLimit );

    // finalize and return the extracted string
    *pc = 0;                                    // append a terminal null
    return static_cast<INT32>(pc - capture);    // return the number of captured characters
}

/// [private] method getQmetadataReadGroupIndex
INT32 baseEncode::getQmetadataReadGroupIndex( char* pqrg, INT32 cb )
{
    /* Rather than compute a true hash value, we simply convert the last 4 bytes of each read-group info string
        to an integer "hash".  This relies on the fact that, in a FASTQ file, the read-group info strings from
        each record are formatted in a way that differences among them (e.g., lane number) occur at the end
        of the read-group info string.
    */

    // compute the "hash"
    UINT32 hashKey;
    if( cb >= 4 )
        hashKey = *reinterpret_cast<UINT32*>(pqrg+cb-4);
    else
    {
        // special handling when there are fewer than 4 bytes in the entire read-group ID string
        *reinterpret_cast<UINT32*>(pqrg+cb) = 0;            // pad with zeros
        hashKey = *reinterpret_cast<UINT32*>(pqrg);         // use whatever nonzero bytes there are
    }

    // probe the lookup tables for the specified read-group ID string
    INT32 h = static_cast<INT32>(m_ofsqmdRG.n);             // (m_ofsqmdRG.n tracks the number of RGIDs)
    while( (--h) >= 0 )
    {
        if( (hashKey == m_hashqmdRG.p[h]) && (0 == strcmp( pqrg, m_qmdRG.p+m_ofsqmdRG.p[h] )) )
            break;
    }

    // if the specified read-group ID string is not currently in the lookup tables...
    if( h < 0 )
    {
        // sanity check
        if( m_ofsqmdRG.n == _UI8_MAX )
        {
            for( UINT32 i=0; i<m_ofsqmdRG.n; ++i )
                CDPrint( cdpCD0, "%3u: %s", i, m_qmdRG.p+m_ofsqmdRG.p[i] );
            throw new ApplicationException( __FILE__, __LINE__, "the maximum number of read group identifiers (%u) has been exceeded", static_cast<INT32>(_UI8_MAX) );
        }

        // conditionally grow the buffers
        if( m_hashqmdRG.Count == m_ofsqmdRG.n )
        {
            m_hashqmdRG.Realloc( m_hashqmdRG.Count+16, true );
            m_ofsqmdRG.Realloc( m_hashqmdRG.Count, true );
        }

        UINT32 ofs = m_qmdRG.n;
        if( m_qmdRG.Count < (ofs+cb+1) )
            m_qmdRG.Realloc( m_qmdRG.Count+16*(cb), true );

        // update the lookup tables
        m_hashqmdRG.p[m_ofsqmdRG.n] = hashKey;              // append to the hash table
        m_ofsqmdRG.p[m_ofsqmdRG.n] = m_qmdRG.n;             // append to the list of string-buffer offsets
        strcpy_s( m_qmdRG.p+ofs, m_qmdRG.cb-ofs, pqrg );    // append to the list of read-group ID strings
        m_qmdRG.n += (cb + 1);

#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s: %2d %s", __FUNCTION__, m_ofsqmdRG.n, pqrg );
#endif

        // track the number of items in the lookup tables
        h = static_cast<INT32>(m_ofsqmdRG.n++);
    }

    return h;
}
#pragma endregion

#pragma region protected methods
/// [protected] method readInputFile
INT64 baseEncode::readInputFile( char* pRead, INT64 cbRead )
{
    HiResTimer hrt(us);

    cbRead-- ;                                  // leave room for a trailing null byte
    cbRead = min2( m_cbUnread, cbRead );        // compute the number of bytes to read
    cbRead = m_inFile.Read( pRead, cbRead );    // read
    m_cbUnread -= cbRead;                       // track the number of bytes not yet read
    pRead[cbRead] = 0;                          // append a trailing null byte

    // performance metrics
    InterlockedExchangeAdd64( &AriocE::PerfMetrics.usReadRaw, hrt.GetElapsed( false ) );

    return cbRead;                              // return the number of bytes read
}

/// [protected] method findSOL
char* baseEncode::findSOL( char* p, char** ppLimit )
{
    while( (p < *ppLimit) && (*p <= ' ') )  // skip consecutive control characters and spaces
        ++p;

    if( (p == *ppLimit) && (m_cbUnread > 0) )
    {
        // read another chunk of data
        INT64 cbRead = readInputFile( m_inBuf.p, m_inBuf.cb );

        // resume at the start of the buffer
        p = m_inBuf.p;
        *ppLimit = m_inBuf.p + cbRead;

        // look again for the start of the line
        while( (p < *ppLimit) && (*p <= ' ') )  // skip consecutive control characters and spaces
            ++p;
    }

    return ((p < *ppLimit) ? p : NULL);
}

/// [protected] method findEOL
char* baseEncode::findEOL( char** pp, char** ppLimit )
{
    char* pEOL = strpbrk( *pp, "\r\n\x1A" );    // --> first end-of-line byte (0x0D or 0x0A) or end-of-file byte (0x1A)
    if( pEOL == NULL )                          // if there is no end-of-line byte, assume that the end of the buffer is the end of the "line"
        pEOL = *ppLimit;

#if TODO_CHOP_WHEN_DEBUGGED
    // find the next end-of-line byte (0x0A or 0x0D), end-of-file byte (0x1A), or null (0x00)
    char* pEOL = *pp;
    while( (pEOL < *ppLimit) )
    {
        if( (*pEOL == '\n') || (*pEOL == '\r') || (*pEOL == '\x1A') || (*pEOL == 0) )
            break;
        ++pEOL;
    }
#endif

    if( (pEOL == *ppLimit) && (m_cbUnread > 0) )
    {
        // slide the remaining data in the buffer to the start of the buffer
        INT64 cbTail = *ppLimit - *pp;
        memmove_s( m_inBuf.p, cbTail, *pp, cbTail );
        char* pTo = m_inBuf.p + cbTail;                                 // --> first free byte in the buffer

        // read another chunk of data
        INT64 cbRead = readInputFile( pTo, m_inBuf.cb-cbTail );

        // resume at the start of the buffer
        *pp = m_inBuf.p;
        *ppLimit = pTo + cbRead;

        // look again for the end of the line
        pEOL = strpbrk( *pp, "\r\n\x1A" );
        if( pEOL == NULL )
            pEOL = *ppLimit;

#if TODO_CHOP_WHEN_DEBUGGED
        // find the next end-of-line byte (0x0A or 0x0D), end-of-file byte (0x1A), or null (0x00)
        pEOL = *pp;
        while( (pEOL < *ppLimit) )
        {
            if( (*pEOL == '\n') || (*pEOL == '\r') || (*pEOL == '\x1A') || (*pEOL == 0) )
                break;
            ++pEOL;
        }
#endif
    }

    // null-terminate the buffer contents
    *pEOL = 0;

    // return a pointer to the end of the current line of data in the buffer
    return pEOL;
}

/// [protected] method flushSqm
void baseEncode::flushSqm()
{
    CDPrint( cdpCDb, "%s writes %d bytes", __FUNCTION__, m_outBufSqm.n );

    // write the contents of the metadata buffer
    m_outFileSqm.Write( m_outBufSqm.p, m_outBufSqm.n );

    // reset the byte count
    m_outBufSqm.n = 0;
}

/// [protected] method writeSqmR
void baseEncode::writeSqmR( INT64 readId, char* pRaw, INT64 cb )
{
    // build the sequence ID using the user-specified subunit ID instead of the specified read ID; the pair flag is set if the sequence is a reverse complement
    INT64 sqId = AriocDS::SqId::PackSqId( m_subId, m_isRminus, 0, m_psip->DataSourceId );

    // build a short string that indicates whether the strand is forward (+) or reverse complement (-)
    char attrs[64];
    strcpy_s( attrs, sizeof attrs, (cb ? "; " : "") );
    sprintf_s( attrs, sizeof attrs, "%sstrand=(%c)", attrs, (m_isRminus ? '-' : '+') );
    INT16 cbAttrs = static_cast<INT16>(strlen( attrs ));
    INT16 cbMetadata = static_cast<INT16>(cb) + cbAttrs;

    // compute the number of bytes to be added to the buffer
    INT32 cbRow = sizeof(INT64) + sizeof(INT16) + cbMetadata;

    // if the buffer is full, flush it
    if( (m_outBufSqm.n+cbRow) > m_outBufSqm.cb )
        flushSqm();

    /* Copy the specified metadata into the buffer in the following format:
        byte 0-7:  sequence ID
        byte 8-9:  number of bytes in metadata string
        byte A- :  metadata string (NOT null terminated)

       This happens to be the binary format used to import and export SQL Server bulk data to a table with the following schema:
        sqId         bigint         not null,
        sqMetadata   varchar(256)   not null
    */
    char* p = m_outBufSqm.p + m_outBufSqm.n;
    
    *reinterpret_cast<INT64*>(p) = sqId;        // sqId
    p += sizeof(INT64);                         // point past sqId
    *reinterpret_cast<INT16*>(p) = cbMetadata;  // length of metadata string
    p += sizeof(INT16);                         // point past length of metadata string

    #pragma warning( push )
    #pragma warning( disable:4996 )     // (don't nag us about memcpy being "unsafe")
    memcpy( p, pRaw, cb );
    memcpy( p+cb, attrs, cbAttrs );
    #pragma warning( pop )

    // track the total number of bytes in the buffer
    m_outBufSqm.n += cbRow;

    if( !m_isRminus )
    {
        // emit the metadata for a SAM file
        m_pscw->AppendReferenceMetadata( m_subId, pRaw );
    }
}

/// [protected] method writeSqmQ
void baseEncode::writeSqmQ( INT64 readId, char* pRaw, INT64 cb )
{
    // assume that the entire Q-sequence metadata string is to be used as the SAM QNAME
    char* pQNAME = pRaw;
    INT32 cbQNAME = static_cast<INT32>(cb);
    INT32 iRG = -1;     // default value is 0xFFFFFFFF (or 0xFFFF when truncated to 16 bits)

    char qnameBuf[QMDBUFFERSIZE];
    if( m_psip->qmdQNAME && *m_psip->qmdQNAME )
    {
        // parse the Q-sequence metadata to obtain what will eventually become the SAM QNAME
        cbQNAME = parseQmetadata( qnameBuf, pRaw, static_cast<INT32>(cb), m_psip->qmdQNAME );
        if( cbQNAME < 0 )
        {
            char buf[256];
            strncpy_s( buf, sizeof buf, pRaw, cb );
            throw new ApplicationException( __FILE__, __LINE__, "unable to parse Q-sequence metadata \"%s\" using QNAME pattern \"%s\"", buf, m_psip->qmdQNAME );
        }

        // at this point qnameBuf contains the QNAME
        pQNAME = qnameBuf;
    }

    // if the .cfg files declared one or more read groups...
    if( m_psip->ifgRaw.InputFile.p[this->m_iInputFile].QmdRG.n )
    {
        // if we have a read-group pattern (i.e., no per-file read groups)...
        if( m_psip->ifgRaw.InputFile.p[this->m_iInputFile].RGOrdinal < 0 )
        {
            /* Parse the Q-sequence metadata to obtain a read-group ID (a 1-based index into the list of read groups
                extracted from the Q-sequence metadata according to the user-specified pattern). */
            char qrgBuf[QMDBUFFERSIZE];
            INT32 cbRG = parseQmetadata( qrgBuf, pRaw, static_cast<INT32>(cb), m_psip->ifgRaw.InputFile.p[this->m_iInputFile].QmdRG.p );
            if( cbRG < 0 )
            {
                char buf[256];
                strncpy_s( buf, sizeof buf, pRaw, cb );
                throw new ApplicationException( __FILE__, __LINE__, "unable to parse Q-sequence metadata \"%s\" using pattern \"%s\"", buf, m_psip->ifgRaw.InputFile.p[this->m_iInputFile].QmdRG.p );
            }

            // at this point qrgBuf contains read-group info captured from the Q-sequence metadata
            iRG = getQmetadataReadGroupIndex( qrgBuf, cbRG );
        }
        else
        {
            // use the per-file read group ID
            iRG = m_psip->ifgRaw.InputFile.p[this->m_iInputFile].RGOrdinal;
        }
    }

    // build the sequence ID
    INT64 sqId = AriocDS::SqId::PackSqId( readId, m_pairFlag, m_subId, m_psip->DataSourceId );

    // compute the number of bytes to be added to the buffer
    UINT32 cbRow = static_cast<UINT32>(sizeof( MetadataRowHeader ) + cbQNAME);

    // if the buffer is full, flush it
    if( (m_outBufSqm.n+cbRow) > m_outBufSqm.cb )
        flushSqm();

    /* Copy the specified metadata into the buffer in the following format:
        byte 0-7:  sequence ID
        byte 8:    read-group index
        byte 9-A:  number of bytes in metadata string
        byte B- :  metadata string (NOT null terminated)

       This happens to be the binary format used to import and export SQL Server bulk data to a table with the following schema:
        sqId         bigint         not null,
        rgIndex      tinyint        not null,
        sqMetadata   varchar(256)   not null

        This data layout is the same for both Q-sequence metadata (from FASTQ deflines) and base quality scores.  (See the
         definition of struct MetadataRowHeader, which is used for both kinds of metadata.)  The Arioc aligner implementation
         expects this so as not to have separate implementations for loading Q-sequence metadata and BQSs.
    */
    MetadataRowHeader* pmrh = reinterpret_cast<MetadataRowHeader*>(m_outBufSqm.p + m_outBufSqm.n);
    pmrh->sqId = sqId;
    pmrh->byteVal = static_cast<UINT8>(iRG);
    pmrh->cb = static_cast<INT16>(cbQNAME);

#pragma warning( push )
#pragma warning( disable:4996 )         // (don't nag us about memcpy being "unsafe")
    memcpy( pmrh+1, pQNAME, cbQNAME );
#pragma warning( pop )

    // track the total number of bytes in the buffer
    m_outBufSqm.n += cbRow;
}

// [protected] method writeRawR
void baseEncode::writeRawR( INT64 readId, char* pRaw, INT64 cb )
{
    // build the sequence ID using the user-specified subunit ID instead of the specified read ID
    INT64 sqId = AriocDS::SqId::PackSqId( m_subId, m_isRminus, 0, m_psip->DataSourceId );

    // compute the number of bytes to be added to the buffer
    INT64 cbRow = (m_cbRawCurrent ? 0 : sizeof(INT64)+sizeof(INT64)) + cb;

    // sanity check: the buffer should always be large enough to accommodate the data
    if( (m_cbRawCurrent+cbRow) > static_cast<INT64>(m_outBufRaw.cb) )
        throw new ApplicationException( __FILE__, __LINE__, "buffer capacity exceeded (%lld/%lld bytes)", m_cbRawCurrent+cbRow, m_outBufRaw.cb );

    /* Copy the specified symbol data into the buffer in the following format:
        byte 0-7:  sequence ID
        byte 8-F:  number of bytes in symbol string
        byte A- :  symbol string (NOT null terminated)

       This happens to be the binary format used to import and export SQL Server bulk data to a table with the following schema:
        sqId         bigint        not null,
        sq           varchar(max)  not null
    */
    char* p = m_outBufRaw.p;

    if( m_cbRawCurrent )
        p += m_cbRawCurrent;
    else
    {
        *reinterpret_cast<INT64*>(p) = sqId;    // sqId
        p += sizeof(INT64) + sizeof(INT64);     // point past sqId and len(sq)
    }

#pragma warning( push )
#pragma warning( disable:4996 )                 // (don't nag us about memcpy being "unsafe")
    memcpy( p, pRaw, cb );
#pragma warning( pop )

    // track the total number of bytes in the current row; the total number of bytes in the buffer is m_outBufRaw.n+m_cbRawCurrent (see flushRaw)
    m_cbRawCurrent += cbRow;
}

// [private] method writeRawQ
void baseEncode::writeRawQ( INT64 readId, char* pRaw, INT64 cb )
{
    // build the sequence ID
    INT64 sqId = AriocDS::SqId::PackSqId( readId, m_pairFlag, m_subId, m_psip->DataSourceId );

    // compute the number of bytes to be added to the buffer
    INT64 cbRow = (m_cbRawCurrent ? 0 : sizeof(INT64)+sizeof(INT16)) + cb;

    // if the buffer is full, flush it
    if( (m_outBufRaw.n+static_cast<UINT32>(m_cbRawCurrent+cbRow)) > static_cast<UINT32>(m_outBufRaw.cb) )
        flushRaw( false );

    /* Copy the specified symbol data into the buffer in the following format:
        byte 0-7:  sequence ID
        byte 8-9:  number of bytes in symbol string
        byte A- :  symbol string (NOT null terminated)

       This happens to be the binary format used to import and export SQL Server bulk data to a table with the following schema:
        sqId         bigint      not null,
        sq           varchar(n)  not null       -- (n <= 8000)
    */
    char* p = m_outBufRaw.p + m_outBufRaw.n;

    if( m_cbRawCurrent )
        p += m_cbRawCurrent;
    else
    {
        *reinterpret_cast<INT64*>(p) = sqId;    // sqId
        p += sizeof(INT64) + sizeof(INT16);     // point past sqId and len(sq)
    }

#pragma warning( push )
#pragma warning( disable:4996 )                 // (don't nag us about memcpy being "unsafe")
    memcpy( p, pRaw, cb );
#pragma warning( pop )

    // performance metrics
    InterlockedExchangeAdd( &m_psip->nSymbolsEncoded, cb );

    // track the total number of bytes in the current row; the total number of bytes in the buffer is m_outBufRaw.n+m_cbRawCurrent (see flushRaw)
    m_cbRawCurrent += cbRow;
}

// [private] method writeSqqR
void baseEncode::writeSqqR( INT64 readId, char* pSqq, INT64 cb )
{
    throw new ApplicationException( __FILE__, __LINE__, "not implemented" );
}

// [private] method writeSqqQ
void baseEncode::writeSqqQ( INT64 readId, char* pSqq, INT64 cb )
{
    // build the sequence ID
    INT64 sqId = AriocDS::SqId::PackSqId( readId, m_pairFlag, m_subId, m_psip->DataSourceId );

    // compute the number of bytes to be added to the buffer
    INT64 cbRow = (m_cbSqqCurrent ? 0 : sizeof(MetadataRowHeader)) + cb;

    // if the buffer is full, flush it
    if( (m_outBufSqq.n+static_cast<UINT32>(m_cbSqqCurrent+cbRow)) > static_cast<UINT32>(m_outBufSqq.cb) )
        flushSqq( false );

    /* Copy the specified symbol data into the buffer in the following format:
        byte 0-7:  sequence ID
        byte 8:    quality-score bias
        byte 9-A:  number of bytes in binary quality-score string
        byte B- :  binary quality-score string

       This happens to be the binary format used to import and export SQL Server bulk data to a table with the following schema:
        sqId         bigint        not null,
        qsb          smallint      not null,
        sqq          varbinary(n)  not null       -- (n <= 8000)

       This data layout is the same for both Q-sequence metadata (from FASTQ deflines) and base quality scores.  (See the
        definition of struct MetadataRowHeader, which is used for both kinds of metadata.)  The Arioc aligner implementation
        expects this so as not to have separate implementations for loading Q-sequence metadata and BQSs.
    */
    char* p = m_outBufSqq.p + m_outBufSqq.n;

    if( m_cbSqqCurrent )
        p += m_cbSqqCurrent;
    else
    {
        MetadataRowHeader* pmrh = reinterpret_cast<MetadataRowHeader*>(p);
        pmrh->sqId = sqId;
        pmrh->byteVal = m_psip->QualityScoreBias;
        pmrh->cb = static_cast<INT16>(cb);
        p += sizeof(MetadataRowHeader);
    }

#pragma warning( push )
#pragma warning( disable:4996 )                 // (don't nag us about memcpy being "unsafe")
    memcpy( p, pSqq, cb );
#pragma warning( pop )

    // track the total number of bytes in the current row; the total number of bytes in the buffer is m_outBufSqq.n+m_cbSqqCurrent (see flushSqq)
    m_cbSqqCurrent += cbRow;
}

// [private] method writeAriocR
void baseEncode::writeAriocR( INT64 readId, char* pRaw, INT64 nSymbols )
{
    // build the sequence ID
    INT64 sqId = AriocDS::SqId::PackSqId( m_subId, m_isRminus, 0, m_psip->DataSourceId );
                 
    // compute the number of bytes required for the encoded row
    UINT32 cbBinary = static_cast<UINT32>(sizeof(INT64)*blockdiv(nSymbols, 21));   // the number of bytes used for the 64-bit values that encode the symbols in the row 
    UINT32 cbRow = sizeof(INT64) +      // sqId
                    sizeof(INT64) +     // N
                    sizeof(INT64) +     // len(B2164)
                    cbBinary;           // B2164

    // sanity check: the buffer should always be large enough to accommodate the encoded data
    if( (m_outBufA21.n+cbRow) > static_cast<UINT32>(m_outBufA21.cb) )
        throw new ApplicationException( __FILE__, __LINE__, "buffer capacity exceeded (%u/%lld bytes)", m_outBufA21.n+cbRow, m_outBufA21.cb );

    /* Copy the specified encoded sequence data into the buffer in the following format:
        byte 0-7       :  sequence ID
        byte 8-0x0F    :  number of symbols in the sequence
        byte 0x10-0x17 :  number of bytes in the binary string
        byte 0x18-     :  binary string (representing a vector of bigint (64-bit) values)

       This happens to be the binary format used to import and export SQL Server bulk data to a table with the following schema:
        sqId         bigint          not null,
        N            bigint          not null,
        B2164        varbinary(max)  not null
    */
    INT64* p = reinterpret_cast<INT64*>(m_outBufA21.p);
    *(p++) = sqId;          // sqId
    *(p++) = nSymbols;      // N
    *(p++) = cbBinary;      // len(B2164)

    // save a copy of the pointer to the first encoded 64-bit value
    INT64* pB2164 = p;

    // encode the bytes
    INT64 b21 = 0;                                  // packed 3-bit encoded values
    INT32 shl = 0;                                  // bit offset
    for( INT32 n=0; n<nSymbols; ++n )
    {
        // encode the nth symbol
        UINT32 rawSymbol = *(pRaw++);
        b21 |= (static_cast<INT64>(m_psip->SymbolEncode[rawSymbol])) << shl;

        // update the bit offset
        if( shl == 60 )
        {
            *(p++) = b21;   // copy the encoded ("packed") symbols into the buffer and increment the buffer pointer
            b21 = 0;        // reset the bits
            shl = 0;        // reset the bit shift
        }
        else
            shl += 3;       // adjust the bit shift
    }

    // if necessary, write the final 64-bit encoded value
    if( b21 )
        *p = b21;

    // track the total number of bytes in the buffer
    m_outBufA21.n += cbRow;

    // write kmers and count hash keys
    if( m_psip->pa21sb->seedInterval )
        writeKmersR( sqId, pB2164, nSymbols );
}

// [private] method writeAriocQ
void baseEncode::writeAriocQ( INT64 readId, char* pRaw, INT64 nSymbols )
{
    // build the sequence ID
    INT64 sqId = AriocDS::SqId::PackSqId( readId, m_pairFlag, m_subId, m_psip->DataSourceId );

    // compute the number of bytes required for the encoded row
    UINT32 cbBinary = static_cast<UINT32>(sizeof(INT64)*blockdiv(nSymbols, 21));   // the number of bytes used for the 64-bit values that encode the symbols in the row 
    UINT32 cbRow = sizeof(INT64) +      // sqId
                    sizeof(INT16) +     // N
                    sizeof(INT16) +     // len(B2164)
                    cbBinary;           // B2164

    // if the buffer is full, flush it
    if( (m_outBufA21.n+cbRow) > m_outBufA21.cb )
        flushA21();

    /* Copy the specified encoded sequence data into the buffer in the following format:
        byte 0-7:  sequence ID
        byte 8-9:  number of symbols in the sequence
        byte A-B:  number of bytes in binary string
        byte C- :  binary string (representing a vector of bigint (64-bit) values)

       This happens to be the binary format used to import and export SQL Server bulk data to a table with the following schema:
        sqId         bigint        not null,
        N            smallint      not null,
        B2164        varbinary(n)  not null       -- (n <= 8000)
    */
    INT64* p = reinterpret_cast<INT64*>(m_outBufA21.p + m_outBufA21.n);
    *(p++) = sqId;                              // sqId
    *reinterpret_cast<INT16*>(p) = static_cast<INT16>(nSymbols);    // N
    p = PINT64INCR( p, sizeof(INT16) );                             // point past N
    *reinterpret_cast<INT16*>(p) = cbBinary;                        // len(B2164)
    p = PINT64INCR( p, sizeof(INT16) );                             // point past len(B2164)

    // save a copy of the pointer to the first encoded 64-bit value
    INT64* pB2164 = p;

    // encode the bytes
    INT64 b21 = 0;                                  // packed 3-bit encoded values
    INT32 shl = 0;                                  // bit offset
    for( INT32 n=0; n<nSymbols; ++n )
    {
        // encode the nth symbol
        UINT32 rawSymbol = *(pRaw++);
        b21 |= (static_cast<INT64>(m_psip->SymbolEncode[rawSymbol])) << shl;

        // update the bit offset
        if( shl == 60 )
        {
            *(p++) = b21;   // copy the encoded ("packed") symbols into the buffer and increment the buffer pointer
            b21 = 0;        // reset the bits
            shl = 0;        // reset the bit shift
        }
        else
            shl += 3;       // adjust the bit shift
    }

    // if necessary, write the final 64-bit encoded value
    if( b21 )
        *p = b21;

    // track the total number of bytes in the buffer
    m_outBufA21.n += cbRow;

    if( m_psip->pa21sb->seedInterval )
        writeKmersQ( sqId, pB2164, nSymbols );
}

// [protected] method endRowRawR
void baseEncode::endRowRawR()
{
    // point to the the current row in the output buffer; the sequence ID is the first field in the row
    INT64* pSqId = reinterpret_cast<INT64*>(m_outBufRaw.p);

    // point to the string length (number of symbols) field in the buffer
    INT64* pcb = pSqId + 1;

    // point to the symbol string
    char* pRaw = reinterpret_cast<char*>(pcb + 1);
    
    // update the string length in the buffer; this is the total row size less the number of bytes in the fixed-size fields
    *pcb = m_cbRawCurrent - (sizeof(INT64)+sizeof(INT64));

    // accumulate the total number of symbols
    InterlockedExchangeAdd( &m_psip->nSymbolsEncoded, *pcb );

    // update the buffer byte counts
    m_outBufRaw.n = static_cast<UINT32>(m_cbRawCurrent);
    m_cbRawCurrent = 0;

    // if we're doing the reverse complement, now is the time to compute it
    if( m_isRminus )
        reverseComplement( pRaw, *pcb );

    // write the encoded sequence data into its buffer
    writeAriocR( *pSqId, pRaw, *pcb );

    if( !m_isRminus )
    {
        // emit the metadata for a SAM file
        m_pscw->AppendReferenceLength( m_subId, *pcb );
    }
}

// [protected] method endRowRawQ
void baseEncode::endRowRawQ()
{
    // point to the the current row in the raw output buffer; the sequence ID is the first field in the row
    INT64* pSqId = reinterpret_cast<INT64*>(m_outBufRaw.p + m_outBufRaw.n);

    // point to the string length (number of symbols) field in the buffer
    INT16* pcb = reinterpret_cast<INT16*>(pSqId + 1);

    // point to the symbol string
    char* pRowData = reinterpret_cast<char*>(pcb + 1);
    
    // update the string length in the buffer; this is the total row size less the number of bytes in the fixed-size fields
    *pcb = static_cast<INT16>(m_cbRawCurrent - (sizeof(INT64)+sizeof(INT16)));

    // update the buffer byte counts
    m_outBufRaw.n += static_cast<UINT32>(m_cbRawCurrent);
    m_cbRawCurrent = 0;

    // write the encoded sequence data into its buffer
    writeAriocQ( AriocDS::SqId::GetReadId(*pSqId), pRowData, *pcb );
}

/// [protected] method endRowSqqQ
void baseEncode::endRowSqqQ()
{
    // point to the the current row in the quality-score output buffer; the sequence ID is the first field in the row
    // TODO: CHOP: INT64* pSqId = reinterpret_cast<INT64*>(m_outBufSqq.p + m_outBufSqq.n);
    MetadataRowHeader* pmrh = reinterpret_cast<MetadataRowHeader*>(m_outBufSqq.p + m_outBufSqq.n);

// TODO: CHOP    // point to the string length (number of symbols) field in the buffer
// TODO: CHOP   INT16* pcb = reinterpret_cast<INT16*>(pSqId + 1);

    // point to the quality-score string
// TODO: CHOP    char* pRowData = reinterpret_cast<char*>(pcb + 1);
    char* pRowData = reinterpret_cast<char*>(pmrh+1);
    
    // update the string length in the buffer; this is the total row size less the number of bytes in the fixed-size fields
// TODO: CHOP    *pcb = static_cast<INT16>(m_cbSqqCurrent - sizeof(MetadataRowHeader);
    pmrh->cb = static_cast<INT16>(m_cbSqqCurrent - sizeof( MetadataRowHeader ));

    /* Map the quality scores to 0-based binary values:
        - to minimize memory accesses, we do this in two loops, the first of which processes eight quality scores at a time
        - there's no error checking here; if the user specifies an incorrect quality-score bias, the result is garbage
    */
// TODO: CHOP    INT16 iCut = (*pcb) & (-static_cast<INT16>(sizeof(INT64)));
    INT16 iCut = pmrh->cb & (-static_cast<INT16>(sizeof( INT64 )));
    for( INT16 i=0; i<iCut; i+=sizeof(INT64) )
    {
        *reinterpret_cast<INT64*>(pRowData) -= m_psip->QSB8;
        pRowData += sizeof(INT64);
    }

    // TODO: CHOP        for( INT16 i=iCut; i<(*pcb); ++i )
    for( INT16 i=iCut; i<(pmrh->cb); ++i )
        *(pRowData++) -= m_psip->QualityScoreBias;

    // update the buffer byte counts
    m_outBufSqq.n += static_cast<UINT32>(m_cbSqqCurrent);
    m_cbSqqCurrent = 0;
}

/// [protected] method flushRaw
void baseEncode::flushRaw( bool flushCurrent )
{
    HiResTimer hrt(ms);

    if( flushCurrent )
    {
        CDPrint( cdpCDb, "%s(true) writes %d bytes", __FUNCTION__, m_outBufRaw.n+m_cbRawCurrent );

        // write the contents of the symbols buffer, including current symbols
        m_outFileRaw.Write( m_outBufRaw.p, m_outBufRaw.n+m_cbRawCurrent );

        // reset the byte counts
        m_outBufRaw.n = 0;
        m_cbRawCurrent = 0;
    }

    else
    {
        CDPrint( cdpCDb, "%s(false) writes %d bytes", __FUNCTION__, m_outBufRaw.n );

        // write the contents of the symbols buffer, excluding current symbols
        m_outFileRaw.Write( m_outBufRaw.p, m_outBufRaw.n );

#pragma warning( push )
#pragma warning( disable:4996 )                 // (don't nag us about memcpy being "unsafe")
        // copy the current symbols to the start of the buffer
        memcpy( m_outBufRaw.p, m_outBufRaw.p+m_outBufRaw.n, m_cbRawCurrent );
#pragma warning( pop )

        // reset the byte count
        m_outBufRaw.n = 0;
    }

    // performance metrics
    InterlockedExchangeAdd( reinterpret_cast<volatile LONG*>(&AriocE::PerfMetrics.msWriteSq), hrt.GetElapsed( false ) );
}

/// [protected] method flushSqq
void baseEncode::flushSqq( bool flushCurrent )
{
    HiResTimer hrt(ms);

    if( flushCurrent )
    {
        // write the contents of the symbols buffer, including current symbols
        m_outFileSqq.Write( m_outBufSqq.p, m_outBufSqq.n+m_cbSqqCurrent );

        // reset the byte counts
        m_outBufSqq.n = 0;
        m_cbSqqCurrent = 0;
    }

    else
    {
        // write the contents of the symbols buffer, excluding current symbols
        m_outFileSqq.Write( m_outBufSqq.p, m_outBufSqq.n );

#pragma warning( push )
#pragma warning( disable:4996 )                 // (don't nag us about memcpy being "unsafe")
        // copy the current symbols to the start of the buffer
        memcpy( m_outBufSqq.p, m_outBufSqq.p+m_outBufSqq.n, m_cbSqqCurrent );
#pragma warning( pop )

        // reset the byte count
        m_outBufSqq.n = 0;
    }

    // performance metrics
    InterlockedExchangeAdd( reinterpret_cast<volatile LONG*>(&AriocE::PerfMetrics.msWriteSq), hrt.GetElapsed( false ) );
}

/// [protected] method flushA21
void baseEncode::flushA21()
{
    CDPrint( cdpCDb, "%s writes %d bytes", __FUNCTION__, m_outBufA21.n );

    HiResTimer hrt(ms);

    // write the contents of the encoded-data buffer
    m_outFileA21.Write( m_outBufA21.p, m_outBufA21.n );

    // performance metrics
    InterlockedExchangeAdd( reinterpret_cast<volatile LONG*>(&AriocE::PerfMetrics.msWriteSq), hrt.GetElapsed( false ) );
    InterlockedExchangeAdd64( &AriocE::PerfMetrics.cbArioc, m_outBufA21.n );

    // reset the byte count
    m_outBufA21.n = 0;
}

/// [protected] method flushKmers
void baseEncode::flushKmers()
{
    if( m_psip->pa21sb->seedInterval )
    {
#if TODO_CHOP_WHEN_DEBUGGED
        CDPrint( cdpCD0, "%s writes %d bytes", __FUNCTION__, m_outBufKmers.n );
#endif
        HiResTimer hrt(ms);
        
        if( m_psip->emitKmers )
        {
            // write the contents of the kmers buffer
            m_outFileKmers.Write( m_outBufKmers.p, m_outBufKmers.n );
        }

        // reset the byte count
        m_outBufKmers.n = 0;

        // performance metrics
        InterlockedExchangeAdd( reinterpret_cast<volatile LONG*>(&AriocE::PerfMetrics.msWriteKmers), hrt.GetElapsed( false ) );
    }
}

/// [protected] method writeConfigFile
void baseEncode::writeConfigFile()
{
    // finalize the contents of the config file
    m_pscw->AppendExecutionTime( m_hrt.GetElapsed(false) );
    m_pscw->AppendQualityScoreBias( m_psip->QualityScoreBias );
    m_pscw->AppendReadGroupInfo( &m_psip->pam->RGMgr, &m_qmdRG, &m_ofsqmdRG, m_psip->ifgRaw.InputFile.p+m_iInputFile );

    // write the config file
    m_pscw->Write();
}
#pragma endregion
