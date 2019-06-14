/*
  baseValidateA21.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// <summary>
/// Base implementation for encoded sequence data validation.
/// </summary>
/// <param name="psip">Reference to a common parameter structure</param>
/// <param name="sqCat">sequence category</param>
/// <param name="iInputFile">index of input file</param>
/// <param name="psem">Reference to an <c>RaiiSemaphore</c> instance</param>
baseValidateA21::baseValidateA21( AriocEncoderParams* psip, SqCategory sqCat, INT16 iInputFile, RaiiSemaphore* psem ) :
                                    m_psip( psip ),
                                    m_iInputFile( iInputFile ),
                                    m_srcId( psip->DataSourceId ),
                                    m_subId( psip->ifgRaw.InputFile.p[iInputFile].SubId ),
                                    m_psemComplete( psem ),
                                    m_inBuf(A21BUFFERSIZE,false)
{
    char drive[_MAX_DRIVE];
    char dir[_MAX_DIR];
    char ext[_MAX_EXT];
    char stubFileSpec[FILENAME_MAX];
    char baseName[256];

#pragma warning ( push )
#pragma warning ( disable : 4996 )		// don't nag us about _splitpath being "unsafe"
    // use the input filename as the "base name" for the associated output file
    _splitpath( psip->ifgRaw.InputFile.p[iInputFile].Filespec, drive, dir, baseName, ext );
    if( *baseName == 0 )
        throw new ApplicationException( __FILE__, __LINE__, "%s: no base name in '%s'", __FUNCTION__, psip->ifgRaw.InputFile.p[iInputFile].Filespec );
#pragma warning ( pop )

    strcpy_s( stubFileSpec, FILENAME_MAX, m_psip->OutFilespecStubSq );
    strcat_s( stubFileSpec, FILENAME_MAX, baseName );
    strcat_s( stubFileSpec, FILENAME_MAX, "$" );

    // create a filename extension for the current worker thread
    char outFileExt[32];

    // for a reverse-complement reference sequence, include "rc" in the file specification
    if( sqCat == sqCatRminus )
        strcpy_s( outFileExt, sizeof outFileExt, ".rc.sbf" );
    else
        strcpy_s( outFileExt, sizeof outFileExt, ".sbf" );

    // open the encoded data output file for input
    char inFileSpec[FILENAME_MAX];
    strcpy_s( inFileSpec, FILENAME_MAX, stubFileSpec );
    strcat_s( inFileSpec, FILENAME_MAX, "a21" );
    strcat_s( inFileSpec, FILENAME_MAX, outFileExt );
    m_inFile.OpenReadOnly( inFileSpec );

    // set up for buffered input
    m_inBuf.n = static_cast<UINT32>(m_inBuf.cb);
}

/// [public] destructor
baseValidateA21::~baseValidateA21()
{
}
#pragma endregion

#pragma region protected methods
/// [protected] method getNext
UINT8* baseValidateA21::getNext( INT32 cb )
{
    /* The first call to this method (to initialize the buffer) should have m_inBuf.n = m_inBuf.cb. */

    // if all of the requested bytes are not in the buffer...
    if( (m_inBuf.n+cb) > m_inBuf.cb )
    {
        // move any remaining bytes to the start of the buffer
        size_t cbx = m_inBuf.cb - m_inBuf.n;
        memmove( m_inBuf.p, m_inBuf.p+m_inBuf.n, cbx );
        m_inBuf.n = 0;

        // fill the remaining bytes in the buffer
        INT64 cbRead = m_inFile.Read( m_inBuf.p+cbx, m_inBuf.cb-cbx );
        if( cbRead == 0 )
            throw new ApplicationException( __FILE__, __LINE__, "unexpected end of file: %s", m_inFile.FileSpec.p );
    }

    // return a pointer to the requested bytes
    UINT8* rval = m_inBuf.p + m_inBuf.n;

    // update the current offset within the buffer
    m_inBuf.n += cb;

    // return a pointer to the data bytes
    return rval;
}
#pragma endregion
