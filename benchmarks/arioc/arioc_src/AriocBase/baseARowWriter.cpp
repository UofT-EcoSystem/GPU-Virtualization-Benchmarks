/*
  baseARowWriter.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// constructor
baseARowWriter::baseARowWriter( AriocBase* pab ) : m_pab(pab),
                                                   m_ofi("","","",oftUnknown,arfNone,0),
                                                   m_outFileSeqNo(0),
                                                   m_nBufs(pab->nGPUs),
                                                   m_nA(0),
                                                   m_outBuf(NULL),
                                                   IsActive(false),
                                                   TotalA(0)
{
    memset( m_filespecStub, 0, sizeof m_filespecStub );
    memset( m_filespecExt, 0, sizeof m_filespecExt );

    m_outBuf = new WinGlobalPtr<char>[m_nBufs];
}

/// destructor
baseARowWriter::~baseARowWriter()
{
    if( m_outBuf )
        delete[] m_outBuf;
}
#pragma endregion

#pragma region virtual base method implementations
/// [public] method Lock
char* baseARowWriter::Lock( INT16 iBuf, UINT32 cb )
{
    static char bitBucket[BITBUCKET_SIZE];

    return bitBucket;
}

/// [public] method Release
void baseARowWriter::Release( INT16 iBuf, UINT32 cb )
{
    // count the total number of reported alignments
    InterlockedIncrement( reinterpret_cast<volatile UINT64*>(&this->TotalA) );
}

/// [public] method Flush
void baseARowWriter::Flush( INT16 iBuf )
{
}

/// [public] method Close
void baseARowWriter::Close()
{
}
#pragma endregion
