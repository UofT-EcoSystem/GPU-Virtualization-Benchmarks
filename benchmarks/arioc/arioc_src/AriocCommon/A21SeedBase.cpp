/*
  A21SeedBase.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructor and destructor
/// [protected] constructor
A21SeedBase::A21SeedBase() : m_seedIndex(-1),
                             iError(-1),
                             iNone(-1),
                             maskNonN(0),
                             seedMask(0),
                             hashBitsMask(0),
                             maxMismatches(0),
                             seedWidth(0),
                             hashKeyWidth(0),
                             seedInterval(0),
                             minNonN(0)
{
}

/// [protected] destructor
A21SeedBase::~A21SeedBase()
{
}
#pragma endregion

#pragma region private methods
/// [private] method convertNone
UINT64 A21SeedBase::convertNone( const UINT64 v )
{
    return v;
}

/// [private] method convertCT
UINT64 A21SeedBase::convertCT( const UINT64 v )
{
    /* Convert all instances of C (101) to T (111) by ORing with a bitmask wherein...
        - bit 1 of each 3-bit field is set only when that subfield's value is 101
        - bits 0 and 2 of each 3-bit field are zero
    */
    return v | (((v >> 1) & (~v) & (v << 1)) & MASK010);
}
#pragma endregion

#pragma region protected methods
/// [protected] method baseInit
void A21SeedBase::baseInit( const char* _si )
{
    // save the special-case index values
    this->iError = this->SI.IndexOf( "error" );
    this->iNone = this->SI.IndexOf( "none" );

    // handle the case where the specified seed ID is null
    if( _si == NULL )
        _si = "none";

    // save the index of the specified string
    m_seedIndex = this->SI.IndexOf( _si );
    if( m_seedIndex < 0 )
        m_seedIndex = this->iError;

    // save an ID string for the specified seed info
    strcpy_s( this->IdString, sizeof this->IdString, this->SI.Key(m_seedIndex) );

    // set up to do seed base symbol conversion
    if( this->SI.Value(m_seedIndex).baseConversion == A21SeedBase::bcCT )
    {
        this->baseConvert = A21SeedBase::bcCT;
        this->fnBaseConvert = &A21SeedBase::convertCT;
    }
    else
    {
        this->baseConvert = A21SeedBase::bcNone;
        this->fnBaseConvert = &A21SeedBase::convertNone;
    }
}
#pragma endregion

#pragma region public methods
/// [public] method StringToSeedInfo
SeedInfo A21SeedBase::StringToSeedInfo( const char* s )
{
    INT32 i = this->SI.IndexOf( s );
    if( i >= 0 )
        return this->SI.Value( i );

    return this->SI["error"];
}

/// [public] method IsNull
bool A21SeedBase::IsNull()
{
    // return true if the specified index refers to "error" or "none" seed info
    return (m_seedIndex == this->iError) || (m_seedIndex == this->iNone);
}
#pragma endregion
