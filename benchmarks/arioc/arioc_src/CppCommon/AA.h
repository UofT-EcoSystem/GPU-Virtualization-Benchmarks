/*
  AA.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AA__

#ifndef __RaiiCriticalSection__
#include "RaiiCriticalSection.h"
#endif

/// <summary>
/// Template <c>AA</c> implements a simple associative array
/// </summary>
template <typename T> class AA
{
    private:
        struct KVP
        {
            UINT32  ofsK;       // offset into the Key string buffer
            UINT32  ofsV;       // offset into the Value buffer
        };

    private:
        WinGlobalPtr<char>  m_K;    // key strings
        WinGlobalPtr<T>     m_V;    // values
        WinGlobalPtr<KVP>   m_KVP;  // key pointers and values

    public:
        UINT32  Count;

    public:
        AA( UINT32 initialCapacity=32 );
        virtual ~AA( void );
        T& operator[]( const char* _k );
        INT32 IndexOf( const char* _k );
        const char* Key( const size_t i );
        T& Value( const size_t i );
        void Reset( void );
};

// default constructor
template <typename T> AA<T>::AA( UINT32 initialCapacity ) : m_V(initialCapacity,true),  m_KVP(initialCapacity,true), Count(0)
{
}

/// destructor
template <typename T> AA<T>::~AA()
{
    // discard the lists
    m_KVP.Free();
    m_K.Free();
}

/// <summary>
/// Dereferences a specified key and returns the corresponding value, using C++ array subscript syntax.
/// </summary>
/// <remarks>Assignment is not thread-safe!</remarks>
template <typename T> T& AA<T>::operator[]( const char* _k )
{
    RaiiCriticalSection< AA<T> > rcs;

    // look for the specified key-value pair
    INT32 i = IndexOf( _k );

    // if the specified key exists, return the corresponding value
    if( i >= 0 ) 
        return m_V.p[m_KVP.p[i].ofsV];

    // at this point the key was not found
    UINT32 insertAt = static_cast<UINT32>(i&0x7FFFFFFF);    // index at which to insert the new Key and Value

    // expand the buffers if necessary
    if( this->Count == m_KVP.Count )
    {
        m_KVP.Realloc( this->Count+32, true );
        m_V.Realloc( this->Count+32, true );
    }

    // expand the Key string buffer if necessary
    size_t cb = strlen(_k) + 1;             // size of new Key
    size_t tail = m_K.n + cb;               // new size of the portion of the Key buffer used for Key strings
    if( tail >= m_K.Count )
        m_K.Realloc( m_K.Count+2048, true );

    // append the new Key string
    strcpy_s( m_K.p+m_K.n, m_K.cb-m_K.n, _k );

    // insert a new key-value pair at the correct position in the list; no insert needed if the insertion point is at the end of the list
    if( insertAt < this->Count )
        memmove( m_KVP.p+insertAt+1, m_KVP.p+insertAt, sizeof(KVP)*(this->Count-insertAt) );

    // update the offsets
    m_KVP.p[insertAt].ofsK = m_K.n;
    m_KVP.p[insertAt].ofsV = this->Count;

    // increment the count
    this->Count++ ;

    // save the new buffer size
    m_K.n = static_cast<UINT32>(tail);

    // return the new Value
    return Value( insertAt );
}

/// <summary>
/// Returns the index of the specified key; if the key is not found, returns the index of the insertion point
///  with bit 31 set.
/// </summary>
template <typename T> INT32 AA<T>::IndexOf( const char* _k )
{
    // look for the specified Key (no hashtable or binary search here -- it's pure brute force and suitable only for small arrays)
    INT32 i = 0;
    for( ; i<static_cast<INT32>(this->Count); ++i )
    {
        INT32 rval = strcmp( _k, this->Key(i) );

        // if the specified Key is in the list, return its index
        if( rval == 0 )
            return i;

        /* stop iterating if the specified Key is not in the list; the current value of i is the index of the
            the next larger Key in the list (i.e. the index at which the specified Key would be inserted) */
        if( rval < 0 )
            break;
    }

    // at this point the specified Key is not in the list, so return the index of the end of the list
    return (i | 0x80000000);
}

/// <summary>
/// Returns the key at the specified index
/// </summary>
template <typename T> const char* AA<T>::Key( const size_t i )
{
    return m_K.p + m_KVP.p[i].ofsK;
}

/// <summary>
/// Returns the value at the specified index
/// </summary>
template <typename T> T& AA<T>::Value( const size_t i )
{
    return m_V.p[m_KVP.p[i].ofsV];
}

/// <summary>
/// Resets the internal buffer contents
/// </summary>
template <typename T> void AA<T>::Reset( void )
{
    memset( m_K.p, 0, m_K.cb );
    memset( m_V.p, 0, m_V.cb );
    memset( m_KVP.p, 0, m_KVP.cb );
    m_K.n = 0;
}