/*
  RaiiPtr.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RaiiPtr__

/// <summary>
/// Class <c>RaiiPtr</c> provides a simple "resource acquisition is initialization" wrapper for a pointer.
/// </summary>
template<class T> class RaiiPtr
{
    private:
        T*  m_pT;

    public:
        /// default constructor
        RaiiPtr() : m_pT(NULL)
        {
        }

        /// constructor (T)
        RaiiPtr( T* _pT ) : m_pT(_pT)
        {
            /* Usage:

                RaiiPtr<T> pT( new T() );
                *pT = <some_value>;
            */
        }

        /// destructor
        ~RaiiPtr()
        {
            if( m_pT )
            {
                delete m_pT;
                m_pT = NULL;
            }
        }

        /// operator =
        void operator=( const T* _pT )
        {
            if( _pT == NULL )
                throw new ApplicationException( __FILE__, __LINE__, "Null pointer value assigned to RaiiPtr" );

            m_pT = _pT;
        }

        /// operator *
        T& operator*()
        {
            return *m_pT;
        }

        /// operator*
        T* operator->()
        {
            return m_pT;
        }
};
