/*
  tuUnloadLUT.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuUnloadLUT__

template <class T>
class tuUnloadLUT : public tuBaseS
{
    private:
        CudaPinnedPtr<T>*   m_pbuf;

    private:
        /// [private] constructor
        tuUnloadLUT( void );

    public:
        /// constructor
        tuUnloadLUT( CudaPinnedPtr<T>* pbuf ) : m_pbuf(pbuf)
        {
        }

        /// destructor
        virtual ~tuUnloadLUT( void )
        {
        }

        /// <summary>
        /// Frees a page-locked ("pinned") buffer
        /// </summary>
        virtual void main( void )
        {
            CRVALIDATOR;

            try
            {
                CREXEC( m_pbuf->Free() );
            }
            catch( ApplicationException* pex )
            {
                CRTHROW;
            }
        }
};
