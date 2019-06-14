/*
  tuLoadLUT.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuLoadLUT__


/// <summary>
/// Class <c>tuLoadLUT</c> loads a specified file into page-locked ("pinned") memory so that it can be mapped into CUDA address space
///
template <class T>
class tuLoadLUT : public tuBaseA
{
    private:
        char                m_fileSpec[FILENAME_MAX];
        WinGlobalPtr<T>*    m_pwgp;
        CudaPinnedPtr<T>*   m_pcpp;
        UINT32              m_cbPad;

    private:
        /// [private] constructor
        tuLoadLUT( void );

    public:
        /// constructor (CudaPinnedPtr&lt;T&gt;*, char*, UINT32)
        tuLoadLUT( CudaPinnedPtr<T>* _pcpp, char* fileSpec, UINT32 cbPad ) : m_pwgp(NULL), m_pcpp(_pcpp), m_cbPad(cbPad)
        {
            if( (fileSpec == NULL) || (*fileSpec == 0) )
                throw new ApplicationException( __FILE__, __LINE__, "missing file specification" );

            // copy the specified file specification to a member variable
            strcpy_s( m_fileSpec, sizeof m_fileSpec, fileSpec );
        }

        /// constructor (WinGlobalPtr&lt;T&gt;*, char*, UINT32)
        tuLoadLUT( WinGlobalPtr<T>* _pwgp, char* fileSpec, UINT32 cbPad ) : m_pwgp(_pwgp), m_pcpp(NULL), m_cbPad(cbPad)
        {
            if( (fileSpec == NULL) || (*fileSpec == 0) )
                throw new ApplicationException( __FILE__, __LINE__, "missing file specification" );

            // copy the specified file specification to a member variable
            strcpy_s( m_fileSpec, sizeof m_fileSpec, fileSpec );
        }

        /// destructor
        virtual ~tuLoadLUT( void )
        {
        }

        /// <summary>
        /// Loads a specified lookup-table file into either page-locked ("pinned") memory or host memory
        /// </summary>
        virtual void main( void )
        {
            HiResTimer hrt(ms);

            // get the file size
            RaiiFile rf( m_fileSpec, true );
            INT64 cb = rf.FileSize();

            CDPrint( cdpCD4, "%s: fileSpec=%s (%lld bytes) ...", __FUNCTION__, m_fileSpec, cb );

            // compute the number of T-sized elements in the array
            INT64 cel = blockdiv(cb, sizeof(T));

            // append the specified amount of padding
            cel += blockdiv(m_cbPad, sizeof(T));

            T* p = NULL;
            size_t cbBuf = 0;
            if( m_pcpp )
            {
                // allocate a page-locked ("pinned") buffer whose address maps into GPU global address space
                m_pcpp->Realloc( cel, true );

                // set a flag to indicate that the buffer will need to be unpinned at some point
                m_pcpp->n = 1;

                p = m_pcpp->p;
                cbBuf = m_pcpp->cb;
            }
            else
            {
                // allocate a buffer in host memory
                m_pwgp->Realloc( cel, true );

                p = m_pwgp->p;
                cbBuf = m_pwgp->cb;
            }

            CDPrint( cdpCD4, "%s: allocated %s buffer: %lld bytes (%3.1fGB) at 0x%016llx for filespec=%s in %dms",
                             __FUNCTION__, (m_pcpp ? "pinned" : "staging"),
                             cbBuf, cbBuf/GIGADOUBLE, p, m_fileSpec, hrt.GetElapsed(false) );

            // read the data
            rf.Read( p, cb );

            INT32 msElapsed = hrt.GetElapsed( false );
            CDPrint( cdpCD2, "%s: loaded %lld bytes (%3.1fGB) in %dms from %s",
                                __FUNCTION__, cbBuf, cbBuf/GIGADOUBLE, msElapsed, (*m_fileSpec ? m_fileSpec : "buffer") );
        }
};
