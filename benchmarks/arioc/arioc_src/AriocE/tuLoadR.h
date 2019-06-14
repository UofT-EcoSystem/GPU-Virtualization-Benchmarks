/*
  tuLoadR.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuLoadR__

/// <summary>
/// Class <c>tuLoadR</c> loads R (reference) sequence data.
/// </summary>
class tuLoadR : public tuBaseA
{
    typedef bool (tuLoadR::*mfnComputeKmerHash)( UINT64& hashKey, INT64*& p64, INT32& shr, INT32& shl );

    private:
        static const INT32 INPUTBUFFERSIZE = (10 * 1024*1024);  // 10Mb

        AriocEncoderParams*             m_psip;
        SqCategory                      m_sqCategory;   // Q, R+, R-
        INT16                           m_iInputFile;   // index of input file specification
        volatile UINT32*                m_WaitBits;     // bit flags used for spin waits
        INT64                           m_cbUnread;
        RaiiFile                        m_inFileArioc;
        WinGlobalPtr<INT64>             m_inBufArioc;
        INT32                           m_maxEncodeHash;
        INT32                           m_subId;
        bool                            m_isRminus;
        mfnComputeKmerHash              m_computeKmerHash;
        INT16                           m_cbHash;       // number of bytes required for a hash value of size SeedBaseParameters::hashBitsWidth
        RaiiSemaphore*                  m_psemComplete;
        HiResTimer                      m_hrt;

    private:
        bool computeKmerHash21( UINT64& hashKey, INT64*& p64, INT32& shr, INT32& shl );
        bool computeKmerHash42( UINT64& hashKey, INT64*& p64, INT32& shr, INT32& shl );
        bool computeKmerHash84( UINT64& hashKey, INT64*& p64, INT32& shr, INT32& shl );

    protected:
        tuLoadR( void );
        void main( void );
        void buildJR( void );
        void buildJQ( void );

    public:
        tuLoadR( AriocEncoderParams* psip, SqCategory sqCat, INT16 iInputFile, volatile UINT32* waitBits, RaiiSemaphore* psem );
        virtual ~tuLoadR( void );
};