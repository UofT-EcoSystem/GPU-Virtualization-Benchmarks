/*
  tuSortJcpu.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuSortJcpu__

/// <summary>
/// Class <c>tuSortJcpu</c> sorts J lists.
/// </summary>
class tuSortJcpu : public tuBaseA
{
    protected:
        AriocEncoderParams* m_psip;
        INT64               m_celH;
        volatile INT64*     m_pnSortedH;

    private:
        static int JvalueComparer( const void*, const void* );

    protected:
        tuSortJcpu( void );
        void main( void );

    public:
        tuSortJcpu( AriocEncoderParams* psip, INT64 celH, volatile INT64* pnSortedJ );
        virtual ~tuSortJcpu( void );
};