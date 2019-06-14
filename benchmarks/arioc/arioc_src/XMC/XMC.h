/*
  XMC.h

    Copyright (c) 2018-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
     in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
     The contents of this file, in whole or in part, may only be copied, modified, propagated, or
     redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __XMC__

class XMC
{
    private:
        AppMain*            m_pam;
        WinGlobalPtr<char>  m_buf;      // buffer that contains SAM data

    public:
        INT64                               SamRowOffset;
        WinGlobalPtr<SamFilePartitionInfo>  SFPI;

    private:
        void sniffSamFile( void );
        void partitionSamFile( void );

    public:
        XMC( AppMain* _pam );
        virtual ~XMC( void );
        static char* FindSOL( char* p, char* pLimit );
        static char* FindEOL( char* p, char* pLimit );
        void Main( void );
};
