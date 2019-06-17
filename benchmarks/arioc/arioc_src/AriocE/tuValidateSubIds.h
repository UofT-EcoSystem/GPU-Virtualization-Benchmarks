/*
  tuValidateSubIds.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuValidateSubIds__

/// <summary>
/// Class <c>tuValidateSubIds</c> validates the J table structure and counts subunit IDs.
/// </summary>
class tuValidateSubIds : public tuBaseA
{
    private:
        INT64                   m_iC0;          // first C-table offset value for the current worker thread
        INT64                   m_iClimit;      // C-table offset limit for the current worker thread
        WinGlobalPtr<Cvalue>*   m_pC;           // C values
        WinGlobalPtr<Hvalue8>*  m_pH;           // H values
        WinGlobalPtr<Jvalue8>*  m_pJ;           // J values
        size_t                  m_cbJ;          // size of J table in bytes

    public:
        INT8                    MinSubId;       // minimum subId
        INT8                    MaxSubId;       // maximum subId
        WinGlobalPtr<UINT32>    SubIdsPerH;     // subId counts for each hash value
        WinGlobalPtr<UINT64>    TotalJ;         // total J values

    protected:
        tuValidateSubIds( void );
        void main( void );

    public:
        tuValidateSubIds( INT64 _iC0, INT64 _iClimit, WinGlobalPtr<Cvalue>* _pC, WinGlobalPtr<Hvalue8>* _pH, WinGlobalPtr<Jvalue8>* _pJ, size_t _cbJ );
        virtual ~tuValidateSubIds( void );
};