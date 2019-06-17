/*
  RGManager.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __RGManager__

class RGManager
{
    public:
        WinGlobalPtr<char>      RG;                         // SAM RG info (one XML element per read group)
        WinGlobalPtr<UINT32>    OfsRG;                      // offsets of SAM read group info strings (XML elements)
        char                    SBFfileSpec[FILENAME_MAX];  // output file specification
        INT32                   RGOrdinal;                  // (see SAMConfigWriter)

    private:
        bool isRGforOrdinal( tinyxml2::XMLElement* pelRG, UINT32 expectedOrdinal );
        tinyxml2::XMLElement* getRGforOrdinal( tinyxml2::XMLElement* pelRG, UINT32 nextOrdinal );

    public:
        RGManager( void );
        virtual ~RGManager( void );
        void LoadReadGroupInfo( InputFileGroup* pifg );
        void SaveRGIDs( void );
        void WriteReadGroupInfo( WinGlobalPtr<OutputFileInfo>& ofi );
};
