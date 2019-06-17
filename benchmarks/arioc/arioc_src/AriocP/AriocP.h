/*
  AriocP.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocP__

class AriocP : public AriocBase
{
    public:
        AriocP( const char* _pathR, A21SpacedSeed _a21ss, A21HashedSeed _a21hs, AlignmentControlParameters _acp, AlignmentScoreParameters _asp,
                InputFileGroup* _pifgQ,
                WinGlobalPtr<OutputFileInfo>& _ofi,
                UINT32 _gpuMask, INT32 _maxDOP, UINT32 _batchSize, INT32 _kmerSize, CIGARfmtType _cft, MDfmtType _mft,
                AppMain* _pam );
        virtual ~AriocP( void );
        void Main( void );
};
