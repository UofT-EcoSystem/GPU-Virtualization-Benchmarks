/*
  tuEncodeFASTQ.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __tuEncodeFASTQ__

/// <summary>
/// Class <c>tuEncodeFASTQ</c> encodes FASTQ-formatted sequence data.
/// </summary>
class tuEncodeFASTQ : public baseEncode
{
    private:
        WinGlobalPtr<UINT32>    m_nN;       // number of reads per read length

    private:
        void summarizeReadLengthDistribution( void );

    protected:
        tuEncodeFASTQ( void );
        void main( void );

    public:
        tuEncodeFASTQ( AriocEncoderParams* psip, SqCategory sqCat, INT16 iInputFile, RaiiSemaphore* psem, SAMConfigWriter* pscw );
        virtual ~tuEncodeFASTQ( void );
};