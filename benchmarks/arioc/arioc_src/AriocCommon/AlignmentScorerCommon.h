/*
  AlignmentScorerCommon.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AlignmentScorerCommon__

#define MAX_N   (16*1024)   // number of symbols in the largest Q sequence supported by this implementation

#pragma region enums
enum Wmxgs      // predefined alignment scores
{   
    // NCBI BLAST predefines           Wm Wx Wg Ws  (see http://blast.ncbi.nlm.nih.gov/Blast.cgi?PROGRAM=blastn&BLAST_PROGRAMS=megaBlast&PAGE_TYPE=BlastSearch (February 2012))
    ncbiW_1_1_5_2 =  0x01010502,    //  1 -1  5  2
    ncbiW_1_1_3_2 =  0x01010302,    //  1 -1  3  2
    ncbiW_1_1_2_2 =  0x01010202,    //  1 -1  2  2
    ncbiW_1_1_1_2 =  0x01010102,    //  1 -1  1  2
    ncbiW_1_1_0_2 =  0x01010002,    //  1 -1  0  2
    ncbiW_1_1_4_1 =  0x01010401,    //  1 -1  4  1
    ncbiW_1_1_3_1 =  0x01010301,    //  1 -1  3  1
    ncbiW_1_1_2_1 =  0x01010201,    //  1 -1  2  1

    ncbiW_1_2_5_2 =  0x01020502,    //  1 -2  5  2
    ncbiW_1_2_2_2 =  0x01020202,    //  1 -2  2  2
    ncbiW_1_2_1_2 =  0x01020102,    //  1 -2  1  2
    ncbiW_1_2_0_2 =  0x01020002,    //  1 -2  0  2
    ncbiW_1_2_3_1 =  0x01020301,    //  1 -2  3  1
    ncbiW_1_2_2_1 =  0x01020201,    //  1 -2  2  1
    ncbiW_1_2_1_1 =  0x01020101,    //  1 -2  1  1

    ncbiW_1_3_5_2 =  0x01030502,    //  1 -3  5  2
    ncbiW_1_3_2_2 =  0x01030202,    //  1 -3  2  2
    ncbiW_1_3_1_2 =  0x01030102,    //  1 -3  1  2
    ncbiW_1_3_0_2 =  0x01030002,    //  1 -3  0  2
    ncbiW_1_3_2_1 =  0x01030201,    //  1 -3  2  1
    ncbiW_1_3_1_1 =  0x01030101,    //  1 -3  1  1

    ncbiW_1_4_5_2 =  0x01040502,    //  1 -4  5  2
    ncbiW_1_4_1_2 =  0x01040102,    //  1 -4  1  2
    ncbiW_1_4_0_2 =  0x01040002,    //  1 -4  0  2
    ncbiW_1_4_2_1 =  0x01040201,    //  1 -4  2  1
    ncbiW_1_4_1_1 =  0x01040101,    //  1 -4  1  1

    ncbiW_2_3_4_4 =  0x02030404,    //  2 -3  4  4
    ncbiW_2_3_2_4 =  0x02030204,    //  2 -3  2  4
    ncbiW_2_3_0_4 =  0x02030004,    //  2 -3  0  4
    ncbiW_2_3_3_3 =  0x02030303,    //  2 -3  3  3
    ncbiW_2_3_6_2 =  0x02030602,    //  2 -3  6  2
    ncbiW_2_3_5_2 =  0x02030502,    //  2 -3  5  2
    ncbiW_2_3_4_2 =  0x02030402,    //  2 -3  4  2
    ncbiW_2_3_2_2 =  0x02030202,    //  2 -3  2  2

    ncbiW_4_5_12_8 = 0x04050C08,    //  4 -5 12  8
    ncbiW_4_5_6_5 =  0x04050605,    //  4 -5  6  5
    ncbiW_4_5_5_5 =  0x04050505,    //  4 -5  5  5
    ncbiW_4_5_4_5 =  0x04050405,    //  4 -5  4  5
    ncbiW_4_5_3_5 =  0x04050305,    //  4 -5  3  5

    // Bowtie 2 defaults
    bt2dW_2_6_5_3 =  0x02060503,    //  2 -6  5  3

    // test                            Wm Wx Wg Ws
    testW_5_4_11_1 = 0x05040B01,    //  5 -4 11  1    (ftp:   //ftp.ncbi.nih.gov/blast/matrices/NUC.4.4)

    // unknown
    Wmxgs_unknown = 0xFFFFFFFF
};

enum ScoreFunctionType  // (see Bowtie2 class SimpleFunc)
{                       // given a constant term B, a coefficient A, and a ScoreFunctionType F, the function is f(x) = A*F(x) + B
    sftC = 1,           // constant   : B
    sftL = 2,           // linear     : A*x + B
    sftS = 3,           // square root: A*sqrt(x) + B
    sftG = 4,           // natural log: A*ln(x) + B
    sftUnknown = 99
};
#pragma endregion


/// <summary>
/// Class <c>AlignmentScorerCommon</c> wraps common scoring and control functionality for AriocU, AriocP, and ASH
/// </summary>
class AlignmentScorerCommon
{
    private:
        static void trimTrailingZeros( char* s );

    public:
        AlignmentScorerCommon( void );
        ~AlignmentScorerCommon( void );
        static Wmxgs StringToWmxgs( const char* s );
        static ScoreFunctionType StringToScoreFunctionType( const char* s );
        static void StringToScoreFunction( const char* p, ScoreFunctionType& sft, double& sfCoef, double& sfConst );
        static char* ScoreFunctionToString( ScoreFunctionType _sft, double _sfA, double _sfB );
};
