/*
  AlignmentScorerCommon.cpp

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#include "stdafx.h"

#pragma region constructors and destructor
/// default constructor
AlignmentScorerCommon::AlignmentScorerCommon()
{
}

/// [public] destructor
AlignmentScorerCommon::~AlignmentScorerCommon()
{
}
#pragma endregion

#pragma region static methods
/// [private static] method trimTrailingZeros
void AlignmentScorerCommon::trimTrailingZeros( char* p )
{
    p += strlen(p) - 1;     // point to the end of the specified string

    while( *p == '0' )      // strip trailing zeros
    {
        *p = 0;
        --p;
    }

    if( *p == '.' )         // strip trailing decimal point
        *p = 0;
}

/// [static] method StringToWmxgs
Wmxgs AlignmentScorerCommon::StringToWmxgs( const char* s )
{
    INT32 Wm = 0;
    INT32 Wx = 0;
    INT32 Wg = 0;
    INT32 Ws = 0;

    /* If the string starts with a digit, assume it is formatted as "m_x_g_s".
       Otherwise, assume it's one of the values defined in the Wmxgs enumeration.
    */
    const char* fmt = (isdigit(*s)) ? "%d %*c %d %*c %d %*c %d" :       // m_x_g_s
                                    "%*6s %d %*c %d %*c %d %*c %d";     // wwwwww_m_x_g_s
    sscanf_s( s, fmt, &Wm, &Wx, &Wg, &Ws );

    // convert negative penalty values to positive integers
    Wx = abs( Wx );
    Wg = abs( Wg );
    Ws = abs( Ws );

    if( (Wm == 0) || (Wx == 0) )
        return Wmxgs_unknown;

    return static_cast<Wmxgs>((Wm<<24)|(Wx<<16)|(Wg<<8)|Ws);
}

/// [static] method StringToScoreFunctionType
ScoreFunctionType AlignmentScorerCommon::StringToScoreFunctionType( const char* s )
{
    if( strlen(s) == 4 )
    {
        switch( *reinterpret_cast<const INT32*>(s) )    // use type punning to do a "string comparison" on the four characters in s
        {
            case 0x43746673:        
                return sftC;

            case 0x4C746673:
                return sftL;

            case 0x53746673:
                return sftS;

            case 0x47746673:
                return sftG;
        }
    }

    throw new ApplicationException( __FILE__, __LINE__, "unrecognized ScoreFunctionType '%s'", s );
}

/// [static] method StringToScoreFunction
void AlignmentScorerCommon::StringToScoreFunction( const char* p, ScoreFunctionType& sft, double& sfCoef, double& sfConst )
{
    /* The specified string may be in either of two formats:
        - a non-negative integer value      (example: "491")
        - [CLSG],constant,coefficient       (example: "G,20,8")
    */

    // if a comma is present, look for the function type, coefficient, and constant
    const char* pComma = strchr( p, ',' );
    if( pComma )
    {
        char c = 0;
#ifdef _WIN32
        if( 3 != sscanf_s( p, "%c,%lf,%lf", &c, static_cast<UINT32>(sizeof c), &sfConst, &sfCoef ) )
#endif
#ifdef __GNUC__
        if( 3 != sscanf( p, "%c,%lf,%lf", &c, &sfConst, &sfCoef ) )
#endif
            throw new ApplicationException( __FILE__, __LINE__, "unable to parse V-score function string: %s", p );

        switch( toupper(c) )
        {
            case 'C':
                sft = sftC;
                sfCoef = 0;
                break;

            case 'L':
                sft = sftL;
                break;

            case 'S':
                sft = sftS;
                break;

            case 'G':
                sft = sftG;
                break;

            default:
                throw new ApplicationException( __FILE__, __LINE__, "unrecognized V-score function type: %c", c );
        }
    }

    else    // no comma, so assume it's a constant
    {
        sft = sftC;
        sfCoef = 0;
        sfConst = static_cast<double>(atof( p ));
    }
}


#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable:4996 )     // (don't nag us about sprintf being "unsafe")
#endif

/// [static] method ScoreFunctionToString
char* AlignmentScorerCommon::ScoreFunctionToString( ScoreFunctionType _sft, double _sfA, double _sfB )
{
    // use a static buffer (i.e. this method is NOT thread-safe)
    static const size_t cbBuf = 128;
    static char buf[cbBuf];

    /* build the score function as a string (e.g., "10*N + 6") */

    if( _sft == sftC )                       // sftC: constant
        sprintf( buf, "%6.4f", fabs(_sfB) );

    else
    {
        // coefficient
        sprintf( buf, "%6.4f", _sfA );  // write the coefficient with 4 decimal places
        trimTrailingZeros( buf );

        // get the sign of the constant term
        char cSign = (_sfB >= 0.0) ? '+' : '-';

        // format the remainder of the string
        switch( _sft )
        {
            case sftG:              // sftG: ln
                sprintf( buf, "%s*ln(N) %c %6.4f", buf, cSign, fabs(_sfB) );
                break;

            case sftL:              // sftL: linear
                sprintf( buf, "%s*N %c %6.4f", buf, cSign, fabs(_sfB) );
                break;

            case sftS:              // sftS: sqrt
                sprintf( buf, "%s*sqrt(N) %c %6.4f", buf, cSign, fabs(_sfB) );
                break;

            default:                // sftC: constant
                throw new ApplicationException( __FILE__, __LINE__, "unexpected ScoreFunctionType value: %d", _sft );
        }
    }

    trimTrailingZeros( buf );

    return buf;
}

#ifdef _WIN32
#pragma warning (pop)
#endif
