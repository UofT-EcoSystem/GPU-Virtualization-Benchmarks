/* 
  MT19937.h

   Copyright (C) 1997-2002, Makoto Matsumoto and Takuji Nishimura.  See notes in MT19937.cpp.
*/

#pragma once
#define __MT19937__


/// <summary>
/// Class <c>MT19937</c> implements a 32-bit random number generator.
/// </summary>
class MT19937
{
    private:
        static const int N = 624;
        static const int M = 397;
        static const UINT32 MATRIX_A = 0x9908b0dfUL;    /* constant vector a */
        static const UINT32 UPPER_MASK = 0x80000000UL;  /* most significant w-r bits */
        static const UINT32 LOWER_MASK = 0x7fffffffUL;  /* least significant r bits */

        // the following were static in the original C implementation; they are instance members here for thread safety
        UINT32 mt[N];                                   /* the array for the state vector  */
        INT32 mti;                                      /* mti==N+1 means mt[N] is not initialized */

    private:
        MT19937( void );
        void init_genrand( UINT32 s );
        void init_by_array( UINT32 init_key[], INT32 key_length );

    public:
        MT19937( UINT32 _seed );
        MT19937( UINT32 init_key[], INT32 key_length );
        virtual ~MT19937( void );
        UINT32 genrand_int32( void );
        INT32 genrand_int31();
        double genrand_real1(void);
        double genrand_real2(void);
        double genrand_real3(void);
        double genrand_res53();
};
