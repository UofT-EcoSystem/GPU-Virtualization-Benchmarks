/*
  TSX.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __TSX__

#include <thrust/tuple.h>
#include <limits>

class TSX
{
    public:

        /* unary predicate
        */
        struct isMappedDvalue
        {
            __host__ __device__
            bool operator()( const UINT64& Dvalue )
            {
                return ((Dvalue & AriocDS::D::flagMapped) != 0);
            }
        };

        /* unary predicate
        */
        struct isUnmappedDvalue
        {
            __host__ __device__
            bool operator()( const UINT64& Dvalue )
            {
                return ((Dvalue & AriocDS::D::flagMapped) == 0);
            }
        };

        /* unary predicate
        */
        struct isCandidateDvalue
        {
            __host__ __device__
            bool operator()( const UINT64& Dvalue )
            {
                return ((Dvalue & AriocDS::D::flagCandidate) != 0);
            }
        };

        /* unary predicate
        */
        struct isNotCandidateDvalue
        {
            __host__ __device__
            bool operator()( const UINT64& Dvalue )
            {
                return ((Dvalue & AriocDS::D::flagCandidate) == 0);
            }
        };

        /* unary predicate
        */
        struct isMappedCandidateDvalue
        {
            __host__ __device__
            bool operator()( const UINT64& Dvalue )
            {
                return ((Dvalue & (AriocDS::D::flagCandidate|AriocDS::D::flagMapped)) == (AriocDS::D::flagCandidate|AriocDS::D::flagMapped));
            }
        };

        /* unary predicate
        */
        struct isNotMappedCandidateDvalue
        {
            __host__ __device__
            bool operator()( const UINT64& Dvalue )
            {
                return ((Dvalue & (AriocDS::D::flagCandidate|AriocDS::D::flagMapped)) != (AriocDS::D::flagCandidate|AriocDS::D::flagMapped));
            }
        };

        /* unary predicate
        */
        struct isUnmappedCandidateDvalue
        {
            __host__ __device__
            bool operator()( const UINT64& Dvalue )
            {
                return ((Dvalue & (AriocDS::D::flagCandidate|AriocDS::D::flagMapped)) == AriocDS::D::flagCandidate);
            }
        };

        /* unary predicate
        */
        struct isFlagX
        {
            __host__ __device__
            bool operator()( const UINT64& Dvalue )
            {
                return ((Dvalue & AriocDS::D::flagX) != 0);
            }
        };

        /* unary predicate
        */
        template <typename T>
        struct isZero
        {
            isZero<T>()
            {
            }

            __host__ __device__
            bool operator()( const T& x )
            {
                return (x == 0);
            }
        };

        /* unary predicate
        */
        template <typename T>
        struct isNonzero
        {
            isNonzero<T>()
            {
            }

            __host__ __device__
            bool operator()( const T& x )
            {
                return (x != 0);
            }
        };
        
        /* unary predicate
        */
        template <typename T>
        struct testMask
        {
            const T m_mask;

            testMask( const T _mask ) : m_mask(_mask)
            {
            }

            __host__ __device__
            bool operator()( const T& x )
            {
                return ((x & m_mask) != 0);
            }
        };

        /* unary predicate
        */
        template <typename T>
        struct testMaskZero
        {
            const T m_mask;

            testMaskZero( const T _mask ) : m_mask(_mask)
            {
            }

            __host__ __device__
            bool operator()( const T& x )
            {
                return ((x & m_mask) == 0);
            }
        };

        /* unary predicate
        */
        template <typename T>
        struct isEven
        {
            isEven<T>()
            {
            }

            __host__ __device__
            bool operator()( const T& x )
            {
                return ((x & 1) == 0);
            }
        };

        /* unary predicate (compare to constant)
        */
        template<typename T>
        struct isEqualTo
        {
            private:
                const T m_compareTo;

            public:
                isEqualTo( const T compareTo ) : m_compareTo(compareTo)
                {
                }

            public:
                __host__ __device__
                bool operator()( const T& x ) const
                {
                    return (x == m_compareTo);
                }
        };

        /* unary predicate (compare to constant)
        */
        template<typename T>
        struct isNotEqualTo
        {
            private:
                const T m_compareTo;

            public:
                isNotEqualTo( const T compareTo ) : m_compareTo(compareTo)
                {
                }

            public:
                __host__ __device__
                bool operator()( const T& x ) const
                {
                    return (x != m_compareTo);
                }
        };

        /* unary predicate (tuple)
        */
        struct isUnmappedDV
        {
            typedef thrust::tuple<UINT64,INT16> DVtuple;    // D value, Vmax

            __host__ __device__
            bool operator()( const DVtuple& dv ) const
            {
                // test the "mapped" flag in the D value
                return ((thrust::get<0>(dv) & AriocDS::D::flagMapped) == 0);
            }
        };

        /* unary predicate (tuple)
        */
        struct isCandidateDC
        {
            typedef thrust::tuple<UINT64,UINT32> DCtuple;    // D value, coverage

            __host__ __device__
            bool operator()( const DCtuple& dv ) const
            {
                // test the "candidate" flag in the D value
                return ((thrust::get<0>(dv) & AriocDS::D::flagCandidate) != 0);
            }
        };

        /* unary predicate (tuple)
        */
        struct isUnmappedODV
        {
            typedef thrust::tuple<UINT32,UINT64,INT16> ODVtuple;    // offset, D value, Vmax

            __host__ __device__
            bool operator()( const ODVtuple& odv ) const
            {
                // test the "mapped" flag in the D value
                return ((thrust::get<1>(odv) & AriocDS::D::flagMapped) == 0);
            }
        };

        /* binary predicate
        */
        struct isEqualD
        {
            __host__ __device__
            bool operator()( const UINT64& d1, const UINT64& d2 ) const
            {
                /* The D values are bitmapped like this...
                        UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
                        UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
                        UINT64  subId :  7;     // bits 32..38: subId
                        UINT64  qid   : 21;     // bits 39..59: QID
                        UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
                        UINT64  flags :  3;     // bits 61..63: flags
                   ... so we can do the comparison by XORing the specified values and zeroing the bits that contain the flags.
                */
                return (((d1 ^ d2) & (~AriocDS::D::maskFlags)) == 0);
            }
        };

        /* binary predicate
        */
        struct isLessD
        {
            __host__ __device__
            bool operator()( const UINT64& d1, const UINT64& d2 ) const
            {
                /* D values are bitmapped like this...
                        UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
                        UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
                        UINT64  subId :  7;     // bits 32..38: subId
                        UINT64  qid   : 21;     // bits 39..59: QID
                        UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
                        UINT64  flags :  3;     // bits 61..63: flags
                        
                   Df values are bitmapped like this...
                        UINT64  qlo   :  1;     // bits 00..00: low-order bit of QID (0: mate 1; 1: mate 2)
                        UINT64  s     :  1;     // bits 01..01: strand (0: forward; 1: reverse complement)
                        UINT64  d     : 31;     // bits 02..32: 0-based position relative to the forward strand
                        UINT64  subId :  7;     // bits 33..39: subId
                        UINT64  qhi   : 20;     // bits 40..59: high-order bits of QID
                        UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
                        UINT64  flags :  3;     // bits 61..63: flags

                   ... so we can do the comparison for either one by masking off the flag bits and comparing the rest.
                */
                return ((d1 & ~AriocDS::D::maskFlags) < (d2 & ~AriocDS::D::maskFlags));
            }
        };

        /* binary predicate
        */
        struct isLessDf
        {
            __host__ __device__
            bool operator()( const UINT64& df1, const UINT64& df2 ) const
            {
                /* Df values are bitmapped like this...
                        UINT64  qlo   :  1;     // bits 00..00: low-order bit of QID (0: mate 1; 1: mate 2)
                        UINT64  s     :  1;     // bits 01..01: strand (0: forward; 1: reverse complement)
                        UINT64  d     : 31;     // bits 02..32: 0-based position relative to the forward strand
                        UINT64  subId :  7;     // bits 33..39: subId
                        UINT64  qhi   : 20;     // bits 40..59: high-order bits of QID
                        UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
                        UINT64  flags :  3;     // bits 61..63: flags

                   We can do the comparison by masking off bits 60..63 and comparing the rest.
                */
                return ((df1 & ~(AriocDS::Df::maskFlags|AriocDS::Df::flagRCq)) < (df2 & ~(AriocDS::Df::maskFlags|AriocDS::Df::flagRCq)));
            }
        };

        /* binary predicate
        */
        struct isLessDvalueQID
        {
            __host__ __device__
            bool operator()( const UINT64& d1, const UINT64& d2 ) const
            {
                /* The D values are bitmapped like this...
                        UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
                        UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
                        UINT64  subId :  7;     // bits 32..38: subId
                        UINT64  qid   : 21;     // bits 39..59: QID
                        UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
                        UINT64  flags :  3;     // bits 61..63: flags
                   ... so we can do the comparison by masking the D values and comparing the masked bits.
                */
                return ((d1 & AriocDS::D::maskQID) < (d2 & AriocDS::D::maskQID));
            }
        };

        /* binary predicate
        */
        struct isEqualDvalueQID
        {
            __host__ __device__
            bool operator()( const UINT64& d1, const UINT64& d2 ) const
            {
                /* The D values are bitmapped like this...
                        UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
                        UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
                        UINT64  subId :  7;     // bits 32..38: subId
                        UINT64  qid   : 21;     // bits 39..59: QID
                        UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
                        UINT64  flags :  3;     // bits 61..63: flags
                   ... so we can do the comparison by XORing the Df values and masking the bits that contain the QID.
                */
                return (((d1 ^ d2) & AriocDS::D::maskQID) == 0);
            }
        };

        /* unary function
        */
        struct xformDvalueToQID
        {
            __host__ __device__
            UINT32 operator()( const UINT64& d ) const
            {
                /* The D values are bitmapped like this:
                        UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
                        UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
                        UINT64  subId :  7;     // bits 32..38: subId
                        UINT64  qid   : 21;     // bits 39..59: QID
                        UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
                        UINT64  flags :  3;     // bits 61..63: flags
                */
                return (static_cast<UINT32>(d >> 39) & AriocDS::QID::maskQID);
            }
        };

        /* unary function
        */
        struct initializeDflags
        {
            private:
                const UINT64  m_newFlags;

            public:
                initializeDflags( const UINT64 newFlags ) : m_newFlags(newFlags)
                {
                }

                __host__ __device__
                void operator()( UINT64& d ) const
                {
                    /* The D values are bitmapped like this:
                        UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
                        UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
                        UINT64  subId :  7;     // bits 32..38: subId
                        UINT64  qid   : 21;     // bits 39..59: QID
                        UINT64  rc    :  1;     // bits 60..60: seed from reverse complement
                        UINT64  flags :  3;     // bits 61..63: flags
                    */
                    d &= (~AriocDS::D::maskFlags);
                    d |= m_newFlags;
                }
        };

        /* unary function
        */
        struct subUInt64
        {
            private:
                const UINT64 m_decr;

            public:
                subUInt64( const UINT64 decr ) : m_decr(decr)
                {
                }

                __host__ __device__
                void operator()( UINT64& u64 ) const
                {
                    u64 -= m_decr;
                }
        };

        /* unary function
        */
        struct andHiLoUInt64
        {
            __host__ __device__
            UINT64 operator()( const UINT64 u64 ) const
            {
                /* Do a boolean AND on the high-order and low-order 32-bit values.
                */
                const UINT64 andHiLo = (u64 & 0xFFFFFFFF) & (u64 >> 32);
                return (andHiLo | (andHiLo << 32));
            }
        };

        /* unary function
        */
        struct setHigh16
        {
            private:
                const UINT32 m_newHigh;

            public:
                setHigh16( const UINT32 val ) : m_newHigh(val << 16)
                {
                }

            __host__ __device__
            UINT32 operator()( const UINT32 u32 ) const
            {
                /* Replace bits 16-31 with the value specified in the constructor.
                */
                return (u32 & 0x0000FFFF) | m_newHigh;
            }
        };

        /* binary function
        */
        struct maxLow16
        {
            public:
                __host__ __device__
                UINT32 operator()( const UINT32& C1, const UINT32& C2 ) const
                {
                    /* Treat each specified value as a pair of 16-bit bitmaps:
                        - compute the unsigned maximum of the 16 low-order bits
                        - compute the sum of the 16 high-order bits
                    */
                    UINT16 lo1 = static_cast<UINT16>(C1);
                    UINT16 lo2 = static_cast<UINT16>(C2);
                    UINT32 hi1 = C1 >> 16;
                    UINT32 hi2 = C2 >> 16;
                    UINT32 newHigh = min2( 0x0000FFFF, hi1+hi2 );    // "saturated" result (clamped at 0x0000FFFF)
                    return (newHigh << 16) | static_cast<UINT32>(max2(lo1,lo2));
                }
        };

        /* binary function
        */
        struct addDvalueFlags
        {
            public:
                __host__ __device__
                UINT64 operator()( const UINT64& d1, const UINT64& d2 ) const
                {
                    /* Sum the flag bits in the specified 64-bit D values; this function is designed to be
                        used in computing a segmented prefix sum where the "flag" bits (identified by
                        AriocDS::D::maskFlags) are summed and the remaining bits comprise a "key" (i.e.,
                        they are the same for both of the specified D values and should remain unchanged
                        by the operation implemented here).
                    */

                    /* A saturated sum should not be needed since there are 3 flag bits and the maximum
                        value should never exceed 7.  But if we ever do need to do a saturated sum,
                        we can do something like this:
                    
                            UINT64 rval = d1 + (d2 & AriocDS::D::maskFlags);
                            return (rval < d1) ? (rval |= AriocDS::D::maskFlags) : rval;

                            See http://locklessinc.com/articles/sat_arithmetic/
                    */

                    return d1 + (d2 & AriocDS::D::maskFlags);
                }
        };
};
