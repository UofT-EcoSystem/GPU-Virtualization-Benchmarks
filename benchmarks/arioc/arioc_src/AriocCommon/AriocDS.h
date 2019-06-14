/*
  AriocDS.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __AriocDS__

/// <summary>
/// Class <c>AriocDS</c> defines commonly-used data structures
class AriocDS
{
    public:

        /// <summary>
        /// Class <c>SqId</c> defines a sequence identifier.
        /// </summary>
        class SqId
        {
            private:
                struct map
                {
                    /* The following struct is defined for reference but not used in code because the NVidia C++ compiler doesn't
                        know about bitfields.

                       The mate flag derives from the "mate" attribute in the <file> element of an AriocE .cfg file.
                       The data source ID derives from the "srcId" attribute in the <dataIn> element of an AriocE .cfg file.
                       The data source subunit ID derives from the "subId" attribute in the <file> element of an AriocE .cfg file.

                       The data source and subunit IDs might be used as follows:
                        - for a reference sequence: srcId = reference genome; subId = chromosome
                        - for a query sequence: srcId = sample number; subId = file number or parent/child/sibling number

                       We place the "mate" flag in the low-order bit in an attempt to locate both mates near to one another
                        in SQL tables and indexes.
                    */
                    UINT64 flagSqIdMate :  1;   // bits 00..00: 0: mate 1; 1: mate 2 (for paired-end reads)
                    UINT64 flagSecondary:  1;   // bits 01..01: 0: primary alignment; 1: secondary alignment
                    UINT64 readId       : 33;   // bits 02..34: read ID (maximum value 8,589,934,591)
                    UINT64 subId        :  7;   // bits 35..41: data source subunit ID (maximum value 127)
                    UINT64 srcId        : 21;   // bits 42..62: data source ID (maximum value 2,097,151)
                    UINT64              :  1;   // bits 63..63: (unused; if necessary, however, this could become the high-order bit of the data source ID)
                };

            public:
                static const UINT64 MaskMateId = static_cast<UINT64>(0x0000000000000001);
                static const UINT64 MaskSecBit = static_cast<UINT64>(0x0000000000000002);
                static const UINT64 MaskReadId = static_cast<UINT64>(0x00000007FFFFFFFC);
                static const UINT64 MaskSubId =  static_cast<UINT64>(0x000003F800000000);
                static const UINT64 MaskSrcId =  static_cast<UINT64>(0x7FFFFC0000000000);
                static const UINT64 MaskDataSource = (AriocDS::SqId::MaskSrcId | AriocDS::SqId::MaskSubId);

                static const INT32 MaxSubId = (1 << 7) - 1;
                static const INT32 MaxSrcId = (1 << 21) - 1;
                static const INT64 AdjacentSqIdDiff = 4;

                static inline UINT32 GetMateId( UINT64 sqId ) { return static_cast<UINT32>( sqId        & 1); };
                static inline UINT32 GetSecBit( UINT64 sqId ) { return static_cast<UINT32>((sqId >>  1) & 1); };
                static inline UINT64 GetReadId( UINT64 sqId ) { return static_cast<UINT64>((sqId >>  2) & 0x00000001FFFFFFFF); };
                static inline UINT32 GetSubId( UINT64 sqId )  { return static_cast<UINT32>((sqId >> 35) & 0x7F); };
                static inline UINT32 GetSrcId( UINT64 sqId )  { return static_cast<UINT32>((sqId >> 42) & 0x001FFFFF); };

                static inline INT64 PackSqId( INT64 readId, bool pairFlag, UINT8 subId, INT32 srcId, bool secBit = 0 )
                {
                    return static_cast<INT64>(pairFlag & 1) |
                           static_cast<INT64>(secBit << 1) |
                           (readId << 2) |
                           (static_cast<INT64>(subId) << 35) |
                           (static_cast<INT64>(srcId) << 42);
                }

                static inline bool IsAdjacent( const INT64 sqId1, const INT64 sqId2 )
                {
                    const INT64 diff = (sqId1 & MaskReadId) - (sqId2 & MaskReadId);
                    return (abs(diff) == AdjacentSqIdDiff);
                }
        };


        /// <summary>
        /// Struct <c>QID</c> defines a 21-bit value that contains a reference an element in a linear list of Q sequences.
        /// </summary>
        struct QID
        {
            UINT32  iq :  5;    //  0..4:  offset into iw'th Qwarp
            UINT32  iw : 16;    //  5..20: offset into Qwarp buffer
            UINT32     : 10;    // 21..30: (unused)
            UINT32  rc :  1;    // 31..31: mapping derived from reverse complement of read (bsDNA alignments only)


            QID()
            {
                *reinterpret_cast<UINT32*>(this) = 0;
            }

            QID( UINT32 _iw, INT16 _iq )
            {
                this->iq = _iq;
                this->iw = _iw;
            }

#define QID_ID(qid) (qid & 0x001FFFFF)
#define QID_IW(qid) (QID_ID(qid) >> 5)
#define QID_IQ(qid) (qid & 0x0000001F)
#define PACK_QID(iw,iq) ((static_cast<UINT32>(iw) << 5) + static_cast<UINT32>(iq))

            static inline UINT32 GetW( UINT32 qid ) { return QID_IW(qid); };
            static inline UINT32 GetQ( UINT32 qid ) { return QID_IQ(qid); };
            static inline UINT32 Pack( UINT32 iw, UINT32 iq ) { return PACK_QID(iw,iq); };

            static const INT32 limitQID = (1 << 21);    // (the limit magnitude of a QID is 21 bits)
            static const UINT32 maskQID = (static_cast<UINT32>(limitQID)-1);
            static const UINT32 maskRC = 0x80000000;    // bit 31
        };


        /// <summary>
        /// Class <c>Df</c> defines a 64-bit value that contains a sequence identifier and reference position.
        /// </summary>
        class Df
        {
            private:
                /* The following struct is defined for reference but not used in code because the NVidia C++ compiler doesn't know about bitfields.
                */
                struct map
                {
                    UINT64  qlo   :  1;     // bits 00..00: low-order bit of QID (0: mate 1; 1: mate 2)
                    UINT64  s     :  1;     // bits 01..01: strand (0: forward; 1: reverse complement)
                    UINT64  d     : 31;     // bits 02..32: 0-based position relative to the forward strand
                    UINT64  subId :  7;     // bits 33..39: subId
                    UINT64  qhi   : 20;     // bits 40..59: high-order bits of QID
                    UINT64  rc    :  1;     // bits 60..60: seed from reverse complement (for bsDNA alignment only)
                    UINT64  flags :  3;     // bits 61..63: flags
                };

            public:
                static inline UINT32 GetQid( UINT64 df ) { return static_cast<UINT32>(((df >> 40) << 1) | (df & 1)) & QID::maskQID; };

                static const UINT64 flagRCq =       (static_cast<UINT64>(1) << 60);
                static const UINT64 flagX =         (static_cast<UINT64>(1) << 61);
                static const UINT64 flagCandidate = (static_cast<UINT64>(1) << 62);
                static const UINT64 flagMapped =    (static_cast<UINT64>(1) << 63);
                static const UINT64 maskFlags =     (flagX|flagCandidate|flagMapped);
                static const INT32  shrFlags =      61;                 // number of bits to shift flags into LSB
                static const UINT64 maskQID =       0x0FFFFF0000000001; // bits 0 and 40-59
                static const UINT64 shrRCq =        59;                 // number of bits to shift flagRCq into bit 1
                static const UINT64 maskStrand =    0x0000000000000002; // bit 1
        };

        /// <summary>
        /// Class <c>D</c> defines a 64-bit value that contains a sequence identifier and reference position.
        /// </summary>
        class D
        {
            private:
                /* The following struct is defined for reference but not used in code because the NVidia C++ compiler doesn't know about bitfields.
                */
                struct map
                {
                    UINT64  d     : 31;     // bits 00..30: 0-based position relative to the strand designated in the s field (bit 31)
                    UINT64  s     :  1;     // bits 31..31: strand (0: forward; 1: reverse complement)
                    UINT64  subId :  7;     // bits 32..38: subId
                    UINT64  qid   : 21;     // bits 39..59: QID
                    UINT64  rc    :  1;     // bits 60..60: seed from reverse complement (for bsDNA alignment only)
                    UINT64  flags :  3;     // bits 61..63: flags
                };

            public:
                static inline UINT32 GetQid( UINT64 D ) { return static_cast<UINT32>(D >> 39) & QID::maskQID; };
                static const UINT64 flagRCq =       Df::flagRCq;
                static const UINT64 flagX =         Df::flagX;
                static const UINT64 flagCandidate = Df::flagCandidate;
                static const UINT64 flagMapped =    Df::flagMapped;
                static const UINT64 maskFlags =     Df::maskFlags;
                static const INT32  shrFlags =      Df::shrFlags;
                static const INT32  shrRC =         60-31;              // bits to shift Qrc bit into position of strand bit
                static const UINT64 maskPos =       0x000000007FFFFFFF; // bits 0-30
                static const UINT64 maskStrand =    0x0000000080000000; // bit 31
                static const UINT64 maskSubId =     0x0000007F00000000; // bits 32-38
                static const UINT64 maskQID =       0x0FFFFF8000000000; // bits 39-59
        };

        /// <summary>
        /// Class <c>Dq</c> defines a 64-bit value that contains a sequence identifier and seed position.
        /// </summary>
        class Dq
        {
            private:
                /* The following struct is defined for reference but not used in code because the NVidia C++ compiler doesn't know about bitfields.
                */
                struct map
                {
                    UINT64  ij    : 27;     // bits 00..26: 0-based index into the J list for the seed
                    UINT64  spos  : 15;     // bits 27..41: 0-based seed position of the seed wrt start of the Q sequence
                    UINT64  qid   : 21;     // bits 42..62: QID
                    UINT64  rc    :  1;     // bits 63..63: seed from reverse complement (for bsDNA alignment only)
                };

#define PACK_DQ(rc,qid,spos,ij) ((static_cast<UINT64>(rc) << 63) | (static_cast<UINT64>(qid) << 42) | (static_cast<UINT64>(spos) << 27) | ij)

            public:
                static inline UINT32 GetIj( UINT64 Dq ) { return static_cast<UINT32>(Dq & 0x07FFFFFF); }
                static inline UINT32 GetSpos( UINT64 Dq ) { return static_cast<UINT32>(Dq >> 27) & 0x7FFF; }
                static inline UINT32 GetQid( UINT64 Dq ) { return static_cast<UINT32>(Dq >> 42) & QID::maskQID; }
                static inline bool IsRC( UINT64 Dq ) { return ((Dq & 0x8000000000000000) != 0); }
                static const UINT32 limitIj = (1 << 27);  // (the limit magnitude of the ij field is 27 bits)
                static const UINT64 maskRC =        (static_cast<UINT64>(1) << 63);
        };

#if TODO_CHOP_IF_UNUSED
        /// <summary>
        /// Struct <c>SXC</c> defines a 4-byte (32-bit) value that describes seed-and-extend seed coverage.
        /// </summary>
        struct SXC
        {
            private:
                /* The following struct is defined for reference but not used in code because the NVidia C++ compiler doesn't know about bitfields.
                */
                struct map
                {
                    UINT32  c     : 15;     // bits 00..14: number of positions covered
                    UINT32  x     :  1;     // bits 15..15: list management flag
                    UINT32  i     : 16;     // bits 16..31: 0-based seed position relative to the start of the Q sequence
                };

#define SXC_MASK_C  0x00007FFF
#define SXC_MASK_X  0x00008000
#define SXC_MASK_I  0xFFFF0000
#define SXC_PACK(c,x,i)     ((i << 16) | (x << 15) | c)
        };
#endif
};
