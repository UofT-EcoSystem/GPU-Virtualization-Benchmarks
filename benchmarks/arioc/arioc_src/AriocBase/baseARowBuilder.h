/*
  baseARowBuilder.h

    Copyright (c) 2015-2019 Johns Hopkins University.  All rights reserved.

    This file is part of the Arioc software distribution.  It is subject to the license terms
    in the LICENSE.txt file found in the top-level directory of the Arioc software distribution.
    The contents of this file, in whole or in part, may only be copied, modified, propagated, or
    redistributed in accordance with the license terms contained in LICENSE.txt.
*/
#pragma once
#define __baseARowBuilder__

/// <summary>
/// Class <c>baseARowBuilder</c> defines a base class for the alignment output "builder" classes (e.g. SBFBuilder*, SAMBuilder*)
/// </summary>
class baseARowBuilder
{
#pragma region enums
    protected:
    enum MappingCategory    // mapping categories for paired-end reads
    {
        mcNull =            0x00,   // (e.g., not a paired-end read)
        mcConcordant =      0x01,   // concordant mapping
        mcDiscordant =      0x02,   // discordant mapping
        mcRejected =        0x04,   // both mates mapped but no concordant mapping was found, and one mate has two or more mappings
        mcUnmapped =        0x08,   // read unmapped
        mcOppMateUnmapped = 0x10,   // opposite mate unmapped

        mcReservedBit6 =    0x40,   // (bits 6 and 7 reserved for use in downstream applications)
        mcReservedBit7 =    0x80
    };
#pragma endregion

    private:
        static INT16    m_init;
        static INT16    m_ddc[64];
        static UINT32   m_ddcut32[32];
        static UINT64   m_ddcut64[64];
        static UINT16   m_hexCharPair[256];

    protected:
        QBatch*                 m_pqb;
        AriocBase*              m_pab;
        INT16                   m_cbRow;
        INT32                   m_cbInvariantFields1;
        INT32                   m_cbInvariantFields2;
        const INT16             m_nEmitFields;
        bool                    m_emitA21TraceFields;
        UINT16                  m_flag;                     // i.e., the SAM FLAG field
        WinGlobalPtr<BRLEAbyte> m_revBRLEA;
        static char             m_symbolDecode[];
        static char             m_symbolDecodeRC[];

    private:
        static INT16 initLUTs( void );

    protected:
        INT16 szToSBF( char* p, char* s );
        INT16 u32ToString( char* p, UINT32 u );
        INT16 i32ToString( char* p, INT32 n );
        INT16 u64ToString( char* p, UINT64 u );
        void reverseBRLEA( BRLEAheader* pBH );
        UINT64 setSqIdPairBit( const UINT64 sqId, const QAI* pQAI );
        UINT64 setSqIdSecBit( const UINT64 sqId, const PAI* pPAI );
        UINT64 setSqIdSecBit( const UINT64 sqId, const QAI* pQAI );
        UINT32 stringizeSEQf( char* pbuf, QAI* pQAI );
        UINT32 stringizeSEQr( char* pbuf, QAI* pQAI );
        UINT32 computeA21Hash32( QAI* pQAI );
        UINT64 computeA21Hash64( QAI* pQAI );

        template <typename T> UINT32 uintToHexString( char* p, T u )
        {
            /* This method builds a string representation of an unsigned integer value (i.e. a 4-, 8-, or 16-byte hexadecimal string without a preceding "0x"). */
            UINT16* pDigitPair = reinterpret_cast<UINT16*>(p + 2*sizeof(T));
            while( --pDigitPair >= reinterpret_cast<UINT16*>(p) )
            {
                *pDigitPair = m_hexCharPair[u & 0xFF];
                u >>= 8;
            }

            return 2*sizeof(T);
        }

        template <typename T> UINT32 intToSBF( char* p, T u )
        {
            /* this method emits a typed integer value */
            *reinterpret_cast<T*>(p) = u;
            return sizeof(T);
        }

    public:
        baseARowBuilder( QBatch* pqb, INT16 nEmitFields, bool emitA21TraceFields );
        virtual ~baseARowBuilder( void );
        virtual INT64 WriteRowUu( baseARowWriter* pw, INT64 sqId, QAI* pQAI );   // unpaired: unmapped read
        virtual INT64 WriteRowUm( baseARowWriter* pw, INT64 sqId, QAI* pQAI );   // unpaired: mapped read
        virtual INT64 WriteRowPc( baseARowWriter* pw, INT64 sqId, PAI* pPAI );   // paired: concordant pair
        virtual INT64 WriteRowPd( baseARowWriter* pw, INT64 sqId, PAI* pPAI );   // paired: discordant pair
        virtual INT64 WriteRowPr( baseARowWriter* pw, INT64 sqId, PAI* pPAI );   // paired: rejected pair
        virtual INT64 WriteRowPu( baseARowWriter* pw, INT64 sqId, PAI* pPAI );   // paired: unmapped pair
};
