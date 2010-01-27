/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Soeren Sonnenburg
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <string.h>
#include <math.h>

#include "features/Alphabet.h"
#include "lib/io.h"

using namespace shogun;

//define numbers for the bases
const uint8_t CAlphabet::B_A=0;
const uint8_t CAlphabet::B_C=1;
const uint8_t CAlphabet::B_G=2;
const uint8_t CAlphabet::B_T=3;
const uint8_t CAlphabet::MAPTABLE_UNDEF=0xff;
const char* CAlphabet::alphabet_names[12]={"DNA", "RAWDNA", "RNA", "PROTEIN", "BINARY", "ALPHANUM", "CUBE", "RAW", "IUPAC_NUCLEIC_ACID", "IUPAC_AMINO_ACID", "NONE", "UNKNOWN"};

/*
CAlphabet::CAlphabet()
: CSGObject()
{

}
*/

CAlphabet::CAlphabet(char* al, int32_t len)
: CSGObject()
{
	EAlphabet alpha=NONE;

	if (len>=(int32_t) strlen("DNA") && !strncmp(al, "DNA", strlen("DNA")))
		alpha = DNA;
	else if (len>=(int32_t) strlen("RAWDNA") && !strncmp(al, "RAWDNA", strlen("RAWDNA")))
		alpha = RAWDNA;
	else if (len>=(int32_t) strlen("RNA") && !strncmp(al, "RNA", strlen("RNA")))
		alpha = RNA;
	else if (len>=(int32_t) strlen("PROTEIN") && !strncmp(al, "PROTEIN", strlen("PROTEIN")))
		alpha = PROTEIN;
	else if (len>=(int32_t) strlen("BINARY") && !strncmp(al, "BINARY", strlen("IBINARY")))
		alpha = BINARY;
	else if (len>=(int32_t) strlen("ALPHANUM") && !strncmp(al, "ALPHANUM", strlen("ALPHANUM")))
		alpha = ALPHANUM;
	else if (len>=(int32_t) strlen("CUBE") && !strncmp(al, "CUBE", strlen("CUBE")))
		alpha = CUBE;
	else if ((len>=(int32_t) strlen("BYTE") && !strncmp(al, "BYTE", strlen("BYTE"))) ||
			(len>=(int32_t) strlen("RAW") && !strncmp(al, "RAW", strlen("RAW"))))
		alpha = RAWBYTE;
	else if (len>=(int32_t) strlen("IUPAC_NUCLEIC_ACID") && !strncmp(al, "IUPAC_NUCLEIC_ACID", strlen("IUPAC_NUCLEIC_ACID")))
		alpha = IUPAC_NUCLEIC_ACID;
	else if (len>=(int32_t) strlen("IUPAC_AMINO_ACID") && !strncmp(al, "IUPAC_AMINO_ACID", strlen("IUPAC_AMINO_ACID")))
		alpha = IUPAC_AMINO_ACID;
	else {
      SG_ERROR( "unknown alphabet %s\n", al);
   }

	set_alphabet(alpha);
}

CAlphabet::CAlphabet(EAlphabet alpha)
: CSGObject()
{
	set_alphabet(alpha);
}

CAlphabet::CAlphabet(CAlphabet* a)
: CSGObject()
{
	ASSERT(a);
	set_alphabet(a->get_alphabet());
	copy_histogram(a);
}

CAlphabet::~CAlphabet()
{
}

bool CAlphabet::set_alphabet(EAlphabet alpha)
{
	bool result=true;
	alphabet=alpha;

	switch (alphabet)
	{
		case DNA:
		case RAWDNA:
			num_symbols = 4;
			break;
		case RNA:
			num_symbols = 4;
			break;
		case PROTEIN:
			num_symbols = 26;
			break;
		case BINARY:
			num_symbols = 2;
			break;
		case ALPHANUM:
			num_symbols = 36;
			break;
		case CUBE:
			num_symbols = 6;
			break;
		case RAWBYTE:
			num_symbols = 256;
			break;
		case IUPAC_NUCLEIC_ACID:
			num_symbols = 16;
			break;
		case IUPAC_AMINO_ACID:
			num_symbols = 23;
			break;
		case NONE:
			num_symbols = 0;
			break;
		default:
			num_symbols = 0;
			result=false;
			break;
	}

	num_bits=(int32_t) ceil(log((float64_t) num_symbols)/log((float64_t) 2));
	init_map_table();
    clear_histogram();

	SG_DEBUG( "initialised alphabet %s\n", get_alphabet_name(alphabet));

	return result;
}

void CAlphabet::init_map_table()
{
	int32_t i;
	for (i=0; i<(1<<(8*sizeof(uint8_t))); i++)
	{
		maptable_to_bin[i] = MAPTABLE_UNDEF;
		maptable_to_char[i] = MAPTABLE_UNDEF;
		valid_chars[i] = false;
	}

	switch (alphabet)
	{
		case CUBE:
			valid_chars[(uint8_t) '1']=true;
			valid_chars[(uint8_t) '2']=true;
			valid_chars[(uint8_t) '3']=true;
			valid_chars[(uint8_t) '4']=true;
			valid_chars[(uint8_t) '5']=true;
			valid_chars[(uint8_t) '6']=true;	//Translation '123456' -> 012345

			maptable_to_bin[(uint8_t) '1']=0;
			maptable_to_bin[(uint8_t) '2']=1;
			maptable_to_bin[(uint8_t) '3']=2;
			maptable_to_bin[(uint8_t) '4']=3;
			maptable_to_bin[(uint8_t) '5']=4;
			maptable_to_bin[(uint8_t) '6']=5;	//Translation '123456' -> 012345

			maptable_to_char[(uint8_t) 0]='1';
			maptable_to_char[(uint8_t) 1]='2';
			maptable_to_char[(uint8_t) 2]='3';
			maptable_to_char[(uint8_t) 3]='4';
			maptable_to_char[(uint8_t) 4]='5';
			maptable_to_char[(uint8_t) 5]='6';	//Translation 012345->'123456'
			break;

		case PROTEIN:
			{
				int32_t skip=0 ;
				for (i=0; i<21; i++)
				{
					if (i==1) skip++ ;
					if (i==8) skip++ ;
					if (i==12) skip++ ;
					if (i==17) skip++ ;
					valid_chars['A'+i+skip]=true;
					maptable_to_bin['A'+i+skip]=i ;
					maptable_to_char[i]='A'+i+skip ;
				} ;                   //Translation 012345->acde...xy -- the protein code
			} ;
			break;

		case BINARY:
			valid_chars[(uint8_t) '0']=true;
			valid_chars[(uint8_t) '1']=true;

			maptable_to_bin[(uint8_t) '0']=0;
			maptable_to_bin[(uint8_t) '1']=1;

			maptable_to_char[0]=(uint8_t) '0';
			maptable_to_char[1]=(uint8_t) '1';
			break;

		case ALPHANUM:
			{
				for (i=0; i<26; i++)
				{
					valid_chars['A'+i]=true;
					maptable_to_bin['A'+i]=i ;
					maptable_to_char[i]='A'+i ;
				} ;
				for (i=0; i<10; i++)
				{
					valid_chars['0'+i]=true;
					maptable_to_bin['0'+i]=26+i ;
					maptable_to_char[26+i]='0'+i ;
				} ;        //Translation 012345->acde...xy0123456789
			} ;
			break;

		case RAWBYTE:
			{
				//identity
				for (i=0; i<256; i++)
				{
					valid_chars[i]=true;
					maptable_to_bin[i]=i;
					maptable_to_char[i]=i;
				}
			}
			break;

		case DNA:
			valid_chars[(uint8_t) 'A']=true;
			valid_chars[(uint8_t) 'C']=true;
			valid_chars[(uint8_t) 'G']=true;
			valid_chars[(uint8_t) 'T']=true;

			maptable_to_bin[(uint8_t) 'A']=B_A;
			maptable_to_bin[(uint8_t) 'C']=B_C;
			maptable_to_bin[(uint8_t) 'G']=B_G;
			maptable_to_bin[(uint8_t) 'T']=B_T;

			maptable_to_char[B_A]='A';
			maptable_to_char[B_C]='C';
			maptable_to_char[B_G]='G';
			maptable_to_char[B_T]='T';
			break;
		case RAWDNA:
			{
				//identity
				for (i=0; i<4; i++)
				{
					valid_chars[i]=true;
					maptable_to_bin[i]=i;
					maptable_to_char[i]=i;
				}
			}
			break;

		case RNA:
			valid_chars[(uint8_t) 'A']=true;
			valid_chars[(uint8_t) 'C']=true;
			valid_chars[(uint8_t) 'G']=true;
			valid_chars[(uint8_t) 'U']=true;

			maptable_to_bin[(uint8_t) 'A']=B_A;
			maptable_to_bin[(uint8_t) 'C']=B_C;
			maptable_to_bin[(uint8_t) 'G']=B_G;
			maptable_to_bin[(uint8_t) 'U']=B_T;

			maptable_to_char[B_A]='A';
			maptable_to_char[B_C]='C';
			maptable_to_char[B_G]='G';
			maptable_to_char[B_T]='U';
			break;

		case IUPAC_NUCLEIC_ACID:
			valid_chars[(uint8_t) 'A']=true; // A	Adenine
			valid_chars[(uint8_t) 'C']=true; // C	Cytosine
			valid_chars[(uint8_t) 'G']=true; // G	Guanine
			valid_chars[(uint8_t) 'T']=true; // T	Thymine
			valid_chars[(uint8_t) 'U']=true; // U	Uracil
			valid_chars[(uint8_t) 'R']=true; // R	Purine (A or G)
			valid_chars[(uint8_t) 'Y']=true; // Y	Pyrimidine (C, T, or U)
			valid_chars[(uint8_t) 'M']=true; // M	C or A
			valid_chars[(uint8_t) 'K']=true; // K	T, U, or G
			valid_chars[(uint8_t) 'W']=true; // W	T, U, or A
			valid_chars[(uint8_t) 'S']=true; // S	C or G
			valid_chars[(uint8_t) 'B']=true; // B	C, T, U, or G (not A)
			valid_chars[(uint8_t) 'D']=true; // D	A, T, U, or G (not C)
			valid_chars[(uint8_t) 'H']=true; // H	A, T, U, or C (not G)
			valid_chars[(uint8_t) 'V']=true; // V	A, C, or G (not T, not U)
			valid_chars[(uint8_t) 'N']=true; // N	Any base (A, C, G, T, or U)

			maptable_to_bin[(uint8_t) 'A']=0; // A	Adenine
			maptable_to_bin[(uint8_t) 'C']=1; // C	Cytosine
			maptable_to_bin[(uint8_t) 'G']=2; // G	Guanine
			maptable_to_bin[(uint8_t) 'T']=3; // T	Thymine
			maptable_to_bin[(uint8_t) 'U']=4; // U	Uracil
			maptable_to_bin[(uint8_t) 'R']=5; // R	Purine (A or G)
			maptable_to_bin[(uint8_t) 'Y']=6; // Y	Pyrimidine (C, T, or U)
			maptable_to_bin[(uint8_t) 'M']=7; // M	C or A
			maptable_to_bin[(uint8_t) 'K']=8; // K	T, U, or G
			maptable_to_bin[(uint8_t) 'W']=9; // W	T, U, or A
			maptable_to_bin[(uint8_t) 'S']=10; // S	C or G
			maptable_to_bin[(uint8_t) 'B']=11; // B	C, T, U, or G (not A)
			maptable_to_bin[(uint8_t) 'D']=12; // D	A, T, U, or G (not C)
			maptable_to_bin[(uint8_t) 'H']=13; // H	A, T, U, or C (not G)
			maptable_to_bin[(uint8_t) 'V']=14; // V	A, C, or G (not T, not U)
			maptable_to_bin[(uint8_t) 'N']=15; // N	Any base (A, C, G, T, or U)

			maptable_to_char[0]=(uint8_t) 'A'; // A	Adenine
			maptable_to_char[1]=(uint8_t) 'C'; // C	Cytosine
			maptable_to_char[2]=(uint8_t) 'G'; // G	Guanine
			maptable_to_char[3]=(uint8_t) 'T'; // T	Thymine
			maptable_to_char[4]=(uint8_t) 'U'; // U	Uracil
			maptable_to_char[5]=(uint8_t) 'R'; // R	Purine (A or G)
			maptable_to_char[6]=(uint8_t) 'Y'; // Y	Pyrimidine (C, T, or U)
			maptable_to_char[7]=(uint8_t) 'M'; // M	C or A
			maptable_to_char[8]=(uint8_t) 'K'; // K	T, U, or G
			maptable_to_char[9]=(uint8_t) 'W'; // W	T, U, or A
			maptable_to_char[10]=(uint8_t) 'S'; // S	C or G
			maptable_to_char[11]=(uint8_t) 'B'; // B	C, T, U, or G (not A)
			maptable_to_char[12]=(uint8_t) 'D'; // D	A, T, U, or G (not C)
			maptable_to_char[13]=(uint8_t) 'H'; // H	A, T, U, or C (not G)
			maptable_to_char[14]=(uint8_t) 'V'; // V	A, C, or G (not T, not U)
			maptable_to_char[15]=(uint8_t) 'N'; // N	Any base (A, C, G, T, or U)
			break;

		case IUPAC_AMINO_ACID:
			valid_chars[(uint8_t) 'A']=true; //A	Ala	Alanine
			valid_chars[(uint8_t) 'R']=true; //R	Arg	Arginine
			valid_chars[(uint8_t) 'N']=true; //N	Asn	Asparagine
			valid_chars[(uint8_t) 'D']=true; //D	Asp	Aspartic acid
			valid_chars[(uint8_t) 'C']=true; //C	Cys	Cysteine
			valid_chars[(uint8_t) 'Q']=true; //Q	Gln	Glutamine
			valid_chars[(uint8_t) 'E']=true; //E	Glu	Glutamic acid
			valid_chars[(uint8_t) 'G']=true; //G	Gly	Glycine
			valid_chars[(uint8_t) 'H']=true; //H	His	Histidine
			valid_chars[(uint8_t) 'I']=true; //I	Ile	Isoleucine
			valid_chars[(uint8_t) 'L']=true; //L	Leu	Leucine
			valid_chars[(uint8_t) 'K']=true; //K	Lys	Lysine
			valid_chars[(uint8_t) 'M']=true; //M	Met	Methionine
			valid_chars[(uint8_t) 'F']=true; //F	Phe	Phenylalanine
			valid_chars[(uint8_t) 'P']=true; //P	Pro	Proline
			valid_chars[(uint8_t) 'S']=true; //S	Ser	Serine
			valid_chars[(uint8_t) 'T']=true; //T	Thr	Threonine
			valid_chars[(uint8_t) 'W']=true; //W	Trp	Tryptophan
			valid_chars[(uint8_t) 'Y']=true; //Y	Tyr	Tyrosine
			valid_chars[(uint8_t) 'V']=true; //V	Val	Valine
			valid_chars[(uint8_t) 'B']=true; //B	Asx	Aspartic acid or Asparagine
			valid_chars[(uint8_t) 'Z']=true; //Z	Glx	Glutamine or Glutamic acid
			valid_chars[(uint8_t) 'X']=true; //X	Xaa	Any amino acid

			maptable_to_bin[(uint8_t) 'A']=0;  //A	Ala	Alanine
			maptable_to_bin[(uint8_t) 'R']=1;  //R	Arg	Arginine
			maptable_to_bin[(uint8_t) 'N']=2;  //N	Asn	Asparagine
			maptable_to_bin[(uint8_t) 'D']=3;  //D	Asp	Aspartic acid
			maptable_to_bin[(uint8_t) 'C']=4;  //C	Cys	Cysteine
			maptable_to_bin[(uint8_t) 'Q']=5;  //Q	Gln	Glutamine
			maptable_to_bin[(uint8_t) 'E']=6;  //E	Glu	Glutamic acid
			maptable_to_bin[(uint8_t) 'G']=7;  //G	Gly	Glycine
			maptable_to_bin[(uint8_t) 'H']=8;  //H	His	Histidine
			maptable_to_bin[(uint8_t) 'I']=9;  //I	Ile	Isoleucine
			maptable_to_bin[(uint8_t) 'L']=10; //L	Leu	Leucine
			maptable_to_bin[(uint8_t) 'K']=11; //K	Lys	Lysine
			maptable_to_bin[(uint8_t) 'M']=12; //M	Met	Methionine
			maptable_to_bin[(uint8_t) 'F']=13; //F	Phe	Phenylalanine
			maptable_to_bin[(uint8_t) 'P']=14; //P	Pro	Proline
			maptable_to_bin[(uint8_t) 'S']=15; //S	Ser	Serine
			maptable_to_bin[(uint8_t) 'T']=16; //T	Thr	Threonine
			maptable_to_bin[(uint8_t) 'W']=17; //W	Trp	Tryptophan
			maptable_to_bin[(uint8_t) 'Y']=18; //Y	Tyr	Tyrosine
			maptable_to_bin[(uint8_t) 'V']=19; //V	Val	Valine
			maptable_to_bin[(uint8_t) 'B']=20; //B	Asx	Aspartic acid or Asparagine
			maptable_to_bin[(uint8_t) 'Z']=21; //Z	Glx	Glutamine or Glutamic acid
			maptable_to_bin[(uint8_t) 'X']=22; //X	Xaa	Any amino acid

			maptable_to_char[0]=(uint8_t) 'A';  //A	Ala	Alanine
			maptable_to_char[1]=(uint8_t) 'R';  //R	Arg	Arginine
			maptable_to_char[2]=(uint8_t) 'N';  //N	Asn	Asparagine
			maptable_to_char[3]=(uint8_t) 'D';  //D	Asp	Aspartic acid
			maptable_to_char[4]=(uint8_t) 'C';  //C	Cys	Cysteine
			maptable_to_char[5]=(uint8_t) 'Q';  //Q	Gln	Glutamine
			maptable_to_char[6]=(uint8_t) 'E';  //E	Glu	Glutamic acid
			maptable_to_char[7]=(uint8_t) 'G';  //G	Gly	Glycine
			maptable_to_char[8]=(uint8_t) 'H';  //H	His	Histidine
			maptable_to_char[9]=(uint8_t) 'I';  //I	Ile	Isoleucine
			maptable_to_char[10]=(uint8_t) 'L'; //L	Leu	Leucine
			maptable_to_char[11]=(uint8_t) 'K'; //K	Lys	Lysine
			maptable_to_char[12]=(uint8_t) 'M'; //M	Met	Methionine
			maptable_to_char[13]=(uint8_t) 'F'; //F	Phe	Phenylalanine
			maptable_to_char[14]=(uint8_t) 'P'; //P	Pro	Proline
			maptable_to_char[15]=(uint8_t) 'S'; //S	Ser	Serine
			maptable_to_char[16]=(uint8_t) 'T'; //T	Thr	Threonine
			maptable_to_char[17]=(uint8_t) 'W'; //W	Trp	Tryptophan
			maptable_to_char[18]=(uint8_t) 'Y'; //Y	Tyr	Tyrosine
			maptable_to_char[19]=(uint8_t) 'V'; //V	Val	Valine
			maptable_to_char[20]=(uint8_t) 'B'; //B	Asx	Aspartic acid or Asparagine
			maptable_to_char[21]=(uint8_t) 'Z'; //Z	Glx	Glutamine or Glutamic acid
			maptable_to_char[22]=(uint8_t) 'X'; //X	Xaa	Any amino acid
			break;

		default:
			break; //leave uninitialised
	};
}

void CAlphabet::clear_histogram()
{
	memset(histogram, 0, sizeof(histogram));
    print_histogram();
}

int32_t CAlphabet::get_max_value_in_histogram()
{
	int32_t max_sym=-1;
	for (int32_t i=(int32_t) (1 <<(sizeof(uint8_t)*8))-1;i>=0; i--)
	{
		if (histogram[i])
		{
			max_sym=i;
			break;
		}
	}

	return max_sym;
}

int32_t CAlphabet::get_num_symbols_in_histogram()
{
	int32_t num_sym=0;
	for (int32_t i=0; i<(int32_t) (1 <<(sizeof(uint8_t)*8)); i++)
	{
		if (histogram[i])
			num_sym++;
	}

	return num_sym;
}

int32_t CAlphabet::get_num_bits_in_histogram()
{
	int32_t num_sym=get_num_symbols_in_histogram();
	if (num_sym>0)
		return (int32_t) ceil(log((float64_t) num_sym)/log((float64_t) 2));
	else
		return 0;
}

void CAlphabet::print_histogram()
{
	for (int32_t i=0; i<(int32_t) (1 <<(sizeof(uint8_t)*8)); i++)
	{
		if (histogram[i])
			SG_PRINT( "hist[%d]=%lld\n", i, histogram[i]);
	}
}

bool CAlphabet::check_alphabet(bool print_error)
{
	bool result = true;

	for (int32_t i=0; i<(int32_t) (1 <<(sizeof(uint8_t)*8)); i++)
	{
		if (histogram[i]>0 && valid_chars[i]==0)
		{
			result=false;
			break;
		}
	}

	if (!result && print_error)
	{
		print_histogram();
		SG_ERROR( "ALPHABET does not contain all symbols in histogram\n");
	}

	return result;
}

bool CAlphabet::check_alphabet_size(bool print_error)
{
	if (get_num_bits_in_histogram() > get_num_bits())
	{
		if (print_error)
		{
			print_histogram();
			fprintf(stderr, "get_num_bits_in_histogram()=%i > get_num_bits()=%i\n", get_num_bits_in_histogram(), get_num_bits()) ;
         SG_ERROR( "ALPHABET too small to contain all symbols in histogram\n");
		}
		return false;
	}
	else
		return true;

}

void CAlphabet::copy_histogram(CAlphabet* a)
{
	memcpy(histogram, a->get_histogram(), sizeof(histogram));
}

const char* CAlphabet::get_alphabet_name(EAlphabet alphabet)
{

	int32_t idx;
	switch (alphabet)
	{
		case DNA:
			idx=0;
			break;
		case RAWDNA:
			idx=1;
			break;
		case RNA:
			idx=2;
			break;
		case PROTEIN:
			idx=3;
			break;
		case BINARY:
			idx=4;
			break;
		case ALPHANUM:
			idx=5;
			break;
		case CUBE:
			idx=6;
			break;
		case RAWBYTE:
			idx=7;
			break;
		case IUPAC_NUCLEIC_ACID:
			idx=8;
			break;
		case IUPAC_AMINO_ACID:
			idx=9;
			break;
		case NONE:
			idx=10;
			break;
		default:
			idx=11;
			break;
	}
	return alphabet_names[idx];
}

#ifdef HAVE_BOOST_SERIALIZATION
std::string CAlphabet::to_string() const
{
	std::ostringstream s;
	::boost::archive::text_oarchive oa(s);
	oa << this;
	return s.str();
}

void CAlphabet::from_string(std::string str)
{
	std::istringstream is(str);
	::boost::archive::text_iarchive ia(is);

	//cast away constness
	CAlphabet* tmp = const_cast<CAlphabet*>(this);

	ia >> tmp;
	*this = *tmp;
}

void CAlphabet::to_file(std::string filename) const
{
	std::ofstream os(filename.c_str(), std::ios::binary);
	::boost::archive::binary_oarchive oa(os);
	oa << this;
}

void CAlphabet::from_file(std::string filename)
{
	std::ifstream is(filename.c_str(), std::ios::binary);
	::boost::archive::binary_iarchive ia(is);
	CAlphabet* tmp= const_cast<CAlphabet*>(this);
	ia >> tmp;
}
#endif //HAVE_BOOST_SERIALIZATION
