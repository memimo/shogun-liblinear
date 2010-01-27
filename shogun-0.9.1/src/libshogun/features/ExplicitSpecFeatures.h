/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _EXPLICITSPECFEATURES_H___
#define _EXPLICITSPECFEATURES_H___

#include "lib/common.h"
#include "lib/io.h"
#include "features/DotFeatures.h"
#include "features/StringFeatures.h"

namespace shogun
{
template <class ST> class CStringFeatures;

/**@brief Features that compute the Spectrum Kernel feature space explicitly.
 *
 * \sa CCommWordStringKernel
 */
class CExplicitSpecFeatures : public CDotFeatures
{
	public:

		/** constructor
		 *
		 * @param str stringfeatures (of words)
		 * @param normalize whether to use sqrtdiag normalization
		 */
		CExplicitSpecFeatures(CStringFeatures<uint16_t>* str, bool normalize=true);

		/** copy constructor */
		CExplicitSpecFeatures(const CExplicitSpecFeatures & orig);

		/** destructor */
		virtual ~CExplicitSpecFeatures();

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		inline virtual int32_t get_dim_feature_space()
		{
			return spec_size;
		}

		/** compute dot product between vector1 and vector2,
		 * appointed by their indices
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, int32_t vec_idx2);

		/** compute dot product between vector1 and a dense vector
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 */
		virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len);

		/** add vector 1 multiplied with alpha to dense vector2
		 *
		 * @param alpha scalar alpha
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 * @param abs_val if true add the absolute value
		 */
		virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false);

		/** get number of non-zero features in vector
		 *
		 * @param num which vector
		 * @return number of non-zero features in vector
		 */
		virtual inline int32_t get_nnz_features_for_vector(int32_t num)
		{
			SG_NOTIMPLEMENTED;
			return 0;
		}

		/** get feature type
		 *
		 * @return templated feature type
		 */
		inline virtual EFeatureType get_feature_type()
		{
			return F_UNKNOWN;
		}

		/** get feature class
		 *
		 * @return feature class
		 */
		inline virtual EFeatureClass get_feature_class()
		{
			return C_SPEC;
		}

		/** get number of strings
		 *
		 * @return number of strings
		 */
		inline virtual int32_t get_num_vectors()
		{
			return num_strings;
		}

		/** get size of one element
		 *
		 * @return size of one element
		 */
		inline virtual int32_t get_size()
		{
			return sizeof(float64_t);
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "ExplicitSpecFeatures"; }

	protected:
		/** obtain the k-spectrum from a list of strings
		 *
		 * @param str the string feature object already in k-mer format
		 */
		void obtain_kmer_spectrum(CStringFeatures<uint16_t>* str);

		/** free kmer spectrum */
		void delete_kmer_spectrum();

	protected:
		/** use sqrtdiag normalization */
		bool use_normalization;
		/** number of strings */
		int32_t num_strings;
		/** size of alphabet */
		int32_t alphabet_size;

		/** size of k-mer spectrum*/
		int32_t spec_size;
		/** k-mer counts for all strings */
		float64_t** k_spectrum;
};
}
#endif // _EXPLICITSPECFEATURES_H___
