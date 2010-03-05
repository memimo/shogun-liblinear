/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Mathematics.h"
#include "kernel/LinearByteKernel.h"
#include "features/SimpleFeatures.h"

using namespace shogun;

CLinearByteKernel::CLinearByteKernel()
: CSimpleKernel<uint8_t>(0), normal(NULL)
{
}

CLinearByteKernel::CLinearByteKernel(CSimpleFeatures<uint8_t>* l, CSimpleFeatures<uint8_t>* r)
: CSimpleKernel<uint8_t>(0), normal(NULL)
{
	init(l, r);
}

CLinearByteKernel::~CLinearByteKernel()
{
	cleanup();
}

bool CLinearByteKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<uint8_t>::init(l, r);
	return init_normalizer();
}

void CLinearByteKernel::cleanup()
{
	delete_optimization();

	CKernel::cleanup();
}

void CLinearByteKernel::clear_normal()
{
	int32_t num = lhs->get_num_vectors();

	for (int32_t i=0; i<num; i++)
		normal[i]=0;
}

void CLinearByteKernel::add_to_normal(int32_t idx, float64_t weight)
{
	int32_t vlen;
	bool vfree;
	uint8_t* vec=((CSimpleFeatures<uint8_t>*) lhs)->get_feature_vector(idx, vlen, vfree);

	for (int32_t i=0; i<vlen; i++)
		normal[i]+= weight*normalizer->normalize_lhs(vec[i], idx);

	((CSimpleFeatures<uint8_t>*) lhs)->free_feature_vector(vec, idx, vfree);
}

float64_t CLinearByteKernel::compute(int32_t idx_a, int32_t idx_b)
{
  int32_t alen, blen;
  bool afree, bfree;

  uint8_t* avec=((CSimpleFeatures<uint8_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
  uint8_t* bvec=((CSimpleFeatures<uint8_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
  ASSERT(alen==blen);

  float64_t result=CMath::dot(avec,bvec, alen);

  ((CSimpleFeatures<uint8_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSimpleFeatures<uint8_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}

bool CLinearByteKernel::init_optimization(
	int32_t num_suppvec, int32_t* sv_idx, float64_t* alphas)
{
	int32_t alen;
	bool afree;

	int32_t num_feat=((CSimpleFeatures<uint8_t>*) lhs)->get_num_features();
	ASSERT(num_feat);

	normal=new float64_t[num_feat];
	for (int32_t i=0; i<num_feat; i++)
		normal[i]=0;

	for (int32_t i=0; i<num_suppvec; i++)
	{
		uint8_t* avec=((CSimpleFeatures<uint8_t>*) lhs)->get_feature_vector(sv_idx[i], alen, afree);
		ASSERT(avec);

		for (int32_t j=0; j<num_feat; j++)
			normal[j]+= alphas[i] * normalizer->normalize_lhs(((float64_t) avec[j]), sv_idx[i]);

		((CSimpleFeatures<uint8_t>*) lhs)->free_feature_vector(avec, 0, afree);
	}

	set_is_initialized(true);
	return true;
}

bool CLinearByteKernel::delete_optimization()
{
	delete[] normal;
	normal=NULL;

	set_is_initialized(false);

	return true;
}

float64_t CLinearByteKernel::compute_optimized(int32_t idx_b)
{
	int32_t blen;
	bool bfree;

	uint8_t* bvec=((CSimpleFeatures<uint8_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);

	float64_t result=0;
	{
		for (int32_t i=0; i<blen; i++)
			result+= normal[i] * ((float64_t) bvec[i]);
	}

	((CSimpleFeatures<uint8_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return normalizer->normalize_rhs(result, idx_b);
}
