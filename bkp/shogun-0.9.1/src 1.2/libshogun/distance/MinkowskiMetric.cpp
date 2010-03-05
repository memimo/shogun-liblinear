/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2009 Christian Gehl
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "distance/MinkowskiMetric.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"

using namespace shogun;

CMinkowskiMetric::CMinkowskiMetric(float64_t k_)
: CSimpleDistance<float64_t>(), k(k_)
{
}

CMinkowskiMetric::CMinkowskiMetric(
	CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r, float64_t k_)
: CSimpleDistance<float64_t>(), k(k_)
{
	init(l, r);
}

CMinkowskiMetric::~CMinkowskiMetric()
{
	cleanup();
}

bool CMinkowskiMetric::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleDistance<float64_t>::init(l,r);

	return result;
}

void CMinkowskiMetric::cleanup()
{
}

float64_t CMinkowskiMetric::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(avec);
	ASSERT(bvec);
	ASSERT(alen==blen);

	float64_t absTmp = 0;
	float64_t result=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			absTmp=fabs(avec[i]-bvec[i]);
			result+=pow(absTmp,k);
		}

	}

	((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return pow(result,1/k);
}
