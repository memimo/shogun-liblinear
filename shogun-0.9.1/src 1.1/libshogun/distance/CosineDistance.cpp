/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008-2009 Christian Gehl
 * Copyright (C) 2008-2009 Fraunhofer Institute FIRST
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "distance/CosineDistance.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"

using namespace shogun;

CCosineDistance::CCosineDistance()
: CSimpleDistance<float64_t>()
{
}

CCosineDistance::CCosineDistance(CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r)
: CSimpleDistance<float64_t>()
{
	init(l, r);
}

CCosineDistance::~CCosineDistance()
{
	cleanup();
}

bool CCosineDistance::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleDistance<float64_t>::init(l,r);

	return result;
}

void CCosineDistance::cleanup()
{
}

float64_t CCosineDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen);
	float64_t s=0;
	float64_t ab=0;
	float64_t sa=0;
	float64_t sb=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			ab+=avec[i]*bvec[i];
			sa+=pow(fabs(avec[i]),2);
			sb+=pow(fabs(bvec[i]),2);
		}
	}

	((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);
	
	s=sqrt(sa)*sqrt(sb);
	
	// trap division by zero
	if(s==0)
		return 0;

	s=1-ab/s;
	if(s<0)
		return 0;
	else
		return s ;
}
