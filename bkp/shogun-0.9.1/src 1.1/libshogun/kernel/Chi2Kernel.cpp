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
#include "kernel/Chi2Kernel.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"

using namespace shogun;

CChi2Kernel::CChi2Kernel(int32_t size, float64_t w)
: CSimpleKernel<float64_t>(size), width(w)
{
}

CChi2Kernel::CChi2Kernel(
	CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r, float64_t w, int32_t size)
: CSimpleKernel<float64_t>(size), width(w)
{
	init(l,r);
}

CChi2Kernel::~CChi2Kernel()
{
	cleanup();
}

bool CChi2Kernel::init(CFeatures* l, CFeatures* r)
{
	bool result=CSimpleKernel<float64_t>::init(l,r);
	init_normalizer();
	return result;
}

float64_t CChi2Kernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen);

	float64_t result=0;
	for (int32_t i=0; i<alen; i++)
	{
		float64_t n=avec[i]-bvec[i];
		float64_t d=avec[i]+bvec[i];
		if (d!=0)
			result+=n*n/d;
	}

	result=exp(-result/width);

	((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
