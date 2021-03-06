/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2009 Christian Gehl
 * Copyright (C) 2007-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/DistanceKernel.h"
#include "features/SimpleFeatures.h"

using namespace shogun;

CDistanceKernel::CDistanceKernel(int32_t size, float64_t w, CDistance* d)
: CKernel(size), distance(d), width(w)
{
	ASSERT(distance);
	SG_REF(distance);
}

CDistanceKernel::CDistanceKernel(
	CFeatures *l, CFeatures *r, float64_t w , CDistance* d)
: CKernel(10), distance(d), width(w)
{
	ASSERT(distance);
	SG_REF(distance);
	init(l, r);
}

CDistanceKernel::~CDistanceKernel()
{
	// important to have the cleanup of CKernel first, it calls get_name which
	// uses the distance
	cleanup();
	SG_UNREF(distance);
}

bool CDistanceKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance);
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

float64_t CDistanceKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=distance->distance(idx_a, idx_b);
	return exp(-result/width);
}
