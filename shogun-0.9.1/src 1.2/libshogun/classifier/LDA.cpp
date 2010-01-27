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

#ifdef HAVE_LAPACK
#include "classifier/Classifier.h"
#include "classifier/LinearClassifier.h"
#include "classifier/LDA.h"
#include "features/Labels.h"
#include "lib/Mathematics.h"
#include "lib/lapack.h"

using namespace shogun;

CLDA::CLDA(float64_t gamma)
: CLinearClassifier(), m_gamma(gamma)
{
}

CLDA::CLDA(float64_t gamma, CSimpleFeatures<float64_t>* traindat, CLabels* trainlab)
: CLinearClassifier(), m_gamma(gamma)
{
	set_features(traindat);
	set_labels(trainlab);
}


CLDA::~CLDA()
{
}

bool CLDA::train(CFeatures* data)
{
	ASSERT(labels);
	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n");
		set_features((CDotFeatures*) data);
	}
	ASSERT(features);
	int32_t num_train_labels=0;
	int32_t* train_labels=labels->get_int_labels(num_train_labels);
	ASSERT(train_labels);

	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();
	ASSERT(num_vec==num_train_labels);

	int32_t* classidx_neg=new int32_t[num_vec];
	int32_t* classidx_pos=new int32_t[num_vec];

	int32_t i=0;
	int32_t j=0;
	int32_t num_neg=0;
	int32_t num_pos=0;
	for (i=0; i<num_train_labels; i++)
	{
		if (train_labels[i]==-1)
			classidx_neg[num_neg++]=i;
		else if (train_labels[i]==+1)
			classidx_pos[num_pos++]=i;
		else
		{
			SG_ERROR( "found label != +/- 1 bailing...");
			return false;
		}
	}

	if (num_neg<=0 && num_pos<=0)
	{
      SG_ERROR( "whooooo ? only a single class found\n");
		return false;
	}

	delete[] w;
	w=new float64_t[num_feat];
	w_dim=num_feat;

	float64_t* mean_neg=new float64_t[num_feat];
	memset(mean_neg,0,num_feat*sizeof(float64_t));

	float64_t* mean_pos=new float64_t[num_feat];
	memset(mean_pos,0,num_feat*sizeof(float64_t));

	/* calling external lib */
	double* scatter=new double[num_feat*num_feat];
	double* buffer=new double[num_feat*CMath::max(num_neg, num_pos)];
	int nf = (int) num_feat;

	CSimpleFeatures<float64_t>* rf = (CSimpleFeatures<float64_t>*) features;
	//mean neg
	for (i=0; i<num_neg; i++)
	{
		int32_t vlen;
		bool vfree;
		float64_t* vec=
			rf->get_feature_vector(classidx_neg[i], vlen, vfree);
		ASSERT(vec);

		for (j=0; j<vlen; j++)
		{
			mean_neg[j]+=vec[j];
			buffer[num_feat*i+j]=vec[j];
		}

		rf->free_feature_vector(vec, classidx_neg[i], vfree);
	}

	for (j=0; j<num_feat; j++)
		mean_neg[j]/=num_neg;

	for (i=0; i<num_neg; i++)
	{
		for (j=0; j<num_feat; j++)
			buffer[num_feat*i+j]-=mean_neg[j];
	}
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nf, nf,
		(int) num_neg, 1.0, buffer, nf, buffer, nf, 0, scatter, nf);
	
	//mean pos
	for (i=0; i<num_pos; i++)
	{
		int32_t vlen;
		bool vfree;
		float64_t* vec=
			rf->get_feature_vector(classidx_pos[i], vlen, vfree);
		ASSERT(vec);

		for (j=0; j<vlen; j++)
		{
			mean_pos[j]+=vec[j];
			buffer[num_feat*i+j]=vec[j];
		}

		rf->free_feature_vector(vec, classidx_pos[i], vfree);
	}

	for (j=0; j<num_feat; j++)
		mean_pos[j]/=num_pos;

	for (i=0; i<num_pos; i++)
	{
		for (j=0; j<num_feat; j++)
			buffer[num_feat*i+j]-=mean_pos[j];
	}
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, nf, nf, (int) num_pos,
		1.0/(num_train_labels-1), buffer, nf, buffer, nf,
		1.0/(num_train_labels-1), scatter, nf);

	float64_t trace=CMath::trace((float64_t*) scatter, num_feat, num_feat);

	double s=1.0-m_gamma; /* calling external lib; indirectly */
	for (i=0; i<num_feat*num_feat; i++)
		scatter[i]*=s;

	for (i=0; i<num_feat; i++)
		scatter[i*num_feat+i]+= trace*m_gamma/num_feat;

	double* inv_scatter= (double*) CMath::pinv(
		scatter, num_feat, num_feat, NULL);

	float64_t* w_pos=buffer;
	float64_t* w_neg=&buffer[num_feat];

	cblas_dsymv(CblasColMajor, CblasUpper, nf, 1.0, inv_scatter, nf,
		(double*) mean_pos, 1, 0., (double*) w_pos, 1);
	cblas_dsymv(CblasColMajor, CblasUpper, nf, 1.0, inv_scatter, nf,
		(double*) mean_neg, 1, 0, (double*) w_neg, 1);
	
	bias=0.5*(CMath::dot(w_neg, mean_neg, num_feat)-CMath::dot(w_pos, mean_pos, num_feat));
	for (i=0; i<num_feat; i++)
		w[i]=w_pos[i]-w_neg[i];

#ifdef DEBUG_LDA
	SG_PRINT("bias: %f\n", bias);
    CMath::display_vector(w, num_feat, "w");
    CMath::display_vector(w_pos, num_feat, "w_pos");
    CMath::display_vector(w_neg, num_feat, "w_neg");
    CMath::display_vector(mean_pos, num_feat, "mean_pos");
    CMath::display_vector(mean_neg, num_feat, "mean_neg");
#endif

	delete[] train_labels;
	delete[] mean_neg;
	delete[] mean_pos;
	delete[] scatter;
	delete[] inv_scatter;
	delete[] classidx_neg;
	delete[] classidx_pos;
	delete[] buffer;
	return true;
}
#endif
