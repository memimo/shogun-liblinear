/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "distributions/LinearHMM.h"
#include "lib/common.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

using namespace shogun;

CLinearHMM::CLinearHMM(CStringFeatures<uint16_t>* f)
: CDistribution(), transition_probs(NULL), log_transition_probs(NULL)
{
	features=f;
	sequence_length = f->get_vector_length(0);
	num_symbols     = (int32_t) f->get_num_symbols();
	num_params      = sequence_length*num_symbols;
}

CLinearHMM::CLinearHMM(int32_t p_num_features, int32_t p_num_symbols)
: CDistribution(), transition_probs(NULL), log_transition_probs(NULL)
{
	sequence_length = p_num_features;
	num_symbols     = p_num_symbols;
	num_params      = sequence_length*num_symbols;
}

CLinearHMM::~CLinearHMM()
{
	delete[] transition_probs;
	delete[] log_transition_probs;
}

bool CLinearHMM::train(CFeatures* data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STRING ||
				data->get_feature_type() != F_WORD)
		{
			SG_ERROR("Expected features of class string type word!\n");
		}
		set_features(data);
	}
	delete[] transition_probs;
	delete[] log_transition_probs;
	int32_t* int_transition_probs=new int32_t[num_params];

	int32_t vec;
	int32_t i;

	for (i=0; i< num_params; i++)
		int_transition_probs[i]=0;

	for (vec=0; vec<features->get_num_vectors(); vec++)
	{
		int32_t len;
		bool free_vec;

		uint16_t* vector=((CStringFeatures<uint16_t>*) features)->
			get_feature_vector(vec, len, free_vec);

		//just count the symbols per position -> transition_probsogram
		for (int32_t feat=0; feat<len ; feat++)
			int_transition_probs[feat*num_symbols+vector[feat]]++;

		((CStringFeatures<uint16_t>*) features)->
			free_feature_vector(vector, vec, free_vec);
	}

	//trade memory for speed
	transition_probs=new float64_t[num_params];
	log_transition_probs=new float64_t[num_params];

	for (i=0;i<sequence_length;i++)
	{
		for (int32_t j=0; j<num_symbols; j++)
		{
			float64_t sum=0;
			int32_t offs=i*num_symbols+
				((CStringFeatures<uint16_t> *) features)->
					get_masked_symbols((uint16_t)j,(uint8_t) 254);
			int32_t original_num_symbols=(int32_t)
				((CStringFeatures<uint16_t> *) features)->
					get_original_num_symbols();

			for (int32_t k=0; k<original_num_symbols; k++)
				sum+=int_transition_probs[offs+k];

			transition_probs[i*num_symbols+j]=
				(int_transition_probs[i*num_symbols+j]+pseudo_count)/
				(sum+((CStringFeatures<uint16_t> *) features)->
					get_original_num_symbols()*pseudo_count);
			log_transition_probs[i*num_symbols+j]=
				log(transition_probs[i*num_symbols+j]);
		}
	}

	delete[] int_transition_probs;
	return true;
}

bool CLinearHMM::train(
	const int32_t* indizes, int32_t num_indizes, float64_t pseudo)
{
	delete[] transition_probs;
	delete[] log_transition_probs;
	int32_t* int_transition_probs=new int32_t[num_params];
	int32_t vec;
	int32_t i;

	for (i=0; i< num_params; i++)
		int_transition_probs[i]=0;

	for (vec=0; vec<num_indizes; vec++)
	{
		int32_t len;
		bool free_vec;

		ASSERT(indizes[vec]>=0 &&
			indizes[vec]<((CStringFeatures<uint16_t>*) features)->
				get_num_vectors());
		uint16_t* vector=((CStringFeatures<uint16_t>*) features)->
			get_feature_vector(indizes[vec], len, free_vec);
		((CStringFeatures<uint16_t>*) features)->
			free_feature_vector(vector, indizes[vec], free_vec);

		//just count the symbols per position -> transition_probsogram
		//
		for (int32_t feat=0; feat<len ; feat++)
			int_transition_probs[feat*num_symbols+vector[feat]]++;
	}

	//trade memory for speed
	transition_probs=new float64_t[num_params];
	log_transition_probs=new float64_t[num_params];

	for (i=0;i<sequence_length;i++)
	{
		for (int32_t j=0; j<num_symbols; j++)
		{
			float64_t sum=0;
			int32_t original_num_symbols=(int32_t)
				((CStringFeatures<uint16_t> *) features)->
					get_original_num_symbols();
			for (int32_t k=0; k<original_num_symbols; k++)
			{
				sum+=int_transition_probs[i*num_symbols+
					((CStringFeatures<uint16_t>*) features)->
						get_masked_symbols((uint16_t)j,(uint8_t) 254)+k];
			}

			transition_probs[i*num_symbols+j]=
				(int_transition_probs[i*num_symbols+j]+pseudo)/
				(sum+((CStringFeatures<uint16_t>*) features)->
					get_original_num_symbols()*pseudo);
			log_transition_probs[i*num_symbols+j]=
				log(transition_probs[i*num_symbols+j]);
		}
	}

	delete[] int_transition_probs;
	return true;
}

float64_t CLinearHMM::get_log_likelihood_example(uint16_t* vector, int32_t len)
{
	float64_t result=log_transition_probs[vector[0]];

	for (int32_t i=1; i<len; i++)
		result+=log_transition_probs[i*num_symbols+vector[i]];
	
	return result;
}

float64_t CLinearHMM::get_log_likelihood_example(int32_t num_example)
{
	int32_t len;
	bool free_vec;
	uint16_t* vector=((CStringFeatures<uint16_t>*) features)->
		get_feature_vector(num_example, len, free_vec);
	float64_t result=log_transition_probs[vector[0]];

	for (int32_t i=1; i<len; i++)
		result+=log_transition_probs[i*num_symbols+vector[i]];

	((CStringFeatures<uint16_t>*) features)->
		free_feature_vector(vector, num_example, free_vec);

	return result;
}

float64_t CLinearHMM::get_likelihood_example(uint16_t* vector, int32_t len)
{
	float64_t result=transition_probs[vector[0]];

	for (int32_t i=1; i<len; i++)
		result*=transition_probs[i*num_symbols+vector[i]];
	
	return result;
}

float64_t CLinearHMM::get_log_derivative(int32_t num_param, int32_t num_example)
{
	int32_t len;
	bool free_vec;
	uint16_t* vector=((CStringFeatures<uint16_t>*) features)->
		get_feature_vector(num_example, len, free_vec);
	float64_t result=0;
	int32_t position=num_param/num_symbols;
	ASSERT(position>=0 && position<len);
	uint16_t sym=(uint16_t) (num_param-position*num_symbols);

	if (vector[position]==sym && transition_probs[num_param]!=0)
		result=1.0/transition_probs[num_param];
	((CStringFeatures<uint16_t>*) features)->
		free_feature_vector(vector, num_example, free_vec);

	return result;
}

void CLinearHMM::get_transition_probs(float64_t** dst, int32_t* num)
{
	*num=num_params;
	size_t sz=sizeof(*transition_probs)*(*num);
	*dst=(float64_t*) malloc(sz);
	ASSERT(dst);

	memcpy(*dst, transition_probs, sz);
}

bool CLinearHMM::set_transition_probs(const float64_t* src, int32_t num)
{
	if (num!=-1)
		ASSERT(num==num_params);

	if (!log_transition_probs)
		log_transition_probs=new float64_t[num_params];

	if (!transition_probs)
		transition_probs=new float64_t[num_params];

	for (int32_t i=0; i<num_params; i++)
	{
		transition_probs[i]=src[i];
		log_transition_probs[i]=log(transition_probs[i]);
	}

	return true;
}

void CLinearHMM::get_log_transition_probs(float64_t** dst, int32_t* num)
{
	*num=num_params;
	size_t sz=sizeof(*log_transition_probs)*(*num);
	*dst=(float64_t*) malloc(sz);
	ASSERT(dst);

	memcpy(*dst, log_transition_probs, sz);
}

bool CLinearHMM::set_log_transition_probs(const float64_t* src, int32_t num)
{
	if (num!=-1)
		ASSERT(num==num_params);

	if (!log_transition_probs)
		log_transition_probs=new float64_t[num_params];

	if (!transition_probs)
		transition_probs=new float64_t[num_params];

	for (int32_t i=0; i< num_params; i++)
	{
		log_transition_probs[i]=src[i];
		transition_probs[i]=exp(log_transition_probs[i]);
	}

	return true;
}




