/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Written (W) 2009 Marius Kloft
 * Copyright (C) 2009 TU Berlin and Max-Planck-Society
 */

#ifndef _SCATTERSVM_H___
#define _SCATTERSVM_H___

#include "lib/common.h"
#include "classifier/svm/MultiClassSVM.h"
#include "classifier/svm/SVM_libsvm.h"

#include <stdio.h>

namespace shogun
{
/** @brief ScatterSVM - Multiclass SVM
 *
 * The ScatterSVM is an unpublished experimental
 * true multiclass SVM. Details are availabe
 * in the following technical report.
 *
 * Robert Jenssen and Marius Kloft and Alexander Zien and S\"oren Sonnenburg and
 *           Klaus-Robert M\"{u}ller,
 * A Multi-Class Support Vector Machine Based on Scatter Criteria, TR 014-2009
 * TU Berlin, 2009
 *
 * */
class CScatterSVM : public CMultiClassSVM
{
	public:
		/** constructor */
		CScatterSVM();
		/** constructor
		 *
		 * @param C constant C
		 * @param k kernel
		 * @param lab labels
		 */
		CScatterSVM(float64_t C, CKernel* k, CLabels* lab);

		/** default destructor */
		virtual ~CScatterSVM();

		/** train SVM classifier
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);

		/** get classifier type
		 *
		 * @return classifier type LIBSVM
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_SCATTERSVM; }

		/** classify one example
		 *
		 * @param num number of example to classify
		 * @return resulting classification
		 */
		virtual float64_t classify_example(int32_t num);

		/** classify one vs rest
		 *
		 * @return resulting labels
		 */
		virtual CLabels* classify_one_vs_rest();

		/** @return object name */
		inline virtual const char* get_name() const { return "ScatterSVM"; }

	private:
		void compute_norm_wc();

	protected:
		/** SVM problem */
		svm_problem problem;
		/** SVM param */
		svm_parameter param;

		/** SVM model */
		struct svm_model* model;

		/** norm of w_c */
		float64_t* norm_wc;

		/** norm of w_cw */
		float64_t* norm_wcw;

		/** ScatterSVM rho */
		float64_t rho;
};
}
#endif // ScatterSVM
