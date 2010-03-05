/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
%define REGRESSION_DOCSTR
"The `Regression` module gathers all regression methods available in the SHOGUN toolkit."
%enddef

%module(docstring=REGRESSION_DOCSTR) Regression

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "Regression_doxygen.i"
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
 #include <shogun/regression/Regression.h>
 #include <shogun/classifier/Classifier.h>
 #include <shogun/classifier/KernelMachine.h>
 #include <shogun/regression/KRR.h>
 #include <shogun/classifier/svm/SVM.h>
 #include <shogun/classifier/svm/LibSVM.h>
 #include <shogun/regression/svr/LibSVR.h>
 #include <shogun/classifier/mkl/MKL.h>
 #include <shogun/regression/svr/MKLRegression.h>
%}

/* Typemaps */
%apply (int32_t** ARGOUT1, int32_t* DIM1) {(int32_t** svs, int32_t* num)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** alphas, int32_t* d1)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* alphas, int32_t d)};
%apply (int32_t* IN_ARRAY1, int32_t DIM1) {(int32_t* svs, int32_t d)};

/* Remove C Prefix */
%rename(Regression) CRegression;
%rename(Classifier) CClassifier;
%rename(KernelMachine) CKernelMachine;
%rename(KRR) CKRR;
%rename(SVM) CSVM;
%rename(LibSVM) CLibSVM;
%rename(LibSVR) CLibSVR;
%rename(MKL) CMKL;
%rename(MKLRegression) CMKLRegression;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/regression/Regression.h>
%include <shogun/classifier/Classifier.h>
%include <shogun/classifier/KernelMachine.h>
%include <shogun/regression/KRR.h>
%include <shogun/classifier/svm/SVM.h>
%include <shogun/classifier/svm/LibSVM.h>
%include <shogun/regression/svr/LibSVR.h>
%include <shogun/classifier/mkl/MKL.h>
%include <shogun/regression/svr/MKLRegression.h>

#ifdef USE_SVMLIGHT
%{
 #include <shogun/classifier/svm/SVM_light.h>
%}

%rename(SVMLight) CSVMLight;

%ignore VERSION;
%ignore VERSION_DATE;
%ignore MAXSHRINK;
%ignore SHRINK_STATE;
%ignore MODEL;
%ignore LEARN_PARM;
%ignore TIMING;

%include <shogun/classifier/svm/SVM_light.h>
%{
 #include <shogun/regression/svr/SVR_light.h>
%}

%rename(SVRLight) CSVRLight;

%include <shogun/regression/svr/SVR_light.h>
#endif //USE_SVMLIGHT
