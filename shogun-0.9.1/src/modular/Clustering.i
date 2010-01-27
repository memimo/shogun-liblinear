/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

%define DOCSTR
"The `Clustering` module gathers all clustering methods available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Clustering

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "Clustering_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%feature("autodoc", "get_radi(self) -> numpy 1dim array of float") get_radi;
%feature("autodoc", "get_centers(self) -> numpy 2dim array of float") get_centers;
%feature("autodoc", "get_merge_distance(self) -> [] of float") get_merge_distance;
%feature("autodoc", "get_pairs(self) -> [] of float") get_pairs;
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
#include <shogun/classifier/DistanceMachine.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/clustering/Hierarchical.h>
%}

/* Typemaps */
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** radii, int32_t* num)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** centers, int32_t* dim, int32_t* num)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** dist, int32_t* num)};
%apply (int32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(int32_t** tuples, int32_t* rows, int32_t* num)};

/* Remove C Prefix */
%rename(DistanceMachine) CDistanceMachine;
%rename(Hierarchical) CHierarchical;
%rename(KMeans) CKMeans;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/classifier/Classifier.h> 
%include <shogun/classifier/DistanceMachine.h>
%include <shogun/clustering/KMeans.h>
%include <shogun/clustering/Hierarchical.h>
