/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CUSTOMKERNEL_H___
#define _CUSTOMKERNEL_H___

#include "lib/Mathematics.h"
#include "lib/common.h"
#include "kernel/Kernel.h"
#include "features/Features.h"

namespace shogun
{
/** @brief The Custom Kernelallows for custom user provided kernel matrices.
 *
 * For squared training matrices it allows to store only the upper triangle of
 * the kernel to save memory: Full symmetric kernel matrices can be stored as
 * is or can be internally converted into (or directly given in) upper triangle
 * representation. Also note that values are stored as 32bit floats.
 *
 */
class CCustomKernel: public CKernel
{
	public:
		/** default constructor */
		CCustomKernel();

		/** constructor
		 *
		 * compute custom kernel from given kernel matrix
		 * @param k kernel matrix
		 */
		CCustomKernel(CKernel* k);

		/** constructor
		 *
		 * sets full kernel matrix from full kernel matrix
		 *
		 * @param km kernel matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		CCustomKernel(
			const float64_t* km, int32_t rows, int32_t cols);

		virtual ~CCustomKernel();

		/** initialize kernel with dummy features
		 *
		 * Kernels always need feature objects assigned. As the custom kernel
		 * does not really require this it creates some magic dummy features
		 * that only know about the number of vectors
		 *
		 * @param rows features of left-hand side
		 * @param cols features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool dummy_init(int32_t rows, int32_t cols);

		/** initialize kernel
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if initializing was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** clean up kernel */
		virtual void cleanup();

		/** return what type of kernel we are
		 *
		 * @return kernel type CUSTOM
		 */
		inline virtual EKernelType get_kernel_type() { return K_CUSTOM; }

		/** return feature type the kernel can deal with
		 *
		 * @return feature type ANY
		 */
		inline virtual EFeatureType get_feature_type() { return F_ANY; }

		/** return feature class the kernel can deal with
		 *
		 * @return feature class ANY
		 */
		inline virtual EFeatureClass get_feature_class() { return C_ANY; }

		/** return the kernel's name
		 *
		 * @return name Custom
		 */
		virtual const char* get_name() const { return "Custom"; }

		/** set kernel matrix (only elements from upper triangle)
		 * from elements of upper triangle (concat'd), including the
		 * main diagonal
		 *
		 * @param km kernel matrix
		 * @param len denotes the size of the array and should match len=cols*(cols+1)/2
		 * @return if setting was successful
		 */
		bool set_triangle_kernel_matrix_from_triangle(
			const float64_t* km, int32_t len);

		/** set kernel matrix (only elements from upper triangle)
		 * from squared matrix
		 *
		 * @param km kernel matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		bool set_triangle_kernel_matrix_from_full(
			const float64_t* km, int32_t rows, int32_t cols);

		/** set full kernel matrix from full kernel matrix
		 *
		 * @param km kernel matrix
		 * @param rows number of rows in matrix
		 * @param cols number of cols in matrix
		 * @return if setting was successful
		 */
		bool set_full_kernel_matrix_from_full(
			const float64_t* km, int32_t rows, int32_t cols);

		/** get number of vectors of lhs features
		 *
		 * @return number of vectors of left-hand side
		 */
		virtual inline int32_t get_num_vec_lhs()
		{
			return num_rows;
		}

		/** get number of vectors of rhs features
		 *
		 * @return number of vectors of right-hand side
		 */
		virtual inline int32_t get_num_vec_rhs()
		{
			return num_cols;
		}

		/** test whether features have been assigned to lhs and rhs
		 *
		 * @return true if features are assigned
		 */
		virtual inline bool has_features()
		{
			return (num_rows>0) && (num_cols>0);
		}

	protected:
		/** compute kernel function
		 *
		 * @param row row
		 * @param col col
		 * @return computed kernel function
		 */
		inline virtual float64_t compute(int32_t row, int32_t col)
		{
			ASSERT(kmatrix);

			if (upper_diagonal)
			{
				if (row <= col)
					return kmatrix[row*num_cols - row*(row+1)/2 + col];
				else
					return kmatrix[col*num_cols - col*(col+1)/2 + row];
			}
			else
				return kmatrix[row*num_cols+col];
		}

	private:
		/** only cleanup stuff specific to Custom kernel */
		void cleanup_custom();

	protected:
		/** kernel matrix */
		float32_t* kmatrix;
		/** number of rows */
		int32_t num_rows;
		/** number of columns */
		int32_t num_cols;
		/** upper diagonal */
		bool upper_diagonal;
};
}
#endif /* _CUSTOMKERNEL_H__ */
