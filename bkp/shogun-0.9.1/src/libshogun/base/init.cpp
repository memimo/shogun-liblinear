/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "base/init.h"
#include "lib/Mathematics.h"
#include "lib/memory.h"
#include "lib/Set.h"
#include "base/Parallel.h"
#include "base/Version.h"

namespace shogun
{
	CParallel* sg_parallel=NULL;
	CIO* sg_io=NULL;
	CVersion* sg_version=NULL;
	CMath* sg_math=NULL;
#ifdef TRACE_MEMORY_ALLOCS
	CSet<CMemoryBlock>* sg_mallocs=NULL;
#endif

	/// function called to print normal messages
	void (*sg_print_message)(FILE* target, const char* str) = NULL;

	/// function called to print warning messages
	void (*sg_print_warning)(FILE* target, const char* str) = NULL;

	/// function called to print error messages
	void (*sg_print_error)(FILE* target, const char* str) = NULL;

	/// function called to cancel things
	void (*sg_cancel_computations)(bool &delayed, bool &immediately)=NULL;

	void init_shogun(void (*print_message)(FILE* target, const char* str),
			void (*print_warning)(FILE* target, const char* str),
			void (*print_error)(FILE* target, const char* str),
			void (*cancel_computations)(bool &delayed, bool &immediately))
	{
		if (!sg_io)
			sg_io = new shogun::CIO();
		if (!sg_parallel)
			sg_parallel=new shogun::CParallel();
		if (!sg_version)
			sg_version = new shogun::CVersion();
		if (!sg_math)
			sg_math = new shogun::CMath();
#ifdef TRACE_MEMORY_ALLOCS
		if (!sg_mallocs)
			sg_mallocs = new shogun::CSet<CMemoryBlock>();

		SG_REF(sg_mallocs);
#endif
		SG_REF(sg_io);
		SG_REF(sg_parallel);
		SG_REF(sg_version);
		SG_REF(sg_math);

		sg_print_message=print_message;
		sg_print_warning=print_warning;
		sg_print_error=print_error;
		sg_cancel_computations=cancel_computations;
	}

	void exit_shogun()
	{
		sg_print_message=NULL;
		sg_print_warning=NULL;
		sg_print_error=NULL;
		sg_cancel_computations=NULL;

		SG_UNREF(sg_math);
		SG_UNREF(sg_version);
		SG_UNREF(sg_parallel);
		SG_UNREF(sg_io);

		// will leak memory alloc statistics on exit
	}
}
