/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "GUILabels.h"
#include "SGInterface.h"

#include <shogun/lib/config.h>
#include <shogun/lib/io.h>
#include <shogun/features/Labels.h>

#include <string.h>

using namespace shogun;

CGUILabels::CGUILabels(CSGInterface* ui_)
: CSGObject(), ui(ui_), train_labels(NULL), test_labels(NULL)
{
}

CGUILabels::~CGUILabels()
{
	SG_UNREF(train_labels);
	SG_UNREF(test_labels);
}

bool CGUILabels::load(char* filename, char* target)
{
	CLabels* labels=NULL;

	if (strncmp(target, "TEST", 4)==0)
		labels=test_labels;
	else if (strncmp(target, "TRAIN", 5)==0)
		labels=train_labels;
	else
		SG_ERROR("Invalid target %s.\n", target);

	if (labels)
	{
		SG_UNREF(labels);
		labels=new CLabels(filename);

		if (labels)
		{
			if (strncmp(target, "TEST", 4)==0)
				set_test_labels(labels);
			else
				set_train_labels(labels);

			return true;
		}
		else
			SG_ERROR("Loading labels failed.\n");
	}

	return false;
}

bool CGUILabels::save(char* param)
{
	bool result=false;
	return result;
}
